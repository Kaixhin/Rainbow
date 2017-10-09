import argparse
import random
import gym
import torch
from torch.autograd import Variable

from agent import Agent
from memory import ReplayMemory
from test import test
from utils import state_to_tensor


parser = argparse.ArgumentParser(description='Rainbow')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--game', type=str, default='Pong', help='ATARI game')
parser.add_argument('--T-max', type=int, default=50000, metavar='STEPS', help='Number of training steps')  # TODO: 5e7
parser.add_argument('--max-episode-length', type=int, default=100, metavar='LENGTH', help='Max episode length')
parser.add_argument('--history-length', type=int, default=4, metavar='T', help='Number of consecutive states processed')  # TODO: Cyclic buffer
parser.add_argument('--hidden-size', type=int, default=256, metavar='SIZE', help='Network hidden size')
parser.add_argument('--model', type=str, metavar='PARAMS', help='Pretrained model (state dict)')
parser.add_argument('--memory-capacity', type=int, default=10000, metavar='CAPACITY', help='Experience replay memory capacity')  # TODO: 1e6
parser.add_argument('--replay-frequency', type=int, default=4, metavar='k', help='Frequency of sampling from memory')
# TODO: Memory prioritisation (w/ alpha and beta hyperparams)?
parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
parser.add_argument('--epsilon-start', type=float, default=1, metavar='ε', help='Initial value of greediness')
parser.add_argument('--epsilon-end', type=float, default=0.01, metavar='ε', help='Final value of greediness')
parser.add_argument('--epsilon-steps', type=int, default=10000, metavar='STEPS', help='Number of steps over which to decay greediness')  # TODO: 1e6
parser.add_argument('--target-update', type=int, default=1000, metavar='τ', help='Number of steps after which to update target network')  # TODO: 30000
parser.add_argument('--reward-clip', type=int, default=1, metavar='VALUE', help='Reward clipping (0 to disable)')
parser.add_argument('--lr', type=float, default=0.00025, metavar='η', help='Learning rate')
parser.add_argument('--batch-size', type=int, default=32, metavar='SIZE', help='Batch size')  # Assumed to be < learn_start
parser.add_argument('--learn-start', type=int, default=1000, metavar='STEPS', help='Number of steps before starting training')  # TODO: 5e4
parser.add_argument('--max-gradient-norm', type=float, default=10, metavar='VALUE', help='Max value of gradient L2 norm for gradient clipping')
parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
parser.add_argument('--evaluation-interval', type=int, default=1000, metavar='STEPS', help='Number of training steps between evaluations')  # TODO: 25000
parser.add_argument('--evaluation-episodes', type=int, default=10, metavar='N', help='Number of evaluation episodes to average over')
parser.add_argument('--evaluation-size', type=int, default=500, metavar='N', help='Number of transitions to use for validating Q')  # TODO: Val replay mem
parser.add_argument('--render', action='store_true', help='Render evaluation agent')


# Setup
args = parser.parse_args()
print(' ' * 26 + 'Options')
for k, v in vars(args).items():
  print(' ' * 26 + k + ': ' + str(v))
torch.manual_seed(args.seed)
env = gym.make(args.game + 'Deterministic-v4')  # TODO: args.max_episode_length
env.seed(args.seed)


# Agent
action_size = 4  # TODO: Pass properly
dqn = Agent(args)
mem = ReplayMemory(args.memory_capacity)


# TODO: History buffer
# Training setup
epsilon = args.epsilon_start
epsilon_decrease = (args.epsilon_start - args.epsilon_end) / args.epsilon_steps
T, done = 0, True

# Construct validation memory
val_mem = ReplayMemory(args.evaluation_size)
while T < args.evaluation_size:
  if done:
    state, done = state_to_tensor(env.reset()), False

  next_state, _, done, _ = env.step(random.randint(0, action_size - 1))
  T += 1
  val_mem.append(state, None, None, None)
  state = state_to_tensor(next_state)


if args.evaluate:
  dqn.eval()  # Set DQN (policy network) to evaluation mode
  avg_reward, avg_Q = test(args, 0, dqn, val_mem, evaluate=True)  # Test
  print('Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_Q))
else:
  # Training loop
  T, done = 0, True
  while T < args.T_max:
    if done:
      state, done = Variable(state_to_tensor(env.reset())), False

    action = dqn.act(state, epsilon)  # Choose an action with ε-greedy

    next_state, reward, done, _ = env.step(action)  # Step
    next_state = state_to_tensor(next_state)
    if args.reward_clip > 0:
      reward = max(min(reward, args.reward_clip), -args.reward_clip)  # Clip rewards
    T += 1

    mem.append(state.data, action, next_state, reward)  # Append transition to memory

    # Train and test
    if T >= args.learn_start:
      # Only decay greediness ε once learning has started
      epsilon = max(epsilon - epsilon_decrease, args.epsilon_end)  # Decay greediness ε

      if T % args.replay_frequency == 0:
        dqn.learn(mem)  # Train with double-Q learning

      if T % args.evaluation_interval == 0:
        dqn.eval()  # Set DQN (policy network) to evaluation mode
        avg_reward, avg_Q = test(args, T, dqn, val_mem)  # Test
        print('Evaluation @ T=' + str(T) + ' | ε: ' + str(epsilon) + ' | Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_Q))
        dqn.train()  # Set DQN (policy network) back to training mode

    # Update target network
    if T % args.target_update == 0:
      dqn.update_target_net()

    state = Variable(next_state)

env.close()
