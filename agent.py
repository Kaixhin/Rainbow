import os
import random
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F

from memory import Transition
from model import DQN


class Agent():
  def __init__(self, args, env):
    self.action_space = env.action_space()
    self.batch_size = args.batch_size
    self.discount = args.discount
    self.max_gradient_norm = args.max_gradient_norm

    self.policy_net = DQN(args, self.action_space)
    if args.model and os.path.isfile(args.model):
      self.policy_net.load_state_dict(torch.load(args.model))
    self.policy_net.train()

    self.target_net = DQN(args, self.action_space)
    self.update_target_net()
    self.target_net.eval()

    self.optimiser = optim.Adam(self.policy_net.parameters(), lr=args.lr)

  def act(self, state, epsilon):
    if random.random() > epsilon:
      return self.policy_net(state.unsqueeze(0)).max(1)[1].data[0]
    else:
      return random.randint(0, self.action_space - 1)

  def learn(self, mem):
    transitions = mem.sample(self.batch_size)
    batch = Transition(*zip(*transitions))  # Transpose the batch

    states = Variable(torch.stack(batch.state, 0))
    actions = Variable(torch.LongTensor(batch.action).unsqueeze(1))
    rewards = Variable(torch.Tensor(batch.reward))
    non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))  # Only process non-terminal next states
    next_states = Variable(torch.stack(tuple(s for s in batch.next_state if s is not None), 0), volatile=True)  # Prevent backpropagating through expected action values

    Qs = self.policy_net(states).gather(1, actions)  # Q(s_t, a_t; θpolicy)
    next_state_argmax_indices = self.policy_net(next_states).max(1, keepdim=True)[1]  # Perform argmax action selection using policy network: argmax_a[Q(s_t+1, a; θpolicy)]
    Qns = Variable(torch.zeros(self.batch_size))  # Q(s_t+1, a) = 0 if s_t+1 is terminal
    Qns[non_final_mask] = self.target_net(next_states).gather(1, next_state_argmax_indices)  # Q(s_t+1, argmax_a[Q(s_t+1, a; θpolicy)]; θtarget)
    Qns.volatile = False  # Remove volatile flag to prevent propagating it through loss
    target = rewards + (self.discount * Qns)  # Double-Q target: Y = r + γ.Q(s_t+1, argmax_a[Q(s_t+1, a; θpolicy)]; θtarget)

    loss = F.smooth_l1_loss(Qs, target)  # Huber loss on TD-error δ: δ = Y - Q(s_t, a_t)
    # TODO: TD-error clipping?
    self.policy_net.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm(self.policy_net.parameters(), self.max_gradient_norm)  # Clamp gradients
    self.optimiser.step()

  def update_target_net(self):
    self.target_net.load_state_dict(self.policy_net.state_dict())

  def save(self, path):
    torch.save(self.policy_net.state_dict(), os.path.join(path, 'model.pth'))

  def evaluate_q(self, state):
    return self.policy_net(state.unsqueeze(0)).max(1)[0].data[0]

  def train(self):
    self.policy_net.train()

  def eval(self):
    self.policy_net.eval()
