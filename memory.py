import random
from collections import deque, namedtuple


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


# TODO: Discretise memory?
class ReplayMemory():
  def __init__(self, capacity, history_length, multi_step):
    self.capacity = capacity
    self.history = history_length
    self.n = multi_step
    self.t = 0  # Internal episode timestep counter
    self.states = deque([], maxlen=capacity)
    self.actions = deque([], maxlen=capacity)
    self.rewards = deque([], maxlen=capacity)
    self.timesteps = deque([], maxlen=capacity)

  # Add empty states to prepare for new episode
  def preappend(self):
    self.t = 0
    # Blank transitions from before episode
    for h in range(-self.history + 1, 0):
      self.timesteps.append(h)
      self.states.append(None)  # TODO: Blank state
      self.actions.append(None)
      self.rewards.append(None)

  def append(self, state, action, reward):
    # Add state, action and reward at time t
    self.timesteps.append(self.t)
    self.states.append(state)
    self.actions.append(action)
    self.rewards.append(reward)  # Technically from time t + 1, but kept at t for all buffers to be in sync
    self.t += 1

  def sample(self, batch_size):
    # Find indices for valid samples
    valid = list(map(lambda x: x >= 0, self.timesteps))  # Valid frames by timestep
    valid = [a and b for a, b in zip(valid, valid[1:] + [False])]  # Cannot use terminal states/state at end of memory
    valid[:self.history - 1] = [False] * (self.history - 1)  # Cannot form stack from initial frames
    inds = random.sample([i for i, v in zip(range(len(valid)), valid) if v], batch_size)
    states = [self.states[i] for i in inds]
    actions = [self.actions[i] for i in inds]
    rewards = [self.rewards[i] for i in inds]
    next_states = [self.states[i + 1] for i in inds]
    return states, actions, rewards, next_states

# TODO: Prioritised experience replay memory
