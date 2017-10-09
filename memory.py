import random
from collections import deque, namedtuple


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


# TODO: Discretise memory?
class ReplayMemory():
  def __init__(self, capacity):
    self.capacity = capacity
    self.memory = deque([], maxlen=capacity)

  def append(self, *args):
    self.memory.append(Transition(*args))

  def sample(self, batch_size):
    return random.sample(self.memory, batch_size)

  def __len__(self):
    return len(self.memory)

  def __getitem__(self, key):
    return self.memory[key]

# TODO: Prioritised experience replay memory
