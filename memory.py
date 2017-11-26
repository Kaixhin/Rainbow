import random
from collections import deque
import torch
from torch.autograd import Variable


# Segment tree data structure where parent node values are sum of children node values
class SumTree():
  def __init__(self, size):
    self.index = 0
    self.size = size
    self.tree = [0] * (2 * size - 1)  # Initialise tree with zeros as no priorities at start to build tree
    self.data = [0] * size  # Wrap-around cyclic buffer
    self.max = 0  # Store max value for fast retrieval

  def _propagate(self, index, update):
    parent = (index - 1) // 2
    self.tree[parent] += update  # Propagate change in value rather than absolute value
    if parent != 0:
      self._propagate(parent, update)

  def update(self, index, value):
    update = value - self.tree[index]
    self.tree[index] = value  # Set new value
    self._propagate(index, update)  # Propagate change
    self.max = max(self.max, value)  # Update max

  def append(self, value):
    self.data[self.index] = value  # Store value in underlying data structure
    self.update(self.index + self.size - 1, value)  # Update tree
    self.index = (self.index + 1) % self.size  # Update index

  # Searches for the location of a value
  def _retrieve(self, index, value):
    left, right = 2 * index + 1, 2 * index + 2
    if left >= len(self.tree):
      return index
    elif value <= self.tree[left]:
      return self._retrieve(left, value)
    else:
      return self._retrieve(right, value - self.tree[left])

  # Searches for a value and returns location and other data
  def get(self, value):
    index = self._retrieve(0, value)  # Search for index of item from root
    data_index = index - self.size + 1
    return (data_index, index, self.tree[index])  # Return data index, tree index, priority

  def total(self):
    return self.tree[0]


class ReplayMemory():
  def __init__(self, args, capacity):
    self.dtype_byte = torch.cuda.ByteTensor if args.cuda else torch.ByteTensor
    self.dtype_long = torch.cuda.LongTensor if args.cuda else torch.LongTensor
    self.dtype_float = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor

    self.capacity = capacity
    self.history = args.history_length
    self.discount = args.discount
    self.n = args.multi_step
    self.priority_weight = args.priority_weight  # Initial importance sampling weight β, annealed to 1 over course of training
    self.t = 0  # Internal episode timestep counter
    self.states = deque([], maxlen=capacity)
    self.actions = deque([], maxlen=capacity)
    self.rewards = deque([], maxlen=capacity)
    self.timesteps = deque([], maxlen=capacity)
    self.nonterminals = deque([], maxlen=capacity)  # Non-terminal states
    self.priorities = SumTree(capacity)  # Store priorities in a wrap-around cyclic buffer within a sum tree

  # Add empty states to prepare for new episode
  def preappend(self):
    self.t = 0
    # Blank transitions from before episode
    for h in range(-self.history + 1, 0):
      self.timesteps.append(h)
      self.states.append(torch.ByteTensor(84, 84).zero_())  # Add blank state
      self.actions.append(None)
      self.rewards.append(None)
      self.nonterminals.append(True)
      self.priorities.append(0)  # Store zero priority

  def append(self, state, action, reward):
    # Add state, action and reward at time t
    self.timesteps.append(self.t)
    self.states.append(state[-1].mul(255).byte().cpu())  # Only store last frame and discretise to save memory
    self.actions.append(action)
    self.rewards.append(reward)  # Technically from time t + 1, but kept at t for all buffers to be in sync
    self.nonterminals.append(True)
    self.t += 1
    self.priorities.append(max(self.priorities.max, 1))  # Store new transition with maximum priority (or use initial priority 1)

  # Add empty state at end of episode
  def postappend(self):
    self.timesteps.append(self.t)
    self.states.append(torch.ByteTensor(84, 84).zero_())  # Add blank state (used to replace terminal state)
    self.actions.append(None)
    self.rewards.append(None)
    self.nonterminals.append(False)
    self.priorities.append(0)  # Store zero priority

  def sample(self, batch_size):
    p_total = self.priorities.total()  # Retrieve sum of all priorities (used to create a normalised probability distribution)
    segment = p_total / batch_size  # Batch size number of segments, based on sum over all probabilities
    samples = [random.uniform(i * segment, (i + 1) * segment) for i in range(batch_size)]  # Uniformly sample an element from each segment
    batch = [self.priorities.get(s) for s in samples]  # Retrieve samples from tree
    idxs, tree_idxs, probs = zip(*batch)  # Unpack data indices, tree indices, unnormalised probabilities (priorities)
    probs = torch.Tensor(probs) / p_total  # Calculate normalised probabilities
    weights = (self.capacity * probs) ** -self.priority_weight  # Compute importance-sampling weights w
    weights = weights / weights.max()   # Normalise by max importance-sampling weight

    # TODO: Convert buffer idxs into deque idxs
    # TODO: Make sure samples are valid
    """
    # Find indices for valid samples
    valid = list(map(lambda x: x >= 0, self.timesteps))  # Valid frames by timestep
    # TODO: Alternative is to allow terminal states (- n+1) but truncate multi-step returns appropriately
    valid = [a and b for a, b in zip(valid, valid[self.n:] + [False] * self.n)]  # Cannot use terminal states (- n+1)/state at end of memory
    valid[:self.history - 1] = [False] * (self.history - 1)  # Cannot form stack from initial frames
    idxs = random.sample([i for i, v in zip(range(len(valid)), valid) if v], batch_size)
    """

    # Create stack of states and nth next states
    state_stack, next_state_stack = [], []
    for h in reversed(range(self.history)):
      state_stack.append(torch.stack([self.states[i - h] for i in idxs], 0))
      next_state_stack.append(torch.stack([self.states[i + self.n - h] for i in idxs], 0))  # nth next state
    states = Variable(torch.stack(state_stack, 1).type(self.dtype_float).div_(255))  # Un-discretise
    next_states = Variable(torch.stack(next_state_stack, 1).type(self.dtype_float).div_(255), volatile=True)

    actions = self.dtype_long([self.actions[i] for i in idxs])

    # Calculate truncated n-step discounted return R^n = Σ_k=0->n-1 (γ^k)R_t+k+1
    returns = [self.rewards[i] for i in idxs]
    for n in range(1, self.n):
      returns = [R + self.discount ** n * self.rewards[i + n] for R, i in zip(returns, idxs)]
    returns = self.dtype_float(returns)

    nonterminals = self.dtype_float([self.nonterminals[i + self.n] for i in idxs]).unsqueeze(1)  # Mask for non-terminal nth next states

    return tree_idxs, states, actions, returns, next_states, nonterminals, weights

  def update_priorities(self, idxs, priorities):
    [self.sum_tree.update(idx, priority) for idx, priority in zip(idxs, priorities)]

  # Set up internal state for iterator
  def __iter__(self):
    # Find indices for valid samples
    valid = list(map(lambda x: x >= 0, self.timesteps))  # Valid frames by timestep
    valid = [a and b for a, b in zip(valid, valid[1:] + [False])]  # Cannot use terminal states/state at end of memory
    valid[:self.history - 1] = [False] * (self.history - 1)  # Cannot form stack from initial frames
    self.current_idx = 0
    self.valid_idxs = [i for i, v in zip(range(len(valid)), valid) if v]
    return self

  # Return valid states for validation
  def __next__(self):
    if self.current_idx == len(self.valid_idxs):
      raise StopIteration
    # Create stack of states and nth next states
    state_stack = []
    for h in reversed(range(self.history)):
      state_stack.append(self.states[self.valid_idxs[self.current_idx - h]])
    state = Variable(torch.stack(state_stack, 0).type(self.dtype_float).div_(255), volatile=True)  # Agent will turn into batch
    self.current_idx += 1
    return state
