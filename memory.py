import math
import random
from collections import deque
import torch
from torch.autograd import Variable


# Segment tree data structure where parent node values are sum of children node values
class SumTree():
  def __init__(self, size):
    # Tree structure (represented by an array)
    self.tree = [0] * 2 ** (math.ceil(math.log(size, 2)) + 1)  # Initialise with zeros as no priorities at start to build tree
    self.leaf_width = len(self.tree) // 2 - 1  # Width of bottom layer (leaves)

  def _query(self, node, i, j, left, right):
    if left >= i and right <= j:
      return self.tree[node]
    elif j < left or i > right:
      return None
    else:
      left_child, right_child, middle = 2 * node, 2 * node + 1, (left + right) // 2
      left = self._query(left_child, i, j, left, middle)
      right = self._query(right_child, i, j, middle + 1, right)
      if left is None and right is None:
        return 0  # TODO: Return 0, INF or None?
      elif left is not None and right is not None:
        return left + right
      elif left is None:
        return right
      else:
        return left

  def query(self, i, j):
    return self._query(0, i, j, 0, self.leaf_width)  # Recursively search from root

  def _update(self, node, i, value, left, right):
    if left == i and right == i:
      self.tree[node] = value  # Update leaf node
    elif i < left or i > right:
      return
    else:
      left_child, right_child, middle = 2 * node, 2 * node + 1, (left + right) // 2
      self._update(left_child, i, value, left, middle)  # Update left child
      self._update(right_child, i, value, middle + 1, right)  # Update right child
      self.tree[node] = self.tree[left_child] + self.tree[right_child]  # Update parent as sum of children

  def update(self, i, value):
    self._update(0, i, value, 0, self.leaf_width)  # Recursively update from root


# Cyclic buffer (where appends wrap around so that the index of only one element changes)
class CyclicBuffer():
  def __init__(self, size):
    self.size = size
    self.index = 0
    self.arr = [0] * size  # Use "empty" integer array
    # TODO: Deal with insertion of element with max priority more efficiently

  def append(self, item):
    index = self.index
    self.arr[index] = item
    self.index = (index + 1) % self.size  # Index for next item wraps around to 0
    return index  # Return index utilised

  def __getitem__(self, key):
    return self.arr[key]
  """
  # Perform indexing as if underlying implementation were deque (no need for setting as only append used)
  def __getitem__(self, key):
    return (self.arr[self.index:] + self.arr[:self.index])[key]  # Creates new array at each call, very expensive
  """


class ReplayMemory():
  def __init__(self, args, capacity, prioritised=False):
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

    # Set up prioritised experience replay if needed
    self.prioritised = prioritised
    if prioritised:
      self.priorities = CyclicBuffer(capacity)  # Store priorities in a way that only the index of one element is updated per append
      self.sum_tree = SumTree(capacity)

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
      if self.prioritised:
        p_index = self.priorities.append(0)  # Store zero priority
        self.sum_tree.update(p_index, 0)

  def append(self, state, action, reward):
    # Add state, action and reward at time t
    self.timesteps.append(self.t)
    self.states.append(state[-1].mul(255).byte().cpu())  # Only store last frame and discretise to save memory
    self.actions.append(action)
    self.rewards.append(reward)  # Technically from time t + 1, but kept at t for all buffers to be in sync
    self.nonterminals.append(True)
    self.t += 1
    if self.prioritised:
      max_priority = max(max(self.priorities), 1e-8)  # Use max of max priority or small constant
      p_index = self.priorities.append(max_priority)  # Store new transition with maximum priority
      self.sum_tree.update(p_index, max_priority)

  # Add empty state at end of episode
  def postappend(self):
    self.timesteps.append(self.t)
    self.states.append(torch.ByteTensor(84, 84).zero_())  # Add blank state (used to replace terminal state)
    self.actions.append(None)
    self.rewards.append(None)
    self.nonterminals.append(False)
    if self.prioritised:
      p_index = self.priorities.append(0)  # Store zero priority
      self.sum_tree.update(p_index, 0)

  def sample(self, batch_size):
    # Find indices based on sampling from a probability distribution defined by normalised priorities
    p_total = self.sum_tree.tree[0]  # Sum over all priorities stored in root node
    p_bins = torch.linspace(0, p_total, batch_size + 1)  # Create a batch size + 1 number of equally-sized bins
    probs = [random.uniform(p_bins[i], p_bins[i + 1]) for i in range(batch_size)]  # Sample values uniformly from this range (unnormalised probabilities)
    print(probs)
    inds = [self.sum_tree.query(p, p) for p in probs]  # TODO: Retrieve indices of corresponding transitions
    print(inds)
    quit()
    probs = torch.Tensor(probs) / p_total  # Calculate normalised probabilities
    weights = (self.capacity * probs) ** -self.priority_weight  # Compute importance-sampling weights
    weights = weights / weights.max()   # Normalise by max weight

    """
    # Find indices for valid samples
    valid = list(map(lambda x: x >= 0, self.timesteps))  # Valid frames by timestep
    # TODO: Alternative is to allow terminal states (- n+1) but truncate multi-step returns appropriately
    valid = [a and b for a, b in zip(valid, valid[self.n:] + [False] * self.n)]  # Cannot use terminal states (- n+1)/state at end of memory
    valid[:self.history - 1] = [False] * (self.history - 1)  # Cannot form stack from initial frames
    inds = random.sample([i for i, v in zip(range(len(valid)), valid) if v], batch_size)
    """

    # Create stack of states and nth next states
    state_stack, next_state_stack = [], []
    for h in reversed(range(self.history)):
      state_stack.append(torch.stack([self.states[i - h] for i in inds], 0))
      next_state_stack.append(torch.stack([self.states[i + self.n - h] for i in inds], 0))  # nth next state
    states = Variable(torch.stack(state_stack, 1).type(self.dtype_float).div_(255))  # Un-discretise
    next_states = Variable(torch.stack(next_state_stack, 1).type(self.dtype_float).div_(255), volatile=True)

    actions = self.dtype_long([self.actions[i] for i in inds])

    # Calculate truncated n-step discounted return R^n = Σ_k=0->n-1 (γ^k)R_t+k+1
    returns = [self.rewards[i] for i in inds]
    for n in range(1, self.n):
      returns = [R + self.discount ** n * self.rewards[i + n] for R, i in zip(returns, inds)]
    returns = self.dtype_float(returns)

    nonterminals = self.dtype_float([self.nonterminals[i + self.n] for i in inds]).unsqueeze(1)  # Mask for non-terminal nth next states

    return inds, states, actions, returns, next_states, nonterminals, weights

  def update_priorities(self, inds, priorities):
    [self.sum_tree.update(i, priority) for i, priority in zip(inds, priorities)]

  # Set up internal state for iterator
  def __iter__(self):
    # Find indices for valid samples
    valid = list(map(lambda x: x >= 0, self.timesteps))  # Valid frames by timestep
    valid = [a and b for a, b in zip(valid, valid[1:] + [False])]  # Cannot use terminal states/state at end of memory
    valid[:self.history - 1] = [False] * (self.history - 1)  # Cannot form stack from initial frames
    self.current_ind = 0
    self.valid_inds = [i for i, v in zip(range(len(valid)), valid) if v]
    return self

  # Return valid states for validation
  def __next__(self):
    if self.current_ind == len(self.valid_inds):
      raise StopIteration
    # Create stack of states and nth next states
    state_stack = []
    for h in reversed(range(self.history)):
      state_stack.append(self.states[self.valid_inds[self.current_ind - h]])
    state = Variable(torch.stack(state_stack, 0).type(self.dtype_float).div_(255), volatile=True)  # Agent will turn into batch
    self.current_ind += 1
    return state
