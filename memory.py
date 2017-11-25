import math
import random
from collections import deque
import torch
from torch.autograd import Variable


class ReplayMemory():
  def __init__(self, args, capacity, prioritised=False):
    self.dtype_byte = torch.cuda.ByteTensor if args.cuda else torch.ByteTensor
    self.dtype_long = torch.cuda.LongTensor if args.cuda else torch.LongTensor
    self.dtype_float = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor

    self.capacity = capacity
    self.history = args.history_length
    self.discount = args.discount
    self.n = args.multi_step
    self.priority_exponent = args.priority_exponent
    self.priority_weight = args.priority_weight  # Initial value, annealed to 1 over course of training
    self.t = 0  # Internal episode timestep counter
    self.states = deque([], maxlen=capacity)
    self.actions = deque([], maxlen=capacity)
    self.rewards = deque([], maxlen=capacity)
    self.timesteps = deque([], maxlen=capacity)
    self.nonterminals = deque([], maxlen=capacity)  # Non-terminal states

    # Set up prioritised experience replay if needed
    self.prioritised = prioritised
    if prioritised:
      self.priorities = deque([0] * capacity, maxlen=capacity)
      self.sum_tree = SegmentTree(self.priorities)

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

  def append(self, state, action, reward):
    # Add state, action and reward at time t
    self.timesteps.append(self.t)
    self.states.append(state[-1].mul(255).byte().cpu())  # Only store last frame and discretise to save memory
    self.actions.append(action)
    self.rewards.append(reward)  # Technically from time t + 1, but kept at t for all buffers to be in sync
    self.nonterminals.append(True)
    self.t += 1

  # Add empty state at end of episode
  def postappend(self):
    self.timesteps.append(self.t)
    self.states.append(torch.ByteTensor(84, 84).zero_())  # Add blank state (used to replace terminal state)
    self.actions.append(None)
    self.rewards.append(None)
    self.nonterminals.append(False)

  def sample(self, batch_size):
    # Find indices for valid samples
    valid = list(map(lambda x: x >= 0, self.timesteps))  # Valid frames by timestep
    # TODO: Alternative is to allow terminal states (- n+1) but truncate multi-step returns appropriately
    valid = [a and b for a, b in zip(valid, valid[self.n:] + [False] * self.n)]  # Cannot use terminal states (- n+1)/state at end of memory
    valid[:self.history - 1] = [False] * (self.history - 1)  # Cannot form stack from initial frames
    inds = random.sample([i for i, v in zip(range(len(valid)), valid) if v], batch_size)

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

    return states, actions, returns, next_states, nonterminals

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


# TODO: Prioritised experience replay memory; disable if not needed (for validation memory)
class SegmentTree():
  def __init__(self, array):
    self.tree = [None] * 2 ** (math.ceil(math.log(len(array), 2)) + 1)  # Tree structure (represented by an array)
    self.leaf_width = len(self.tree) // 2 - 1  # Width of bottom layer (leaves)
    self._build(1, 0, self.leaf_width, array)  # Build tree

  def _build(self, node, left, right, array):
    print(node, left, right)
    if left == right:
      try:
        self.tree[node] = array[left]  # Leaf node is raw value
      except IndexError:
        self.tree[node] = 0  # TODO: Set to INF?
    else:
      left_child, right_child, middle = 2 * node, 2 * node + 1, (left + right) // 2
      self._build(left_child, left, middle, array)  # Recurse on left
      self._build(right_child, middle + 1, right, array)  # Recurse on right
      self.tree[node] = self.tree[left_child] + self.tree[right_child]  # Internal node is sum of its children

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
        return None  # TODO: Return INF?
      elif left is not None and right is not None:
        return min(left, right)
      elif left is None:
        return right
      else:
        return left

  def query(self, i, j):
    return self._query(1, i, j, 0, self.leaf_width)

  def _update(self, node, i, value, left, right):
    if left == i and right == i:
      self.tree[node] = value
    elif i < left or i > right:
      return
    else:
      left_child, right_child, middle = 2 * node, 2 * node + 1, (left + right) // 2
      self._update(left_child, i, value, left, middle)
      self._update(right_child, i, value, middle + 1, right)

  def update(self, i, value):
    self._update(1, i, value, 0, self.leaf_width)
