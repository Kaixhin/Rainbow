import random
from collections import namedtuple
import torch
from torch.autograd import Variable


Transition = namedtuple('Transition', ('timestep', 'state', 'action', 'reward', 'nonterminal'))


# Segment tree data structure where parent node values are sum/max of children node values
class SegmentTree():
  def __init__(self, size):
    self.index = 0
    self.size = size
    self.full = False  # Used to track actual capacity
    self.sum_tree = [0] * (2 * size - 1)  # Initialise fixed size tree with all (priority) zeros
    self.max_tree = [1] * (2 * size - 1)  # Initialise with all ones (incorrect as max will always be >= 1, but sum tree is used for sampling)
    self.data = [Transition(-1, torch.ByteTensor(84, 84).zero_(), None, 0, True)] * size  # Wrap-around cyclic buffer filled with (zero-priority) blank transitions

  # Propagates value up tree given a tree index
  def _propagate(self, index, value):
    parent = (index - 1) // 2
    left, right = 2 * parent + 1, 2 * parent + 2
    self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
    self.max_tree[parent] = max(self.max_tree[left], self.max_tree[right])
    if parent != 0:
      self._propagate(parent, value)

  # Updates value given a tree index
  def update(self, index, value):
    self.sum_tree[index], self.max_tree[index] = value, value  # Set new value
    self._propagate(index, value)  # Propagate value

  def append(self, data, value):
    self.data[self.index] = data  # Store data in underlying data structure
    self.update(self.index + self.size - 1, value)  # Update tree
    self.index = (self.index + 1) % self.size  # Update index
    self.full = self.full or self.index == 0  # Save when capacity reached

  # Searches for the location of a value in sum tree
  def _retrieve(self, index, value):
    left, right = 2 * index + 1, 2 * index + 2
    if left >= len(self.sum_tree):
      return index
    elif value <= self.sum_tree[left]:
      return self._retrieve(left, value)
    else:
      return self._retrieve(right, value - self.sum_tree[left])

  # Searches for a value in sum tree and returns value, data index and tree index
  def find(self, value):
    index = self._retrieve(0, value)  # Search for index of item from root
    data_index = index - self.size + 1
    return (self.sum_tree[index], data_index, index)  # Return value, data index, tree index

  # Returns data given a data index
  def get(self, data_index):
    return self.data[data_index % self.size]

  def total(self):
    return self.sum_tree[0]

  def max(self):
    return self.max_tree[0]


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
    self.transitions = SegmentTree(capacity)  # Store transitions in a wrap-around cyclic buffer within a sum tree for querying priorities

  # Add empty states to prepare for new episode
  def preappend(self):
    self.t = 0
    for h in range(-self.history + 1, 0):  # Blank transitions from before episode
      self.transitions.append(Transition(h, torch.ByteTensor(84, 84).zero_(), None, 0, True), 0)  # Add blank state with zero priority

  # Adds state, action and reward at time t (technically reward from time t + 1, but kept at t for all buffers to be in sync)
  def append(self, state, action, reward):
    state = state[-1].mul(255).byte().cpu()  # Only store last frame and discretise to save memory
    self.transitions.append(Transition(self.t, state, action, reward, True), self.transitions.max())  # Store new transition with maximum priority
    self.t += 1

  # Add empty state at end of episode
  def postappend(self):
    for _ in range(self.n):  # Add blank transitions (used to replace terminal state) with zero priority; simplifies truncated n-step discounted return
      self.transitions.append(Transition(self.t, torch.ByteTensor(84, 84).zero_(), None, 0, False), 0)

  # Returns a valid sample from a segment
  def _get_sample_from_segment(self, segment, i):
    valid = False
    while not valid:
      sample = random.uniform(i * segment, (i + 1) * segment)  # Uniformly sample an element from within a segment
      prob, idx, tree_idx = self.transitions.find(sample)  # Retrieve sample from tree with un-normalised probability
      # Resample if transition straddled current index or probablity 0 TODO: Separate out pre- and post-index
      if abs(idx - self.transitions.index) > max(self.history, self.n) and prob != 0:
        valid = True

    # Retrieve all required transition data (from t - h to t + n)
    transition = [self.transitions.get(idx + t) for t in range(1 - self.history, self.n + 1)]

    # Create un-discretised state and nth next state
    state = torch.stack([trans.state for trans in transition[:self.history]]).type(self.dtype_float).div_(255)
    next_state = torch.stack([trans.state for trans in transition[self.n:self.n + self.history]]).type(self.dtype_float).div_(255)  # nth next state

    action = self.dtype_long([transition[self.history - 1].action])

    # Calculate truncated n-step discounted return R^n = Σ_k=0->n-1 (γ^k)R_t+k+1
    R = self.dtype_float([transition[self.history - 1].reward])
    for n in range(1, self.n):
      # Invalid nth next states have reward 0 and hence do not affect calculation
      R += self.discount ** n * transition[self.history + n - 1].reward

    nonterminal = self.dtype_float([transition[self.history + self.n - 1].nonterminal])  # Mask for non-terminal nth next states

    return prob, idx, tree_idx, state, action, R, next_state, nonterminal

  def sample(self, batch_size):
    p_total = self.transitions.total()  # Retrieve sum of all priorities (used to create a normalised probability distribution)
    segment = p_total / batch_size  # Batch size number of segments, based on sum over all probabilities
    batch = [self._get_sample_from_segment(segment, i) for i in range(batch_size)]  # Get batch of valid samples
    probs, idxs, tree_idxs, states, actions, returns, next_states, nonterminals = zip(*batch)
    states, next_states, = Variable(torch.stack(states)), Variable(torch.stack(next_states), volatile=True)
    actions, returns, nonterminals = torch.cat(actions), torch.cat(returns), torch.stack(nonterminals)
    probs = Variable(self.dtype_float(probs)) / p_total  # Calculate normalised probabilities
    capacity = self.capacity if self.transitions.full else self.transitions.index
    weights = (capacity * probs) ** -self.priority_weight  # Compute importance-sampling weights w
    weights = weights / weights.max()   # Normalise by max importance-sampling weight from batch
    return tree_idxs, states, actions, returns, next_states, nonterminals, weights

  def update_priorities(self, idxs, priorities):
    [self.transitions.update(idx, priority) for idx, priority in zip(idxs, priorities)]

  # Set up internal state for iterator
  def __iter__(self):
    # Find indices for valid samples
    self.valid_idxs = []
    for t in range(self.capacity):
      if self.transitions.data[t].timestep >= 0:
        self.valid_idxs.append(t)
    self.current_idx = 0
    return self

  # Return valid states for validation
  def __next__(self):
    if self.current_idx == len(self.valid_idxs):
      raise StopIteration
    # Create stack of states and nth next states
    state_stack = []
    for t in reversed(range(self.history)):
      state_stack.append(self.transitions.data[self.valid_idxs[self.current_idx - t]].state)
    state = Variable(torch.stack(state_stack, 0).type(self.dtype_float).div_(255), volatile=True)  # Agent will turn into batch
    self.current_idx += 1
    return state
