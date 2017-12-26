import random
from collections import namedtuple
import torch
from torch.autograd import Variable


# Segment tree data structure where parent node values are sum of children node values
class SumTree():
  def __init__(self, size):
    self.index = 0
    self.size = size
    self.tree = [0] * (2 * size - 1)  # Initialise fixed size tree with all (priority) zeros
    self.data = [Transition(-1, torch.ByteTensor(84, 84).zero_(), None, 0, True)] * size  # Wrap-around cyclic buffer filled with (zero-priority) blank transitions
    self.max = 0  # Store max value for fast retrieval

  # Propagates value up tree given a tree index
  def _propagate(self, index, update):
    parent = (index - 1) // 2
    self.tree[parent] += update  # Propagate change in value rather than absolute value
    if parent != 0:
      self._propagate(parent, update)

  # Updates value given a tree index
  def update(self, index, value):
    update = value - self.tree[index]
    self.tree[index] = value  # Set new value
    self._propagate(index, update)  # Propagate change
    self.max = max(self.max, value)  # Update max

  def append(self, data, value):
    self.data[self.index] = data  # Store data in underlying data structure
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

  # Searches for a value and returns value, data index and tree index
  def find(self, value):
    index = self._retrieve(0, value)  # Search for index of item from root
    data_index = index - self.size + 1
    return (self.tree[index], data_index, index)  # Return value, data index, tree index

  # Returns data given a data index
  def get(self, data_index):
    return self.data[data_index % self.size]

  def total(self):
    return self.tree[0]


Transition = namedtuple('Transition', ('timestep', 'state', 'action', 'reward', 'nonterminal'))


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
    self.transitions = SumTree(capacity)  # Store transitions in a wrap-around cyclic buffer within a sum tree for querying priorities

  # Add empty states to prepare for new episode
  def preappend(self):
    self.t = 0
    # Blank transitions from before episode
    for h in range(-self.history + 1, 0):
      # Add blank state with zero priority
      self.transitions.append(Transition(h, torch.ByteTensor(84, 84).zero_(), None, 0, True), 0)

  # Adds state, action and reward at time t (technically reward from time t + 1, but kept at t for all buffers to be in sync)
  def append(self, state, action, reward):
    state = state[-1].mul(255).byte().cpu()  # Only store last frame and discretise to save memory
    # Store new transition with maximum priority (or use initial priority 1)
    self.transitions.append(Transition(self.t, state, action, reward, True), max(self.transitions.max, 1))
    self.t += 1

  # Add empty state at end of episode
  def postappend(self):
    # Add blank state (used to replace terminal state) with zero priority
    self.transitions.append(Transition(self.t, torch.ByteTensor(84, 84).zero_(), None, 0, False), 0)

  def sample(self, batch_size):
    p_total = self.transitions.total()  # Retrieve sum of all priorities (used to create a normalised probability distribution)
    segment = p_total / batch_size  # Batch size number of segments, based on sum over all probabilities
    samples = [random.uniform(i * segment, (i + 1) * segment) for i in range(batch_size)]  # Uniformly sample an element from each segment
    batch = [self.transitions.find(s) for s in samples]  # Retrieve samples from tree
    probs, idxs, tree_idxs = zip(*batch)  # Unpack unnormalised probabilities (priorities), data indices, tree indices
    # TODO: Check that transitions with 0 probability are not returned/make sure samples are valid

    # Retrieve all required transition data (from t - h to t + n)
    full_transitions = [[self.transitions.get(i + t) for i in idxs] for t in range(1 - self.history, self.n + 1)]  # Time x batch

    # Create stack of states and nth next states
    state_stack, next_state_stack = [], []
    for t in range(self.history):
      state_stack.append(torch.stack([transition.state for transition in full_transitions[t]], 0))
      next_state_stack.append(torch.stack([transition.state for transition in full_transitions[t + self.n]], 0))  # nth next state
    states = Variable(torch.stack(state_stack, 1).type(self.dtype_float).div_(255))  # Un-discretise
    next_states = Variable(torch.stack(next_state_stack, 1).type(self.dtype_float).div_(255), volatile=True)

    actions = self.dtype_long([transition.action for transition in full_transitions[self.history - 1]])

    # Calculate truncated n-step discounted return R^n = Σ_k=0->n-1 (γ^k)R_t+k+1
    returns = [transition.reward for transition in full_transitions[self.history - 1]]
    for n in range(1, self.n):
      # Invalid nth next states have reward 0 and hence do not affect calculation
      returns = [R + self.discount ** n * transition.reward for R, transition in zip(returns, full_transitions[self.history + n])]
    returns = self.dtype_float(returns)  # TODO: Make sure this doesn't cause issues around current buffer index

    nonterminals = [transition.nonterminal for transition in full_transitions[self.history + self.n - 1]] # Mask for non-terminal nth next states
    for t in range(self.history, self.history + self.n):  # Hack: if nth next state is invalid (overlapping transition), treat it as terminal
      nonterminals = [nonterm and (trans.timestep - pre_trans.timestep) == 1 for nonterm, trans, pre_trans in zip(nonterminals, full_transitions[t], full_transitions[t - 1])]
    nonterminals = self.dtype_float(nonterminals).unsqueeze(1) 

    probs = Variable(torch.Tensor(probs)) / p_total  # Calculate normalised probabilities
    weights = (self.capacity * probs) ** -self.priority_weight  # Compute importance-sampling weights w
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
