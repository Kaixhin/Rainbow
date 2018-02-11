import random
from collections import namedtuple
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler


Transition = namedtuple('Transition', ('timestep', 'state', 'action', 'reward', 'nonterminal'))


# Segment tree data structure where parent node values are sum of children node values
class SumTree():
  def __init__(self, size):
    self.index = 0
    self.size = size
    self.tree = [0] * (2 * size - 1)  # Initialise fixed size tree with all (priority) zeros
    self.data = [Transition(-1, torch.ByteTensor(84, 84).zero_(), None, 0, True)] * size  # Wrap-around cyclic buffer filled with (zero-priority) blank transitions
    self.max = 1  # Store max value (initialised at 1) for fast retrieval

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


class PrioritySampler(Sampler):
  def __init__(self, transitions, batch_size, history, n):
    self.transitions = transitions
    self.batch_size = batch_size
    self.history = history
    self.n = n

  def __iter__(self):
    p_total = self.transitions.total()  # Retrieve sum of all priorities (used to create a normalised probability distribution)
    segment = p_total / self.batch_size  # Batch size number of segments, based on sum over all probabilities
    samples = [random.uniform(i * segment, (i + 1) * segment) for i in range(self.batch_size)]  # Uniformly sample an element from each segment
    batch = [self.transitions.find(s) for s in samples]  # Retrieve samples from tree
    probs, idxs, tree_idxs = zip(*batch)  # Unpack unnormalised probabilities (priorities), data indices, tree indices
    probs, idxs, tree_idxs = torch.FloatTensor(probs), torch.LongTensor(idxs), torch.LongTensor(tree_idxs)
    # If any transitions straddle current index, remove them (simpler than replacing with unique valid transitions) TODO: Separate out pre- and post-index
    valid_idxs = idxs.sub(self.transitions.index).abs_() > max(self.history, self.n)
    # If any transitions have 0 probability (priority), remove them (may not be necessary check)
    valid_idxs.mul_(probs != 0)
    probs, idxs, tree_idxs = probs[valid_idxs], idxs[valid_idxs], tree_idxs[valid_idxs]
    yield idxs

  def __len__(self):
    return 0


class ReplayMemory(Dataset):
  def __init__(self, args, capacity):
    self.cuda = args.cuda
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
    self.batch_size = args.batch_size

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
    # Store new transition with maximum priority
    self.transitions.append(Transition(self.t, state, action, reward, True), self.transitions.max)
    self.t += 1

  # Add empty state at end of episode
  def postappend(self):
    # Add blank transitions (used to replace terminal state) with zero priority; simplifies truncated n-step discounted return calculations
    for _ in range(self.n):
      self.transitions.append(Transition(self.t, torch.ByteTensor(84, 84).zero_(), None, 0, False), 0)

  def __getitem__(self, idx):
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

    return state, action, R, next_state, nonterminal

  def __len__(self):
    return self.capacity  # TODO: Return number of valid samples?

  def get_loader(self):
    return iter(DataLoader(self, batch_sampler=PrioritySampler(self.transitions, self.batch_size, self.history, self.n), num_workers=8, pin_memory=self.cuda))

    """
    probs = Variable(probs) / p_total  # Calculate normalised probabilities
    weights = (self.capacity * probs) ** -self.priority_weight  # Compute importance-sampling weights w
    weights = weights / weights.max()   # Normalise by max importance-sampling weight from batch

    return tree_idxs, states, actions, returns, next_states, nonterminals, weights
    """

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
