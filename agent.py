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
    self.atoms = args.atoms
    self.Vmin = args.V_min
    self.Vmax = args.V_max
    self.support = torch.linspace(args.V_min, args.V_max, args.atoms)  # Support (range) of Z
    self.delta_Z = (args.V_max - args.V_min) / (args.atoms - 1)
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
      return (self.policy_net(state.unsqueeze(0)).data * self.support).sum(2).max(1)[1][0]
    else:
      return random.randint(0, self.action_space - 1)

  def learn(self, mem):
    transitions = mem.sample(self.batch_size)
    batch = Transition(*zip(*transitions))  # Transpose the batch

    states = Variable(torch.stack(batch.state, 0))
    actions = torch.LongTensor(batch.action)
    rewards = torch.Tensor(batch.reward)
    non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))  # Only process non-terminal next states
    next_states = Variable(torch.stack(tuple(s for s in batch.next_state if s is not None), 0), volatile=True)

    # TODO: Tidy this section up
    # Compute probabilities of Q(s,a*)
    q_probs = self.policy_net(states)
    qa_probs = q_probs[range(self.batch_size), actions]

    # Compute distribution of Q(s_,a)
    # Compute probabilities p(x, a)
    # TODO: Use n-step distributional loss
    probs = self.target_net(next_states).data
    qs = self.support.expand_as(probs) * probs
    # TODO: Use double-Q action selection
    argmax_a = qs.sum(2).max(1)[1]
    qa_probs2 = probs[range(self.batch_size), argmax_a]

    # Compute projection of the application of the Bellman operator.
    bellman_op = rewards.unsqueeze(1) + non_final_mask.float().unsqueeze(1) * self.discount * self.support.unsqueeze(0)
    bellman_op = torch.clamp(bellman_op, self.Vmin, self.Vmax)

    # Compute categorical indices for distributing the probability
    m = torch.zeros(self.batch_size, self.atoms)
    b = (bellman_op - self.Vmin) / self.delta_Z
    l, u = b.floor().long(), b.ceil().long()

    # Distribute probability
    offset = torch.linspace(0, ((self.batch_size - 1) * self.atoms), self.batch_size).long().unsqueeze(1).expand(self.batch_size, self.atoms)
    m.view(-1).index_add_(0, (l + offset).view(-1), (qa_probs2 * (u.float() - b)).view(-1))
    m.view(-1).index_add_(0, (u + offset).view(-1), (qa_probs2 * (b - l.float())).view(-1))

    loss = -torch.sum(Variable(m) * qa_probs.log())
    """
    Zs = self.policy_net(states)[range(self.batch_size), actions]  # Z(s_t, a_t; θpolicy)
    next_state_argmax_indices = self.policy_net(next_states).sum(2).max(1, keepdim=True)[1]  # Perform argmax action selection using policy network: argmax_a[Z(s_t+1, a; θpolicy)]
    print(Zs.size(), next_state_argmax_indices.size())
    Zns = Variable(torch.zeros(self.batch_size))  # Z(s_t+1, a) = 0 if s_t+1 is terminal
    Zns[non_final_mask] = self.target_net(next_states).gather(1, next_state_argmax_indices)  # Z(s_t+1, argmax_a[Z(s_t+1, a; θpolicy)]; θtarget)
    Zns.volatile = False  # Remove volatile flag to prevent propagating it through loss
    target = rewards + (self.discount * Zns)  # Double-Q target: Y = r + γ.Z(s_t+1, argmax_a[Z(s_t+1, a; θpolicy)]; θtarget)

    loss = F.smooth_l1_loss(Zs, target)  # Huber loss on TD-error δ: δ = Y - Q(s_t, a_t)
    """
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
    return (self.policy_net(state.unsqueeze(0)).data * self.support).sum(2).max(1)[0][0]

  def train(self):
    self.policy_net.train()

  def eval(self):
    self.policy_net.eval()
