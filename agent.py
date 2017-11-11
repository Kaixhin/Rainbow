import os
import torch
from torch import nn, optim
from torch.autograd import Variable

from model import DQN


class Agent():
  def __init__(self, args, env):
    self.action_space = env.action_space()
    self.atoms = args.atoms
    self.Vmin = args.V_min
    self.Vmax = args.V_max
    self.support = torch.linspace(args.V_min, args.V_max, args.atoms)  # Support (range) of z
    self.delta_z = (args.V_max - args.V_min) / (args.atoms - 1)
    self.batch_size = args.batch_size
    self.n = args.multi_step
    self.discount = args.discount
    self.max_gradient_norm = args.max_gradient_norm

    self.policy_net = DQN(args, self.action_space)
    if args.model and os.path.isfile(args.model):
      self.policy_net.load_state_dict(torch.load(args.model))
    self.policy_net.train()

    self.target_net = DQN(args, self.action_space)
    self.update_target_net()
    self.target_net.eval()

    self.optimiser = optim.Adam(self.policy_net.parameters(), lr=args.lr, eps=args.adam_eps)

  # Resets noisy weights in all linear layers
  def reset_noise(self):
    for name, module in self.policy_net.named_children():
      if 'fc' in name:
        module.reset_noise()

  def act(self, state):
    return (self.policy_net(state.unsqueeze(0)).data * self.support).sum(2).max(1)[1][0]

  def learn(self, mem):
    states, actions, returns, next_states = mem.sample(self.batch_size)

    states = Variable(torch.stack(states, 0))
    actions = torch.LongTensor(actions)
    returns = torch.Tensor(returns)
    non_final_mask = torch.Tensor(tuple(map(lambda s: s is not None, next_states)))  # Only process non-terminal nth next states
    next_states = Variable(torch.stack(tuple(s for s in next_states if s is not None), 0), volatile=True)  # nth next state

    # Calculate current state probabilities
    ps = self.policy_net(states)  # Probabilities p(s_t, ·; θpolicy)
    ps_a = ps[range(self.batch_size), actions]  # p(s_t, a_t; θpolicy)

    # Calculate nth next state probabilities
    pns = self.policy_net(next_states).data  # Probabilities p(s_t+n, ·; θpolicy)
    dns = self.support.expand_as(pns) * pns  # Distribution d_t+n = (z, p(s_t+n, ·; θpolicy))
    argmax_indices_ns = dns.sum(2).max(1)[1]  # Perform argmax action selection using policy network: argmax_a[(z, p(s_t+n, a; θpolicy))]
    pns = self.target_net(next_states).data  # Probabilities p(s_t+n, ·; θtarget)
    pns_a = ps_a.data.new(ps_a.size()).zero_()
    pns_a.index_copy_(0, non_final_mask.nonzero().squeeze(1), pns[range(pns.size(0)), argmax_indices_ns])  # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θpolicy))]; θtarget)

    # Compute Tz (Bellman operator T applied to z)
    Tz = returns.unsqueeze(1) + non_final_mask.unsqueeze(1) * (self.discount ** self.n) * self.support.unsqueeze(0)  # Tz = R^n + (γ^n)z (accounting for terminal states)
    Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)  # Clamp between supported values
    # Compute L2 projection of Tz onto fixed support z
    b = (Tz - self.Vmin) / self.delta_z  # b = (Tz - Vmin) / Δz
    l, u = b.floor().long(), b.ceil().long()

    # Distribute probability of Tz
    m = torch.zeros(self.batch_size, self.atoms)
    offset = torch.linspace(0, ((self.batch_size - 1) * self.atoms), self.batch_size).long().unsqueeze(1).expand(self.batch_size, self.atoms)
    m.view(-1).index_add_(0, (l + offset).view(-1), (pns_a * (u.float() - b)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
    m.view(-1).index_add_(0, (u + offset).view(-1), (pns_a * (b - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)

    loss = -torch.sum(Variable(m) * ps_a.log())  # Cross-entropy loss (minimises Kullback-Leibler divergence)
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
