import os
import random
import torch
from torch import nn, optim

from model import DQN


class Agent():
  def __init__(self, args, env):
    self.action_space = env.action_space()
    self.quantile = args.quantile
    self.atoms = args.quantiles if args.quantile else args.atoms
    if args.quantile:
      self.cumulative_density = (2 * torch.arange(self.atoms).to(device=args.device) + 1) / (2 * self.atoms)  # Quantile cumulative probability weights τ
    else:
      self.Vmin = args.V_min
      self.Vmax = args.V_max
      self.support = torch.linspace(args.V_min, args.V_max, self.atoms).to(device=args.device)  # Support (range) of z
      self.delta_z = (args.V_max - args.V_min) / (self.atoms - 1)
    self.batch_size = args.batch_size
    self.n = args.multi_step
    self.discount = args.discount
    self.norm_clip = args.norm_clip

    self.online_net = DQN(args, self.action_space, args.quantile).to(device=args.device)
    if args.model and os.path.isfile(args.model):
      # Always load tensors onto CPU by default, will shift to GPU if necessary
      self.online_net.load_state_dict(torch.load(args.model, map_location='cpu'))
    self.online_net.train()

    self.target_net = DQN(args, self.action_space, args.quantile).to(device=args.device)
    self.update_target_net()
    self.target_net.train()
    for param in self.target_net.parameters():
      param.requires_grad = False

    self.optimiser = optim.Adam(self.online_net.parameters(), lr=args.lr, eps=args.adam_eps)

  # Resets noisy weights in all linear layers (of online net only)
  def reset_noise(self):
    self.online_net.reset_noise()

  # Acts based on single state (no batch)
  def act(self, state):
    with torch.no_grad():
      return (self.online_net(state.unsqueeze(0)) * ((1 / self.atoms) if self.quantile else self.support)).sum(2).argmax(1).item()

  # Acts with an ε-greedy policy
  def act_e_greedy(self, state, epsilon=0.05):
    return random.randrange(self.action_space) if random.random() < epsilon else self.act(state)

  def learn(self, mem):
    # Sample transitions
    idxs, states, actions, returns, next_states, nonterminals, weights = mem.sample(self.batch_size)

    # Calculate current state probabilities (online network noise already sampled)
    ps = self.online_net(states)  # Probabilities p(s_t, ·; θonline)/quantile probabilities θ(s_t, ·; θonline)
    ps_a = ps[range(self.batch_size), actions]  # p(s_t, a_t; θonline)

    with torch.no_grad():
      # Calculate nth next state probabilities
      pns = self.online_net(next_states)  # Probabilities p(s_t+n, ·; θonline)
      dns = ((1 / self.atoms) if self.quantile else self.support.expand_as(pns)) * pns  # Distribution d_t+n = (z, p(s_t+n, ·; θonline))
      argmax_indices_ns = dns.sum(2).argmax(1)  # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
      self.target_net.reset_noise()  # Sample new target net noise
      pns = self.target_net(next_states)  # Probabilities p(s_t+n, ·; θtarget)
      pns_a = pns[range(self.batch_size), argmax_indices_ns]  # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)

      if self.quantile:
        # Compute distributional Bellman target Tθ = R^n + (γ^n)p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)
        Ttheta = returns.unsqueeze(1) + nonterminals * (self.discount ** self.n) * pns_a  # (accounting for terminal states)
      else:
        # Compute Tz (Bellman operator T applied to z)
        Tz = returns.unsqueeze(1) + nonterminals * (self.discount ** self.n) * self.support.unsqueeze(0)  # Tz = R^n + (γ^n)z (accounting for terminal states)
        Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)  # Clamp between supported values
        # Compute L2 projection of Tz onto fixed support z
        b = (Tz - self.Vmin) / self.delta_z  # b = (Tz - Vmin) / Δz
        l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
        # Fix disappearing probability mass when l = b = u (b is int)
        l[(u > 0) * (l == u)] -= 1
        u[(l < (self.atoms - 1)) * (l == u)] += 1

        # Distribute probability of Tz
        m = states.new_zeros(self.batch_size, self.atoms)
        offset = torch.linspace(0, ((self.batch_size - 1) * self.atoms), self.batch_size).unsqueeze(1).expand(self.batch_size, self.atoms).to(actions)
        m.view(-1).index_add_(0, (l + offset).view(-1), (pns_a * (u.float() - b)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
        m.view(-1).index_add_(0, (u + offset).view(-1), (pns_a * (b - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)

    if self.quantile:
      u = Ttheta - ps_a  # Residual u
      kappa_cond = (u < 1).to(torch.float32)  # |u| ≤ κ
      huber_loss = 0.5 * u ** 2 * kappa_cond + (u.abs() - 0.5) * (1 - kappa_cond)  # Huber loss Lκ(u)
      loss = torch.sum(torch.abs(self.cumulative_density - (u < 0).to(torch.float32)) * huber_loss, 1)  # Quantile Huber loss ρκτ(u) = |τ − δ{u<0}|Lκ(u)
    else:
      loss = -torch.sum(m * ps_a.log(), 1)  # Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))
    loss = weights * loss  # Importance weight losses
    self.online_net.zero_grad()
    loss.mean().backward()  # Backpropagate minibatch loss
    self.optimiser.step()
    nn.utils.clip_grad_norm_(self.online_net.parameters(), self.norm_clip)  # Clip gradients by L2 norm
    if self.quantile:
      loss = (self.atoms * loss).clamp(max=5)  # Heuristic for prioritised replay

    mem.update_priorities(idxs, loss.detach())  # Update priorities of sampled transitions

  def update_target_net(self):
    self.target_net.load_state_dict(self.online_net.state_dict())

  # Save model parameters on current device (don't move model between devices)
  def save(self, path):
    torch.save(self.online_net.state_dict(), os.path.join(path, 'model.pth'))

  # Evaluates Q-value based on single state (no batch)
  def evaluate_q(self, state):
    with torch.no_grad():
      return (self.online_net(state.unsqueeze(0)) * ((1 / self.atoms) if self.quantile else self.support)).sum(2).max(1)[0].item()

  def train(self):
    self.online_net.train()

  def eval(self):
    self.online_net.eval()
