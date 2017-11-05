import math
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F


class DQN(nn.Module):
  def __init__(self, args, action_space):
    super().__init__()
    self.action_space = action_space
    self.relu = nn.ReLU()
    self.softmax = nn.Softmax()

    self.conv1 = nn.Conv2d(args.history_length, 32, 8, stride=4, padding=1)
    self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
    self.conv3 = nn.Conv2d(64, 64, 3)
    self.fc1 = nn.Linear(3136, args.hidden_size)  # TODO: Noisy linear layers
    self.fc_z_v = nn.Linear(args.hidden_size, args.atoms)
    self.fc_z_a = nn.Linear(args.hidden_size, action_space * args.atoms)

  def forward(self, x):
    x = self.relu(self.conv1(x))
    x = self.relu(self.conv2(x))
    x = self.relu(self.conv3(x))
    x = x.view(x.size(0), -1)
    x = self.relu(self.fc1(x))
    v, a = self.fc_z_v(x), self.fc_z_a(x)  # Calculate value and advantage streams
    x = v.repeat(1, self.action_space) + a - a.mean(1, keepdim=True).expand_as(a)
    p = torch.stack([self.softmax(p) for p in x.chunk(self.action_space, 1)], 1)  # Probabilities with action over second dimension
    return p.clamp(min=1e-8, max=1 - 1e-8)  # Use clipping to prevent NaNs
