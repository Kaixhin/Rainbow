from torch import nn
from torch.nn import init


class DQN(nn.Module):
  def __init__(self, hidden_size, action_size):
    super().__init__()
    self.relu = nn.ReLU()
    self.conv1 = nn.Conv2d(1, 16, 8, stride=4, padding=1)
    self.conv2 = nn.Conv2d(16, 32, 4, stride=2, padding=2)
    self.fc1 = nn.Linear(3872, hidden_size)  # TODO: Noisy linear layers
    self.fc_v = nn.Linear(hidden_size, 1)
    self.fc_a = nn.Linear(hidden_size, action_size)
    # TODO: Distributional version

    # Orthogonal weight initialisation
    for name, p in self.named_parameters():
      if 'weight' in name:
        init.orthogonal(p)
      elif 'bias' in name:
        init.constant(p, 0)

  def forward(self, x):
    x = self.relu(self.conv1(x))
    x = self.relu(self.conv2(x))
    x = x.view(x.size(0), -1)
    x = self.relu(self.fc1(x))
    v = self.fc_v(x)
    a = self.fc_a(x)
    return v.expand_as(a) + a - a.mean(1, keepdim=True).expand_as(a)
