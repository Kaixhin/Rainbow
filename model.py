from torch import nn


class DQN(nn.Module):
  def __init__(self, args, action_space):
    super().__init__()
    self.relu = nn.ReLU()
    self.conv1 = nn.Conv2d(args.history_length, 32, 8, stride=4, padding=1)
    self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
    self.conv3 = nn.Conv2d(64, 64, 3)
    self.fc1 = nn.Linear(3136, args.hidden_size)  # TODO: Noisy linear layers
    self.fc_v = nn.Linear(args.hidden_size, 1)
    self.fc_a = nn.Linear(args.hidden_size, action_space)
    # TODO: Distributional version

  def forward(self, x):
    x = self.relu(self.conv1(x))
    x = self.relu(self.conv2(x))
    x = self.relu(self.conv3(x))
    x = x.view(x.size(0), -1)
    x = self.relu(self.fc1(x))
    v = self.fc_v(x)
    a = self.fc_a(x)
    return v.expand_as(a) + a - a.mean(1, keepdim=True).expand_as(a)
