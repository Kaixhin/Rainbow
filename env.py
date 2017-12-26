from collections import deque
import atari_py
import torch
from torch.nn import functional as F


class Env():
  def __init__(self, args):
    super().__init__()
    self.dtype = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor
    self.ale = atari_py.ALEInterface()
    self.ale.loadROM(atari_py.get_game_path(args.game))
    self.ale.setInt('max_num_frames', args.max_episode_length)
    self.ale.setFloat('repeat_action_probability', 0)  # Disable sticky actions
    self.ale.setInt('frame_skip', 4)  # TODO: Should input max over last 2 frames of 4
    actions = self.ale.getMinimalActionSet()
    self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions))
    self.lives = 0  # Life counter (used in DeepMind training)
    self.window = args.history_length  # Number of frames to concatenate
    self.buffer = deque([], maxlen=args.history_length)
    self.training = True  # Consistent with model training mode

  def _get_state(self):
    state = self.dtype(self.ale.getScreenGrayscale()).div_(255).view(1, 1, 210, 160)
    state = F.upsample(state, size=(84, 84), mode='bilinear')  # TODO: Check resizing is same as original/discretises to [0, 255] properly
    return state.squeeze().data

  def _reset_buffer(self):
    for t in range(self.window):
      self.buffer.append(self.dtype(84, 84).zero_())

  def reset(self):
    # Reset internals
    self._reset_buffer()
    self.ale.reset_game()
    self.lives = self.ale.lives()
    # Process and return initial state
    observation = self._get_state()
    # TODO: 30 random no-op starts?
    self.buffer.append(observation)
    return torch.stack(self.buffer, 0)

  def step(self, action):
    # Process state
    reward = self.ale.act(self.actions.get(action))
    observation = self._get_state()
    self.buffer.append(observation)
    done = self.ale.game_over()
    # Detect loss of life as terminal in training mode
    if self.training:
      lives = self.ale.lives()
      if lives < self.lives:
        done = True  # TODO: Prevent resetting env on loss of life
      else:
        self.lives = lives
    # Return state, reward, done
    return torch.stack(self.buffer, 0), reward, done

  # Uses loss of life as terminal signal
  def train(self):
    self.training = True

  # Uses standard terminal signal
  def eval(self):
    self.training = False

  def action_space(self):
    return len(self.actions)

  def seed(self, seed):
    self.ale.setInt('random_seed', seed)

  def render(self):
    pass  # TODO
    # self.env.render()

  def close(self):
    pass  # TODO
    # self.env.close()
