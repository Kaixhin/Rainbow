from skimage import color, transform
import torch


def state_to_tensor(state):
  gray_img = color.rgb2gray(state)
  downsized_img = transform.resize(gray_img, (84, 84), mode='constant')
  return torch.from_numpy(downsized_img).float().div_(255).unsqueeze(0)
