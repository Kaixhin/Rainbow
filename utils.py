from skimage import color, transform
import torch


def state_to_tensor(state):
  gray_img = color.rgb2gray(state)  # TODO: Check image conversion doesn't cause problems
  downsized_img = transform.resize(gray_img, (84, 84), mode='constant')  # TODO: Check resizing doesn't cause problems
  return torch.from_numpy(downsized_img).unsqueeze(0)  # Return 3D image tensor
