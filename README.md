Rainbow
=======
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.md)

Rainbow: Combining Improvements in Deep Reinforcement Learning [[1]](#references). Includes quantile regression loss [[2]](#references): run with `--quantile --atoms 200`.

Results and pretrained models can be found in the [releases](https://github.com/Kaixhin/Rainbow/releases).

- [x] DQN [[3]](#references)
- [x] Double DQN [[4]](#references)
- [x] Prioritised Experience Replay [[5]](#references)
- [x] Dueling Network Architecture [[6]](#references)
- [x] Multi-step Returns [[7]](#references)
- [x] Distributional RL [[8]](#references)
- [x] Noisy Nets [[9]](#references)

Requirements
------------

- [atari-py](https://github.com/openai/atari-py)
- [OpenCV Python](https://pypi.python.org/pypi/opencv-python)
- [Plotly](https://plot.ly/)
- [PyTorch](http://pytorch.org/)

To install all dependencies with Anaconda run `conda env create -f environment.yml` and use `source activate rainbow` to activate the environment.

Acknowledgements
----------------

- [@floringogianu](https://github.com/floringogianu) for [categorical-dqn](https://github.com/floringogianu/categorical-dqn)
- [@jvmancuso](https://github.com/jvmancuso) for [Noisy layer](https://github.com/pytorch/pytorch/pull/2103)
- [@jaara](https://github.com/jaara) for [AI-blog](https://github.com/jaara/AI-blog)
- [@openai](https://github.com/openai) for [Baselines](https://github.com/openai/baselines)
- [@mtthss](https://github.com/mtthss) for [implementation details](https://github.com/Kaixhin/Rainbow/wiki/Matteo's-Notes)
- [@ShangtongZhang](https://github.com/ShangtongZhang) for [DeepRL](https://github.com/ShangtongZhang/DeepRL)

References
----------

[1] [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298)  
[2] [Distributional Reinforcement Learning with Quantile Regression](https://arxiv.org/abs/1710.10044)  
[3] [Playing Atari with Deep Reinforcement Learning](http://arxiv.org/abs/1312.5602)  
[4] [Deep Reinforcement Learning with Double Q-learning](http://arxiv.org/abs/1509.06461)  
[5] [Prioritized Experience Replay](http://arxiv.org/abs/1511.05952)  
[6] [Dueling Network Architectures for Deep Reinforcement Learning](http://arxiv.org/abs/1511.06581)  
[7] [Reinforcement Learning: An Introduction](http://www.incompleteideas.net/sutton/book/ebook/the-book.html)  
[8] [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887)  
[9] [Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295)  
