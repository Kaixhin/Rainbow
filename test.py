import os
import plotly
from plotly.graph_objs import Scatter, Line
import torch
from torch.autograd import Variable

from env import Env


# Globals
Ts, rewards, Qs = [], [], []


# Test DQN
def test(args, T, dqn, val_mem, evaluate=False):
  env = Env(args)
  env.seed(args.seed)
  env.eval()
  Ts.append(T)
  T_rewards, T_Qs = [], []

  # Test performance over several episodes
  done = True
  for ep in range(args.evaluation_episodes):
    while True:
      if done:
        state = Variable(env.reset(), volatile=True)
        reward_sum = 0
        done = False

      if args.render:
        env.render()

      action = dqn.act(state)  # Choose an action  greedily
      state, reward, done = env.step(action)  # Step
      state = Variable(state, volatile=True)
      reward_sum += reward

      if done:
        T_rewards.append(reward_sum)
        break
  env.close()

  # Test Q-values over validation memory
  for state in val_mem.states:  # TODO: Update once states stored efficiently
    if state is not None:
      T_Qs.append(dqn.evaluate_q(Variable(state, volatile=True)))

  if not evaluate:
    # Append to results
    rewards.append(T_rewards)
    Qs.append(T_Qs)

    # Plot
    _plot_line(Ts, rewards, 'Reward', path='results')
    _plot_line(Ts, Qs, 'Q', path='results')

    # Save model weights
    dqn.save('results')

  # Return average reward and Q-value
  return sum(T_rewards) / len(T_rewards), sum(T_Qs) / len(T_Qs)


# Plots min, max and mean + standard deviation bars of a population over time
def _plot_line(xs, ys_population, title, path=''):
  max_colour = 'rgb(0, 132, 180)'
  mean_colour = 'rgb(0, 172, 237)'
  std_colour = 'rgba(29, 202, 255, 0.2)'

  ys = torch.Tensor(ys_population)
  ys_min = ys.min(1)[0].squeeze()
  ys_max = ys.max(1)[0].squeeze()
  ys_mean = ys.mean(1).squeeze()
  ys_std = ys.std(1).squeeze()
  ys_upper, ys_lower = ys_mean + ys_std, ys_mean - ys_std

  trace_max = Scatter(x=xs, y=ys_max.numpy(), line=Line(color=max_colour, dash='dash'), name='Max')
  trace_upper = Scatter(x=xs, y=ys_upper.numpy(), line=Line(color='transparent'), name='+1 Std. Dev.', showlegend=False)
  trace_mean = Scatter(x=xs, y=ys_mean.numpy(), fill='tonexty', fillcolor=std_colour, line=Line(color=mean_colour), name='Mean')
  trace_lower = Scatter(x=xs, y=ys_lower.numpy(), fill='tonexty', fillcolor=std_colour, line=Line(color='transparent'), name='-1 Std. Dev.', showlegend=False)
  trace_min = Scatter(x=xs, y=ys_min.numpy(), line=Line(color=max_colour, dash='dash'), name='Min')

  plotly.offline.plot({
    'data': [trace_upper, trace_mean, trace_lower, trace_min, trace_max],
    'layout': dict(title=title,
                   xaxis={'title': 'Step'},
                   yaxis={'title': title})
  }, filename=os.path.join(path, title + '.html'), auto_open=False)
