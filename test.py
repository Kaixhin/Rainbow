from datetime import datetime
import os
import plotly
from plotly.graph_objs import Scatter, Line
import torch
from torch.autograd import Variable

from env import Env


# Simple timestamped logger
def log(s):
  print('[' + str(datetime.now().time()) + '] ' + s)


# Test DQN
def test(args, eval_count, T, dqn, val_mem, Ts, rewards, Qs, evaluate=False):
  env = Env(args)
  env.eval()
  Ts[eval_count] = T
  T_rewards, T_Qs = [], []

  # Test performance over several episodes
  dqn.eval()  # Set DQN (online network) to evaluation mode
  done = True
  for _ in range(args.evaluation_episodes):
    while True:
      if done:
        state, reward_sum, done = Variable(env.reset(), volatile=True), 0, False

      action = dqn.act_e_greedy(state)  # Choose an action ε-greedily
      state, reward, done = env.step(action)  # Step
      state = Variable(state, volatile=True)
      reward_sum += reward
      if args.render:
        env.render()

      if done:
        T_rewards.append(reward_sum)
        break
  env.close()

  # Test Q-values over validation memory
  for state in val_mem:  # Iterate over valid states
    T_Qs.append(dqn.evaluate_q(state))
  dqn.train()  # Set DQN (online network) back to training mode

  if not evaluate:
    # Append to results
    rewards[eval_count] = torch.Tensor(T_rewards)
    Qs[eval_count] = torch.Tensor(T_Qs)

    # Plot
    _plot_line(Ts[:eval_count + 1], rewards[:eval_count + 1], 'Reward', path='results')
    _plot_line(Ts[:eval_count + 1], Qs[:eval_count + 1], 'Q', path='results')

    # Save model weights
    dqn.save('results')

  # Log average reward and Q-value
  avg_reward, avg_Q = sum(T_rewards) / len(T_rewards), sum(T_Qs) / len(T_Qs)
  log('T = ' + str(T) + ' / ' + str(args.T_max) + ' | Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_Q))


# Plots min, max and mean + standard deviation bars of a population over time
def _plot_line(xs, ys_population, title, path=''):
  max_colour, mean_colour, std_colour = 'rgb(0, 132, 180)', 'rgb(0, 172, 237)', 'rgba(29, 202, 255, 0.2)'

  ys = torch.Tensor(ys_population)
  ys_min, ys_max, ys_mean, ys_std = ys.min(1)[0].squeeze(), ys.max(1)[0].squeeze(), ys.mean(1).squeeze(), ys.std(1).squeeze()
  ys_upper, ys_lower = ys_mean + ys_std, ys_mean - ys_std

  trace_max = Scatter(x=xs, y=ys_max.numpy(), line=Line(color=max_colour, dash='dash'), name='Max')
  trace_upper = Scatter(x=xs, y=ys_upper.numpy(), line=Line(color='transparent'), name='+1 Std. Dev.', showlegend=False)
  trace_mean = Scatter(x=xs, y=ys_mean.numpy(), fill='tonexty', fillcolor=std_colour, line=Line(color=mean_colour), name='Mean')
  trace_lower = Scatter(x=xs, y=ys_lower.numpy(), fill='tonexty', fillcolor=std_colour, line=Line(color='transparent'), name='-1 Std. Dev.', showlegend=False)
  trace_min = Scatter(x=xs, y=ys_min.numpy(), line=Line(color=max_colour, dash='dash'), name='Min')

  plotly.offline.plot({
    'data': [trace_upper, trace_mean, trace_lower, trace_min, trace_max],
    'layout': dict(title=title, xaxis={'title': 'Step'}, yaxis={'title': title})
  }, filename=os.path.join(path, title + '.html'), auto_open=False)
