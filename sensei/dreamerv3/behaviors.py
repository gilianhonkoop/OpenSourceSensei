import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

from . import agent
from . import expl
from . import ninjax as nj
from . import jaxutils


class Greedy(nj.Module):

  def __init__(self, wm, act_space, config, task_reward='reward'):

    if task_reward == 'reward':
      rewfn = lambda s: wm.heads['reward'](s).mean()[1:]
    else:
      rewfn = lambda s: jnp.squeeze(wm.heads[task_reward](s).mean()[1:])
    if config.critic_type == 'vfunction':
      critics = {'extr': agent.VFunction(rewfn, config, name='critic')}
    else:
      raise NotImplementedError(config.critic_type)
    self.ac = agent.ImagActorCritic(
        critics, {'extr': 1.0}, act_space, config, name='ac')

  def initial(self, batch_size):
    return self.ac.initial(batch_size)

  def policy(self, latent, state):
    return self.ac.policy(latent, state)

  def train(self, imagine, start, data):
    return self.ac.train(imagine, start, data)

  def report(self, data):
    return {}


class Random(nj.Module):

  def __init__(self, wm, act_space, config, task_reward=''):
    self.config = config
    self.act_space = act_space

  def initial(self, batch_size):
    return jnp.zeros(batch_size)

  def policy(self, latent, state):
    batch_size = len(state)
    shape = (batch_size,) + self.act_space.shape
    if self.act_space.discrete:
      dist = jaxutils.OneHotDist(jnp.zeros(shape))
    else:
      dist = tfd.Uniform(-jnp.ones(shape), jnp.ones(shape))
      dist = tfd.Independent(dist, 1)
    return {'action': dist}, state

  def train(self, imagine, start, data):
    return None, {}

  def report(self, data):
    return {}


class Explore(nj.Module):

  REWARDS = {
      'disag': expl.Disag,
  }

  def __init__(self, wm, act_space, config, task_reward='reward'):
    self.config = config
    self.rewards = {}
    critics = {}
    # Rewards that the exploration policy should optimize
    for key, scale in config.expl_rewards.items():
      if not scale and not config.expl_value_without_scale:
        continue
      if key == 'extr':
        # Extrinsic reward
        rewfn = lambda s: wm.heads[task_reward](s).mean()[1:]
        critics[key] = agent.VFunction(rewfn, config, name=key)
      elif key == 'vlm_reward':
        # Predicted vlm reward / motif
        if config.vlm_reward_key == task_reward:
          rewfn = lambda s: wm.heads[task_reward](s).mean()[1:]
        else:
          rewfn = lambda s: jnp.squeeze(wm.heads[config.vlm_reward_key](s).mean()[1:])
        critics[key] = agent.VFunction(rewfn, config, name=key)
      else:
        # Ensemble disagreement
        rewfn = self.REWARDS[key](
            wm, act_space, config, name=key + '_reward')
        critics[key] = agent.VFunction(rewfn, config, name=key)
        self.rewards[key] = rewfn
    scales = {k: v for k, v in config.expl_rewards.items() if v}
    self.ac = agent.ImagActorCritic(
      critics, scales, act_space, config,
      config.expl_reward_ops == 'sum',
      **config.perc_based_scaling,
      name='ac',
    )

  def initial(self, batch_size):
    return self.ac.initial(batch_size)

  def policy(self, latent, state):
    return self.ac.policy(latent, state)

  def train(self, imagine, start, data):
    metrics = {}
    for key, rewfn in self.rewards.items():
      mets = rewfn.train(data)
      metrics.update({f'{key}_k': v for k, v in mets.items()})
    traj, mets = self.ac.train(imagine, start, data)
    metrics.update(mets)
    return traj, metrics

  def report(self, data):
    return {}
