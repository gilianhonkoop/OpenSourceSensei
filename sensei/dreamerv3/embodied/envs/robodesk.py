import os

import embodied


class RoboDesk(embodied.Env):

  def __init__(self, task, mode, repeat=1, length=500, camera='top', log_all_rewards=False, resets=True):
    assert mode in ('train', 'eval')
    # TODO: This env variable is meant for headless GPU machines but may fail
    # on CPU-only machines.
    if 'MUJOCO_GL' not in os.environ:
      os.environ['MUJOCO_GL'] = 'egl'
    try:
      from my_robodesk import robodesk
    except ImportError:
      import my_robodesk as robodesk
    task, reward = task.rsplit('_', 1)
    if mode == 'eval':
      reward = 'success'
    assert reward in ('dense', 'sparse', 'success'), reward
    self._gymenv = robodesk.RoboDesk(task, reward, repeat, length,
                                     camera=camera, fix_reset=(mode == 'eval'), log_rewards=log_all_rewards)
    from . import from_gym
    from . import env_wrappers
    self._env = from_gym.FromGym(env_wrappers.RemoveObsWrapper(self._gymenv, ['hr_image']))  # remove hr_image

  @property
  def obs_space(self):
    return self._env.obs_space

  @property
  def act_space(self):
    return self._env.act_space

  def step(self, action):
    obs = self._env.step(action)
    obs['is_terminal'] = False
    return obs


class MotifRoboDesk(embodied.Env):

  def __init__(self, task, mode, repeat=1, length=500, camera='top', log_all_rewards=False, sliding_avg=0,
               deriv_scaling=False, deriv_ema_alpha = 0.09, deriv_scaling_alpha = 0.35,
               clipping_min=None, clipping_max=None,
               motif_model_dir='/is/rg/al/Datasets/robodesk/Robodesk/motif_reward/motif_highres_image', model_cpt_id="49", resets=True):
    assert mode in ('train', 'eval')
    # TODO: This env variable is meant for headless GPU machines but may fail
    # on CPU-only machines.
    if 'MUJOCO_GL' not in os.environ:
      os.environ['MUJOCO_GL'] = 'egl'
    try:
      from my_robodesk import robodesk
    except ImportError:
      import my_robodesk as robodesk
    task, reward = task.rsplit('_', 1)
    if mode == 'eval':
      reward = 'success'
    assert reward in ('dense', 'sparse', 'success'), reward
    self._roboenv = robodesk.RoboDesk(task, reward, repeat, length,
                                     camera=camera, fix_reset=(mode == 'eval'), log_rewards=log_all_rewards)

    from .motif_env_wrapper import RewardWrapper
    self._gymenv = RewardWrapper(self._roboenv, model_dir=motif_model_dir, img_key='hr_image', sliding_avg=sliding_avg,
                                 deriv_scaling=deriv_scaling, deriv_ema_alpha = deriv_ema_alpha, deriv_scaling_alpha = deriv_scaling_alpha,
                                 clipping_min=clipping_min, clipping_max=clipping_max, model_cpt_id=model_cpt_id)

    from . import from_gym
    from . import env_wrappers
    self._env = from_gym.FromGym(
      env_wrappers.RemoveObsWrapper(self._gymenv, ['hr_image']),  # remove hr_image from observation again to save RAM
    )

  @property
  def obs_space(self):
    return self._env.obs_space

  @property
  def act_space(self):
    return self._env.act_space

  def step(self, action):
    obs = self._env.step(action)
    obs['is_terminal'] = False
    return obs
