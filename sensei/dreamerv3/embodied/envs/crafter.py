import embodied
import numpy as np

class Crafter(embodied.Env):

  def __init__(self, task, size=(64, 64), outdir=None, log_inventory=False, log_glyphs=False, hr_image=True, hr_size=(512, 512),seed=None):
    assert task in ('reward', 'noreward')
    import my_crafter as crafter

    self._hr_image = hr_image
    self._hr_size = hr_size

    self._env = crafter.Env(size=size, reward=(task == 'reward'), seed=seed)
    if outdir:
      outdir = embodied.Path(outdir)
      self._env = crafter.Recorder(
          self._env, outdir,
          save_stats=True,
          save_video=False,
          save_episode=False,
      )
    self._achievements = crafter.constants.achievements.copy()
    self._items = crafter.constants.items.copy()
    self._log_inventory = log_inventory
    self._log_glyphs = log_glyphs
    self._done = True


  @property
  def obs_space(self):
    spaces = {
        'image': embodied.Space(np.uint8, self._env.observation_space.shape),
        'reward': embodied.Space(np.float32),
        'is_first': embodied.Space(bool),
        'is_last': embodied.Space(bool),
        'is_terminal': embodied.Space(bool),
        'log_reward': embodied.Space(np.float32),
    }
    spaces.update({
        f'log_achievement_{k}': embodied.Space(np.int32)
        for k in self._achievements})
    if self._log_inventory:
      spaces.update({f'inventory_{k}': embodied.Space(np.int32) for k in self._items})
    if self._hr_image:
      spaces.update({'hr_image': embodied.Space(np.uint8, (self._hr_size[0], self._hr_size[1], 3))})
    if self._log_glyphs:
      spaces.update({'terrain': embodied.Space(np.uint8, (9, 7)),
                     'objects': embodied.Space(np.uint8, (9, 7)),
                     'daylight': embodied.Space(np.float32),
                     'sleep': embodied.Space(bool)})

    return spaces

  @property
  def act_space(self):
    return {
        'action': embodied.Space(np.int32, (), 0, self._env.action_space.n),
        'reset': embodied.Space(bool),
    }

  def step(self, action):
    if action['reset'] or self._done:
      self._done = False
      image = self._env.reset()
      return self._obs(image, 0.0, {}, is_first=True)
    image, reward, self._done, info = self._env.step(action['action'])
    reward = np.float32(reward)
    return self._obs(
        image, reward, info,
        is_last=self._done,
        is_terminal=info['discount'] == 0)

  def _obs(
      self, image, reward, info,
      is_first=False, is_last=False, is_terminal=False):
    log_achievements = {
        f'log_achievement_{k}': info['achievements'][k] if info else 0
        for k in self._achievements}
    if self._log_inventory:
      log_inventory = {f'inventory_{k}': info['inventory'][k] if info else 0 for k in self._items}
      log_achievements.update(log_inventory)

    if self._hr_image:
        log_achievements.update({'hr_image': self._env.render(self._hr_size)})

    if self._log_glyphs:
        terrains, objs, daylight, sleep = self._env.get_glyphs()
        log_achievements.update({'terrain': terrains, 'objects': objs,
                                 'daylight': daylight, 'sleep': sleep})

    return dict(
        image=image,
        reward=reward,
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal,
        log_reward=np.float32(info['reward'] if info else 0.0),
        **log_achievements,
    )

  def render(self):
    return self._env.render()