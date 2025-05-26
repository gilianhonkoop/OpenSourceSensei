import os

import copy
import gym


class RemoveObsWrapper(gym.Wrapper):
    def __init__(self, env, blacklist_keys=[]):
        """Removes parts of the observation

        Args:
            blacklist_keys: Keys of observation space to remove
        """
        gym.Wrapper.__init__(self, env)

        self.blacklist_keys = blacklist_keys

        # update observation space
        wrapped_observation_space = env.observation_space
        assert isinstance(wrapped_observation_space, gym.spaces.Dict)
        observation_space = {}
        for name, space in wrapped_observation_space.spaces.items():
            if name not in blacklist_keys:
                observation_space.update({name: copy.deepcopy(space)})
        self.observation_space = gym.spaces.Dict(observation_space)

    def _filter_obs(self, obs):
        return {key: value for key, value in obs.items() if key not in self.blacklist_keys}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._filter_obs(obs), reward, done, info

    def reset(self):
        return self._filter_obs(self.env.reset())
