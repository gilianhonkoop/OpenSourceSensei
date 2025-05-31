import copy
import os

import gym
import numpy as np
import smart_settings
import torch
from gym import Wrapper

from motif.eval_utils import get_batch_dict_for_reward_model, reward_model_with_mb
from motif.reward_model import RewardModel


class RewardWrapper(gym.Wrapper):
    def __init__(self, env, model_dir='/is/rg/al/Datasets/robodesk/Robodesk/motif_reward/motif_image', img_key='image', model_cpt_id=49, device=torch.device("cpu")):
        """Constructor for the Reward wrapper.

        Args:
            env: Environment to be wrapped.
            img_key: Key of obs to get correct image size (image vs. hr_image in Dreamer codebase)
        """
        Wrapper.__init__(self, env)

        self.reward_key = "motif_reward"

        # update observation space
        wrapped_observation_space = env.observation_space
        assert isinstance(wrapped_observation_space, gym.spaces.Dict)
        observation_space = {name: copy.deepcopy(space) for name, space in wrapped_observation_space.spaces.items()}
        observation_space[self.reward_key] = gym.spaces.Box(-np.inf, np.inf, (), np.float32)
        observation_space['state'] = gym.spaces.Box(-np.inf, np.inf, (73,), np.float64)
        self.observation_space = gym.spaces.Dict(observation_space)

        self.img_key = img_key

        params = smart_settings.load(os.path.join(model_dir, "settings.json"), make_immutable=False)
        self.motif_reward_model = RewardModel(params["reward_model_params"]["model_params"], device=device)
        self.motif_reward_model.load(
                    os.path.join(model_dir, f"checkpoint_{model_cpt_id}")
                )

    def step(
        self, action
    ):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        obs_dict, reward, done, info = self.env.step(action)
        return self.update_obs_dict(obs_dict), reward, done, info
    
    def reset(self):
        obs_dict = self.env.reset()
        return self.update_obs_dict(obs_dict)

    def update_obs_dict(self, obs_dict):
        # Saving the simulator state as well for re-rendering afterwards
        obs_dict["state"] = self.physics.get_state()
        return self.compute_motif_reward(obs_dict)

    
    @torch.no_grad()
    def compute_motif_reward(self, obs_dict):
        # for now only support for one option!
        if self.motif_reward_model.encoder.use_obs_vec:
            rollout_images = None
            rollout_images_left = None
            obs_vec = np.concatenate(
                [
                    obs_dict["qpos_robot"],
                    obs_dict["qvel_robot"],
                    obs_dict["end_effector"],
                    obs_dict["qpos_objects"],
                    obs_dict["qvel_objects"],
                ]
            )
            # I put this here as a dummy thing because right now it actually doesn't exist in this form!
            # And we would need to make it more generic and non-robodesk specific with the keys!
        elif self.motif_reward_model.encoder.use_image:
            im_size = self.motif_reward_model.encoder.image_resolution
            assert im_size == obs_dict[self.img_key].shape[0]
            rollout_images = obs_dict[self.img_key].reshape(-1, im_size, im_size, 3)
            rollout_images_left = None
            obs_vec = None
        else:
            raise NotImplementedError
        batch_dict = get_batch_dict_for_reward_model(
            self.motif_reward_model, rollout_images, obs_vec, rollout_images_left
        )
        reward_dict = reward_model_with_mb(self.motif_reward_model, batch_dict)
        obs_dict[self.reward_key] = reward_dict.rewards.squeeze().detach().cpu().numpy().item()
        return obs_dict
    
if __name__ == "__main__":

    import robodesk

    env = robodesk.RoboDesk(task='open_slide', reward='dense', action_repeat=1, episode_length=500, image_size=64)
    obs = env.reset()

    done = False
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

    print("------------- Checking wrapper ---------")
    env_wrapper = RewardWrapper(env)
    obs = env_wrapper.reset()
    for i in range(5):
        action = env_wrapper.action_space.sample()
        obs, reward, done, info = env_wrapper.step(action)    
        print(obs["motif_reward"])
