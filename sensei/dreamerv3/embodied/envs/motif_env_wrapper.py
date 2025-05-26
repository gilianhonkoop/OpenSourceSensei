import copy
import os

import gym
import numpy as np
import smart_settings
import torch
from gym import Wrapper

from motif.eval_utils import get_batch_dict_for_reward_model, reward_model_with_mb
from motif.reward_model import RewardModel

class EMA_Derivative_Estimator:
    def __init__(self, alpha):
        """
        Initialize the estimator.

        Parameters:
        - alpha: The smoothing factor for EMA, between 0 and 1.
        """
        self.alpha = alpha
        self.reset()

    def estimate(self, new_value):
        """
        Estimate the derivative of the signal at the current timestep.

        Parameters:
        - new_value: The new signal value at the current timestep.

        Returns:
        - The estimated derivative at the current timestep.
        """
        if self.previous_ema is None:
            # For the very first call, we don't have a previous EMA, so use the current value
            self.previous_ema = new_value
            return 0.001  # No derivative estimate available at the first step

        # Compute the current EMA
        current_ema = self.alpha * new_value + (1 - self.alpha) * self.previous_ema

        # Estimate the derivative as the difference between the current and previous EMA
        derivative = current_ema - self.previous_ema

        # Update the previous EMA and derivative for the next call
        self.previous_ema = current_ema
        self.previous_derivative = derivative

        return derivative
    
    def reset(self):
        self.previous_ema = None  # Previous EMA value
        self.previous_derivative = 0 # Initialize previous derivative

# Motif reward wrapper for Gym environment
class RewardWrapper(gym.Wrapper):
    def __init__(self, env, model_dir='', img_key='image',
                 sliding_avg=0, deriv_scaling=False, deriv_ema_alpha = 0.09, deriv_scaling_alpha = 0.35,
                 clipping_min=None, clipping_max=None,
                 model_cpt_id=49, device=torch.device("cpu")):
        """Constructor for the Reward wrapper.

        Args:
            env: Environment to be wrapped.
            img_key: Key of obs to get correct image size (image vs hq_img in Dreamer codebase)
        """
        Wrapper.__init__(self, env)

        self.reward_key = "motif_reward"

        # update observation space
        wrapped_observation_space = env.observation_space
        assert isinstance(wrapped_observation_space, gym.spaces.Dict)
        observation_space = {name: copy.deepcopy(space) for name, space in wrapped_observation_space.spaces.items()}
        observation_space[self.reward_key] = gym.spaces.Box(-np.inf, np.inf, (), np.float32)
        self.observation_space = gym.spaces.Dict(observation_space)

        self.img_key = img_key

        assert not (sliding_avg and deriv_scaling), "Both variables should not be True at the same time"

        self.sliding_avg = sliding_avg
        self.window = []

        if deriv_scaling:
            self.ema_deriv_estimator = EMA_Derivative_Estimator(alpha=deriv_ema_alpha)
        else:
            self.ema_deriv_estimator = None
        self.deriv_scaling_alpha = deriv_scaling_alpha

        self.clipping_min = clipping_min if clipping_min is not None else -np.inf
        self.clipping_max = clipping_max if clipping_max is not None else np.inf
        assert self.clipping_max > self.clipping_min, "Max clipping value has to be greater than min!"

        if model_dir:
            params = smart_settings.load(os.path.join(model_dir, "settings.json"), make_immutable=False)
            self.motif_reward_model = RewardModel(params["reward_model_params"]["model_params"], device=device)
            self.motif_reward_model.load(
                os.path.join(model_dir, f"checkpoint_{model_cpt_id}")
            )
        else:
            # empty model dir is provided -> no motif model traiend -> motif reward is always 0
            self.motif_reward_model = None
        # print(next(self.motif_reward_model.parameters()).is_cuda)

    def step(
            self, action
    ):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        obs_dict, reward, done, info = self.env.step(action)
        return self.update_obs_dict(obs_dict), reward, done, info

    def reset(self):
        self.window = []
        if self.ema_deriv_estimator is not None:
            self.ema_deriv_estimator.reset()
        obs_dict = self.env.reset()
        return self.update_obs_dict(obs_dict)

    def update_obs_dict(self, obs_dict):
        # Saving the simulator state as well for re-rendering afterwards
        return self.compute_motif_reward(obs_dict)

    @torch.no_grad()
    def compute_motif_reward(self, obs_dict):

        if self.motif_reward_model is None:
            # no motif reward module provided ("Plan2Explore phase")
            motif_reward = 0.0
        else:
            # for now only support for one option!
            rollout_images = None
            rollout_images_left = None
            obs_vec = None
            if self.motif_reward_model.encoder.use_obs_vec:
                if "qpos_robot" in obs_dict.keys(): 
                    obs_vec = np.concatenate(
                        [
                            obs_dict["qpos_robot"],
                            obs_dict["qvel_robot"],
                            obs_dict["end_effector"],
                            obs_dict["qpos_objects"],
                            obs_dict["qvel_objects"],
                        ]
                    )
                elif "inv_glyphs" in obs_dict.keys(): 
                    key_glyph_id = 2102
                    key_in_inventory = key_glyph_id in obs_dict["inv_glyphs"]
                    obs_vec = np.array([key_in_inventory * 1.])[None]
                else:
                    raise NotImplementedError
                # I put this here as a dummy thing because right now it actually doesn't exist in this form!
                # And we would need to make it more generic and non-robodesk specific with the keys!
            if self.motif_reward_model.encoder.use_image:
                if not self.motif_reward_model.encoder.resize_image:
                    im_size = self.motif_reward_model.encoder.image_resolution
                    assert im_size == obs_dict[self.img_key].shape[0]
                else:
                    im_size = obs_dict[self.img_key].shape[0]
                rollout_images = obs_dict[self.img_key].reshape(-1, im_size, im_size, 3)
            else:
                raise NotImplementedError
            batch_dict = get_batch_dict_for_reward_model(
                self.motif_reward_model, rollout_images, obs_vec, rollout_images_left
            )
            reward_dict = reward_model_with_mb(self.motif_reward_model, batch_dict)

            motif_reward = reward_dict.rewards.squeeze().detach().cpu().numpy().item()

        if self.sliding_avg:
            self.window.append(motif_reward)
            if len(self.window) > self.sliding_avg:
                self.window.pop(0)
            motif_reward = np.mean(self.window)
        elif self.ema_deriv_estimator is not None: 
            ema_deriv = self.ema_deriv_estimator.estimate(motif_reward)
            ema_deriv = np.maximum(1e-3, ema_deriv)
            scale_for_motif = np.exp(-1 / (self.deriv_scaling_alpha * ema_deriv))
            motif_reward = motif_reward * scale_for_motif

        obs_dict[self.reward_key] = np.clip(motif_reward, a_min=self.clipping_min, a_max=self.clipping_max)
        return obs_dict


import functools
import embodied
# Motif reward wrapper for embodied.Env
class RewardWrapperEmbodied(embodied.base.Wrapper):
    def __init__(self, env, model_dir='', img_key='image',
                 sliding_avg=0, deriv_scaling=False, deriv_ema_alpha = 0.09, deriv_scaling_alpha = 0.35,
                 clipping_min=None, clipping_max=None,
                 model_cpt_id=49, device=torch.device("cpu")):
        """Constructor for the Reward wrapper.

        Args:
            env: Environment to be wrapped.
            img_key: Key of obs to get correct image size (image vs hq_img in Dreamer codebase)
        """
        embodied.base.Wrapper.__init__(self, env)

        self.reward_key = "motif_reward"
        self.img_key = img_key

        assert not (sliding_avg and deriv_scaling), "Both variables should not be True at the same time"

        self.sliding_avg = sliding_avg
        self.window = []

        if deriv_scaling:
            self.ema_deriv_estimator = EMA_Derivative_Estimator(alpha=deriv_ema_alpha)
        else:
            self.ema_deriv_estimator = None
        self.deriv_scaling_alpha = deriv_scaling_alpha

        self.clipping_min = clipping_min if clipping_min is not None else -np.inf
        self.clipping_max = clipping_max if clipping_max is not None else np.inf
        assert self.clipping_max > self.clipping_min, "Max clipping value has to be greater than min!"

        if model_dir:
            params = smart_settings.load(os.path.join(model_dir, "settings.json"), make_immutable=False)
            self.motif_reward_model = RewardModel(params["reward_model_params"]["model_params"], device=device)
            self.motif_reward_model.load(
                os.path.join(model_dir, f"checkpoint_{model_cpt_id}"), device=device,
            )
        else:
            # empty model dir is provided -> no motif model trained -> motif reward is always 0
            self.motif_reward_model = None

    @functools.cached_property
    def obs_space(self):
        spaces = self.env.obs_space
        spaces[self.reward_key] = embodied.core.Space(np.float32, shape=(1,))
        return spaces

    def step(self, action):
        if action['reset']:
            self.reset()
        obs_dict = self.env.step(action)
        return self.update_obs_dict(obs_dict)

    def reset(self):
        self.window = []
        if self.ema_deriv_estimator is not None:
            self.ema_deriv_estimator.reset()

    def update_obs_dict(self, obs_dict):
        # Saving the simulator state as well for re-rendering afterwards
        return self.compute_motif_reward(obs_dict)

    @torch.no_grad()
    def compute_motif_reward(self, obs_dict):

        if self.motif_reward_model is None:
            # no motif reward module provided ("Plan2Explore phase")
            motif_reward = 0.0
        else:
            # for now only support for one option!
            rollout_images = None
            rollout_images_left = None
            obs_vec = None
            batch_dim = -1
            if self.motif_reward_model.encoder.use_obs_vec:
                if "qpos_robot" in obs_dict.keys():
                    obs_vec = np.concatenate(
                        [
                            obs_dict["qpos_robot"],
                            obs_dict["qvel_robot"],
                            obs_dict["end_effector"],
                            obs_dict["qpos_objects"],
                            obs_dict["qvel_objects"],
                        ]
                    )
                elif "inv_glyphs" in obs_dict.keys():
                    key_glyph_id = 2102
                    key_in_inventory = key_glyph_id in obs_dict["inv_glyphs"]
                    obs_vec = np.array([key_in_inventory * 1.])[None]
                else:
                    raise NotImplementedError
                # I put this here as a dummy thing because right now it actually doesn't exist in this form!
                # And we would need to make it more generic and non-robodesk specific with the keys!
            if self.motif_reward_model.encoder.use_image:
                if not self.motif_reward_model.encoder.resize_image:
                    im_size = self.motif_reward_model.encoder.image_resolution
                    if len(obs_dict[self.img_key].shape) == 4:
                        batch_dim = obs_dict[self.img_key].shape[0]
                    else:
                        assert im_size == obs_dict[self.img_key].shape[0]
                else:
                    im_size = obs_dict[self.img_key].shape[0]
                rollout_images = obs_dict[self.img_key].reshape(batch_dim, im_size, im_size, 3)
            else:
                raise NotImplementedError
            batch_dict = get_batch_dict_for_reward_model(
                self.motif_reward_model, rollout_images, obs_vec, rollout_images_left
            )
            reward_dict = reward_model_with_mb(self.motif_reward_model, batch_dict)

            motif_reward = reward_dict.rewards.detach().cpu().numpy()  # [0]#.item()
            if batch_dim == -1:
                motif_reward = motif_reward[0]

        if self.sliding_avg:
            self.window.append(motif_reward)
            if len(self.window) > self.sliding_avg:
                self.window.pop(0)
            motif_reward = np.mean(self.window)
        elif self.ema_deriv_estimator is not None:
            ema_deriv = self.ema_deriv_estimator.estimate(motif_reward)
            ema_deriv = np.maximum(1e-3, ema_deriv)
            scale_for_motif = np.exp(-1 / (self.deriv_scaling_alpha * ema_deriv))
            motif_reward = motif_reward * scale_for_motif

        obs_dict[self.reward_key] = np.clip(motif_reward, a_min=self.clipping_min, a_max=self.clipping_max)
        return obs_dict

class MotifStandalone:

    def __init__(self, model_dir='', img_key='image',
                 clipping_min=None, clipping_max=None,
                 model_cpt_id=49, device=torch.device("cuda:0")):

        self.reward_key = "motif_reward"
        self.img_key = img_key
        self.clipping_min = clipping_min if clipping_min is not None else -np.inf
        self.clipping_max = clipping_max if clipping_max is not None else np.inf
        assert self.clipping_max > self.clipping_min, "Max clipping value has to be greater than min!"

        if model_dir:
            params = smart_settings.load(os.path.join(model_dir, "settings.json"), make_immutable=False)
            self.motif_reward_model = RewardModel(params["reward_model_params"]["model_params"], device=device)
            self.motif_reward_model.load(
                os.path.join(model_dir, f"checkpoint_{model_cpt_id}"),
            )
        else:
            # empty model dir is provided -> no motif model traiend -> motif reward is always 0
            self.motif_reward_model = None
        # print(next(self.motif_reward_model.parameters()).is_cuda)

    @torch.no_grad()
    def compute_motif_reward(self, obs_dict):

        if self.motif_reward_model is None:
            # no motif reward module provided ("Plan2Explore phase")
            motif_reward = 0.0
        else:
            # for now only support for one option!
            rollout_images = None
            rollout_images_left = None
            obs_vec = None
            batch_dim = -1
            if self.motif_reward_model.encoder.use_obs_vec:
                if "qpos_robot" in obs_dict.keys():
                    obs_vec = np.concatenate(
                        [
                            obs_dict["qpos_robot"],
                            obs_dict["qvel_robot"],
                            obs_dict["end_effector"],
                            obs_dict["qpos_objects"],
                            obs_dict["qvel_objects"],
                        ]
                    )
                elif "inv_glyphs" in obs_dict.keys():
                    key_glyph_id = 2102
                    key_in_inventory = key_glyph_id in obs_dict["inv_glyphs"]
                    obs_vec = np.array([key_in_inventory * 1.])[None]
                else:
                    raise NotImplementedError
                # I put this here as a dummy thing because right now it actually doesn't exist in this form!
                # And we would need to make it more generic and non-robodesk specific with the keys!
            if self.motif_reward_model.encoder.use_image:
                if not self.motif_reward_model.encoder.resize_image:
                    im_size = self.motif_reward_model.encoder.image_resolution
                    if len(obs_dict[self.img_key].shape) == 4:
                        batch_dim = obs_dict[self.img_key].shape[0]
                    else:
                        assert im_size == obs_dict[self.img_key].shape[0]
                else:
                    im_size = obs_dict[self.img_key].shape[0]
                rollout_images = obs_dict[self.img_key].reshape(batch_dim, im_size, im_size, 3)
            else:
                raise NotImplementedError
            batch_dict = get_batch_dict_for_reward_model(
                self.motif_reward_model, rollout_images, obs_vec, rollout_images_left
            )
            reward_dict = reward_model_with_mb(self.motif_reward_model, batch_dict)

            motif_reward = reward_dict.rewards.detach().cpu().numpy()  # [0]#.item()
            if batch_dim == -1:
                motif_reward = motif_reward[0]

        obs_dict[self.reward_key] = np.clip(motif_reward, a_min=self.clipping_min, a_max=self.clipping_max)
        return obs_dict


