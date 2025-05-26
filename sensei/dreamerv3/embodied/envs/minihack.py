import minihack  # Required, otherwise environments don't register
from minihack.tiles import GlyphMapper
import functools
import numpy as np
from nle import nethack
import embodied
import gym
from embodied.core import base
import os
from dreamerv3.embodied.core import space as spacelib

from dreamerv3.embodied.core.wrappers import ResizeImage
from dreamerv3.embodied.envs.from_gym import FromGym
from dreamerv3.embodied.envs.minihack_utils import MinihackInteractionTracker


TOURIST_ID = 339
WARRIOR_ID = 337
CAVEMAN_ID = 329
PRIESTESS_ID = 335
SAMURAI_ID = 335

STAIRCASE_UP_ID = 2382
FLOOR_ID = 2378
KEY_ID = 2102

# register custom levels
gym.envs.register(
    id="MiniHack-MultiRoomChest-N4-v0",
    entry_point="dreamerv3.embodied.envs.minihack_levels:MiniHackMultiRoomChestN4",
)
gym.envs.register(
    id="MiniHack-MultiRoomChest-N2-v0",
    entry_point="dreamerv3.embodied.envs.minihack_levels:MiniHackMultiRoomChestN2",
)

gym.envs.register(
    id="MiniHack-KeyChestCorridor-S3-v0",
    entry_point="dreamerv3.embodied.envs.minihack_levels:MiniHackKeyChestCorridorS3",
)
gym.envs.register(
    id="MiniHack-KeyChestCorridor-S4-v0",
    entry_point="dreamerv3.embodied.envs.minihack_levels:MiniHackKeyChestCorridorS4",
)
gym.envs.register(
    id="MiniHack-WaterWalkingBoots-v0",
    entry_point="dreamerv3.embodied.envs.minihack_levels:MiniHackWaterBoots",
)

gym.envs.register(
    id="MiniHack-SeaMonsters-v0",
    entry_point="dreamerv3.embodied.envs.minihack_levels:MiniHackSeaMonsters",
)

gym.envs.register(
    id="MiniHack-KeyRoom-S20-v0",
    entry_point="dreamerv3.embodied.envs.minihack_levels:MiniHackKeyRoom20x15",
)

gym.envs.register(
    id="MiniHack-KeyRoom-S25-v0",
    entry_point="dreamerv3.embodied.envs.minihack_levels:MiniHackKeyRoom25x15",
)

gym.envs.register(
    id="MiniHack-KeyRoom-S30-v0",
    entry_point="dreamerv3.embodied.envs.minihack_levels:MiniHackKeyRoom30x15",
)

gym.envs.register(
    id="MiniHack-KeyRoom-S35-v0",
    entry_point="dreamerv3.embodied.envs.minihack_levels:MiniHackKeyRoom35x15",
)

gym.envs.register(
    id="MiniHack-KeyRoom-S40-v0",
    entry_point="dreamerv3.embodied.envs.minihack_levels:MiniHackKeyRoom40x15",
)


class WrappedMiniHack(FromGym):
    def __init__(self, task: str, remap_to_tourist: bool, remap_staircase: bool, **kwargs):
        restr_actions = kwargs.pop("restrict_actions", False)
        kwargs["autopickup"] = kwargs.pop("autopickup", False)

        if 'KeyRoom' in task:
            kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 600)
        elif 'N4' in task:
            kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 120 * 4)
        elif 'N6' in task:
            kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 120 * 6)
        elif 'Boots' in task:
            kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 200)
            kwargs["remap_boots_key"] = KEY_ID  # treat boots as a key when in inventory
        elif 'SeaMonsters' in task:
            kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 200)
            kwargs["remap_armor_key"] = KEY_ID  # treat armor as a key when in inventory

        if kwargs["autopickup"] and 'KeyRoom' in task:
            if restr_actions:
                kwargs['actions'] = tuple(nethack.CompassCardinalDirection) + (nethack.Command.APPLY,)
            else:
                kwargs['actions'] = tuple(nethack.CompassCardinalDirection) + tuple(
                    nethack.CompassIntercardinalDirection) + (nethack.Command.APPLY,)
        elif restr_actions:
            if 'KeyRoom' in task:
                kwargs['actions'] = tuple(nethack.CompassCardinalDirection) + (
                nethack.Command.PICKUP, nethack.Command.APPLY)
            elif 'Quest' in task:
                kwargs['actions'] = tuple(nethack.CompassCardinalDirection) + (
                nethack.Command.ZAP, nethack.Command.RUSH)
                kwargs['character'] = "cav-hum-cha-fem"
            elif 'MultiRoom' in task:
                if 'Locked' in task:
                    kwargs['actions'] = tuple(nethack.CompassCardinalDirection) + (
                    nethack.Command.OPEN, nethack.Command.KICK)
                    # kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 160)
                else:
                    kwargs['actions'] = tuple(nethack.CompassCardinalDirection)
        super().__init__(f"MiniHack-{task}-v0",
                         observation_keys=["pixel_crop", "message", "screen_descriptions_crop", "inv_glyphs",
                                           "inv_strs"], **kwargs)
        self._env._glyph_mapper = GlyphMapperRemapped(remap_to_tourist=remap_to_tourist,
                                                      remap_staircase=remap_staircase)
        self.interaction_tracker = MinihackInteractionTracker()

    @functools.cached_property
    def obs_space(self):
        spaces = super().obs_space
        spaces['image'] = spaces.pop('pixel_crop')
        # spaces.pop('glyphs_crop')
        for k in self.interaction_tracker.interaction_func_dict.keys():
            spaces[f'rew_int_{k}'] = spacelib.Space(np.float32, (1,), -np.inf, np.inf)
        spaces['key_in_inv'] = spacelib.Space(bool, (1,))
        return spaces

    def step(self, action):
        obs = super().step(action)
        obs['image'] = obs.pop('pixel_crop')
        if self._info:
            obs['is_terminal'] = int(self._info['end_status']) >= 1
        obs['key_in_inv'] = np.array([KEY_ID in obs['inv_glyphs']])
        obs = self.interaction_tracker.update_obs(obs)
        return obs


class MiniHack(ResizeImage):
    def __init__(self, *args, size, **kwargs):
        super().__init__(WrappedMiniHack(*args, **kwargs), size=size)
        if 'screen_descriptions_crop' in self._keys:
            self._keys.remove('screen_descriptions_crop')
        if 'inv_strs' in self._keys:
            self._keys.remove('inv_strs')
        self.reset_inventory_length = 0

    @functools.cached_property
    def obs_space(self):
        spaces = super().obs_space
        spaces['reset_inventory_length'] = spacelib.Space(np.int32, (1,), 0, 55)
        return spaces

    def step(self, action):
        obs = super().step(action)
        if action['reset']:
            self.reset_inventory_length = sum(obs["inv_glyphs"] != 5976)
        obs["reset_inventory_length"] = self.reset_inventory_length
        obs["reward"] = np.clip(obs["reward"], 0, 1)
        return obs


# class MotifMiniHack(embodied.Env):
#     def __init__(
#         self,
#         task: str,
#         remap_to_tourist: bool,
#         size,
#         sliding_avg=0,
#         deriv_scaling=False,
#         deriv_ema_alpha=0.09,
#         deriv_scaling_alpha=0.35,
#         clipping_min=None,
#         clipping_max=None,
#         motif_model_dir="/is/rg/al/Datasets/robodesk/Robodesk/motif_reward/motif_highres_image",
#         model_cpt_id="49",
#         **kwargs,
#     ):
#         restr_actions = kwargs.pop("restrict_actions", False)
#         if restr_actions:
#             if 'KeyRoom' in task:
#                 kwargs['actions'] = tuple(nethack.CompassCardinalDirection) + (nethack.Command.PICKUP, nethack.Command.APPLY)
#             elif 'Quest' in task:
#                 kwargs['actions'] = tuple(nethack.CompassCardinalDirection) + (nethack.Command.ZAP, nethack.Command.RUSH)
#                 kwargs['character'] = "cav-hum-cha-fem"
#             elif 'MultiRoom' in task:
#                 if 'Locked' in task:
#                     kwargs['actions'] = tuple(nethack.CompassCardinalDirection) + (nethack.Command.OPEN, nethack.Command.KICK)
#                     kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 160)
#                 else:
#                     kwargs['actions'] = tuple(nethack.CompassCardinalDirection)

#         self._minihack_env = gym.make(f"MiniHack-{task}-v0",  observation_keys=["pixel_crop", "message", "screen_descriptions_crop"], **kwargs)

#         # print(self._minihack_env.observation_space)

#         self._minihack_env._glyph_mapper = GlyphMapperRemapped(remap_to_tourist=remap_to_tourist)
#         self._resized_minihack_env = ResizeImageCustom(self._minihack_env, size=size, ignored_keys=['screen_descriptions_crop'])

#         from .motif_env_wrapper import RewardWrapper
#         self._gymenv = RewardWrapper(
#             self._resized_minihack_env,
#             model_dir=motif_model_dir,
#             img_key="pixel_crop",
#             sliding_avg=sliding_avg,
#             deriv_scaling=deriv_scaling,
#             deriv_ema_alpha=deriv_ema_alpha,
#             deriv_scaling_alpha=deriv_scaling_alpha,
#             clipping_min=clipping_min,
#             clipping_max=clipping_max,
#             model_cpt_id=model_cpt_id,
#         )

#         from . import from_gym
#         self._env = from_gym.FromGym(self._gymenv)

#     @property
#     def obs_space(self):
#         spaces = self._env.obs_space
#         print(spaces.keys())
#         if "image" not in spaces.keys() and "pixel_crop" in spaces.keys():
#             spaces["image"] = spaces.pop("pixel_crop")
#         # spaces.pop('glyphs_crop')
#         return spaces

#     @property
#     def act_space(self):
#         return self._env.act_space

#     def step(self, action):
#         obs = self._env.step(action)
#         print("Inside step of main Motif wrapper!")
#         if "image" not in obs.keys() and "pixel_crop" in obs.keys():
#             obs["image"] = obs.pop("pixel_crop")
#         # obs.pop('glyphs_crop')
#         return obs

class ResizeImageCustom(base.Wrapper):

    def __init__(self, env, ignored_keys=[], size=(64, 64)):
        super().__init__(env)
        self._size = size
        self._keys = [
            k for k, v in env.observation_space.spaces.items()
            if len(v.shape) > 1 and v.shape[:2] != size and k not in ignored_keys]
        print(f'Resizing keys {",".join(self._keys)} to {self._size}.')
        if self._keys:
            from PIL import Image
            self._Image = Image

    @functools.cached_property
    def obs_space(self):
        spaces = self.env.observation_space.spaces
        for key in self._keys:
            shape = self._size + spaces[key].shape[2:]
            spaces[key] = spacelib.Space(np.uint8, shape)
        return spaces

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        for key in self._keys:
            obs[key] = self._resize(obs[key])
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        for key in self._keys:
            obs[key] = self._resize(obs[key])
        return obs

    def _resize(self, image):
        image = self._Image.fromarray(image)
        image = image.resize(self._size, self._Image.NEAREST)
        image = np.array(image)
        return image


#### ============================== FIRST MOTIF without resize wrapper, and wrapper only after!

class WrappedMotifMiniHack(embodied.Env):
    def __init__(
            self,
            task: str,
            remap_to_tourist: bool,
            remap_staircase: bool,
            sliding_avg=0,
            deriv_scaling=False,
            deriv_ema_alpha=0.09,
            deriv_scaling_alpha=0.35,
            clipping_min=None,
            clipping_max=None,
            motif_model_dir="/is/rg/al/Datasets/robodesk/Robodesk/motif_reward/motif_highres_image",
            model_cpt_id="49",
            **kwargs,
    ):
        restr_actions = kwargs.pop("restrict_actions", False)
        kwargs["autopickup"] = kwargs.pop("autopickup", False)

        if 'KeyRoom' in task:
            kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 600)  
        elif 'N4' in task:
            kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 120 * 4)
        elif 'N6' in task:
            kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 120 * 6)

        if kwargs["autopickup"] and 'KeyRoom' in task:
            if restr_actions:
                kwargs['actions'] = tuple(nethack.CompassCardinalDirection) + (nethack.Command.APPLY,)
            else:
                kwargs['actions'] = tuple(nethack.CompassCardinalDirection) + tuple(
                    nethack.CompassIntercardinalDirection) + (nethack.Command.APPLY,)
        elif restr_actions:
            if 'KeyRoom' in task:
                kwargs['actions'] = tuple(nethack.CompassCardinalDirection) + (
                nethack.Command.PICKUP, nethack.Command.APPLY)
            elif 'Quest' in task:
                kwargs['actions'] = tuple(nethack.CompassCardinalDirection) + (
                nethack.Command.ZAP, nethack.Command.RUSH)
                kwargs['character'] = "cav-hum-cha-fem"
            elif 'MultiRoom' in task:
                if 'Locked' in task:
                    kwargs['actions'] = tuple(nethack.CompassCardinalDirection) + (
                    nethack.Command.OPEN, nethack.Command.KICK)
                else:
                    kwargs['actions'] = tuple(nethack.CompassCardinalDirection)

        self._minihack_env = gym.make(f"MiniHack-{task}-v0",
                                      observation_keys=["pixel_crop", "message", "screen_descriptions_crop",
                                                        "inv_glyphs", "inv_strs"], **kwargs)

        self._minihack_env._glyph_mapper = GlyphMapperRemapped(remap_to_tourist=remap_to_tourist,
                                                               remap_staircase=remap_staircase)
        # self._resized_minihack_env = ResizeImageCustom(self._minihack_env, size=size, ignored_keys=['screen_descriptions_crop'])

        # self._rerendered_minihack_env = ReRenderCustom(self._minihack_env, new_key="rerendered_image", new_patch_size=14) 

        from .motif_env_wrapper import RewardWrapper
        self._gymenv = RewardWrapper(
            self._minihack_env,
            model_dir=motif_model_dir,
            img_key="pixel_crop",  # before with rerendered env: rerendered_image
            sliding_avg=sliding_avg,
            deriv_scaling=deriv_scaling,
            deriv_ema_alpha=deriv_ema_alpha,
            deriv_scaling_alpha=deriv_scaling_alpha,
            clipping_min=clipping_min,
            clipping_max=clipping_max,
            model_cpt_id=model_cpt_id,
        )

        from . import from_gym
        # self._env = from_gym.FromGym(self._gymenv)

        # from . import env_wrappers
        # self._env = from_gym.FromGym(
        # env_wrappers.RemoveObsWrapper(self._gymenv, ['rerendered_image']),  # remove rerendered_image from observation again to save RAM
        # )
        self._env = from_gym.FromGym(self._gymenv)
        self.interaction_tracker = MinihackInteractionTracker()

    @property
    def obs_space(self):
        spaces = self._env.obs_space
        if "image" not in spaces.keys() and "pixel_crop" in spaces.keys():
            spaces["image"] = spaces.pop("pixel_crop")
        # spaces.pop('glyphs_crop')
        for k in self.interaction_tracker.interaction_func_dict.keys():
            spaces[f'rew_int_{k}'] = spacelib.Space(np.float32, (1,), -np.inf, np.inf)
        spaces['key_in_inv'] = spacelib.Space(bool, (1,))
        return spaces

    @property
    def act_space(self):
        return self._env.act_space

    def step(self, action):
        obs = self._env.step(action)
        if "image" not in obs.keys() and "pixel_crop" in obs.keys():
            obs["image"] = obs.pop("pixel_crop")
        if self._env._info:
            obs['is_terminal'] = int(self._env._info['end_status']) >= 1
        obs['key_in_inv'] = np.array([KEY_ID in obs['inv_glyphs']])
        obs = self.interaction_tracker.update_obs(obs)
        return obs


class MotifMiniHack(ResizeImage):
    def __init__(self, *args, size, **kwargs):
        super().__init__(WrappedMotifMiniHack(*args, **kwargs), size=size)
        if 'screen_descriptions_crop' in self._keys:
            self._keys.remove('screen_descriptions_crop')
        if 'inv_strs' in self._keys:
            self._keys.remove('inv_strs')
        self.reset_inventory_length = 0

    @functools.cached_property
    def obs_space(self):
        spaces = super().obs_space
        spaces['reset_inventory_length'] = spacelib.Space(np.int32, (1,), 0, 55)
        return spaces

    def step(self, action):
        obs = super().step(action)
        if action['reset']:
            self.reset_inventory_length = sum(obs["inv_glyphs"] != 5976)
        obs["reset_inventory_length"] = self.reset_inventory_length
        obs["reward"] = np.clip(obs["reward"], 0, 1)
        return obs


class ReRenderCustom(base.Wrapper):

    def __init__(self, env, new_key="rerendered_image", new_patch_size=14):
        super().__init__(env)
        self.new_key = new_key
        self.custom_glpyh_mapper = GlyphMapperRemapped(remap_to_tourist=self.env._glyph_mapper.remap_to_tourist,
                                                       patch_size=new_patch_size)
        self.new_size = (new_patch_size * 5, new_patch_size * 5, 3)

    @functools.cached_property
    def obs_space(self):
        spaces = self.env.observation_space.spaces
        spaces[self.new_key] = spacelib.Space(np.uint8, self.new_size)
        return spaces

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs[self.new_key] = self.custom_glpyh_mapper._glyph_to_rgb(obs["glyphs_crop"])
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs[self.new_key] = self.custom_glpyh_mapper._glyph_to_rgb(obs["glyphs_crop"])
        return obs


from minihack.tiles import glyph2tile, MAXOTHTILE
import pkg_resources
import pickle
from PIL import Image


class GlyphMapperRemapped:
    """This class is used to map glyphs to rgb pixels."""

    def __init__(self, patch_size=16, remap_to_tourist=True, remap_staircase=True):
        self.tiles = self.load_tiles()
        self.patch_size = patch_size
        self.remap_to_tourist = remap_to_tourist
        self.remap_staircase = remap_staircase

    def load_tiles(self):
        """This function expects that tile.npy already exists.
        If it doesn't, call make_tiles.py in win/
        """

        tile_rgb_path = os.path.join(
            pkg_resources.resource_filename("minihack", "tiles"),
            "tiles.pkl",
        )

        return pickle.load(open(tile_rgb_path, "rb"))

    def glyph_id_to_rgb(self, glyph_id):
        tile_id = glyph2tile[glyph_id]
        assert 0 <= tile_id <= MAXOTHTILE
        return self.tiles[tile_id]

    def upsample_glyph_tile(self, tile_img, patch_size):
        tile_large = Image.fromarray(tile_img).resize((patch_size, patch_size))
        return np.array(tile_large)

    def _glyph_to_rgb(self, glyphs):
        # Expects glhyphs as two-dimensional numpy ndarray
        cols = None
        col = None

        for i in range(glyphs.shape[1]):
            for j in range(glyphs.shape[0]):
                current_glyph = glyphs[j, i]
                if self.remap_to_tourist and current_glyph in [WARRIOR_ID, CAVEMAN_ID, PRIESTESS_ID, SAMURAI_ID]:
                    current_glyph = TOURIST_ID
                elif self.remap_staircase and current_glyph == STAIRCASE_UP_ID:
                    current_glyph = FLOOR_ID
                rgb = self.glyph_id_to_rgb(
                    current_glyph
                )  # print("patch rgb shape: ", rgb.shape)
                # Each glpyh is 16x16 --> upsample to desired patch size
                if self.patch_size != 16:
                    rgb = self.upsample_glyph_tile(rgb, self.patch_size)
                if col is None:
                    col = rgb
                else:
                    col = np.concatenate((col, rgb))

            if cols is None:
                cols = col
            else:
                cols = np.concatenate((cols, col), axis=1)
            col = None

        return cols

    def to_rgb(self, glyphs):
        return self._glyph_to_rgb(glyphs)