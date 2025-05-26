
import minihack
import gym

import os

from minihack import MiniHackNavigation, LevelGenerator, MiniHackSkill
from minihack.envs.minigrid import MiniGridHack
from minihack.reward_manager import SequentialRewardManager

from minihack.envs.keyroom import MiniHackKeyDoor
from minihack.level_generator import PATH_DAT_DIR

from nle import nethack

# New action spaces
BOOTACTIONS = tuple(nethack.CompassCardinalDirection) + (
    nethack.Command.WEAR,
    nethack.Command.FIRE,
)

ARMORACTIONS = tuple(nethack.CompassCardinalDirection) + (
            nethack.Command.WEAR,
            nethack.Command.FIRE,
            nethack.Command.TAKEOFF,
        )

class KeyRoomGenerator:
    def __init__(self, room_size, subroom_size, lit):
        des_path = os.path.join(PATH_DAT_DIR, "key_and_door_tmp.des")
        with open(des_path) as f:
            df = f.read()

        if room_size <= 15:
            df = df.replace("RS", str(room_size))
        else:
            df = df.replace("RS", str(room_size), 1)
            df = df.replace("RS", str(15))
        df = df.replace("SS", str(subroom_size))
        if not lit:
            df = df.replace("lit", str("unlit"))

        self.des_file = df

    def get_des(self):
        return self.des_file

class MyMiniHackKeyRoom(MiniHackKeyDoor):
    def __init__(self, *args, room_size, subroom_size, lit, **kwargs):
        lev_gen = KeyRoomGenerator(room_size, subroom_size, lit)
        des_file = lev_gen.get_des()
        super().__init__(*args, des_file=des_file, **kwargs)

# New KeyRoom Versions
class MiniHackKeyRoom20x15(MyMiniHackKeyRoom):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, room_size=20, subroom_size=5, lit=True, **kwargs
        )

class MiniHackKeyRoom25x15(MyMiniHackKeyRoom):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, room_size=25, subroom_size=5, lit=True, **kwargs
        )

class MiniHackKeyRoom30x15(MyMiniHackKeyRoom):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, room_size=30, subroom_size=5, lit=True, **kwargs
        )

class MiniHackKeyRoom35x15(MyMiniHackKeyRoom):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, room_size=35, subroom_size=5, lit=True, **kwargs
        )

class MiniHackKeyRoom40x15(MyMiniHackKeyRoom):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, room_size=40, subroom_size=5, lit=True, **kwargs
        )



class MiniHackSeaMonsters(MiniHackSkill):
    def __init__(self, *args, **kwargs):
        large_map = kwargs.pop("large", True)
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 250)
        kwargs["autopickup"] = kwargs.pop("autopickup", True)
        kwargs["character"] = "cav-hum-law-mal"
        if kwargs["autopickup"]:
            kwargs["actions"] = ARMORACTIONS
        else:
            kwargs["actions"] = tuple(nethack.CompassCardinalDirection) + (nethack.Command.PICKUP,
                                                                           nethack.Command.APPLY,
                                                                           nethack.Command.TAKEOFF,
                                                                           nethack.Command.FIRE,
                                                                           )
        self.autodressup = kwargs.pop("autowearing", True)
        self.start_undressed = kwargs.pop("start_undressed", True)
        self.remap_armor = kwargs.pop("remap_armor_key", 0)  # replace armor key in inventory with custom key
        des_file = """
MAZE: "mylevel",' '
INIT_MAP:solidfill,' '
GEOMETRY:center,center
MAP
-----------------------
|.......}}}}}}}}......|
|.......}}}}}}}}......|
|.......}}}}}}}}......|
|.......}}}}}}}}......|
|.......}}}}}}}}......|
-----------------------
ENDMAP
REGION:(0,0,22,6),lit,"ordinary"
$left_bank = selection:fillrect (1,1,3,5)
$water = selection:fillrect (8,1,13,5)
$right_bank = selection:fillrect (16,1,21,5)
BRANCH:(1,1,6,5),(0,0,0,0)
STAIR:rndcoord($right_bank),down
TERRAIN: randline (7, 4), (16, 4), 3, '.'
TERRAIN: randline (7, 4), (16, 4), 3, '.'
OBJECT:('[', "green dragon scale mail"),rndcoord($left_bank),blessed,30
LOOP [4] {
    MONSTER: (';', "piranha"), rndcoord(filter('W', $water)), hostile
}
"""
        if large_map:
            des_file = """
MAZE: "mylevel",' '
INIT_MAP:solidfill,' '
GEOMETRY:center,center
MAP
-------------------------------------------------------------
|..............}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}.............|
|..............}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}.............|
|..............}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}.............|
|..............}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}.............|
|..............}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}.............|
-------------------------------------------------------------
ENDMAP
REGION:(0,0,80,6),lit,"ordinary"
$left_bank = selection:fillrect (1,1,3,5)
$water = selection:fillrect (15,1,46,5)
$right_bank = selection:fillrect (48,1,59,5)
BRANCH:(8,1,12,5),(0,0,0,0)
STAIR:rndcoord($right_bank),down
TERRAIN: randline (14, 4), (48, 4), 3, '.'
TERRAIN: randline (14, 4), (48, 4), 3, '.'
OBJECT:('[', "green dragon scale mail"),rndcoord($left_bank),blessed,30
LOOP [4] {
    MONSTER: (';', "piranha"), rndcoord(filter('W', $water)), hostile
}
            """
        self.pop_inv_obs = False
        self.wearing_armor = False
        if self.autodressup:
            if "inv_glyphs" not in kwargs["observation_keys"]:
                kwargs["observation_keys"] = kwargs.pop("observation_keys") + ("inv_glyphs",)
                self.pop_inv_obs = True
        super().__init__(*args, des_file=des_file, **kwargs)
        if self.autodressup:
            self.action_space = gym.spaces.Discrete(4)

    def reset(self):
        self.wearing_armor = False
        outs = super().reset()
        if not self.start_undressed:
            return outs
        obs, r, done, info = super().step(6)  # take off everything you wear
        return obs

    def step(self, action):

        obs, r, done, info = super().step(action)
        if not done and self.autodressup and not self.wearing_armor and obs["inv_glyphs"][5] != 5976:
            obs, r, done, info = super().step(4)  # puton
            if not done:
                obs, r, done, info = super().step(5)  # armor
                self.wearing_armor = True
        if self.pop_inv_obs:
            obs.pop("inv_glyphs")
        elif self.remap_armor and self.wearing_armor:
            obs["inv_glyphs"][5] = self.remap_armor  # replace armor in inventory
        return obs, r, done, info

class MiniHackWaterBoots(MiniHackSkill):
    def __init__(self, *args, **kwargs):
        random_water = kwargs.pop("random_water", True)
        water_tiles = kwargs.pop("water_tiles", 100)
        mix_tiles = kwargs.pop("mix_tiles", False)
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 200)
        kwargs["autopickup"] = kwargs.pop("autopickup", True)
        kwargs["character"] = "cav-hum-law-mal"
        if kwargs["autopickup"]:
            kwargs["actions"] = BOOTACTIONS
            self.autobootup = kwargs.pop("autowearing", True)
        else:
            kwargs["actions"] = tuple(nethack.CompassCardinalDirection) + (nethack.Command.PICKUP,
                                                                           nethack.Command.APPLY,
                                                                           nethack.Command.FIRE)
            self.autobootup = False  # automatic boot wearing only with autopickup possible

        self.remap_boots = kwargs.pop("remap_boots_key", 0) # replace boots key in inventory with custom key
        xl_map = kwargs.pop("xl_map", True)
        if xl_map:
            des_file = """
MAZE: "mylevel", ' '
FLAGS:hardfloor,premapped
GEOMETRY:center,center
MAP
------------------------------------------------------------------------
|...................................................................PP.|
|...................................................................PP.|
|...................................................................PP.|
|...................................................................PP.|
------------------------------------------------------------------------
ENDMAP
BRANCH: (19,2,19,2),(20,3,20,3)
REGION: (0,0,70,50), lit, "ordinary"
$left = selection: fillrect (1,1,1,4)
$right = selection: fillrect (22,1,69,4)
$exit = selection: fillrect (70,1,70,4)
STAIR: rndcoord($exit),down

OBJECT:('[',"water walking boots"),rndcoord($left),blessed,0
"""
        else:
            des_file = """
MAZE: "mylevel", ' '
FLAGS:hardfloor,premapped
GEOMETRY:center,center
MAP
----------------------------------------------------
|...............................................PP.|
|...............................................PP.|
|...............................................PP.|
|...............................................PP.|
----------------------------------------------------

ENDMAP
BRANCH: (9,2,9,2),(10,3,10,3)
REGION: (0,0,50,50), lit, "ordinary"
$left = selection: fillrect (1,1,2,4)
$right = selection: fillrect (10,1,49,4)
STAIR: (50, 4),down

OBJECT:('[',"water walking boots"), (1,1),blessed,0
"""

        self.pop_inv_obs = False
        self.wearing_boots = False
        if self.autobootup:
            if "inv_glyphs" not in kwargs["observation_keys"]:
                kwargs["observation_keys"] = kwargs.pop("observation_keys") + ("inv_glyphs",)
                self.pop_inv_obs = True

        if random_water: # randomly add water
            for i in range(water_tiles):
                if mix_tiles and i % 2 == 0:
                    des_file = des_file + "\n TERRAIN: rndcoord($right), 'W'"
                else:
                    des_file = des_file + "\n TERRAIN: rndcoord($right), 'P'"
        else:  # checkerboard pattern
            for x_i in range(6, 49):
                for y_i in range(1, 5):
                    if (x_i % 2 == 0 and y_i % 2 == 1) or (x_i % 2 == 1 and y_i % 2 == 0):
                        des_file = des_file + f"\n TERRAIN: ({x_i}, {y_i}), 'P'"
        super().__init__(*args, des_file=des_file, **kwargs)
        if self.autobootup:
            self.action_space = gym.spaces.Discrete(4)

    def reset(self):
        self.wearing_boots = False
        return super().reset()

    def step(self, action):
        obs, r, done, info = super().step(action)
        if not done and self.autobootup and not self.wearing_boots and obs["inv_glyphs"][5] != 5976:
            obs, r, done, info = super().step(4)  # puton
            if not done:
                obs, r, done, info = super().step(5)  # boots
                self.wearing_boots = True
        if self.pop_inv_obs:
            obs.pop("inv_glyphs")
        elif self.remap_boots and self.wearing_boots:
            obs["inv_glyphs"][5] = self.remap_boots  # replace boots in inventory
        return obs, r, done, info

class MiniGridHackChest(MiniGridHack):

    def __init__(self, *args, **kwargs):
        kwargs["autopickup"] = kwargs.pop("autopickup", True)
        if kwargs["autopickup"]:
            kwargs["actions"] = tuple(nethack.CompassCardinalDirection) + (nethack.Command.OPEN,)
        else:
            kwargs["actions"] = tuple(nethack.CompassCardinalDirection) + (nethack.Command.PICKUP, nethack.Command.OPEN)
        reward_manager = SequentialRewardManager()
        reward_manager.add_message_event(["g - a key.", "h - a key."], reward=0)
        reward_manager.add_message_event(["Seems like something lootable over there."], reward=1)
        kwargs["reward_manager"] = reward_manager
        self.rm = reward_manager
        super().__init__(*args, **kwargs)

    def get_env_map(self, env):
        door_pos = []
        goal_pos = None
        empty_strs = 0
        empty_str = True
        env_map = []
        key_pos = None

        for j in range(env.grid.height):
            str = ""
            for i in range(env.width):

                c = env.grid.get(i, j)
                if c is None:
                    str += "."
                    continue
                empty_str = False
                if c.type == "wall":
                    str += self.wall
                elif c.type == "door":
                    str += "+"
                    door_pos.append((i, j - empty_strs))
                elif c.type == "floor":
                    str += "."
                elif c.type == "lava":
                    str += "L"
                elif c.type == "goal":
                    goal_pos = (i, j - empty_strs)
                    str += "."
                elif c.type == "player":
                    str += "."
                elif c.type == "key":
                    str += "."
                    key_pos = (i, j - empty_strs)
                elif c.type == "ball":
                    str += "."
                    goal_pos = (i, j - empty_strs)
            if not empty_str and j < env.grid.height - 1:
                if set(str) != {"."}:
                    str = str.replace(".", " ", str.index(self.wall))
                    inv = str[::-1]
                    str = inv.replace(".", " ", inv.index(self.wall))[::-1]
                    env_map.append(str)
            elif empty_str:
                empty_strs += 1

        start_pos = (int(env.agent_pos[0]), int(env.agent_pos[1]) - empty_strs)
        env_map = "\n".join(env_map)

        return env_map, start_pos, goal_pos, door_pos, key_pos

    def get_env_desc(self):
        self.minigrid_env.reset()
        env = self.minigrid_env

        map, start_pos, goal_pos, door_pos, key_pos = self.get_env_map(env)

        lev_gen = LevelGenerator(map=map)

        lev_gen.set_start_pos(start_pos)
        lev_gen.footer += f"CONTAINER:('(',\"chest\"),not_trapped,{goal_pos}" + "{OBJECT:('`', \"boulder\")}"
        if key_pos is not None:
            lev_gen.add_object("skeleton key", '(', key_pos)
        else:
            lev_gen.add_object("skeleton key", '(', key_pos)
            assert False

        for d in door_pos:
            lev_gen.add_door(self.door_state, d)

        lev_gen.wallify()

        return lev_gen.get_des()

    def reset(self):
        self.rm.current_event_idx = 0
        return super().reset()


class MiniHackMultiRoomChestN2(MiniGridHackChest):
    def __init__(self, *args, **kwargs):
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 100)
        super().__init__(
            *args, env_name="MiniGrid-MultiRoom-N2-S4-v0", **kwargs
        )

class MiniHackMultiRoomChestN4(MiniGridHackChest):
    def __init__(self, *args, **kwargs):
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 400)
        super().__init__(
            *args, env_name="MiniGrid-MultiRoom-N4-S5-v0", **kwargs
        )

class MiniHackKeyChestCorridorS3(MiniGridHackChest):
    def __init__(self, *args, **kwargs):
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 400)
        super().__init__(
            *args, env_name="MiniGrid-KeyCorridorS3R3-v0", **kwargs
        )

class MiniHackKeyChestCorridorS4(MiniGridHackChest):
    def __init__(self, *args, **kwargs):
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 400)
        super().__init__(
            *args, env_name="MiniGrid-KeyCorridorS4R3-v0", **kwargs
        )
