import numpy as np

class MinihackInteractionTracker():
    def __init__(self) -> None:
        self.key_message1 = np.array([103,  32,  45,  32,  97,  32, 107, 101, 121,  32, 110,  97, 109,
        101, 100,  32,  84, 104, 101,  32,  77,  97, 115, 116, 101, 114,
            32,  75, 101, 121,  32, 111, 102,  32,  84, 104, 105, 101, 118,
        101, 114, 121,  46,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0], dtype=np.uint8)
        
        self.key_message2 = np.array([103, 32,  45,  32,  97,  32, 107, 101, 121,  46,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0], dtype=np.uint8)

        self.key_message_start = np.array([ 89, 111, 117,  32, 115, 101, 101,  32, 104, 101, 114, 101,  32,
            97,  32, 107, 101, 121,  32, 110,  97, 109, 101, 100,  32,  84,
        104, 101,  32,  77,  97, 115, 116, 101, 114,  32,  75, 101, 121,
            32, 111, 102,  32,  84, 104, 105, 101, 118, 101, 114, 121,  46,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0], dtype=np.uint8)
       
        self.interaction_func_dict = {
            "is_key_picked_up": self.is_key_picked_up,
            # "is_agent_on_key": self.is_agent_on_key, # will comment out again
            "is_door_open": self.is_door_open,
            "is_infront_of_locked_door": self.is_infront_of_locked_door,
            # "is_infront_of_locked_door_wkey": self.is_infront_of_locked_door_wkey,
            # "is_chest_in_frame_wkey": self.is_chest_in_frame_wkey,
        }

    def is_key_picked_up(self, obs):
        # two message types for key_pickup: 
        #     " 'h - a key named The Master Key of Thievery.',"
        #     and  'g - a key named The Master Key of Thievery.',
        # keymessage2 is now for keychest that doesn't have the part ...named Master Key...
        return (obs["message"][1:]==self.key_message1[1:]).all() or (obs["message"][1:]==self.key_message2[1:]).all()*1

    def is_agent_on_key(self, obs):
        return (obs["message"]==self.key_message_start).all()*1

    def is_door_open(self, obs):
        OPEN_DOOR_ID1 = 2372
        OPEN_DOOR_ID2 = 2373
        return ((obs["glyphs_crop"] == OPEN_DOOR_ID1).any() or (obs["glyphs_crop"] == OPEN_DOOR_ID2).any())*1

    def is_infront_of_locked_door(self, obs):
        CLOSED_DOOR_ID1 = 2374
        CLOSED_DOOR_ID2 = 2375

        # agent_loc = [2,2]
        indices = np.array([[2, 1], [1, 2], [2,3], [3,2], [1,1], [1,3], [3,1], [3,3]])
        elements = obs["glyphs_crop"][tuple(indices.T)]
        if CLOSED_DOOR_ID1 in elements or CLOSED_DOOR_ID2 in elements:
            return 1
        else: 
            return 0

    def is_infront_of_locked_door_wkey(self, obs):
        CLOSED_DOOR_ID1 = 2374
        CLOSED_DOOR_ID2 = 2375

        # agent_loc = [2,2]
        indices = np.array([[2, 1], [1, 2], [2,3], [3,2], [1,1], [1,3], [3,1], [3,3]])
        elements = obs["glyphs_crop"][tuple(indices.T)]
        if (CLOSED_DOOR_ID1 in elements or CLOSED_DOOR_ID2 in elements) and obs["key_in_inv"]:
            return 1
        else: 
            return 0

    def is_chest_in_frame_wkey(self, obs):
        CHEST_ID = 2096
        return ((obs["glyphs_crop"] == CHEST_ID).any() and obs["key_in_inv"])*1

    def update_obs(self, obs):
        for k, v_func in self.interaction_func_dict.items():
            obs[f"rew_int_{k}"] = v_func(obs)
        return obs
    