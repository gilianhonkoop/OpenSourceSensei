import os

import ruamel.yaml as yaml

from dreamerv3.embodied import replay as dreamer_replay
from dreamerv3.embodied.core.basics import unpack
from dreamerv3.embodied.core.config import Config
from dreamerv3.embodied.core.path import Path
import numpy as np
import pickle
import os
from utils import scan, setup_video, concatenate_images, GlyphMapperCustom, generate_pairs, distance_to_previous_ones

import numpy as np
import copy

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
            "is_agent_on_key": self.is_agent_on_key,
            "is_door_open": self.is_door_open,
            "is_infront_of_locked_door": self.is_infront_of_locked_door,
        }

    def is_key_picked_up(self, glyphs_crop, message):
        # two message types for key_pickup: 
        #     " 'h - a key named The Master Key of Thievery.',"
        #     and  'g - a key named The Master Key of Thievery.',

        return (message[1:]==self.key_message1[1:]).all()

    def is_agent_on_key(self, glyphs_crop, message):
        return (message==self.key_message_start).all()

    def is_door_open(self, glyphs_crop, message):
        OPEN_DOOR_ID = 2372
        return (glyphs_crop == OPEN_DOOR_ID).any()

    def is_infront_of_locked_door(self, glyphs_crop, message):
        CLOSED_DOOR_ID = 2374
        # agent_loc = [2,2]
        indices = np.array([[2, 1], [1, 2], [2,3], [3,2]])
        elements = glyphs_crop[tuple(indices.T)]
        if CLOSED_DOOR_ID in elements:
            return 1
        else: 
            return 0

    def get_interaction_dict(self, glyphs_crop, message):
        interaction_dict = {}
        for k, v_func in self.interaction_func_dict.items():
            interaction_dict[f"{k}"] = v_func(glyphs_crop, message)
        return interaction_dict

if __name__ == "__main__":
    num_jobs = 12
    job_statistics = []
    job_lengths = []

    interaction_tracker = MinihackInteractionTracker()

    for job_id in range(num_jobs):
        run_name = "test_minihack_s15_explore_motif_fixed" 

        data_dir = f"/fast/csancaktar/dreamerv3/{run_name}/working_directories/{job_id}/replay"
        filenames = scan(data_dir, capacity=None, shorten=0)

        obs_glyphs = []
        obs_messages = []
        # obs_is_first = []
        # actions = []
        total_length = 0
        # motif_reward = []
        print(f"========== JOB: {job_id} ==========")
        non_finished_uuids = []
        non_finished_uuids_filenames = []
        for filename in filenames:
            with Path(filename).open("rb") as f:
                if filename.stem.split("-")[2] == "0000000000000000000000":
                    non_finished_uuids.append(filename.stem.split("-")[1])
                    non_finished_uuids_filenames.append(filename)
                    pass
                else:
                    try:
                        data = np.load(f)
                        length = int(filename.stem.split("-")[3])
                        # first_indices = np.where(np.logical_and(np.append(np.diff(data["is_first"][:length]), 0)!=0, data["is_first"][:length]))
                        total_length += length
                        # for key, val in data.items():
                        #     print(key, type(val[0]))

                        # # print("message 1: ", "".join(map(chr, data["message"][0])))
                        # print("action: ", data["action"][0])

                        obs_glyphs.extend(data["glyphs_crop"][:length, ::13, ::13])
                        obs_messages.extend(data["message"][:length,...])
                        # obs_is_first.extend(data["is_first"][:length])
                        # actions.extend(data["action"][:length,...])
                        # motif_reward.extend(data["motif_reward"][:length,...])
                        if filename.stem.split("-")[1] in non_finished_uuids:
                            ind = non_finished_uuids.index(filename.stem.split("-")[1])
                            non_finished_uuids.pop(ind)
                            non_finished_uuids_filenames.pop(ind)
                    except:
                        pass
        print("Length of non-finished uuids: ", len(non_finished_uuids_filenames))
        print("Total length without duplicates: ", total_length)

        for filename in non_finished_uuids_filenames:
            with Path(filename).open("rb") as f:
                try:
                    data = np.load(f)
                    length = int(filename.stem.split("-")[3])
                    # first_indices = np.where(np.logical_and(np.append(np.diff(data["is_first"][:length]), 0)!=0, data["is_first"][:length]))
                    total_length += length
                    obs_glyphs.extend(data["glyphs_crop"][:length, ::13, ::13])
                    obs_messages.extend(data["message"][:length,...])
                    # obs_is_first.extend(data["is_first"][:length])
                    # actions.extend(data["action"][:length,...])
                    # motif_reward.extend(data["motif_reward"][:length,...])
                except:
                    pass
        print("Total length after adding non_finished files as well: ", total_length)

        overall_interactions = {
        "is_key_picked_up": 0,
        "is_agent_on_key": 0, 
        "is_door_open": 0,
        "is_infront_of_locked_door" : 0,
        }

        for i_t in range(total_length):
            # getting the interactions per timestep!
            interaction_dict = interaction_tracker.get_interaction_dict(obs_glyphs[i_t], obs_messages[i_t])
            
            for k, v in interaction_dict.items(): 
                overall_interactions[k] += v * 1
            # is_key_picked_up += interaction_dict["is_key_picked_up"] * 1
            # is_agent_on_key += interaction_dict["is_agent_on_key"] * 1 
            # is_door_open += interaction_dict["is_door_open"] * 1 
            # is_infront_of_locked_door += interaction_dict["is_infront_of_locked_door"] * 1
            
            if i_t % 100000 == 0:
                print(f"Finished {i_t} samples!")
            
        for k, v in overall_interactions.items():
            print(f"Total {k} : {v}")

        job_statistics.append(copy.deepcopy(overall_interactions))
        job_lengths.append(total_length)
        
        obs_glyphs = None
        obs_messages = None

    print("ALL JOB RESULTS!")
    for i,job_dict in enumerate(job_statistics):
        print("=======================")
        print(f"Statistics for job {i}: ")
        for k, v in job_dict.items():
            print(f"job {i} total {k} : {v}")
        print("=======================")

    
    print("ALL JOBS RELATIVE!")
    for i,job_dict in enumerate(job_statistics):
        print("=======================")
        print(f"Statistics for job {i}: ")
        for k, v in job_dict.items():
            print(f"job {i} relative {k} : {v/job_lengths[i]}")
        print("=======================")

