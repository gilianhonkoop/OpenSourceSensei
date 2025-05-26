import os
import random
# import rom parent
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)


import ruamel.yaml as yaml
from dreamerv3.my_crafter.env import Env
from dreamerv3.embodied import replay as dreamer_replay
from dreamerv3.embodied.core.basics import unpack
from dreamerv3.embodied.core.config import Config
from dreamerv3.embodied.core.path import Path
import imageio
import numpy as np
import pickle
import os
from PIL import Image
from utils_no_minihack import (
    scan,
    setup_video,
    concatenate_images,
    generate_pairs,
)



if __name__ == "__main__":

    saveimg = True
    obs_keys = ['image', 'terrain', 'objects', 'daylight', 'sleep', 'is_first']
    items = [  'health', 'food', 'drink', 'energy', 'sapling', 'wood', 'stone', 'coal', 'iron', 'diamond',
               'wood_pickaxe', 'stone_pickaxe', 'iron_pickaxe', 'wood_sword', 'stone_sword', 'iron_sword']
    initial_inventory = {'health':9, 'food':9, 'drink':9, 'energy':9}
    for it in items:
        obs_keys.append(f'inventory_{it}')
    data_dir = (
        f"~/logdir/test_craft/replay"
    )
    filenames = scan(data_dir, capacity=None, shorten=0)
    random.shuffle(filenames)
    print("file names=", filenames)

    total_length = 0
    max_length = 2000 # divisble by 100
    num_pairs = 36  #50

    all_obs = {}
    for k in obs_keys:
        all_obs[k] = []

    non_finished_uuids = []
    non_finished_uuids_filenames = []
    for filename in filenames:
        with Path(filename).open("rb") as f:
            if filename.stem.split("-")[2] == "0000000000000000000000":
                non_finished_uuids.append(filename.stem.split("-")[1])
                non_finished_uuids_filenames.append(filename)
                pass
            else:

                data = np.load(f)
                print("KEYS= ", data.keys())
                for k in data.keys():
                    print(k)
                length = int(filename.stem.split("-")[3])
                total_length += length
                print(filename, "with length =", length)
                for k in obs_keys:
                    all_obs[k].extend(data[k][:length, ...])
        if total_length > max_length:
            break

    print("Length of non-finished uuids: ", len(non_finished_uuids_filenames))
    print("Total length without duplicates: ", total_length)

    if total_length < max_length:
        for filename in non_finished_uuids_filenames:
            with Path(filename).open("rb") as f:
                data = np.load(f)
                length = int(filename.stem.split("-")[3])
                # first_indices = np.where(np.logical_and(np.append(np.diff(data["is_first"][:length]), 0)!=0, data["is_first"][:length]))
                total_length += length
                for k in obs_keys:
                    all_obs[k].extend(data[k][:length, ...])
            if total_length > max_length:
                break

    print("Total length after adding non_finished files as well: ", total_length)

    samples_to_keep = max_length
    # reshape the obs_gylphs to pseudo episodes (256 each)

    obs_array = {}
    glyph_array = {}
    for k in obs_keys:
        obs_array[k] = np.asarray(all_obs[k])[:samples_to_keep]

    # obs_descriptions_array.shape: (1000000, 5, 5, 84)
    print("obs image shape: ", obs_array['image'] .shape)

    ep_length = 100 #64 ##1024
    buffer = obs_array['image'].reshape(-1, ep_length, 64, 64, 3)
    terrain_buffer = obs_array['terrain'].reshape(-1, ep_length, 9, 7)
    objects_buffer = obs_array['objects'].reshape(-1, ep_length, 9, 7)
    daylight_buffer = obs_array['daylight'].reshape(-1, ep_length)
    sleeping_buffer = obs_array['sleep'].reshape(-1, ep_length)
    first_buffer = obs_array['is_first'].reshape(-1, ep_length)

    inventory_buffer = {}
    for it in items:
        inventory_buffer[it] = obs_array[f'inventory_{it}'].reshape(-1, ep_length)

    def inventory_message(inventory, game_state=0):
        inventory_used = {k: v for k, v in inventory.items() if v>0}
        items_used = len(inventory_used)
        game_state_number = "first" if game_state == 0 else "second"
        message = f"In the {game_state_number} game state the player has "
        if items_used ==0:
            message += f"nothing in its inventory."
        num = 0
        for k, v in inventory_used.items():
            if num == items_used-1:
                message += f"and {v} {k} in its inventory."
            else:
                message += f"{v} {k}, "
            num += 1
        return message

    print("Buffer shape: ", buffer.shape)
    num_rollouts = len(buffer)
    pair_indices = generate_pairs(num_rollouts, ep_length, num_pairs)
    print(pair_indices.shape)

    crafter_env = Env(size=(512, 512))
    local_view = crafter_env.get_local_view()

    if saveimg:
        img_dir = f"llava_test" #f"crafter_test"
        # img_dir_vertical = os.path.join(img_dir, "vertical_pairs")

        os.makedirs(img_dir, exist_ok=True)
        # os.makedirs(img_dir_vertical, exist_ok=True)

        for pair_i in range(num_pairs):
            rollout1, rollout2, t1, t2 = pair_indices[pair_i]
            #glyph_img1 = glyph_mapper_low_res._glyph_to_rgb(buffer[rollout1, t1, ...])
            #glyph_img2 = glyph_mapper_low_res._glyph_to_rgb(buffer[rollout2, t2, ...])
            #img1 = Image.fromarray(buffer[rollout1, t1, ...])
            #img2 = Image.fromarray(buffer[rollout2, t2, ...])

            #img1.save(os.path.join(img_dir, f"pair_{pair_i}_0.png"))
            #img2.save(os.path.join(img_dir, f"pair_{pair_i}_1.png"))

            # first image
            sampled_iventory1 = {it: invent[rollout1, t1] for it, invent in inventory_buffer.items()}
            if first_buffer[rollout1, t1, ...]:  # due to bug first inventory is incorrectly logged
                sampled_iventory1 = initial_inventory

            canvas1 = Image.fromarray(crafter_env.render_from_glyphs(texture_ids=terrain_buffer[rollout1, t1, ...],
                                                                     object_ids=objects_buffer[rollout1, t1, ...],
                                                                     daylight=daylight_buffer[rollout1, t1],
                                                                     sleeping=sleeping_buffer[rollout1, t1],
                                                                     inventory=sampled_iventory1),)
            canvas1.save(os.path.join(img_dir, f"pair_{pair_i}_0_terrain.png"))
            print(inventory_message(sampled_iventory1), t1, first_buffer[rollout1, t1, ...])

            # second image
            sampled_iventory2 = {it: invent[rollout2, t2] for it, invent in inventory_buffer.items()}
            if first_buffer[rollout2, t2, ...]:  # due to bug first inventory is incorrectly logged
                sampled_iventory2 = initial_inventory

            canvas2 = Image.fromarray(crafter_env.render_from_glyphs(texture_ids=terrain_buffer[rollout2, t2, ...],
                                                                     object_ids=objects_buffer[rollout2, t2, ...],
                                                                     daylight=daylight_buffer[rollout2, t2],
                                                                     sleeping=sleeping_buffer[rollout2, t2],
                                                                     inventory=sampled_iventory2), )
            canvas2.save(os.path.join(img_dir, f"pair_{pair_i}_1_terrain.png"))
            print(inventory_message(sampled_iventory2), t2, first_buffer[rollout2, t2, ...])

            #canvas1 = Image.fromarray(local_view.get_canvas(unit=(56, 56), texture_ids=terrain_buffer[rollout1, t1, ...]))
            #canvas1.save(os.path.join(img_dir, f"pair_{pair_i}_0_terrain.png"))
            #canvas2 = Image.fromarray(local_view.get_canvas(unit=(56, 56), texture_ids=terrain_buffer[rollout2, t2, ...]))
            #canvas2.save(os.path.join(img_dir, f"pair_{pair_i}_1_terrain.png"))

