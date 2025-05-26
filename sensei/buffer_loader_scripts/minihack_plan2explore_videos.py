import os

import ruamel.yaml as yaml

from dreamerv3.embodied import replay as dreamer_replay
from dreamerv3.embodied.core.basics import unpack
from dreamerv3.embodied.core.config import Config
from dreamerv3.embodied.core.path import Path
import imageio
import numpy as np
import pickle
import os
from PIL import Image
from utils import scan, setup_video, concatenate_images, GlyphMapperCustom, generate_pairs, distance_to_previous_ones


if __name__ == "__main__":

    run_name = "test_phase3_minihack_gradhead" #"test_minihack_multiroom_n6_locked" # "test_minihack_multiroom_n6" #"test_minihack2"
    # /is/cluster/fast/csancaktar/dreamerv3/test_minihack_s15_explore_autopickup_inventory/
    # test_minihack_multiroom_monster_n4_explore_restricted
    # test_minihack_multiroom_n2_locked_dreamer_restricted
    job_id = 0
    data_dir = f"/fast/csancaktar/dreamerv3/{run_name}/working_directories/{job_id}/eval_replay"
    filenames = scan(data_dir, capacity=None, shorten=0)

    obs_glyphs = []
    obs_messages = []
    obs_is_first = []
    actions = []
    rewards = []
    total_length = 0

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
                length = int(filename.stem.split("-")[3])
                # first_indices = np.where(np.logical_and(np.append(np.diff(data["is_first"][:length]), 0)!=0, data["is_first"][:length]))
                total_length += length
                # for key, val in data.items():
                #     print(key, type(val[0]))

                # # print("message 1: ", "".join(map(chr, data["message"][0])))
                # print("action: ", data["action"][0])
                obs_glyphs.extend(data["glyphs_crop"][:length, ::13, ::13])
                obs_messages.extend(data["message"][:length,...])
                obs_is_first.extend(data["is_first"][:length])
                actions.extend(data["action"][:length,...])
                rewards.extend(data["reward"][:length,...])
                if filename.stem.split("-")[1] in non_finished_uuids:
                    ind = non_finished_uuids.index(filename.stem.split("-")[1])
                    non_finished_uuids.pop(ind)
                    non_finished_uuids_filenames.pop(ind)
    print("Length of non-finished uuids: ", len(non_finished_uuids_filenames))
    print("Total length without duplicates: ", total_length)

    for filename in non_finished_uuids_filenames:
        with Path(filename).open("rb") as f:
            data = np.load(f)
            length = int(filename.stem.split("-")[3])
            # first_indices = np.where(np.logical_and(np.append(np.diff(data["is_first"][:length]), 0)!=0, data["is_first"][:length]))
            total_length += length
            obs_glyphs.extend(data["glyphs_crop"][:length, ::13, ::13])
            obs_messages.extend(data["message"][:length,...])
            obs_is_first.extend(data["is_first"][:length])
            actions.extend(data["action"][:length,...])
            rewards.extend(data["reward"][:length,...])
    print("Total length after adding non_finished files as well: ", total_length)

    # Compute distances to signal beginning: 
    dist2first = distance_to_previous_ones(obs_is_first)

    # Initialize GlyphMapper
    glyph_mapper_high_res = GlyphMapperCustom(patch_size=14*10, remap_warrior=True)

    ep_length = 801
    samples_to_keep = (total_length // ep_length) * ep_length
    # reshape the obs_gylphs to pseudo episodes (256 each)

    obs_gylphs_array = np.asarray(obs_glyphs)[:samples_to_keep]
    obs_messages_array = np.asarray(obs_messages)[:samples_to_keep]
    dist2first_array = np.asarray(dist2first)[:samples_to_keep]
    actions_array = np.asarray(actions)[:samples_to_keep]
    rewards_array = np.asarray(rewards)[:samples_to_keep]

    print("Task solved: ", np.sum(rewards_array>0.9))
    
    buffer = obs_gylphs_array.reshape(-1, ep_length, 5, 5)
    buffer_messages = obs_messages_array.reshape(-1, ep_length, 256)
    buffer_dist2first = dist2first_array.reshape(-1, ep_length)
    buffer_actions = actions_array.reshape(-1, ep_length, actions_array.shape[-1])
    buffer_rewards = rewards_array.reshape(-1, ep_length)
    print("Buffer shape: ", buffer.shape)

    # save some videos as test!
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 20)
    

    for i  in range(len(buffer)-20,len(buffer)):
        seq = buffer[i,...]
        seq_message = buffer_messages[i,...]
        seq_dist2first = buffer_dist2first[i, :]
        seq_actions = buffer_actions[i,...]
        seq_reward = buffer_rewards[i,:]
        output_path = f"dreamer_minihack_videos_final_runs/{run_name}_job{job_id}_eval"

        video, video_path = setup_video(output_path, f'',"glyph_", 10)
        for t, glpyph in enumerate(seq):
          frame = glyph_mapper_high_res._glyph_to_rgb(glpyph)
          img = Image.fromarray(frame)
          draw = ImageDraw.Draw(img)
          message_t = "".join(map(chr, seq_message[t]))
          # label = f"time: {t}, message: {message_t}"
          # label = f"time: {t}, dist2first: {seq_dist2first[t]}"
          label = f"action: {seq_actions[t]}, dist2first: {seq_dist2first[t]}, reward: {seq_reward[t]}"
          draw.text((0,0), label, (255,255,255), font=font)
          frame = np.array(img)
          video.append_data(frame)
        video.close()

    # num_rollouts = len(buffer)
    # num_pairs = 500
    # pair_indices = generate_pairs(num_rollouts, ep_length, num_pairs)
    # print(pair_indices.shape)
    # num_pairs = pair_indices.shape[0]
    # # Now re-render all pairs with the glyph mapper and save!
    # # pair_indices: [rollout_id1, rollout_id2, t_1, t_2]

    # # observations_paired_dataset: [number_of_samples, 2, 1, env_states_dim]
    # # (250000, 2, 1, 74)

    # saveimg = True

    # #### --------- LOW RES IMAGES --------- ####
    # glyph_mapper_low_res = GlyphMapperCustom(patch_size=14)

    # width = height = glyph_mapper_low_res.patch_size * 5
    # minihack_image_dataset_lowres = np.zeros(
    #     (num_pairs, 2, 1, width, height, 3), dtype=np.uint8
    # )
    # # observations_paired_dataset: [number_of_samples, 2, 1, env_states_dim]
    # # (250000, 2, 1, 74)

    # if saveimg:
    #     img_dir = f"dreamer_minihack_{run_name}_{job_id}_jpeg"
    #     # img_dir_vertical = os.path.join(img_dir, "vertical_pairs")

    #     os.makedirs(img_dir, exist_ok=True)
    #     # os.makedirs(img_dir_vertical, exist_ok=True)

    # for pair_i in range(num_pairs):
    #     rollout1, rollout2, t1, t2 = pair_indices[pair_i]
    #     glyph_img1 = glyph_mapper_low_res._glyph_to_rgb(buffer[rollout1, t1, ...])
    #     glyph_img2 = glyph_mapper_low_res._glyph_to_rgb(buffer[rollout2, t2, ...])

    #     minihack_image_dataset_lowres[pair_i, 0, 0, ...] = glyph_img1
    #     minihack_image_dataset_lowres[pair_i, 1, 0, ...] = glyph_img2

    #     if saveimg:
    #         img1 = Image.fromarray(glyph_img1)
    #         img2 = Image.fromarray(glyph_img2)

    #         img1.save(os.path.join(img_dir, f"pair_{pair_i}_0.jpeg"))
    #         img2.save(os.path.join(img_dir, f"pair_{pair_i}_1.jpeg"))

    #         # img_concat = concatenate_images(
    #         #     img1, img2, "vertical", 28
    #         # )  # Example for vertical concatenation with 50px whitespace
    #         # img_concat.save(os.path.join(img_dir_vertical, f"pair_{pair_i}.png"))

    #     if pair_i % 1000 == 0:
    #         print(f"Rendered {pair_i} pairs already!")

    # with open(os.path.join('/fast/csancaktar/minihack_motif_dataset',f'images_lowres_paired_minihack_{run_name}_{job_id}.pickle'), 'wb') as handle:
    #     pickle.dump(minihack_image_dataset_lowres, handle, protocol=pickle.HIGHEST_PROTOCOL)
