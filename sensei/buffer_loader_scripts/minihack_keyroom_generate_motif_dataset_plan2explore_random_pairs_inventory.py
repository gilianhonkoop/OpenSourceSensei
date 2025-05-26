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
from utils import (
    scan,
    setup_video,
    concatenate_images,
    GlyphMapperCustom,
    generate_pairs,
)
from minihack_utils import *

if __name__ == "__main__":

    run_name = "test_minihack_s15_explore_autopickup_inventory_FINAL_restricted"
    job_id = 2
    data_dir = (
        f"/fast/csancaktar/dreamerv3/{run_name}/working_directories/{job_id}/replay"
    )
    filenames = scan(data_dir, capacity=None, shorten=0)

    obs_glyphs = []
    obs_messages = []
    obs_is_first = []
    obs_descriptions = []
    total_length = 0
    obs_inv_glpyhs = []
    obs_reset_inv_length = []
    obs_inv_strs = []

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
                obs_glyphs.extend(data["glyphs_crop"][:length, ::13, ::13])
                obs_messages.extend(data["message"][:length, ...])
                obs_is_first.extend(data["is_first"][:length])
                obs_descriptions.extend(data["screen_descriptions_crop"][:length, ...])
                
                obs_inv_glpyhs.extend(data["inv_glyphs"][:length,...])
                obs_reset_inv_length.extend(data["reset_inventory_length"][:length,...])
                obs_inv_strs.extend(data["inv_strs"][:length,...])

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
            obs_messages.extend(data["message"][:length, ...])
            obs_is_first.extend(data["is_first"][:length])
            obs_descriptions.extend(data["screen_descriptions_crop"][:length, ...])

            obs_inv_glpyhs.extend(data["inv_glyphs"][:length,...])
            obs_reset_inv_length.extend(data["reset_inventory_length"][:length,...])
            obs_inv_strs.extend(data["inv_strs"][:length,...])

    print("Total length after adding non_finished files as well: ", total_length)

    # Initialize GlyphMapper
    glyph_mapper = GlyphMapperCustom(patch_size=14, remap_warrior=True)

    samples_to_keep = (total_length // 1000) * 1000
    # reshape the obs_gylphs to pseudo episodes (256 each)

    obs_gylphs_array = np.asarray(obs_glyphs)[:samples_to_keep]
    obs_descriptions_array = np.asarray(obs_descriptions)[:samples_to_keep]
    obs_messages_array = np.asarray(obs_messages)[:samples_to_keep]

    obs_inv_glpyhs_array = np.asarray(obs_inv_glpyhs)[:samples_to_keep]
    obs_reset_inv_length_array = np.asarray(obs_reset_inv_length)[:samples_to_keep]
    obs_inv_strs_array = np.asarray(obs_inv_strs)[:samples_to_keep]

    # obs_descriptions_array.shape: (1000000, 5, 5, 84)
    print("obs glpyhs shape: ", obs_gylphs_array.shape)
    print(
        "messages shape: ",
        obs_messages_array.shape,
        type(obs_messages_array[0]),
        type(obs_messages_array[0][0]),
    )

    ep_length = 250

    buffer = obs_descriptions_array.reshape(-1, ep_length, 5, 5, 84)

    buffer_glpyhs = obs_gylphs_array.reshape(-1, ep_length, 5, 5)
    buffer_messages = obs_messages_array.reshape(-1, ep_length, 256)

    buffer_inv_glyphs = obs_inv_glpyhs_array.reshape(-1, ep_length, 55)
    buffer_inv_strs = obs_inv_strs_array.reshape(-1, ep_length, 55, 80)

    print("Buffer obs_descriptions_array shape: ", buffer.shape)

    print("Buffer obs glpyhs shape: ", buffer_glpyhs.shape)

    # Check if there are enough samples for non-both interactions!
    # if not generate more and loop!
    num_rollouts = len(buffer)

    num_pairs = 100000
    pair_indices = generate_pairs(num_rollouts, ep_length, int(num_pairs))
    # each row contains: [rollout_id1, rollout_id2, t_1, t_2]

    save_dataset = True

    if save_dataset:
        motif_dataset_dir = (
            "/fast/csancaktar/sensei_datasets/minihack/keyroom_s15_p2e_run2_random_dataset_100K"
        )
        os.makedirs(motif_dataset_dir, exist_ok=True)

        os.makedirs(os.path.join(motif_dataset_dir, "data"), exist_ok=True)
        os.makedirs(os.path.join(motif_dataset_dir, "preference"), exist_ok=True)

    # ------- Check for interestingness ---------------

    pairs_interestingness = annotate_pair_for_interestingsness_minihack(
        buffer, pair_indices, mode="inventory", buffer_inventory=buffer_inv_glyphs
    )
    
    pairs_interestingness_binary = annotate_pair_for_interestingsness_minihack(
        buffer, pair_indices, mode="binary"
    )
    # pairs_interestingness_num_items = annotate_pair_for_interestingsness_minihack(
    #     buffer, pair_indices, mode="num_items"
    # )
    pairs_interestingness_distance = annotate_pair_for_interestingsness_minihack(
        buffer, pair_indices, mode="distance"
    )

    # contrastive_ind_and_maybe_both = np.where(pairs_interestingness < 2)[0]
    # print(
    #     "Number of contrastive pairs using distance annotation: ",
    #     len(contrastive_ind_and_maybe_both),
    # )

    # if len(contrastive_ind_and_maybe_both) < num_pairs:
    #     both_ind = np.where(pairs_interestingness == 2)[0]
    #     remaining_elements = num_pairs - contrastive_ind_and_maybe_both.shape[0]
    #     print(
    #         f"Adding {remaining_elements} remaining elements where both are interesting from {len(both_ind)} pairs"
    #     )
    #     assert len(both_ind) > remaining_elements

    #     contrastive_ind_and_maybe_both = np.concatenate(
    #         [contrastive_ind_and_maybe_both, both_ind[:remaining_elements]], axis=0
    #     )
    # elif len(contrastive_ind_and_maybe_both) > num_pairs:
    #     contrastive_ind_and_maybe_both = contrastive_ind_and_maybe_both[:num_pairs]

    # pairs_interestingness = pairs_interestingness[contrastive_ind_and_maybe_both]
    # # pairs_interestingness_num_items = pairs_interestingness_num_items[
    # #     contrastive_ind_and_maybe_both
    # # ]
    # pairs_interestingness_distance = pairs_interestingness_distance[
    #     contrastive_ind_and_maybe_both
    # ]

    pair_indices_dataset = pair_indices #[contrastive_ind_and_maybe_both]

    if save_dataset:
        np.save(
            os.path.join(motif_dataset_dir, "preference", "preferences_gt_inventory.npy"),
            pairs_interestingness,
        )
        np.save(
            os.path.join(
                motif_dataset_dir, "preference", "preferences_gt_binary.npy"
            ),
            pairs_interestingness_binary,
        )
        np.save(
            os.path.join(
                motif_dataset_dir, "preference", "preferences_gt_distance.npy"
            ),
            pairs_interestingness_distance,
        )

        np.save(
            os.path.join(motif_dataset_dir, "data", "pair_indices.npy"), pair_indices_dataset
        )

    assert pair_indices_dataset.shape[0] == num_pairs

    # Now re-render all pairs with the glyph mapper and save!
    # pair_indices: [rollout_id1, rollout_id2, t_1, t_2]

    # observations_paired_dataset: [number_of_samples, 2, 1, env_states_dim]
    # (250000, 2, 1, 74)

    saveimg = False

    #### --------- LOW RES IMAGES --------- ####
    glyph_mapper_low_res = GlyphMapperCustom(patch_size=16)

    width = height = glyph_mapper_low_res.patch_size * 5
    minihack_image_dataset_lowres = np.zeros(
        (num_pairs, 2, 1, width, height, 3), dtype=np.uint8
    )
    # observations_paired_dataset: [number_of_samples, 2, 1, env_states_dim]
    # (250000, 2, 1, 74)

    minihack_messages_dataset = np.zeros((num_pairs, 2, 1, 256), dtype=np.uint8)
    minihack_inventory_binary_dataset = np.zeros((num_pairs, 2, 1, 1), dtype=np.uint8)
    minihack_inventory_glyphs_dataset = np.zeros((num_pairs, 2, 1, 10), dtype=np.int64)
    minihack_inventory_strings_dataset = np.zeros((num_pairs, 2, 1, 10, 80), dtype=np.uint8)
    minihack_glyphs_dataset = np.zeros((num_pairs, 2, 1, 5, 5), dtype=np.int64)

    if saveimg:
        img_dir = f"dreamer_minihack_w_interestingness_filtered_new_{run_name}_{job_id}"
        img_dir_vertical = os.path.join(img_dir, "vertical_pairs")

        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(img_dir_vertical, exist_ok=True)

    for pair_i in range(num_pairs):
        rollout1, rollout2, t1, t2 = pair_indices_dataset[pair_i]
        glyph_img1 = glyph_mapper_low_res._glyph_to_rgb(
            buffer_glpyhs[rollout1, t1, ...]
        )
        glyph_img2 = glyph_mapper_low_res._glyph_to_rgb(
            buffer_glpyhs[rollout2, t2, ...]
        )

        minihack_image_dataset_lowres[pair_i, 0, 0, ...] = glyph_img1
        minihack_image_dataset_lowres[pair_i, 1, 0, ...] = glyph_img2

        minihack_messages_dataset[pair_i, 0, 0, ...] = buffer_messages[
            rollout1, t1, ...
        ]
        minihack_messages_dataset[pair_i, 1, 0, ...] = buffer_messages[
            rollout2, t2, ...
        ]

        minihack_inventory_glyphs_dataset[pair_i, 0, 0, ...] = buffer_inv_glyphs[rollout1, t1, :10]
        minihack_inventory_glyphs_dataset[pair_i, 1, 0, ...] = buffer_inv_glyphs[rollout2, t2, :10]

        minihack_inventory_binary_dataset[pair_i, 0, 0, 0] = 1 if 2102 in buffer_inv_glyphs[rollout1, t1, ...] else 0
        minihack_inventory_binary_dataset[pair_i, 1, 0, 0] = 1 if 2102 in buffer_inv_glyphs[rollout2, t2, ...] else 0

        minihack_inventory_strings_dataset[pair_i, 0, 0, ...] = buffer_inv_strs[rollout1, t1, :10, ...]
        minihack_inventory_strings_dataset[pair_i, 1, 0, ...] = buffer_inv_strs[rollout2, t2, :10, ...]

        minihack_glyphs_dataset[pair_i, 0, 0, ...] = buffer_glpyhs[rollout1, t1, ...]
        minihack_glyphs_dataset[pair_i, 1, 0, ...] = buffer_glpyhs[rollout2, t2, ...]


        if saveimg:
            img1 = Image.fromarray(glyph_img1)
            img2 = Image.fromarray(glyph_img2)

            img1.save(os.path.join(img_dir, f"pair_{pair_i}_0.png"))
            img2.save(os.path.join(img_dir, f"pair_{pair_i}_1.png"))

            img_concat = concatenate_images(
                img1, img2, "vertical", 28
            )  # Example for vertical concatenation with 50px whitespace

            annotation = pairs_interestingness[pair_i]
            key_in_inventory1 = minihack_inventory_binary_dataset[pair_i, 0, 0, 0]
            key_in_inventory2 = minihack_inventory_binary_dataset[pair_i, 1, 0, 0]

            img_concat.save(
                os.path.join(
                    img_dir_vertical, f"pair_{pair_i}_annotation_{annotation}_key_first{key_in_inventory1}_second{key_in_inventory2}.png"
                )
            )

            # Decode the messages!
            message1 = "".join(map(chr, minihack_messages_dataset[pair_i, 0, 0, ...]))
            message2 = "".join(map(chr, minihack_messages_dataset[pair_i, 1, 0, ...]))
            print("====================")
            print(f"=====PAIR {pair_i} =======")
            print("Message 1: ", message1)
            print("Message 2: ", message2)
            print("====================")

        if pair_i % 1000 == 0:
            print(f"Rendered {pair_i} pairs already!")

    if save_dataset:

        with open(
            os.path.join(motif_dataset_dir, "data", f"images.pickle"), "wb"
        ) as handle:
            pickle.dump(
                minihack_image_dataset_lowres, handle, protocol=pickle.HIGHEST_PROTOCOL
            )

        np.save(
            os.path.join(motif_dataset_dir, "data", "messages.npy"),
            minihack_messages_dataset,
        )

        np.save(
            os.path.join(motif_dataset_dir, "data", "inv_glyphs.npy"),
            minihack_inventory_glyphs_dataset,
        )

        np.save(
            os.path.join(motif_dataset_dir, "data", "observations.npy"),
            minihack_inventory_binary_dataset,
        )

        np.save(
            os.path.join(motif_dataset_dir, "data", "inv_strings.npy"),
            minihack_inventory_strings_dataset,
        )

        np.save(
            os.path.join(motif_dataset_dir, "data", "obs_glyphs.npy"),
            minihack_glyphs_dataset,
        )