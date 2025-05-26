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
import matplotlib.pyplot as plt
colors = ["#800020", "#0F52BA", "#228B22", "#4B0082", "#CC5500", "#B8860B", "#008080", "#36454F", "#87CEEB"]


if __name__ == "__main__":

    run_name = "test_minihack_s15_explore_motif_test4final"

    job_id = 7
    # data_dir = f"/fast/csancaktar/dreamerv3/{run_name}/working_directories/{job_id}/replay"
    data_dir = f"/fast/csancaktar/dreamerv3/{run_name}/working_directories/{job_id}/eval_replay"
    filenames = scan(data_dir, capacity=None, shorten=0)

    obs_glyphs = []
    obs_messages = []
    obs_is_first = []
    actions = []
    total_length = 0
    motif_reward = []

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
                motif_reward.extend(data["motif_reward"][:length,...])
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
            motif_reward.extend(data["motif_reward"][:length,...])
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
    motif_array = np.asarray(motif_reward)[:samples_to_keep]

    buffer = obs_gylphs_array.reshape(-1, ep_length, 5, 5)
    buffer_messages = obs_messages_array.reshape(-1, ep_length, 256)
    buffer_dist2first = dist2first_array.reshape(-1, ep_length)
    buffer_actions = actions_array.reshape(-1, ep_length, actions_array.shape[-1])
    buffer_motif = motif_array.reshape(-1, ep_length)

    print("Buffer shape: ", buffer.shape)

    # save some videos as test!
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 20)
    
    global_i = 0
    for i  in range(len(buffer)-20,len(buffer)):
        seq = buffer[i,...]
        seq_message = buffer_messages[i,...]
        seq_dist2first = buffer_dist2first[i, :]
        seq_actions = buffer_actions[i,...]
        seq_reward = buffer_motif[i,...]

        output_path = f"dreamer_minihack_rerender_motif_inventory_runs_4FINAL/{run_name}_job{job_id}_eval"

        video, video_path = setup_video(output_path, f'',"glyph_", 10)
        for t, glpyph in enumerate(seq):
          frame = glyph_mapper_high_res._glyph_to_rgb(glpyph)
          img = Image.fromarray(frame)
          draw = ImageDraw.Draw(img)
          message_t = "".join(map(chr, seq_message[t]))
          label = f"time: {t}, motif: {seq_reward[t]}"

          draw.text((0,0), label, (255,255,255), font=font)
          frame = np.array(img)
          video.append_data(frame)
        video.close()

        # Save motif reward! 
        fig, ax = plt.subplots(1, 1, figsize=(3.6, 2.1))
        x_axis = np.arange(1, len(seq) + 1)
        ax.plot(x_axis, seq_reward, color=colors[0], linewidth=1.5)
        ax.set_xlabel(r"Timestep $t$")
        ax.set_ylabel(r"Motif")

        fig.savefig(os.path.join(output_path, f"rollout_motif_{global_i}.png"), dpi=600, bbox_inches="tight")
        plt.close(fig)

        global_i += 1