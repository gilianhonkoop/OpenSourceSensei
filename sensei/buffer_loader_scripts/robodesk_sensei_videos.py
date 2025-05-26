import os

import ruamel.yaml as yaml

from dreamerv3.embodied.core.basics import unpack
from dreamerv3.embodied.core.path import Path
import imageio
import numpy as np
import pickle
import os
from PIL import Image
from utils import scan, setup_video
import matplotlib.pyplot as plt

from dreamerv3.embodied.core.config import Config
from dreamerv3.train import make_env

qvel_objects_inds = {
"drawer_joint": 0,
"slide_joint": 1, 
"red_button": 2,
"green_button": 3,
"blue_button": 4, 
"ball": [8,9,10], #xyz positions
"upright_block": [14, 15, 16],
"flat_block": [20, 21, 22],
}
colors = ["#800020", "#0F52BA", "#228B22", "#4B0082", "#CC5500", "#B8860B", "#008080", "#36454F", "#87CEEB"]


def get_interaction_metrics(qvel_objects):

  interaction_dict = {}
  for key, ind in qvel_objects_inds.items():
    # new_key = f"interaction_{key}"
    if "ball" in key or "block" in key:
      interaction_dict[key] = np.any(np.abs(qvel_objects[ind])>2*1e-2, axis=-1)
    else:
      interaction_dict[key] = np.abs(qvel_objects[ind])>2*1e-2
  return interaction_dict


if __name__ == "__main__":

    # MAKE ENVIRONMENT 
    configs = yaml.YAML(typ="safe").load((Path("dreamerv3/configs.yaml")).read())
    config = Config(configs["defaults"])
    config = config.update(configs["p2x_robodesk"])

    # config = yaml.YAML(typ="safe").load((Path(os.path.join(job_dir,"config.yaml"))).read())
    env = make_env(config)


    import numpy as np
    # job_id = 1
    # run_name = "test_motifrobodesk4_percentile_scaling1_elementwise_test"
    
    # # job_id = 3
    # # run_name = "test_motifrobodesk_final_grid_gpt_new"
    # data_dir = f"/fast/csancaktar/dreamerv3/{run_name}/working_directories/{job_id}/replay"

    job_id = 2
    run_name = "test_motifrobodesk_grid_gpt_new_general_prompt_seeds"
    data_dir = f"/is/cluster/fast/csancaktar/dreamerv3_iclr/{run_name}/working_directories/{job_id}/replay"
    
    # job_id = 8 # and 9
    # run_name = "test_motifrobodesk_grid_gpt_new2"
    # data_dir = f"/fast/csancaktar/dreamerv3_iclr/{run_name}/working_directories/{job_id}/replay"

    filenames = scan(data_dir, capacity=None, shorten=0)

    images = []
    obj_qvel = []
    motif_reward = []
    env_state = [] # also saving the environment states for re-rendering!

    total_length = 0

    non_finished_uuids = []
    non_finished_uuids_filenames = []
    for filename in filenames:
        with Path(filename).open('rb') as f:
            if filename.stem.split('-')[2] == '0000000000000000000000':
                non_finished_uuids.append(filename.stem.split('-')[1])
                non_finished_uuids_filenames.append(filename)
                pass
            else:
              data = np.load(f)
              length = int(filename.stem.split('-')[3])
              if total_length == 0:
                for k, v in data.items():
                    print(f"key: {k}, value type {type(v)}, shape: {v.shape}")
              # first_indices = np.where(np.logical_and(np.append(np.diff(data["is_first"][:length]), 0)!=0, data["is_first"][:length]))
              total_length += length
              images.extend(data["image"][:length,...])
              obj_qvel.extend(data["qvel_objects"][:length,...])
              motif_reward.extend(data["motif_reward"][:length,...])
              env_state.extend(data["state"][:length, ...])
              if filename.stem.split('-')[1] in non_finished_uuids:
                  ind = non_finished_uuids.index(filename.stem.split('-')[1])
                  non_finished_uuids.pop(ind)
                  non_finished_uuids_filenames.pop(ind)

    print("Length of non-finished uuids: ", len(non_finished_uuids_filenames))
    print("Total length without duplicates: ", total_length)

    print(filenames[-100:])
    # for filename in non_finished_uuids_filenames:
    #     with Path(filename).open('rb') as f:
    #         data = np.load(f)
    #         length = int(filename.stem.split('-')[3])
    #         # first_indices = np.where(np.logical_and(np.append(np.diff(data["is_first"][:length]), 0)!=0, data["is_first"][:length]))
    #         total_length += length
    #         images.extend(data["image"][:length,...])
    #         obj_qvel.extend(data["qvel_objects"][:length,...])
    #         motif_reward.extend(data["motif_reward"][:length,...])
    #         env_state.extend(data["state"][:length, ...])
    # print("Total length after adding non_finished files as well: ", total_length)
    # print("Length of images: ", len(images), type(images), type(images[0]))
    # print(images[0].shape)

    ep_length = 1000
    num_rollouts = 50
    for start_i  in range(len(images)-num_rollouts*ep_length, len(images), ep_length):
        seq = images[start_i:start_i+ep_length]
        seq_vel = obj_qvel[start_i:start_i+ep_length]
        seq_reward = motif_reward[start_i:start_i+ep_length]
        seq_state = env_state[start_i:start_i+ep_length]

        # output_path = f"dreamer_robodesk_promotional_videos/{run_name}/{job_id}"
        # output_path = f"dreamer_robodesk_promotional_videos/SENSEI_ceeus_oracle/{run_name}_{job_id}"
        output_path = f"dreamer_robodesk_promotional_videos/SENSEI_general/{run_name}_{job_id}"

        video, video_path = setup_video(output_path, f'_{start_i}','', 60)
        for t, frame in enumerate(seq):
          video.append_data(frame)
        video.close()

        video, video_path = setup_video(output_path, f'_large_rendering_{start_i}','', 60)
        for t, state in enumerate(seq_state):
          env._env._env.physics.set_state(state)
          env._env._env.physics.forward()
          frame_large = env._env._env.render(resize=False)
          video.append_data(frame_large)
        video.close()

        # compute interactions:
        interaction_dict = {key: [] for key in qvel_objects_inds.keys()}
        for obj_qvel_t in seq_vel: 
            new_data = get_interaction_metrics(obj_qvel_t)
            interaction_dict = {key: interaction_dict[key] + [value] for key, value in new_data.items() if key in interaction_dict}

        fig, ax = plt.subplots(1, 1, figsize=(3.6, 2.1))
        x_axis = np.arange(1, len(seq) + 1)

        i = 0
        for key, val in interaction_dict.items():
            ax.plot(x_axis, val, color=colors[i], linewidth=1.5, label=key)
            i +=1 
            
        ax.set_xlabel(r"Timestep $t$")
        ax.set_ylabel(r"interaction log")


        # Shrink current axis's height by 10% on the bottom
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                            box.width, box.height * 0.9])

        # Put a legend below current axis
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35),
                    fancybox=True, shadow=False, ncol=4, frameon=False)

        fig.savefig(os.path.join(output_path, f"rollout{start_i}.png"), dpi=600, bbox_inches="tight")
        plt.close(fig)

        # Save motif reward! 
        fig, ax = plt.subplots(1, 1, figsize=(3.6, 2.1))
        x_axis = np.arange(1, len(seq) + 1)
        ax.plot(x_axis, seq_reward, color=colors[0], linewidth=1.5)
        ax.set_xlabel(r"Timestep $t$")
        ax.set_ylabel(r"Motif")

        fig.savefig(os.path.join(output_path, f"rollout_motif_{start_i}.png"), dpi=600, bbox_inches="tight")
        plt.close(fig)