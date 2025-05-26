import os
import numpy as np
from dreamerv3.embodied.core.path import Path
from PIL import Image

# for the environment initialization
from dreamerv3.embodied.core.config import Config
from dreamerv3.train import make_env
import ruamel.yaml as yaml

def scan(directory, capacity=None, shorten=0):
    directory = Path(directory)
    filenames, total = [], 0
    for filename in sorted(directory.glob('*.npz')):
        if capacity and total >= capacity:
            break
        # print(filename)
        filenames.append(filename)
        total += max(0, int(filename.stem.split('-')[3]) - shorten)
    # print(total)
    return filenames


# Function to overwrite some values in replay buffer, e.g. vlm_rewards from past motif annotations.
# Takes replay buffer from working_dir and applies a function modify( ) to each dict.
# The new replay buffer is stored in f"{target_dir}/replay"
def copy_modified_replay(working_dir, target_dir, modify):
    data_dir = f"{working_dir}/replay"
    tar_dir = f"{target_dir}/replay"
    os.makedirs(tar_dir, exist_ok=True)
    filenames = scan(data_dir, capacity=None, shorten=0)
    for filename in filenames:
        with Path(filename).open("rb") as f:
            x = np.load(f)
            y = modify(dict(**x))
            target_f = os.path.join(tar_dir, filename.stem)
            np.savez(target_f, **y)


def relabel_motif(obs, motif_wrapper):
    obs.pop('motif_reward')
    obs = motif_wrapper.compute_motif_reward(obs)
    return obs

# example modify() function that only keeps red channel in images of replay buffer
def redify(x):
    x['image'][:, :, :, 1:] = 0
    return x

# Function to overwrite some values in replay buffer, e.g. vlm_rewards from past motif annotations.
# Takes replay buffer from working_dir and applies a function modify( ) to each dict.
# The new replay buffer is stored in f"{target_dir}/replay"
def copy_modified_replay_wrerendering(working_dir, target_dir, modify, env):
    data_dir = f"{working_dir}/replay"
    tar_dir = f"{target_dir}/replay"
    os.makedirs(tar_dir, exist_ok=True)
    filenames = scan(data_dir, capacity=None, shorten=0)

    # img_dir = "/fast/csancaktar/dummy_copy_test/img"
    # os.makedirs(img_dir, exist_ok=True)

    num_file = 0
    for filename in filenames:
        with Path(filename).open("rb") as f:
            x = np.load(f)
            x = dict(x)
            hr_images = np.zeros((len(x["image"]), 224,224,3), dtype=np.uint8)
            # (data_len, 64, 64, 3) to (data_len, 224, 224, 3)

            length = int(filename.stem.split('-')[3])
            # first_indices = np.where(np.logical_and(np.append(np.diff(data["is_first"][:length]), 0)!=0, data["is_first"][:length]))

            buffer_og_images = x["image"]
            for sample_i in range(len(x["state"])):
                env._env._env.reset()
                env._env._env.physics.forward()

                # flag_invalid = 0
                if sample_i<length:
                    env._env._env.physics.set_state(x["state"][sample_i])
                    env._env._env.physics.forward()
                    hr_images[sample_i,...] = env._env._env.render(resize=False)
                else:
                    # flag_invalid = 1
                    # print("INVALID!")
                    img_og = Image.fromarray(buffer_og_images[sample_i,...])
                    resized_img = img_og.resize((224, 224), Image.ANTIALIAS)
                    hr_images[sample_i,...] = np.array(resized_img)

                # # if not flag_invalid and sample_i<length:
                # img2 = Image.fromarray(buffer_og_images[sample_i,...])
                # img2.save(os.path.join(img_dir, f"file{num_file}_{sample_i}_og_invalid{flag_invalid}.png"))
                
                # img1 = Image.fromarray(hr_images[sample_i,...])
                # img1.save(os.path.join(img_dir, f"file{num_file}_{sample_i}_hr_invalid{flag_invalid}.png"))
        
            num_file += 1

            x["image"] = hr_images
            y = modify(dict(**x))

            # re-add the old images
            y["image"] = buffer_og_images
            target_f = os.path.join(tar_dir, filename.stem)
            np.savez(target_f, **y)

            print("Finished: ", filename)