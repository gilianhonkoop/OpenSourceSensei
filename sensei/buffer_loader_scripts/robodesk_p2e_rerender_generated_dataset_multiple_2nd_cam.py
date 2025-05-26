import os

import ruamel.yaml as yaml
import random
from dreamerv3.embodied.core.path import Path
import imageio
import numpy as np
import pickle
import os
from PIL import Image
from utils import scan, setup_video, concatenate_images, generate_pairs_weighted, generate_pairs, distance_to_previous_ones

# for the environment initialization
from dreamerv3.embodied.core.config import Config
from dreamerv3.embodied import wrappers
from dreamerv3.train import make_env
from dreamerv3.embodied.core.path import Path
from buffer_loader_scripts.robodesk_utils import get_interaction_from_vel, stack_reward_from_obs, gripper_pos_to_target_distance


if __name__ == "__main__":

    dataset_dir = "/fast/csancaktar/sensei_datasets/robodesk/plan2explore_multiple_runs_contrastive"

    width = height = 224
    save_dataset = True
    saveimg = False

    if saveimg:
        img_dir = f"dreamer_robodesk_left_test2"
        img_dir_vertical = os.path.join(img_dir, "vertical_pairs")

        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(img_dir_vertical, exist_ok=True)


    num_pairs = 200000

    # MAKE ENVIRONMENT 
    configs = yaml.YAML(typ="safe").load((Path("dreamerv3/configs.yaml")).read())
    config = Config(configs["defaults"])
    config = config.update(configs["p2x_robodesk"])

    # config = yaml.YAML(typ="safe").load((Path(os.path.join(job_dir,"config.yaml"))).read())
    env = make_env(config)

    # with open(os.path.join(dataset_dir, "data", "images.pickle"), "rb") as data:
    #     images_array = pickle.load(data)
   
    env_state_dataset = np.load(os.path.join(dataset_dir, "data", "env_states.npy"))
    # env_state_dataset = np.zeros((num_pairs, 2, 1, 73))



    # initialize the empty dataset arrays!
    robodesk_image_dataset_highres = np.zeros((num_pairs, 2, 1, width, height, 3), dtype=np.uint8)
    # env_state_dataset = np.zeros((num_pairs, 2, 1, 73))

    # pairs_interestingness = np.zeros((num_pairs,))
    pairs_interestingness = np.load(os.path.join(dataset_dir, "preference", "preferences_gt.npy"))

    pair_indices_dataset = np.zeros((num_pairs, 2))

    env._env._env.reset()
    env._env._env.physics.forward()
    # env._env._env.camera = "left"
    env._env._env.render(resize=False)

    for pair_i in range(num_pairs):
                
        env_state1 = env_state_dataset[pair_i, 0, 0, :]
        env_state2 = env_state_dataset[pair_i, 1, 0, :]

        env._env._env.physics.set_state(env_state1)
        env._env._env.physics.forward()
        img1 = env._env._env.render(resize=False)
        
        env._env._env.physics.set_state(env_state2)
        env._env._env.physics.forward()
        img2 = env._env._env.render(resize=False)
        
        robodesk_image_dataset_highres[pair_i, 0, 0, ...] = img1
        robodesk_image_dataset_highres[pair_i, 1, 0, ...] = img2
        
        
        if pair_i % 10000 == 0:
            print(f"Rendered {pair_i} pairs already!")

        if saveimg:
            pil_img1 = Image.fromarray(img1)
            pil_img2 = Image.fromarray(img2)

            pil_img1.save(os.path.join(img_dir, f"pair_{pair_i}_0.png"))
            pil_img2.save(os.path.join(img_dir, f"pair_{pair_i}_1.png"))

            annotation = pairs_interestingness[pair_i]
            img_concat = concatenate_images(pil_img1, pil_img2, "vertical", 28)  # Example for vertical concatenation with 50px whitespace
            img_concat.save(os.path.join(img_dir_vertical, f"pair_{pair_i}_preference{annotation}.png"))


    if save_dataset:
        with open(
            os.path.join(dataset_dir, "data", f"images_left.pickle"), "wb"
        ) as handle:
            pickle.dump(
                robodesk_image_dataset_highres, handle, protocol=pickle.HIGHEST_PROTOCOL
            )