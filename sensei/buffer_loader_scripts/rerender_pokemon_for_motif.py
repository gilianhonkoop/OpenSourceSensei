import numpy as np
import os
import pickle
from PIL import Image

from skimage.transform import resize
from buffer_loader_scripts.utils import concatenate_images

def resize_to_obs(img):
    return (255 * resize(img[:, :, :3], (64, 64, 3))).astype(np.uint8)



# dataset_dir = "/is/cluster/fast/csancaktar/sensei_datasets/pokemon/p2e_round1_pokegym_log_sensei6_0"
# dataset_dir = "/is/cluster/fast/csancaktar/sensei_datasets/pokemon/pokegym_sensei_gen1_50K_parallelized"
dataset_dir = "/is/cluster/fast/csancaktar/sensei_datasets/pokemon/pokegym_sensei_gen1_wlevel9_50K_parallelized"

with open(os.path.join(dataset_dir, "data", "images.pickle"), "rb") as data:
    images_array = pickle.load(data)

save_dataset = True
saveimg = False

if save_dataset:
    new_dataset_for_low_res = "/is/cluster/fast/csancaktar/sensei_datasets/pokemon_resized_for_motif/pokegym_sensei_gen1_wlevel9_50K_parallelized"
    # new_dataset_for_low_res = "/is/cluster/fast/csancaktar/sensei_datasets/pokemon_resized_for_motif/pokegym_sensei_gen1_50K_parallelized"

    os.makedirs(new_dataset_for_low_res, exist_ok=True)
    os.makedirs(os.path.join(new_dataset_for_low_res, "data"), exist_ok=True)
    os.makedirs(os.path.join(new_dataset_for_low_res, "preference"), exist_ok=True)

num_pairs = 50000

width = 64 
height = 64

image_dataset_low_res = np.zeros(
    (num_pairs, 2, 1, width, height, 3), dtype=np.uint8
)

if saveimg:
    img_dir = f"dreamer_pokemon_lowres_test_comp"
    img_dir_vertical = os.path.join(img_dir, "vertical_pairs")

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(img_dir_vertical, exist_ok=True)


for pair_i in range(num_pairs):
    img1 = resize_to_obs(images_array[pair_i, 0, 0, ...])
    img2 = resize_to_obs(images_array[pair_i, 1, 0, ...])

    image_dataset_low_res[pair_i, 0, 0, ...] = img1
    image_dataset_low_res[pair_i, 1, 0, ...] = img2

    if saveimg:
        pil_img1 = Image.fromarray(image_dataset_low_res[pair_i, 0, 0, ...])
        pil_img2 = Image.fromarray(image_dataset_low_res[pair_i, 1, 0, ...])

        pil_img1.save(os.path.join(img_dir, f"pair_{pair_i}_0.png"))
        pil_img2.save(os.path.join(img_dir, f"pair_{pair_i}_1.png"))

        img_concat = concatenate_images(
            pil_img1, pil_img2, "vertical", 28
        )  # Example for vertical concatenation with 50px whitespace


        img_concat.save(
            os.path.join(
                img_dir_vertical, f"pair_{pair_i}.png"
            )
        )

        pil_img1 = Image.fromarray(images_array[pair_i, 0, 0, ...])
        pil_img2 = Image.fromarray(images_array[pair_i, 1, 0, ...])

        img_concat = concatenate_images(
            pil_img1, pil_img2, "vertical", 28
        )  # Example for vertical concatenation with 50px whitespace


        img_concat.save(
            os.path.join(
                img_dir_vertical, f"pair_{pair_i}_ogres.png"
            )
        )

    if pair_i % 1000 == 0:
        print(f"Re-rendered {pair_i} pairs already!")


if save_dataset:
    with open(
        os.path.join(new_dataset_for_low_res, "data", f"images.pickle"), "wb"
    ) as handle:
        pickle.dump(
            image_dataset_low_res, handle, protocol=pickle.HIGHEST_PROTOCOL
        )

