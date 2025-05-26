import numpy as np
import os
import pickle
from PIL import Image

from skimage.transform import resize
from buffer_loader_scripts.utils import concatenate_images


# Possible prompts. me1 -> gpt{1,2,3,4,5} -> me2 + two screenshots (gen_pair function)
poke_prompts = {'me1': 'Your task is to help me play the Gameboy Game Pokemon Red. I just obtained my starter Pokemon Squirtle. My goal is to find and beat the gym leader Brock. What do I need to do and which areas do I need to traverse to achieve this goal. Keep it very short.',
                'gpt1': "1. Travel north from Pallet Town to Viridian City. \n2. Stock up on items at the Viridian City Poké Mart and heal at the Pokémon Center. \n3. Head west to the entrance of Viridian Forest. \n4. Navigate through Viridian Forest, battling Bug-type trainers and catching new Pokémon. \n5. Exit Viridian Forest to reach Pewter City. \n6. Heal your team at the Pewter City Pokémon Center. \n7. Enter the Pewter City Gym and battle Brock’s trainers. \n8. Fight and defeat Brock, who uses Rock-type Pokémon, with your Water-type moves. \nGood luck!",
                'gpt2': "1. **Route 1**: Head north from Pallet Town to Viridian City. \n2. **Viridian City**: Heal at the Pokémon Center, then go north through Route 2. \n3. **Viridian Forest**: Traverse through the forest, battling trainers and gaining experience. \n4. **Pewter City**: Once through the forest, you'll arrive in Pewter City. \n5. **Pewter City Gym**: Train Squirtle if needed, then challenge and defeat Brock in the gym. \nSquirtle has a type advantage against Brock’s Rock-type Pokémon. Good luck!",
                'gpt3': "1. Travel north from Pallet Town through Route 1 to reach Viridian City. \n2. Replenish supplies and heal Squirtle at the Pokémon Center in Viridian City. \n3. Head northwest to Route 2 and pass through Viridian Forest. \n4. Exit Viridian Forest to reach Pewter City. \n5. Heal your Pokémon at the Pewter City Pokémon Center. \n6. Train Squirtle to increase its level and learn Water moves. \n7. Challenge and defeat Brock at the Pewter City Gym with Squirtle’s Water moves.",
                'gpt4': "1. Head north from Pallet Town to Viridian City. \n2. Go west to Viridian Forest. \n3. Traverse Viridian Forest and exit to reach Pewter City. \n4. Challenge Brock in the Pewter City Gym. Use Squirtle's Water moves to have an advantage.",
                'gpt5': "1. **Route 1**: Head north from Pallet Town to Viridian City. \n2. **Viridian City**: Heal your Pokémon and stock up on essentials (Potions, Poké Balls). \n3. **Route 2**: Continue north, passing through the Viridian Forest. \n4. **Viridian Forest**: Navigate through the forest by battling Bug-type trainers and Pokémon. \n5. **Pewter City**: Exit the forest and proceed to Pewter City. \n6. **Pewter Gym**: Train Squirtle to at least level 12-14 (Water-type moves like Bubble or Water Gun will help). \n7. **Defeat Brock**: Use Squirtle’s Water-type moves to defeat Brock’s Rock-type Pokémon. \nGood luck!",
                'me2': 'Here are two screenshots from the game. Which image depicts a game state that is closer to my goal? Answer in maximum one word: 0 if the first state is better, 1 if the second state is better.',
                }

# sampling strategies for sampling screenshots of rollouts

# beta distribution discretized.
# With a=3, b=2 it looks like a tilted normal distribution favoring samples around 3/3 of the trajectory:
# .:iIi
# With a= 3, b=1 probability is exponentially increasing with 50% samples in last 30%
# __.:I
def sample_discrete_beta(bins, a=3, b=2):
    return int(np.floor(np.random.beta(a, b) * bins))


# uniform
def sample_discrete_uniform(bins):
    return int(np.floor(np.random.uniform(bins + 1))) - 1


def sample_screenshot(working_dir):
    # working_dir is path to cluster_utils working directory or to run in poke_transfer git repo
    if any([i.name == 'pokemon_scores' for i in os.scandir(working_dir)]):
        pkmn_dir = os.path.join(working_dir, 'pokemon_scores')  # in working directory
    else:
        pkmn_dir = os.path.join(working_dir, '')  # in repo

    # fetch all directories of training replay buffers
    pkmn_replays = [f.path for f in os.scandir(pkmn_dir) if f.is_dir() and f.name.startswith('train_')]

    # collect rollouts with sufficient (>2) screenshots
    pkmn_rollouts = []
    for replay in pkmn_replays:
        replay_i = [f.path for f in os.scandir(replay) if f.is_dir() and len([fi for fi in os.listdir(f)]) > 5]
        pkmn_rollouts += replay_i

    # uniform sampling of rollout
    # rollout_i = sample_discrete_uniform(len(pkmn_rollouts))
    # sampling of rollout with preference on most recent rollouts
    rollout_i = sample_discrete_beta(len(pkmn_rollouts), 3, 1)

    # collect all screenshots
    obs_rollout_i = [f.path for f in os.scandir(pkmn_rollouts[rollout_i]) if f.name.endswith('.png')]

    # uniform sampling of screenshot
    obs_i = sample_discrete_uniform(len(obs_rollout_i))
    # obs_i = sample_discrete_beta(len(obs_rollout_i))

    return obs_rollout_i[obs_i]  # return path to image


# generate and return a pair of two screenshots as numpy arrays
def gen_pair(working_dir):
    img1 = np.asarray(Image.open(sample_screenshot(working_dir)))[:, :, :3]
    img2 = np.asarray(Image.open(sample_screenshot(working_dir)))[:, :, :3]
    return img1, img2

# resize images to same size as obs during training (64 x 64 pixels)
# this should be used for motif inputs because high-res inputs are not
# available during runtime

def resize_to_obs(img):
    return (255 * resize(img[:, :, :3], (64, 64, 3))).astype(np.uint8)

# images are stored as pickle files. gen_pair expects a directory filled with images.
# This function unpacks pickles (from git repo i.e. load_dir), saves them in a target directory
# in the form required by gen_pair
def pickle_to_imgs(load_dir, target_dir):
    replays = [f.name for f in os.scandir(load_dir) if f.is_dir() and f.name.startswith('train_')]
    print(replays)
    for replay in replays:
        replay_dir = os.path.join(load_dir, replay)
        replay_i = [f.name for f in os.scandir(replay_dir) if f.is_dir() and len([fi for fi in os.listdir(f)]) > 1]
        print(replay_i)
        for rep_i in replay_i:
            img_num = 0
            rep_i_dir = os.path.join(replay_dir, rep_i)
            target_dir_replay_i = os.path.join(target_dir, replay, rep_i)
            os.makedirs(target_dir_replay_i)
            pickle_j = [f.name for f in os.scandir(rep_i_dir) if f.name.endswith(".pkl")]
            pickle_j.sort()
            for j in pickle_j:
                pickle_dir = os.path.join(rep_i_dir, j)
                with open(pickle_dir, 'rb') as pd:
                    img_list = pickle.load(pd)
                for img in img_list:
                    PIL_image = Image.fromarray(np.uint8(img))
                    PIL_image.save(os.path.join(target_dir_replay_i, f'frame_{str(img_num).zfill(5)}.png'))
                    img_num = img_num + 1

def pickle_count(load_dir):
    replays = [f.name for f in os.scandir(load_dir) if f.is_dir() and f.name.startswith('train_')]
    print(replays)
    for replay in replays:
        replay_dir = os.path.join(load_dir, replay)
        replay_i = [f.name for f in os.scandir(replay_dir) if f.is_dir() and len([fi for fi in os.listdir(f)]) > 1]
        print(replay_i)
        for rep_i in replay_i:
            img_num = 0
            rep_i_dir = os.path.join(replay_dir, rep_i)

            pickle_j = [f.name for f in os.scandir(rep_i_dir) if f.name.endswith(".pkl")]
            pickle_j.sort()
            for j in pickle_j:
                pickle_dir = os.path.join(rep_i_dir, j)
                with open(pickle_dir, 'rb') as pd:
                    img_list = pickle.load(pd)
                print(f"Length of images in pickle {pickle_dir}: {len(img_list)}")


if __name__ == "__main__":
    # 1. unpickle images 
    # 2. run the dataset generation --> it takes long so there is a script to parallelize it on the cluster 
    #           --> generate_pokemon_dataset.py
    #           --> you can then use: collect_pokemon_dataset.py
    # note that we resize for motif training, same as in the pokemon env here --> so after annotation you need to create a resized copy of the images.pickle
    # see buffer_loader_scripts/rerender_pokemon_for_motif.py


    # ========= Counting the images as sanity =========
    # run_name = "pokegym_log_sensei6_0"

    # load_dir = f"/is/cluster/fast/csancaktar/dreamerv3_icml/pokemon/pokegym_sensei_motif/working_directories/0/motifpokemon_scores"
    # pickle_count(load_dir)

    # ========= Unpickling the images =========
    # run_name = "pokegym_sensei_gen1_wlevel9"
    # # run_name = "pokegym_sensei_gen_wlevel9_750K"

    # # load_dir = f"/fast/csancaktar/dreamerv3_iclr/pokemon_data/poke_transfer2/{run_name}"
    # # load_dir = f"/is/cluster/fast/csancaktar/dreamerv3_icml/pokemon/pokegym_sensei_motif/working_directories/0/motifpokemon_scores"
    # load_dir = "/is/cluster/fast/csancaktar/dreamerv3_icml/pokemon/pokegym_sensei_motif_seeds_hope_continue_save/working_directories/0/motifpokemon_scores"
    # # load_dir = "/is/cluster/fast/csancaktar/dreamerv3_icml/pokemon/pokegym_sensei_motif_seeds_hope_continue750K/working_directories/0/motifpokemon_scores"

    # target_dir = f"/fast/csancaktar/dreamerv3_iclr/pokemon_data/unpickled_images/{run_name}"
    # pickle_to_imgs(load_dir, target_dir)

    # ========= Generating the dataset =========

    # run_name = "pokegym_sensei_gen1"
    run_name = "pokegym_sensei_gen1_wlevel9"
    # run_name = "pokegym_sensei_gen_wlevel9_750K"


    # # load_dir = f"/fast/csancaktar/dreamerv3_iclr/pokemon_data/poke_transfer2/{run_name}"
    target_dir = f"/fast/csancaktar/dreamerv3_iclr/pokemon_data/unpickled_images/{run_name}"
    # # pickle_to_imgs(load_dir, target_dir)

    num_pairs = 100
    save_dataset = False
    saveimg = True

    if save_dataset:
        # motif_dataset_dir = (
        #     "/fast/csancaktar/sensei_datasets/pokemon/p2e_round1_pokegym_log_sensei6_0"
        # )
        motif_dataset_dir = (
            "/fast/csancaktar/sensei_datasets/pokemon/pokegym_sensei_gen1_wlevel9_50K"
        )
        os.makedirs(motif_dataset_dir, exist_ok=True)

        os.makedirs(os.path.join(motif_dataset_dir, "data"), exist_ok=True)
        os.makedirs(os.path.join(motif_dataset_dir, "preference"), exist_ok=True)

    width = 144 
    height = 160
    # For pokemon the image shapes are: (144, 160, 3)

    image_dataset_og_res = np.zeros(
        (num_pairs, 2, 1, width, height, 3), dtype=np.uint8
    )

    if saveimg:
        img_dir = f"dreamer_pokemon_{run_name}_50Kdataset"
        img_dir_vertical = os.path.join(img_dir, "vertical_pairs")

        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(img_dir_vertical, exist_ok=True)


    for pair_i in range(num_pairs):
        # I just need to double check that the same image doesn't show up twice!
        img_path1 = sample_screenshot(target_dir)
        while True:
            img_path2 = sample_screenshot(target_dir)
            if img_path1 != img_path2:
                break
        
        img1 = np.asarray(Image.open(img_path1))[:, :, :3]
        img2 = np.asarray(Image.open(img_path2))[:, :, :3]

        image_dataset_og_res[pair_i, 0, 0, ...] = img1
        image_dataset_og_res[pair_i, 1, 0, ...] = img2

        if saveimg and pair_i<100:
            pil_img1 = Image.fromarray(img1)
            pil_img2 = Image.fromarray(img2)

            pil_img1.save(os.path.join(img_dir, f"pair_{pair_i}_0.png"))
            pil_img2.save(os.path.join(img_dir, f"pair_{pair_i}_1.png"))

            img_concat = concatenate_images(
                pil_img1, pil_img2, "horizontal", 28, background_color=(176,224,230)
            )  # Example for vertical concatenation with 50px whitespace


            img_concat.save(
                os.path.join(
                    img_dir_vertical, f"pair_{pair_i}.png"
                )
            )

        if pair_i % 1000 == 0:
            print(f"Generated {pair_i} pairs already!")

    if save_dataset:
        with open(
            os.path.join(motif_dataset_dir, "data", f"images.pickle"), "wb"
        ) as handle:
            pickle.dump(
                image_dataset_og_res, handle, protocol=pickle.HIGHEST_PROTOCOL
            )
