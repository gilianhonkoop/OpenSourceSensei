# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# import numpy as np

# # Example trajectory: 100 points in a curve
# trajectory = np.cumsum(np.random.randn(100, 2), axis=0)

# # Create figure and axis
# fig, ax = plt.subplots()
# line, = ax.plot([], [], 'bo-', lw=2)  # blue line with dots

# # Set axis limits
# ax.set_xlim(np.min(trajectory[:, 0]) - 1, np.max(trajectory[:, 0]) + 1)
# ax.set_ylim(np.min(trajectory[:, 1]) - 1, np.max(trajectory[:, 1]) + 1)

# # Initialization function
# def init():
#     line.set_data([], [])
#     return line,

# # Update function for animation
# def update(frame):
#     x = trajectory[:frame+1, 0]
#     y = trajectory[:frame+1, 1]
#     line.set_data(x, y)
#     return line,

# # Create animation
# ani = animation.FuncAnimation(
#     fig, update, frames=len(trajectory),
#     init_func=init, blit=True, repeat=False
# )

# # Save as GIF
# ani.save('trajectory.gif', writer='pillow', fps=20)



import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from PIL import Image
import os
import pickle
import colorsys

from skimage.transform import resize

import smart_settings
from dreamerv3.motif.reward_model import RewardModel
from dreamerv3.motif.eval_utils import get_batch_dict_for_reward_model, reward_model_with_mb
device = "cuda:0"

pastelgreen = (225/255, 247/255, 208/255)
darkgreen = (145/255, 170/255, 126/255)

darkwmblue = (38/255, 112/255, 130/255)
darkwmgreen = (93/255, 133/255, 51/255)

newpurple = (101/255, 79/255, 167/255)
midgrey = (100/255, 100/255, 100/255)

def scale_lightness(rgb, scale_l):
    # convert rgb to hls
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # manipulate h, l, s values and return as rgb
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s = s)


def resize_to_obs(img):
    return (255 * resize(img[:, :, :3], (64, 64, 3))).astype(np.uint8)

motif_pokemon_gpt_gen1 = {
    "name": "pokemon1",
    "label": "Pokemon GPT gen 1",
    "motif_model_dir": "/fast/csancaktar/dreamerv3_iclr/pokemon_data/poke_transfer2/pokemon_motif/best_val_motif_wd49",
    "model_cpt_id": 49,
    "color": scale_lightness(newpurple, 1.5),
}

# # motif_pokemon_gpt_gen1 = {
# #     "name": "pokemon1",
# #     "label": "Pokemon GPT gen 1",
# #     "motif_model_dir": "/fast/csancaktar/dreamerv3_iclr/pokemon_data/poke_transfer2/pokemon_motif/best_val_motif_new_wd30",
# #     "model_cpt_id": 49,
# #     "color": newpurple,
# # }


# motif_pokemon_gpt_gen2 = {
#     "name": "pokemon2",
#     "label": "Pokemon GPT gen 2",
#     "motif_model_dir": "/is/cluster/fast/csancaktar/results/motif/motif_pokegym_sensei_gen1_2_150K/grid_pokemon_gpt_motif_trained_gen1_2/working_directories/44",
#     "model_cpt_id": 25,
#     "color": newpurple,
# }

motif_pokemon_gpt_gen2_pure = {
    "name": "pokemon2",
    "label": "Pokemon GPT gen 2",
    # "motif_model_dir": "/is/cluster/fast/csancaktar/results/motif/motif_pokegym_sensei_gen1_2_50K/grid_pokemon_gpt_motif_trained_gen2/working_directories/90",
    "motif_model_dir": "/is/cluster/fast/csancaktar/results/motif/motif_pokegym_sensei_gen1_2_wlevel9/grid_pokemon_gpt_motif_trained_gen2_wlevel9/working_directories/73",
    "model_cpt_id": 49,
    "color": newpurple,
}


motif_pokemon_gpt_gen1_2 = {
    "name": "pokemon2",
    "label": "Pokemon GPT gen 2",
    # "motif_model_dir": "/is/cluster/fast/csancaktar/results/motif/motif_pokegym_sensei_gen1_2_50K/grid_pokemon_gpt_motif_trained_gen2/working_directories/90",
    "motif_model_dir": "/is/cluster/fast/csancaktar/results/motif/motif_pokegym_sensei_gen1_2_wlevel9/grid_pokemon_gpt_motif_trained_gen1_2_wlevel9/working_directories/91",
    "model_cpt_id": 49,
    "color": newpurple,
}


motif_models = [motif_pokemon_gpt_gen1, motif_pokemon_gpt_gen1_2]


width = 64 
height = 64

ep_length = 1000
# Load your sequence of images
# image_folder = "/home/csancaktar/Projects/mydream_er/dreamer_pokemon_pokegym_sensei_gen1_test"
# image_folder = "/fast/csancaktar/dreamerv3_iclr/pokemon_data/unpickled_images/pokegym_sensei_gen1/train_0/train_rollout_20250326T032902" # level6
# image_folder = "/fast/csancaktar/dreamerv3_iclr/pokemon_data/unpickled_images/pokegym_sensei_gen1/train_0/train_rollout_20250326T031831" # random
# image_folder = "/fast/csancaktar/dreamerv3_iclr/pokemon_data/unpickled_images/pokegym_sensei_gen1/train_0/train_rollout_20250326T024116" # random
image_folder = "/fast/csancaktar/dreamerv3_iclr/pokemon_data/unpickled_images/pokegym_sensei_gen_wlevel9_750K/train_0/train_rollout_20250330T223431"

image_files = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.png')])
# print(image_files[1300:2070])
# images = [np.array(Image.open(f)) for f in image_files[1200:1800]]
# images = [np.array(Image.open(f)) for f in image_files[2604:2900]]
images = [np.array(Image.open(f)) for f in image_files[2550:2590]]


image_folder = "/fast/csancaktar/dreamerv3_iclr/pokemon_data/unpickled_images/pokegym_sensei_gen_wlevel9_750K/train_0/train_rollout_20250330T212840"

image_files = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.png')])
# print(image_files[1300:2070])
# images = [np.array(Image.open(f)) for f in image_files[1595:1655]]
# images = [np.array(Image.open(f)) for f in image_files[1893:1923]]
# images = [np.array(Image.open(f)) for f in image_files[254:312]]
images = [np.array(Image.open(f)) for f in image_files[722:865]]

# frame_02550 -- frame_02552 -- frame_02590
def minmax_scale(traj):
    return (traj - np.min(traj)) / (np.max(traj) - np.min(traj))

smoothing = True

from scipy.ndimage import uniform_filter1d


# Define a smoothing function
def smooth_signal(signal, window_size=3):
    return uniform_filter1d(signal, size=window_size)



for motif_model in motif_models:
    motif_model_dir = motif_model["motif_model_dir"]
    model_cpt_id = motif_model["model_cpt_id"]

    params = smart_settings.load(os.path.join(motif_model_dir, "settings.json"), make_immutable=False)
    motif_reward_model = RewardModel(params["reward_model_params"]["model_params"], device=device)
    motif_reward_model.load(
        os.path.join(motif_model_dir, f"checkpoint_{model_cpt_id}")
    )
    motif_model["model"] = motif_reward_model
    motif_model["motif_trajs"] = []



    trajectory = []

    for image_seq in images:
        # shape: t, width, height, 3
        rollout_images = np.asarray(image_seq)
        rollout_images = resize_to_obs(rollout_images)[None]

        obs_vec = None
        rollout_images_left = None
        batch_dict = get_batch_dict_for_reward_model(
            motif_reward_model, rollout_images, obs_vec
        )

        reward_dict = reward_model_with_mb(motif_model["model"], batch_dict)

        motif_reward = reward_dict.rewards.squeeze().detach().cpu().numpy()
        trajectory.append(motif_reward)

    if smoothing:
        motif_model["motif_trajs"] = minmax_scale(smooth_signal(np.asarray(trajectory)).reshape(-1, 1))
    else:
        motif_model["motif_trajs"] = minmax_scale(np.asarray(trajectory).reshape(-1, 1))


print(len(images))

# Set up the figure
fig, (ax_img, ax_plot) = plt.subplots(1, 2, figsize=(6.3, 2.4), gridspec_kw={'width_ratios': [1.7, 3.5]})

# Display first image
img_display = ax_img.imshow(images[0], cmap='gray', aspect='auto')
ax_img.axis('off')

# Line plot: x vs time (y)
line, = ax_plot.plot([], [], 'bo-', lw=2, markersize=1, color=scale_lightness(newpurple, 1.5),label='SENSEI Gen1')
line2, = ax_plot.plot([], [], 'bo-', lw=2, markersize=1, color=newpurple, label='SENSEI Gen2')

# ax_plot.set_ylim(min(np.min(motif_models[0]["motif_trajs"][:, 0]), np.min(motif_models[1]["motif_trajs"][:, 0])) - 1, 
#                  max(np.min(motif_models[0]["motif_trajs"][:, 0]), np.max(motif_models[1]["motif_trajs"][:, 0])) + 1)
ax_plot.set_ylim(-0.05, 1.1)
ax_plot.set_xlim(0, len(motif_models[1]["motif_trajs"]))  # y is time

ax_plot.set_xlabel(r"$t$")
# ax_plot.set_ylabel(r"$r_t^{sem}$")
ax_plot.set_ylabel(r"$r_t^{sem}$", labelpad=1)  # smaller pad = closer to axis
ax_plot.set_yticks([0, 0.5, 1])
# ax_plot.legend(loc='upper left', frameon=False, fontsize=8)
# ax_plot.legend(loc='lower center', frameon=False, fontsize=8)
ax_plot.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, 1.12),  # exactly at the top of the plot area
    frameon=False,
    ncol=2,
    fontsize=8
)

fig.subplots_adjust(wspace=0.45)  # default is usually 0.2

# Init function
def init():
    line.set_data([], [])
    line2.set_data([], [])

    img_display.set_data(images[0])
    return line, img_display


# Update function
def update(frame):
    x_vals = motif_models[0]["motif_trajs"][:frame+1, 0]
    x_vals2 = motif_models[1]["motif_trajs"][:frame+1, 0]

    time_vals = np.arange(frame+1)
    line.set_data(time_vals, x_vals)
    line2.set_data(time_vals, x_vals2)

    img_display.set_data(images[frame])
    return line, img_display

# Animate
ani = animation.FuncAnimation(
    fig, update, frames=len(trajectory),
    init_func=init, blit=True, repeat=False
)

# Save to GIF
ani.save(f'gifs/trajectory_gen2_wlevel9_rollout_gymleader_overlayed8.gif', writer='pillow', fps=10)
