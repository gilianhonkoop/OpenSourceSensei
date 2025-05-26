import imageio
from PIL import Image
from dreamerv3.embodied.core.path import Path
import os
from minihack.tiles import glyph2tile, MAXOTHTILE
import pkg_resources
import numpy as np
import pickle 

WARRIOR_ID = 337
FARMER_ID = 339
STAIRCASE_UP_ID = 2382
FLOOR_ID = 2378

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

def setup_video(output_path, name_suffix, name_prefix, fps):
    os.makedirs(output_path, exist_ok=True)
    file_path = os.path.join(output_path, f"{name_prefix}rollout{name_suffix}.mp4")
    i = 0
    while os.path.isfile(file_path):
        i += 1
        file_path = os.path.join(output_path, f"{name_prefix}rollout{name_suffix}_{i}.mp4")
    print("Record video in {}".format(file_path))
    return (
        imageio.get_writer(file_path, fps=fps, codec="h264", quality=10, pixelformat="yuv420p"), #yuv420p, yuvj422p
        file_path,
    )

def concatenate_images(img1, img2, direction="horizontal", whitespace=10, background_color=(255, 255, 255)):
    """
    Concatenate two images with an optional whitespace between them.

    Parameters:
    - img1_path: Path to the first image.
    - img2_path: Path to the second image.
    - direction: 'horizontal' or 'vertical' concatenation.
    - whitespace: Width of the whitespace in pixels.
    - background_color: Background color for the whitespace (default is white).

    Returns:
    - A new PIL Image object with the concatenated images.
    """

    # Calculate total size
    if direction == "horizontal":
        total_width = img1.width + img2.width + whitespace
        total_height = max(img1.height, img2.height)
    else:  # vertical
        total_width = max(img1.width, img2.width)
        total_height = img1.height + img2.height + whitespace

    # Create a new image with a white background
    new_img = Image.new("RGB", (total_width, total_height), color=background_color)

    # Paste img1
    new_img.paste(img1, (0, 0))

    # Paste img2 with whitespace
    if direction == "horizontal":
        new_img.paste(img2, (img1.width + whitespace, 0))
    else:  # vertical
        new_img.paste(img2, (0, img1.height + whitespace))

    return new_img


class GlyphMapperCustom:
    """This class is used to map glyphs to rgb pixels."""

    def __init__(self, patch_size=16, remap_warrior=True, remap_staircase=True):
        self.tiles = self.load_tiles()
        self.patch_size = patch_size
        self.remap_warrior = remap_warrior
        self.remap_staircase = remap_staircase

    def load_tiles(self):
        """This function expects that tile.npy already exists.
        If it doesn't, call make_tiles.py in win/
        """

        tile_rgb_path = os.path.join(
            pkg_resources.resource_filename("minihack", "tiles"),
            "tiles.pkl",
        )

        return pickle.load(open(tile_rgb_path, "rb"))

    def glyph_id_to_rgb(self, glyph_id):
        tile_id = glyph2tile[glyph_id]
        assert 0 <= tile_id <= MAXOTHTILE
        return self.tiles[tile_id]

    def upsample_glyph_tile(self, tile_img, patch_size):
        tile_large = Image.fromarray(tile_img).resize((patch_size, patch_size))
        return np.array(tile_large)

    def _glyph_to_rgb(self, glyphs):
        # Expects glhyphs as two-dimensional numpy ndarray
        cols = None
        col = None

        for i in range(glyphs.shape[1]):
            for j in range(glyphs.shape[0]):
                current_glyph = glyphs[j, i]
                if self.remap_warrior and current_glyph == WARRIOR_ID:
                    current_glyph = FARMER_ID
                elif self.remap_staircase and current_glyph == STAIRCASE_UP_ID:
                    current_glyph = FLOOR_ID
                rgb = self.glyph_id_to_rgb(
                    current_glyph
                )  # print("patch rgb shape: ", rgb.shape)
                # Each glpyh is 16x16 --> upsample to desired patch size
                if self.patch_size != 16:
                    rgb = self.upsample_glyph_tile(rgb, self.patch_size)
                if col is None:
                    col = rgb
                else:
                    col = np.concatenate((col, rgb))

            if cols is None:
                cols = col
            else:
                cols = np.concatenate((cols, col), axis=1)
            col = None

        return cols

    def to_rgb(self, glyphs):
        return self._glyph_to_rgb(glyphs)


def generate_pairs(num_rollouts, ep_length, n):
    #  n = num_pairs

    # np.random.seed(0)
    i = np.random.choice(num_rollouts, n)
    j = np.random.choice(num_rollouts - 1, n)
    j[j >= i] += 1
    # print(np.any(i == j))
    # False

    i_t = np.random.choice(ep_length, n)
    j_t = np.random.choice(ep_length, n)

    rollout_indices = np.stack([i, j, i_t, j_t], axis=1)
    return np.unique(rollout_indices, axis=0)


def generate_pairs_weighted(num_rollouts, ep_length, n, power=5, rollout_weights=False, time_weights=True, dist2first=None):
    #  n = num_pairs
    if rollout_weights:
        weights_early = generate_weights_for_early_indices(num_rollouts, power=power)
        weights_late = generate_complementary_weights(weights_early, remove_last=True)
        i = np.random.choice(num_rollouts, n, p=weights_early)
        j = np.random.choice(num_rollouts - 1, n, p=weights_late)
    else:
        i = np.random.choice(num_rollouts, n)
        j = np.random.choice(num_rollouts - 1, n)

    j[j >= i] += 1

    if time_weights:
        i_t = np.zeros(n)
        j_t = np.zeros(n)

        sample_no = 0
        for rollout_i, rollout_j in zip(i,j):
            _, early_episode_weigths = get_time_weights(dist2first[rollout_i,:], power=power)
            late_episode_weights, _ = get_time_weights(dist2first[rollout_j,:], power=power)

            sample_i = np.random.choice(ep_length, p=early_episode_weigths)
            sample_j = np.random.choice(ep_length, p=late_episode_weights)
            
            i_t[sample_no] = sample_i
            j_t[sample_no] = sample_j
            
            dist_i = dist2first[rollout_i,sample_i]
            dist_j =dist2first[rollout_j, sample_j]
            # print(f"sample {sample_no}, rollout i: {rollout_i} early ep {sample_i}, dist2first= {dist_i}")
            # print(f"sample {sample_no}, rollout j: {rollout_j} late ep {sample_j}, dist2first= {dist_j}")
            # print("-------")
            sample_no += 1
    else:
        i_t = np.random.choice(ep_length, n)
        j_t = np.random.choice(ep_length, n)

    rollout_indices = np.stack([i, j, i_t, j_t], axis=1)

    return np.unique(rollout_indices, axis=0).astype('int64')


def normalize_weights(weights):
    weights = np.asarray(weights).astype('float64')
    return weights / weights.sum()  # Normalize to get probabilities


def generate_weights_for_early_indices(max_index, power=3):
    """
    The bias strength is controlled by the power parameter.
    """
    weights = np.arange(max_index, 0, -1) ** power
    return normalize_weights(weights)


def generate_complementary_weights(weights, eps=1e-3, remove_last=False):
    if remove_last:
        weights = weights[:-1]
    comp_weights = 1 / (eps + weights)
    return normalize_weights(comp_weights)


def distance_to_previous_ones(is_first):
    distances = []
    last_one_index = None  # We start with no '1' found

    for i, value in enumerate(is_first):
        if value == 1 or last_one_index is None:
            last_one_index = i  # Update the last index of '1'
            distances.append(0)  # Distance to itself is zero
        else:
            distances.append(i - last_one_index)  # Compute the distance

    return distances

def get_time_weights(dist2first, power=1):
    dist2first_early = dist2first.copy()
    
    dist2first[dist2first<100] = 0 
    late_episode_weights = dist2first.squeeze() ** power
    late_episode_weights = normalize_weights(late_episode_weights)

    dist2first_early[dist2first>150] = 0
    early_episode_weigths = dist2first_early.squeeze() ** power
    early_episode_weigths = normalize_weights(early_episode_weigths)

    # early_episode_weigths = generate_complementary_weights(late_episode_weights)

    return late_episode_weights, early_episode_weigths 