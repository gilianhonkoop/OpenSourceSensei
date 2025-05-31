import copy
import json
import os
import pickle
import re
import string
import io
import base64

import numpy as np
from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_PLACEHOLDER,
    IMAGE_TOKEN_INDEX,
)
from llava.eval.run_llava import load_image
from PIL import Image


class AnnotationIdx:
    FIRST = 0
    SECOND = 1
    TIE = 2
    UNKOWN = 3


def concatenate_images(
    img1, img2, direction="horizontal", whitespace=10, background_color=(255, 255, 255)
):
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


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def get_default_args():
    args = type(
        "Args",
        (),
        {
            # "model_path": "liuhaotian/llava-v1.6-34b",
            # "model_base": None,
            # "model_name": get_model_name_from_path(model_path),
            # "query": prompt,
            "conv_mode": None,
            # "image_file": image,
            "sep": ",",
            "temperature": 0.6,
            "top_p": 0.8,
            "num_beams": 1,
            "max_new_tokens": 1024,
        },
    )()
    return args

def update_default_args(args, new_args_dict):
    for key, val in new_args_dict.items():
        if hasattr(args, key):
            args.__setattr__(key,val)
        else:
            print("Unknown key")
            raise KeyError
    return


def get_conv_mode_from_model_name(model_name):
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"
    return conv_mode


def load_gt_annotator(dataset_dir):
    preferences_dataset = np.load(
        os.path.join(dataset_dir, "preference", "preferences_gt.npy"), mmap_mode="r"
    )
    return preferences_dataset


def load_image_pickle_dataset(dataset_dir):
    # TODO check the dataset_dir vs. data structure!
    print("Starting to load image data!")
    with open(os.path.join(dataset_dir, "data", "images.pickle"), "rb") as data:
        images_array = pickle.load(data)
    print("Finished loading image data!")
    return images_array


def prep_query_with_image(query, model):
    # ----------- Prepare image tokens  in first query -----------
    qs = copy.deepcopy(query)

    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    return qs


def assign_llava_output(output):
    if output.lower().translate(str.maketrans("", "", string.punctuation)) == "top":
        llava_annotation = AnnotationIdx.FIRST
    elif (
        output.lower().translate(str.maketrans("", "", string.punctuation)) == "bottom"
    ):
        llava_annotation = AnnotationIdx.SECOND
    elif (
        output.lower().translate(str.maketrans("", "", string.punctuation)) == "tie"
    ):
        llava_annotation = AnnotationIdx.TIE
    else:
        llava_annotation = AnnotationIdx.UNKOWN
    return llava_annotation


def log_result_dict(logdir, result_dict):
    log_index = result_dict["pair"]
    out_file = open(os.path.join(logdir, f"result_{log_index}.json"), "w")
    json.dump(result_dict, out_file, indent=6)
    out_file.close()


def save_img(imgdir, image, log_index):
    image.save(os.path.join(imgdir, f"img_{log_index}.png"))


def convert_images(images_array):
    images = []
    for pair_i in range(images_array.shape[0]):
        # ----------- Get images -----------
        data1 = images_array[pair_i, 0, 0, ...]
        data2 = images_array[pair_i, 1, 0, ...]

        img1 = Image.fromarray(data1)
        img2 = Image.fromarray(data2)

        img_vconcat = concatenate_images(img1, img2, "vertical", 28)  # vertical concatenation with 50px whitespace
        images.append(img_vconcat)
    return images


def encode_array_to_base64(image_array, resize=None):
    # Convert the NumPy array to a PIL Image object
    image = Image.fromarray(np.uint8(image_array))  # Ensure the array is of type uint8
    
    h, _ = image.size

    if resize is not None and resize != h:
        print("image resized!")
        image = image.resize((resize, resize), Image.Resampling.LANCZOS)

    # Save the image to a buffer, instead of a file
    with io.BytesIO() as buffer:
        image.save(buffer, format="PNG")  # You can change JPEG to PNG if you prefer
        # Move to the beginning of the buffer
        buffer.seek(0)
        # Encode the image stored in buffer
        return base64.b64encode(buffer.read()).decode('utf-8')
    
def encode_imgfile_to_base64(image_path, resize=None):
    # Convert the NumPy array to a PIL Image object
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")