import logging
import os
import pickle
import random
import sys
from pathlib import Path
os.environ["PYTHONPATH"] = str(Path(__file__).parents[1]) + "/"
sys.path.append(str(Path(__file__).parents[1]))  # add root dir to path

from typing import Any, List, Optional, Sequence

import numpy as np

# Needs to be imported to register models and envs
import torch
from cluster_utils import read_params_from_cmdline, save_metrics_params
# from cluster_utils.settings import save_settings_to_json

from motif import allogger
from motif.annotator.annotator import GPT4LanguageModel, LlavaVisionLanguageModel, MockLanguageModel

import json
from types import SimpleNamespace
import argparse
from accelerate import init_empty_weights

def convert_namespace_to_dict(obj):
    if isinstance(obj, dict):
        return {k: convert_namespace_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, (SimpleNamespace, argparse.Namespace)):
        return {k: convert_namespace_to_dict(v) for k, v in vars(obj).items()}
    elif isinstance(obj, list):
        return [convert_namespace_to_dict(v) for v in obj]
    else:
        return obj

def save_settings_to_json(params, save_dir):
    """
    Recursively converts Namespace/dict to pure dict and saves as JSON.
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "settings.json")

    settings = convert_namespace_to_dict(params)

    with open(save_path, "w") as f:
        json.dump(settings, f, indent=4)

    print(f"[INFO] Saved settings to: {save_path}")

def default_gpt_params():
    return {
        "model": "gpt-4-turbo-2024-04-09",
        "detail": "low",
        "temperature": 0.2,
        "logdir": None,
        "save_img": False,
    }


def default_llava_params():
    return {
        # "model_path": "liuhaotian/llava-v1.6-34b",
        "model_path": "liuhaotian/llava-v1.5-7b",
        "load_in_8bit": False,
        "logdir": None,
        "save_img": False,
    }


def main(params):
    """Evaluation entry point."""

    logger = allogger.get_logger(scope="main", basic_logging_params={"level": logging.INFO})
    metrics = {}

    # First read the settings for the model!
    torch.manual_seed(params.seed)
    np.random.seed(params.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(params.seed)
    random.seed(params.seed)

    if "device" in params:
        if "cuda" in params.device:
            if torch.cuda.is_available():
                device = torch.device(params.device)
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(params.device)
    else:
        device = torch.device("cuda:0")

    print("Device is set to: ", device)

    if params['annotator'] == 'gpt4':
        annotator_params_dict = default_gpt_params()
        annotator_params_dict.update(params['annotator_params'])
        if params["logging"]:
            annotator_params_dict["logdir"] = os.path.join(params.working_dir, "log")
        annotator = GPT4LanguageModel(**annotator_params_dict)
    elif params['annotator'] == 'llava':
        annotator_params_dict = default_llava_params()
        annotator_params_dict.update(params['annotator_params']['model_params'])
        if params["logging"]:
            annotator_params_dict["logdir"] = os.path.join(params.working_dir, "log")
                
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        annotator = LlavaVisionLanguageModel(**annotator_params_dict, sampling_params=params['annotator_params']['sampling_params'])
    elif params['annotator'] == 'random': 
        annotator = MockLanguageModel()
    else:
        raise NotImplementedError

    # Load image buffer! 
    print("Starting to load image data!")
    with open(os.path.join(params['dataset_dir'], "data", "images.pickle"), "rb") as data:
        images_array = pickle.load(data)
    print("Finished loading image data!")

    # IF there is an oracle, load it to check for accuracy!
    gt_preference_file = Path(os.path.join(params['dataset_dir'], "preference", "preferences_gt.npy"))
    if gt_preference_file.is_file():
        gt_annotations = np.load(gt_preference_file)
    else:
        gt_annotations = None

    # =======================================================================
    # Partition the overall dataset based on partition_size and partition_idx!
    # =======================================================================
    len_dataset = len(images_array)
    num_samples = params['partition_size']
    partition_idx = params['partition_idx']
    sample_start_idx = np.arange(0,len_dataset,num_samples)[partition_idx]

    # Get image partition
    images_partition = images_array[sample_start_idx:min(len_dataset,sample_start_idx+num_samples)]

    # Get VLM annotations!
    annotation_outputs = annotator.generate(images_array[sample_start_idx:min(len_dataset,sample_start_idx+num_samples)])

    np.save(os.path.join(params['working_dir'], f'annotation_model_partition{partition_idx}.npy'), annotation_outputs)
    np.save(os.path.join(params['working_dir'], f'images_model_partition{partition_idx}.npy'), images_partition)

    if gt_annotations is not None:
        gt_annotation_partition = gt_annotations[sample_start_idx:min(len_dataset,sample_start_idx+num_samples)]
        # I am saving this for debugging purposes! :D 
        np.save(os.path.join(params['working_dir'], f'gt_annotation_model_partition{partition_idx}.npy'), gt_annotation_partition)
        metrics["accuracy"] = np.sum(annotation_outputs == gt_annotation_partition)/len(annotation_outputs)

    save_metrics_params(metrics, params)
    allogger.close()
    return 0


if __name__ == "__main__":
    params = read_params_from_cmdline(verbose=True)
    os.makedirs(params.working_dir, exist_ok=True)

    allogger.basic_configure(
        logdir=params.working_dir,
        default_outputs=["tensorboard"],
        manual_flush=True,
        tensorboard_writer_params=dict(min_time_diff_btw_disc_writes=1),
    )

    allogger.utils.report_env(to_stdout=True)
    save_settings_to_json(params, params.working_dir)

    sys.exit(main(params))
