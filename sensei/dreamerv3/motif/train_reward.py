import logging
import os
import random
import sys
from pathlib import Path

os.environ["PYTHONPATH"] = str(Path(__file__).parents[1]) + "/"
sys.path.append(str(Path(__file__).parents[1]))  # add root dir to path

import numpy as np

# Needs to be imported to register models and envs
import torch
from cluster_utils import read_params_from_cmdline, save_metrics_params
from cluster_utils.settings import save_settings_to_json

from motif import allogger
from motif.reward_model import RewardModel
from motif.reward_model_trainer import RewardModelTrainer

# import smart_settings


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
            device = torch.device(params.device)
    else:
        device = torch.device("cuda:0")

    print("Device is set to: ", device)

    reward_model = RewardModel(params["reward_model_params"].model_params, device=device)
    print("initialized reward model!")

    trainer = RewardModelTrainer(
        params["reward_model_params"].train_params,
        reward_model,
        dataset_dir=params["dataset_dir"],
        seed=params["seed"],
    )
    print("initialized trainer!")
    trainer.train(save_cpt=True, working_dir=params.working_dir)

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
