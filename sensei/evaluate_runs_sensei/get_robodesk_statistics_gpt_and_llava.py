import os
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pickle
from evaluate_runs_sensei.buffer_utils import collect_data

basedir = "/home/gilian/Documents/Uni/UVA/master/Deep Learning 2/OpenSourceSensei"

run_sensei_gpt = {
    "name": "sensei_classic_gpt4",
    "working_dirs": [
        f"{basedir}/logdir/gpt/seed10",
        f"{basedir}/logdir/gpt/seed11",
        f"{basedir}/logdir/gpt/seed12",
    ]
}

run_sensei_llava = {
    "name": "sensei_classic_llava",
    "working_dirs": [
        f"{basedir}/logdir/llava/seed10",
        f"{basedir}/logdir/llava/seed11",
        f"{basedir}/logdir/llava/seed12",
    ]
}

runs = [run_sensei_gpt, run_sensei_llava]

def data_dict():
    data_dict = get_reward_keys_dict()
    data_dict.update(get_interaction_keys_dict())
    return data_dict

def get_reward_keys_dict():
    return {
        "rew_open_slide": [],
        "rew_open_slide_easy": [],
        "rew_open_drawer": [],
        "rew_open_drawer_easy": [],
        "rew_open_drawer_medium": [],
        "rew_push_green": [],
        "rew_stack": [],
        "rew_upright_block_off_table": [],
        "rew_flat_block_in_bin": [],
        "rew_flat_block_in_shelf": [],
        "rew_lift_upright_block": [],
        "rew_lift_ball": [],
        "rew_push_blue": [],
        "rew_push_red": [],
        "rew_flat_block_off_table": [],
        "rew_ball_off_table": [],
        "rew_upright_block_in_bin": [],
        "rew_ball_in_bin": [],
        "rew_upright_block_in_shelf": [],
        "rew_ball_in_shelf": [],
        "rew_lift_flat_block": [],
    }

def get_interaction_keys_dict():
    return {
        "rew_interaction_drawer_joint":  [],
        "rew_interaction_slide_joint":  [],
        "rew_interaction_red_button":  [],
        "rew_interaction_green_button":  [],
        "rew_interaction_blue_button":  [],
        "rew_interaction_ball":  [],
        "rew_interaction_upright_block":  [],
        "rew_interaction_flat_block":  [],
    }

if __name__ == "__main__":
    log_stats_dir = f"{basedir}/logdir/statistics"

    os.makedirs(log_stats_dir, exist_ok=True)
    
    for run in runs:
        run_name = run["name"]
        run_dirs = run["working_dirs"]
        
        print(f"Collecting statistics for run {run_name}")

        runs_statistics = []

        for run_dir in run_dirs:
            current_run_stats_dict = {"run_dir": run_dir}
            data_dir = os.path.join(run_dir, "replay")

            current_data_dict = data_dict()
            current_data_dict, data_length = collect_data(current_data_dict, data_dir, include_non_finished=False, discrete_action=False)

            current_run_stats_dict["total_length"] = data_length

            for k in get_interaction_keys_dict().keys(): 
                current_run_stats_dict[k] = np.sum(current_data_dict[k])

            for k in get_reward_keys_dict().keys(): 
                current_run_stats_dict[k] = np.sum(current_data_dict[k])

            print(f"Done with stats for run {run_name} with dir: {run_dir}")
            for k, v in current_run_stats_dict.items():
                print(f"Total {k} : {v}")

            runs_statistics.append(current_run_stats_dict)


        print("Saving statistics for current run!")
        with open(os.path.join(log_stats_dir, f'statistics_{run_name}.pickle'), 'wb') as handle:
            pickle.dump(runs_statistics, handle, protocol=pickle.HIGHEST_PROTOCOL)