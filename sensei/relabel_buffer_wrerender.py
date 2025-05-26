from cluster import read_params_from_cmdline, save_metrics_params, exit_for_resume
from dreamerv3.embodied.envs.motif_env_wrapper import MotifStandalone
from buffer_loader_scripts.overwrite_replay import *

# for the environment initialization
from dreamerv3.embodied.core.config import Config
from dreamerv3.train import make_env
import ruamel.yaml as yaml

if __name__ == '__main__':


    # params = read_params_from_cmdline()
    # buffer_load_dir = params['buffer_load_dir']
    # buffer_save_dir = params['working_dir']
    # motif_dir = params['motif_dir']
    # motif_wrapper = MotifStandalone(model_dir=motif_dir, clipping_min=-100, clipping_max=100)
    # relabel_buffer = lambda obs: relabel_motif(obs, motif_wrapper)
    # copy_modified_replay(working_dir=buffer_load_dir, target_dir=buffer_save_dir, modify=relabel_buffer)
    # # Dummy metrics
    # metrics = {}
    # metrics["result"] = 0.0
    # save_metrics_params(metrics, params)


    # # ===================================================================================================
    buffer_load_dir = "/is/cluster/fast/csancaktar/dreamerv3_iclr/test_motifrobodesk_grid_gpt_new_general_prompt_seeds/working_directories/3"
    # buffer_load_dir = "/is/cluster/fast/csancaktar/dreamerv3/test_robodesk10_interactions_reward_heads_longer/working_directories/0"
    buffer_save_dir = "/is/cluster/fast/csancaktar/sensei_robodesk_gen2_buffer_relabeled_generalprompt_dir3"

    # MAKE ENVIRONMENT 
    configs = yaml.YAML(typ="safe").load((Path("dreamerv3/configs.yaml")).read())
    config = Config(configs["defaults"])
    config = config.update(configs["p2x_robodesk"])
    # MAKE ENVIRONMENT 
    env = make_env(config)

    motif_dir = "/is/cluster/fast/csancaktar/results/motif/robodesk_p2e_data/grid_robodesk_generalprompt_gpt4_gen1_2_125K/working_directories/32"
    motif_wrapper = MotifStandalone(model_dir=motif_dir, clipping_min=-200, clipping_max=200, model_cpt_id=49)
    relabel_buffer = lambda obs: relabel_motif(obs, motif_wrapper)
    copy_modified_replay_wrerendering(working_dir=buffer_load_dir, target_dir=buffer_save_dir, modify=relabel_buffer, env=env)
