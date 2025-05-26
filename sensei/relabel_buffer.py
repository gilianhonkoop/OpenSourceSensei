from cluster import read_params_from_cmdline, save_metrics_params, exit_for_resume
from dreamerv3.embodied.envs.motif_env_wrapper import MotifStandalone
from buffer_loader_scripts.overwrite_replay import *

if __name__ == '__main__':

    params = read_params_from_cmdline()
    buffer_load_dir = params['buffer_load_dir']
    buffer_save_dir = params['working_dir']
    motif_dir = params['motif_dir']
    motif_wrapper = MotifStandalone(model_dir=motif_dir, clipping_min=-100, clipping_max=100)
    relabel_buffer = lambda obs: relabel_motif(obs, motif_wrapper)
    copy_modified_replay(working_dir=buffer_load_dir, target_dir=buffer_save_dir, modify=relabel_buffer)
    # Dummy metrics
    metrics = {}
    metrics["result"] = 0.0
    save_metrics_params(metrics, params)
