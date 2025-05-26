from cluster_utils import read_params_from_cmdline, save_metrics_params
from dreamerv3.train import main

ignoreable_keys = ['id']

def recursive_dict_names(params, key):
    dict_names = []
    value_names = []
    if isinstance(params[key], dict):
        sub_dict = params[key]
        for sub_key in sub_dict.keys():
            sub_dict_names, sub_value_names = recursive_dict_names(sub_dict, sub_key)
            for sdn, svn in zip(sub_dict_names, sub_value_names):
                dict_names.append(key + "." + sdn)
                value_names.append(svn)
    else:
        dict_names.append(key)
        value_names.append(str(params[key]))
    return dict_names, value_names

def cluster_params_to_sysargv(params, append_wandb_id=None):

    argv = []
    for key in params.keys():

        if key not in ignoreable_keys:
            if key == 'working_dir':
                keyname = '--logdir'
                value = str(params[key])
                argv.append(keyname)
                argv.append(value)
            else:
                recursive_keys, recursive_values = recursive_dict_names(params, key)
                for rk, rv in zip(recursive_keys, recursive_values):
                    if append_wandb_id is not None and rk == 'wandb.run_name':
                        rv = f'{rv}_{append_wandb_id}'  # Hack, append cluster id to wandb id to name runs differently
                    keyname = f'--{rk}'
                    argv.append(keyname)
                    argv.append(rv)
    return argv


if __name__ == '__main__':

    params = read_params_from_cmdline()
    cluster_id = params['id']
    new_sysargv = cluster_params_to_sysargv(params, append_wandb_id=cluster_id)
    main(
        argv=new_sysargv,
    )
    # dummy metrics, to run hp optimization replace this with a success metric (eg score)
    metrics = {}
    metrics["result"] = 0.0
    save_metrics_params(metrics, params)
