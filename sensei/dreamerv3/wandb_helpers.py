
# When syncing W&B in online mode at max ~15 runs are synced in parallel. If we want to run more, we can save
# the files in offline mode and sync them manually. For this we need to initialize a directory with
# > wandb init
# then we select the correct W&B project. We can later run W&B in offline mode and provide this directory.
# W&B saves all things in the provided directory and we need to manually sync it with
# > wandb sync offline-run-$FILENAME
# The following dict informs W&B where to save the runs, depending on the project.

wandb_offline_dirs = {'SemGroundPlay': '/is/cluster/fast/csancaktar/SENSEI/SemGroundPlay/',
                      'SemGroundPlay_MiniHack': '/is/cluster/fast/csancaktar/SENSEI/SemGroundPlay_MiniHack/',
                      'SemGroundPlay_MiniHack_final': '/is/cluster/fast/csancaktar/SENSEI/SemGroundPlay_MiniHack_final/',
                      'semgroundplay_minihack_iclr': '/is/cluster/fast/csancaktar/SENSEI/SemGroundPlay_MiniHack_ICLR/',
                      'SemGroundPlay_MiniHack_final_phase3': '/is/cluster/fast/csancaktar/SENSEI/SemGroundPlay_MiniHack_final_phase3/',
                      'SENSEI_percentile_final': '/is/cluster/fast/csancaktar/SENSEI/SENSEI_percentile_final/',
                      'SENSEI_Robodesk_Final': '/is/cluster/fast/csancaktar/SENSEI/SENSEI_Robodesk_Final/',
                      'SENSEI_Robodesk_Final_ablations': '/is/cluster/fast/csancaktar/SENSEI/SENSEI_Robodesk_Final_ablations/',
                      'SENSEI_Robodesk_Phase3_Final': '/is/cluster/fast/csancaktar/SENSEI/SENSEI_Robodesk_Phase3_Final/',
                      'minihack_ablations': '/is/cluster/fast/csancaktar/SENSEI/minihack_ablations/',
                      'minihack_ablations_new': '/is/cluster/fast/csancaktar/SENSEI/minihack_ablations_new/',
                      'robodesk_final_gpt_grid2': '/is/cluster/fast/csancaktar/SENSEI/robodesk_final_gpt_grid2/',
                    #   'sensei_robodesk_phase3_general': '/is/cluster/fast/csancaktar/SENSEI/sensei_robodesk_phase3_general',
                      'sensei_robodesk_phase3_general': '/is/cluster/fast/csancaktar/SENSEI/sensei_robodesk_phase3_general_new',
                      'robodesk_gen2': '/is/cluster/fast/csancaktar/SENSEI/robodesk_gen2/',
                      'pokemon_offline': '/is/cluster/fast/csancaktar/SENSEI/pokemon_offline/',
                      'Robodesk': '/mnt/lustre/work/butz/bst495/sensei_logdir/robodesk_wb/',
                      }
