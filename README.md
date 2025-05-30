# SENSEI - Open source VLM

## Introduction

This paper builds forth upon [SENSEI](https://arxiv.org/pdf/2503.01584). It covers a core challenge of Reinforcement Learning (RL), namely exploration. This is especially challenging when dealing with environments that have sparse or delayed rewards (Tang et al., 2017b; Hao et al., 2024). Using intrinsic motivation, agents can explore an environment without relying on external rewards. The recent RL framework SEmaNtically Sensible ExploratIon (SENSEI) incorporates this motivation by using vision-language models (VLMs) to provide semantic feedback (Sancaktar et al., 2024). This framework is comprised of three main parts. First, GPT-4 is used to annotate image pairs that show the agent interacting with its environment. Given the images and a prompt, the VLM can reflect its notion of 'interestingness', by stating which image is more interesting. As VLMs are trained on human data, it should theoretically reflect the human notion of interestingness. Second, the framework distills an intrinsic reward function from these annotations which can optimize exploration. Lastly, a Dreamer agent is used for task-free exploration of different environments. This exploration combines its world model with guidance from the distilled reward function to decidie its actions. The main focus of the research is to show that:

- SENSEI can explore rich, semantically meaningful behaviors with few prerequisites.
- SENSEI enables enable fast learning of downstream tasks through its learned world model .

Maybe something about (from the OG paper)

1. Does the distilled reward function Rψ from VLM anno-
   tations encourage interesting behavior?
2. Can SENSEI discover semantically meaningful behavior
   during task-free exploration?
3. Is the world model learned via exploration suitable for
   later learning to efficiently solve downstream tasks?

### Related work

Intrinsic motivation techniques in reinforcement learning (RL) enable agents to explore environments without external rewards by focusing on state-space coverage, novelty-driven methods like random network distillation (RND), or information gain strategies to minimize uncertainty about environment dynamics (Bellemare et al., 2016; Pathak et al., 2017; Sancaktar et al., 2022). However, these methods often lack semantic understanding, prioritizing novelty over alignment with downstream tasks (Sancaktar et al., 2024). Recent approaches incorporate human priors via large language models (LLMs) or vision-language models (VLMs) to guide agents toward meaningful behaviors, with VLMs extracting task-relevant signals from visual observations (Du et al., 2023; Baumli et al., 2023). The SENSEI framework leverages GPT-4 to annotate observation pairs for "interestingness," distilling these into a scalar reward function using Motif, integrated with a model-based agent built on Dreamer V3 (Klissarov et al., 2023; Hafner et al., 2023). To balance meaningful and novel exploration, SENSEI combines semantic rewards with an uncertainty bonus from an ensemble of next-state predictors. To reduce reliance on GPT-4, the open-source LLaVA model, combining a CLIP ViT-L/14 encoder and Vicuna LLM, is used for annotations after feature alignment and tuning on multimodal instruction data (Liu et al., 2024). Similar open-source VLM applications, like Omni-JARVIS, unify vision-language-action tokens for reasoning and acting in environments like Minecraft (Wang et al., 2024).

## Contribution

The model proposed in the paper current only utilizes the GPT-4 model as VLM. This is a limitation for SENSEI, as GPT-4 is closed-source and costly. In this project, we explore whether the Large Language and Vision Assistant (LLaVA), an open-source VLM by Liu et al. (2024), could serve as a solid alternative to GPT-4. This replacement can improve accessibility and scalability to semantic exploration. This will also serve as a base for other VLM models to be incorporated in the SENSEI framework, making it easier to switch between different VLMs. Additionally, using LLaVA enables offline-learning without having to make API calls.

## Results

We used the Robodesk tabletop environment to assess the performance of SENSEI using LLaVA. We collected 100K image-pairs from prior Robodesk epsiodes, and annotated them for semantic interestingness using LLaVA. The image-pair annotations we obtained how a bias towards the first image, which is selected in about 80% of the cases. GPT-4 shows a moderate second-image bias, selecting it in about 60% of the cases. The contrast in choice of preferences is detailed in [image]. 

We successfully replicated the GPT-4 distilled baseline from the original SENSEI paper. The task-free Robodesk exploration (250K steps, 3 seeds), showed similar number of object interactions, with minor seed-related variance [see image 2]. 

With the LLaVA-distilled agent, we observed approximately twice more drawer interactions compared to the GPT-4 baseline, indicating that the LLaVA is able to distill a reward function capable of guiding semantically meaningful behavior. However, there were noticably fewer interactions with objects other than the drawer, suggesting limited generalization beyond explicitly mentioned objects in the prompt. The overall behaviour thus demonstrates semantically meaningful exploration that is strongly driven by prompt content, but with narrower object focus. Differences in exploration between the smaller LLaVA model and the larger GPT-4 demonstrate a trade-off between model size and semantic generality. LLaVA yields a strong prompt-focused behavior, but less balanced interest across all objects.

## Conlusion

Despite its smaller size, LLaVA enables a cost-efficient, fully open alternative to GPT-4 for semantic reward distillation within the SENSEI framework. The successful integration into DreamerV3 confirms that open-source VLMs can instill intrinsic "interestingness" for task-free RL exploration. LLaVa distilled reward function can successfuly guide the agent when concepts are clearly specified in the prompt. However, there appears to be a reduced object generalisation and annotation consistency compared to GPT-4, leading to skewed object interactions.

Future work should focus on exploring more advanced open-source VLMs (e.g. LLaVA-Next) to improve annotation balance and reduce prompt bias. Further, the prompt should be refined to encourage broader object coverage. The results should be validated across diverse environemnts (e.g. OpenAI Gym Atari suite) to assess generalizability. Lastly, the distilled reward functions should be thoroughly analyzed to understand the semantic discrepancies beyween open and closed VLM annotations.


## Individual contribution

#### Max

#### Yitjun

#### Urban



#### Gilian

My main focus during this project was adapting and using the previous Sensei Dreamer codebase for our project. This involved resolving pip-package
version mismatches and outdated cuda drivers, fixing bugs in the previous code and actually running the dreamer agent with the new distilled reward
function, Apart from this, I also created the layout / setup for the Read.me file. Additionally, I trimmed and refined the previous SENSEI codebase so
that it can be included in our new public github repository.

Other deliverables for this course, such as the project proposal, project draft, project presenta-
tion and the final report version were split up equally between our group members. For these deliverables, my main focus was the method and results,
but I also tried to coordinate individual parts from members, e.g. our initial project proposal (before handing it in) had a lot of overlap
between the parts each member wrote.

# Codebase

This codebase is comprised of two main components: `SENSEI` and `Motif`, each contained in their respective folders. Results for this experiment are contained in the `logdir` folder.

## Installation

### Motif

### Sensei

In order to install a conda environment containing all the necessary packages with specific versions, simply run the `install_env.sh` bash script. This will automatically create an environment called `sensei_env`.

These specific package versions have been tested on NVIDIA CUDA drivers with version 12.7. For different driver versions, a different jax wheel could potentially be more suitable. On an a100 GPU, a single run of 250K steps took approximately 10 hours.

Additionally, one can install ffmpeg (e.g. `sudo apt-get install ffmpeg`) in order to create image replays of the runs.

## Running experiments

### Motif

#### Reward training

### Sensei

We follow DreamerV3's code base for running experiments. In order to run Dreamer, one can use:

```sh
python dreamerv3/train.py  --logdir ~/logdir/gpt/seed_42 --configs robodesk_sensei --seed 42
```

`configs` are pre-defined configurations that we provide for some experiment setups. Single parameters can be adjusted with the `--` notation, such as `--logdir`, `--seed`, or `--task`.

A different distilled reward function can be selected by changing `motif_model_dir` and `model_cpt_id` in the `env.motifrobodesk` settings of the selected config.

Before running any experiments, directory paths need to be change to the path where this folder is located on the current machine.

#### WandB

Training progress for Sensei can be tracked using Weights & Biases (W&B).

To run SENSEI with W&B, change W&B settings in `sensei/dreamer/configs.yaml` to your username, project name, etc.  
To run the code with W&B simply set `use_wandb: True`

## Acknowledgments

the code for `motif` was developed based on the code base of [SENSEI](https://arxiv.org/pdf/2503.01584).

The code for `sensei` was developed based on the code base of [DreamerV3](https://github.com/danijar/dreamerv3) and [SENSEI](https://arxiv.org/pdf/2503.01584).

## References
