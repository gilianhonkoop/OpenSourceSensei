#!/bin/bash

ENV_NAME=sensei_env

conda create -n $ENV_NAME python=3.10 -y
conda activate $ENV_NAME

# Upgrade pip
pip install --upgrade pip

# Install PyTorch + torchvision with CUDA 11.8
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# Install specific versions of wheel and setuptools
pip install --upgrade wheel==0.38.4 setuptools==65.5.1

# Install gym and dependencies
pip install gym[atari,accept-rom-license,atari-py,ale-py]==0.19.0

# Install DreamerV3 requirements
pip install optax
pip install rich
pip install tensorflow-cpu
pip install tensorflow_probability
pip install ruamel.yaml==0.17.40

# Install Robodesk-related packages
pip install robodesk
pip install Pillow==9.5.0

pip install "cluster_utils[runner]"

pip install wandb

pip install "jax[cuda11_pip]==0.4.25" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install nvidia-cudnn-cu12==8.9.7.29

# For creating replays
# pip install imageio[ffmpeg]

echo "Environment '$ENV_NAME' setup complete!"

