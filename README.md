# Madoc: Multi-Agent Domain Calibration with a Handful of Offline Data

This is the implementation of the NeurIPS 2024 paper "Multi-Agent Domain Calibration with a Handful of Offline Data". 

## Installation instructions

### Install Python environment

Install Python environment with conda:

```bash
conda create -n madoc python=3.7 -y
conda activate madoc
pip install torch torchvision torchaudio
pip install d4rl==1.1
pip install wandb
pip install ml-collections==0.1.1
pip install gym==0.21.0
pip install Cython==3.0.0a10
pip install importlib-metadata==4.13.0
pip install numpy==1.21.5
pip install urllib3==1.26.11
pip install scikit-learn
pip install tensorboard
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```

### Install MuJoCo

```bash
cd ~/
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
mkdir .mujoco
tar -xzvf mujoco210-linux-x86_64.tar.gz -C .mujoco/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
```

### Download Datasets

You can run the script `download_datasets.sh` to download all datasets.

## Run experiments

You can run the script `run_tasks.sh` to perform all experiments.