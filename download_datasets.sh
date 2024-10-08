#!/bin/bash

cd ~/
mkdir .d4rl
cd .d4rl
mkdir datasets
cd datasets
wget http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/halfcheetah_medium-v2.hdf5
wget http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/halfcheetah_medium_replay-v2.hdf5
wget http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/halfcheetah_medium_expert-v2.hdf5
wget http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/hopper_medium-v2.hdf5
wget http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/hopper_medium_replay-v2.hdf5
wget http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/hopper_medium_expert-v2.hdf5
wget http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/walker2d_medium-v2.hdf5
wget http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/walker2d_medium_replay-v2.hdf5
wget http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/walker2d_medium_expert-v2.hdf5
wget http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/ant_medium-v2.hdf5
wget http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/ant_medium_replay-v2.hdf5
wget http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/ant_medium_expert-v2.hdf5

wget http://datasets.polixir.site/v2/HalfCheetah-v3/HalfCheetah-v3-low-1000-train-noise.npz
wget http://datasets.polixir.site/v2/HalfCheetah-v3/HalfCheetah-v3-medium-1000-train-noise.npz
wget http://datasets.polixir.site/v2/HalfCheetah-v3/HalfCheetah-v3-high-1000-train-noise.npz
wget http://datasets.polixir.site/v2/HalfCheetah-v3/Hopper-v3-low-1000-train-noise.npz
wget http://datasets.polixir.site/v2/HalfCheetah-v3/Hopper-v3-medium-1000-train-noise.npz
wget http://datasets.polixir.site/v2/HalfCheetah-v3/Hopper-v3-high-1000-train-noise.npz
wget http://datasets.polixir.site/v2/HalfCheetah-v3/Walker2d-v3-low-1000-train-noise.npz
wget http://datasets.polixir.site/v2/HalfCheetah-v3/Walker2d-v3-medium-1000-train-noise.npz
wget http://datasets.polixir.site/v2/HalfCheetah-v3/Walker2d-v3-high-1000-train-noise.npz

