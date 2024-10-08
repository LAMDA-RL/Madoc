#!/bin/bash

python src/main.py --device cuda --env_name HalfCheetah-v2 --dataset_name halfcheetah_medium-v2 
python src/main.py --device cuda --env_name HalfCheetah-v2 --dataset_name halfcheetah_medium_replay-v2
python src/main.py --device cuda --env_name HalfCheetah-v2 --dataset_name halfcheetah_medium_expert-v2
python src/main.py --device cuda --env_name Hopper-v2 --dataset_name hopper_medium-v2
python src/main.py --device cuda --env_name Hopper-v2 --dataset_name hopper_medium_replay-v2
python src/main.py --device cuda --env_name Hopper-v2 --dataset_name hopper_medium_expert-v2
python src/main.py --device cuda --env_name Walker2d-v2 --dataset_name walker2d_medium-v2
python src/main.py --device cuda --env_name Walker2d-v2 --dataset_name walker2d_medium_replay-v2
python src/main.py --device cuda --env_name Walker2d-v2 --dataset_name walker2d_medium_expert-v2
python src/main.py --device cuda --env_name Ant-v2 --dataset_name ant_medium-v2
python src/main.py --device cuda --env_name Ant-v2 --dataset_name ant_medium_replay-v2
python src/main.py --device cuda --env_name Ant-v2 --dataset_name ant_medium_expert-v2

python src/main.py --device cuda --env_name HalfCheetah-v3 --dataset_name HalfCheetah-v3-low-1000-train-noise --neorl True
python src/main.py --device cuda --env_name HalfCheetah-v3 --dataset_name HalfCheetah-v3-medium-1000-train-noise --neorl True
python src/main.py --device cuda --env_name HalfCheetah-v3 --dataset_name HalfCheetah-v3-high-1000-train-noise --neorl True
python src/main.py --device cuda --env_name Hopper-v3 --dataset_name Hopper-v3-low-1000-train-noise --neorl True
python src/main.py --device cuda --env_name Hopper-v3 --dataset_name Hopper-v3-medium-1000-train-noise --neorl True
python src/main.py --device cuda --env_name Hopper-v3 --dataset_name Hopper-v3-high-1000-train-noise --neorl True
python src/main.py --device cuda --env_name Walker2d-v3 --dataset_name Walker2d-v3-low-1000-train-noise --neorl True
python src/main.py --device cuda --env_name Walker2d-v3 --dataset_name Walker2d-v3-medium-1000-train-noise --neorl True
python src/main.py --device cuda --env_name Walker2d-v3 --dataset_name Walker2d-v3-high-1000-train-noise --neorl True



