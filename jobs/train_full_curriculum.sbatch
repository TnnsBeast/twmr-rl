#!/bin/bash
#SBATCH -J twmr-full
#SBATCH -p gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH -t 24:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=njcf2022@mymail.pomona.edu

set -euo pipefail

cd /bigdata/lab/aclarklab/twmr-rl
mkdir -p logs
source .venv/bin/activate

export PATH="$HOME/.pixi/envs/ffmpeg/bin:$PATH"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false

./.venv/bin/python scripts/full_curriculum.py \
  --suffix_prefix=multiobs-full \
  --phase3_num_timesteps=1000000 \
  --phase3_num_envs=512 \
  --phase3_num_eval_envs=64 \
  --phase3_batch_size=512 \
  --phase3_unroll_length=20 \
  --phase3_num_minibatches=16 \
  --phase3_num_updates_per_batch=8 \
  --phase3_num_evals=8 \
  --phase3_episode_length=1000 \
  --bridge_num_timesteps=1500000 \
  --bridge_num_envs=512 \
  --bridge_num_eval_envs=64 \
  --bridge_batch_size=512 \
  --bridge_unroll_length=20 \
  --bridge_num_minibatches=16 \
  --bridge_num_updates_per_batch=8 \
  --bridge_num_evals=8 \
  --bridge_episode_length=1000 \
  --dual_num_timesteps=10000000 \
  --dual_num_envs=512 \
  --dual_num_eval_envs=64 \
  --dual_batch_size=512 \
  --dual_unroll_length=20 \
  --dual_num_minibatches=16 \
  --dual_num_updates_per_batch=8 \
  --dual_num_evals=12 \
  --dual_episode_length=1600 \
  "$@"
