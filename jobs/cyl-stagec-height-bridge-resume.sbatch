#!/bin/bash
#SBATCH -J cyl-stagec-hbridge-r
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH -t 24:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=njcf2022@mymail.pomona.edu

set -euo pipefail

cd /bigdata/lab/aclarklab/twmr-rl
source .venv/bin/activate
export PATH="$HOME/.pixi/envs/ffmpeg/bin:$PATH"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

initial_ckpt="${1:-}"
if [[ -z "${initial_ckpt}" ]]; then
  initial_ckpt="$(ls -1dt logs/TransformableWheelMobileRobot-*-cyl-stagec-height-bridge-h0p045-l0p2/checkpoints 2>/dev/null | head -n 1 || true)"
fi

if [[ -z "${initial_ckpt}" ]]; then
  echo "No height-bridge h0p045 checkpoint found. Pass checkpoint path as first argument."
  exit 1
fi

echo "Resuming height bridge from: ${initial_ckpt}"

./.venv/bin/python scripts/stagec_bridge_curriculum.py \
  --mode=height \
  --initial_checkpoint_path="${initial_ckpt}" \
  --suffix_prefix=cyl-stagec-height-bridge-resume \
  --heights=0.05,0.055,0.06 \
  --fixed_half_length=0.20 \
  --gate_index=0 \
  --gate_threshold=0.3 \
  --min_success_to_continue=0.015625 \
  --num_timesteps=1000000 \
  --num_envs=512 \
  --num_eval_envs=64 \
  --batch_size=512 \
  --unroll_length=20 \
  --num_minibatches=16 \
  --num_updates_per_batch=8 \
  --num_evals=8 \
  --episode_length=1000 \
  --eval_num_episodes=64 \
  --eval_batch_size=32
