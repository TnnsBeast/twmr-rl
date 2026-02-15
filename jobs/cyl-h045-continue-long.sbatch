#!/bin/bash
#SBATCH -J cyl-h045-long
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

checkpoint_path="$(printf '%s' "${1:-}" | xargs)"
if [[ -z "${checkpoint_path}" ]]; then
  checkpoint_path="$(ls -1dt \
    logs/TransformableWheelMobileRobot-*-cyl-stagec-height-bridge-h0p045-l0p2/checkpoints \
    2>/dev/null | head -n 1 || true)"
fi

if [[ -z "${checkpoint_path}" ]]; then
  echo "No h0p045 training checkpoint directory found. Pass checkpoint path as first argument."
  exit 1
fi

if [[ ! -d "${checkpoint_path}" ]]; then
  echo "Checkpoint directory does not exist: ${checkpoint_path}"
  exit 1
fi

if ! find "${checkpoint_path}" -mindepth 1 -maxdepth 1 -type d -name '[0-9]*' | grep -q .; then
  echo "Checkpoint directory has no saved training steps: ${checkpoint_path}"
  exit 1
fi

echo "Continuing from checkpoint path: $(realpath "${checkpoint_path}")"

suffix="cyl-h045-long"
overrides='{"xml_variant":"box","obstacle_height":0.045,"obstacle_x_position":0.6,"obstacle_half_length":0.2,"success_x_margin":0.1}'

./.venv/bin/python train_jax_ppo.py \
  --env_name=TransformableWheelMobileRobot \
  --playground_config_overrides="${overrides}" \
  --load_checkpoint_path="${checkpoint_path}" \
  --num_timesteps=5000000 \
  --num_envs=512 \
  --num_eval_envs=64 \
  --batch_size=512 \
  --unroll_length=20 \
  --num_minibatches=16 \
  --num_updates_per_batch=8 \
  --num_evals=10 \
  --episode_length=1000 \
  --seed=1 \
  --suffix="${suffix}" \
  --use_wandb=false \
  --use_tb=false \
  --rollout_camera=cam_robot_follow \
  --rollout_width=640 \
  --rollout_height=480

exp_dir="$(ls -1dt logs/TransformableWheelMobileRobot-*-"${suffix}" 2>/dev/null | head -n 1 || true)"
if [[ -z "${exp_dir}" ]]; then
  echo "Could not locate experiment directory for suffix ${suffix}."
  exit 1
fi

ckpt_dir="${exp_dir}/checkpoints"
if [[ ! -d "${ckpt_dir}" ]]; then
  echo "Checkpoint directory missing: ${ckpt_dir}"
  exit 1
fi

if [[ -f rollout0.mp4 ]]; then
  cp -f rollout0.mp4 "${exp_dir}/rollout0.mp4"
  cp -f rollout0.mp4 "${exp_dir}/rollout0-follow10.mp4"
  echo "Saved rollout video to: ${exp_dir}/rollout0-follow10.mp4"
fi

timestamp="$(date +%Y%m%d-%H%M%S)"
eval_json="logs/eval-${suffix}-h0p045-256ep-${timestamp}.json"

./.venv/bin/python scripts/eval_twmr_checkpoint.py \
  --checkpoint_path="${ckpt_dir}" \
  --playground_config_overrides="${overrides}" \
  --num_episodes=256 \
  --episode_length=1000 \
  --batch_size=32 \
  --seed=1 \
  --output_json="${eval_json}"

echo "Wrote consistency eval JSON: ${eval_json}"
