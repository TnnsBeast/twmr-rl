#!/bin/bash
#SBATCH -J cyl-render045
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 02:00:00
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

checkpoint_path="${1:-}"
episode_length="${2:-2000}"
terminate_on_success="${3:-false}"

if [[ "${terminate_on_success}" != "true" && "${terminate_on_success}" != "false" ]]; then
  echo "terminate_on_success must be 'true' or 'false' (received: ${terminate_on_success})"
  exit 1
fi

if [[ -z "${checkpoint_path}" ]]; then
  checkpoint_path="$(ls -1dt logs/TransformableWheelMobileRobot-*-cyl-stagec-height-bridge-h0p045-l0p2/checkpoints 2>/dev/null | head -n 1 || true)"
fi

if [[ -z "${checkpoint_path}" ]]; then
  echo "No h0p045 checkpoint directory found. Pass checkpoint path as first argument."
  exit 1
fi

echo "Rendering from checkpoint path: ${checkpoint_path}"
echo "Render episode_length: ${episode_length}"
echo "Render terminate_on_success: ${terminate_on_success}"

suffix="cyl-render-h0p045-follow10-ep${episode_length}-tos${terminate_on_success}"
overrides="$(printf '{"xml_variant":"box","obstacle_height":0.045,"obstacle_x_position":0.6,"obstacle_half_length":0.2,"success_x_margin":0.1,"terminate_on_success":%s}' "${terminate_on_success}")"
./.venv/bin/python train_jax_ppo.py \
  --env_name=TransformableWheelMobileRobot \
  --playground_config_overrides="${overrides}" \
  --load_checkpoint_path="${checkpoint_path}" \
  --num_timesteps=0 \
  --num_envs=64 \
  --num_eval_envs=8 \
  --batch_size=64 \
  --unroll_length=8 \
  --num_minibatches=4 \
  --num_updates_per_batch=2 \
  --num_evals=1 \
  --episode_length="${episode_length}" \
  --seed=1 \
  --suffix="${suffix}" \
  --use_wandb=false \
  --use_tb=false \
  --rollout_camera=cam_robot_follow \
  --rollout_width=640 \
  --rollout_height=480

exp_dir="$(ls -1dt logs/TransformableWheelMobileRobot-*-"${suffix}" 2>/dev/null | head -n 1 || true)"
if [[ -z "${exp_dir}" ]]; then
  echo "Could not locate render experiment directory for suffix ${suffix}."
  exit 1
fi

if [[ -f rollout0.mp4 ]]; then
  cp -f rollout0.mp4 "${exp_dir}/rollout0.mp4"
  cp -f rollout0.mp4 "${exp_dir}/rollout0-follow10.mp4"
  echo "Saved rollout video to: ${exp_dir}/rollout0-follow10.mp4"
else
  echo "rollout0.mp4 was not generated."
  exit 1
fi
