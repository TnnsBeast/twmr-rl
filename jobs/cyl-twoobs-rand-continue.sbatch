#!/bin/bash
#SBATCH -J cyl-2obs-rand
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
  checkpoint_path="$(
    ls -1dt \
      logs/TransformableWheelMobileRobot-*-cyl-h045-long/checkpoints \
      logs/TransformableWheelMobileRobot-*-cyl-h045-overnight/checkpoints \
      2>/dev/null | head -n 1 || true
  )"
fi

if [[ -z "${checkpoint_path}" ]]; then
  echo "No continuation checkpoint directory found. Pass checkpoint path as first argument."
  exit 1
fi

if [[ ! -d "${checkpoint_path}" ]]; then
  echo "Checkpoint directory does not exist: ${checkpoint_path}"
  exit 1
fi

if [[ "$(basename "${checkpoint_path}")" =~ ^[0-9]+$ ]] && [[ "$(basename "$(dirname "${checkpoint_path}")")" == "checkpoints" ]]; then
  echo "Using explicit checkpoint step directory: ${checkpoint_path}"
elif ! find "${checkpoint_path}" -mindepth 1 -maxdepth 1 -type d -name '[0-9]*' | grep -q .; then
  echo "Checkpoint path must be either:"
  echo "  1) a checkpoints directory containing numeric step subdirs, or"
  echo "  2) an explicit checkpoint step directory (.../checkpoints/<step>)"
  echo "Received: ${checkpoint_path}"
  exit 1
fi

num_timesteps="${2:-10000000}"
if ! [[ "${num_timesteps}" =~ ^[0-9]+$ ]] || [[ "${num_timesteps}" -le 0 ]]; then
  echo "num_timesteps must be a positive integer. Got: ${num_timesteps}"
  exit 1
fi

echo "Continuing from checkpoint path: $(realpath "${checkpoint_path}")"
echo "Timesteps: ${num_timesteps}"

suffix="cyl-twoobs-rand-continue"
overrides='{"xml_variant":"box_dual","enable_second_obstacle":true,"randomize_obstacles":true,"obstacle_local_windowed_reward":true,"obstacle_height":0.045,"obstacle2_height":0.045,"obstacle_half_length":0.2,"obstacle2_half_length":0.2,"randomize_obstacle_x_min":0.5,"randomize_obstacle_x_max":0.85,"randomize_obstacle_gap_min":0.2,"randomize_obstacle_gap_max":0.5,"success_x_margin":0.1}'
rollout_camera="${3:-}"
rollout_args=()
if [[ -n "${rollout_camera}" ]]; then
  rollout_args+=(--rollout_camera="${rollout_camera}")
fi

./.venv/bin/python train_jax_ppo.py \
  --env_name=TransformableWheelMobileRobot \
  --playground_config_overrides="${overrides}" \
  --load_checkpoint_path="${checkpoint_path}" \
  --num_timesteps="${num_timesteps}" \
  --num_envs=512 \
  --num_eval_envs=64 \
  --batch_size=512 \
  --unroll_length=20 \
  --num_minibatches=16 \
  --num_updates_per_batch=8 \
  --num_evals=12 \
  --episode_length=1600 \
  --num_videos=4 \
  --seed=1 \
  --suffix="${suffix}" \
  --use_wandb=false \
  --use_tb=false \
  "${rollout_args[@]}" \
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

shopt -s nullglob
rollouts=(rollout*.mp4)
if (( ${#rollouts[@]} > 0 )); then
  for rollout in "${rollouts[@]}"; do
    cp -f "${rollout}" "${exp_dir}/${rollout}"
  done
  echo "Saved ${#rollouts[@]} rollout video(s) under: ${exp_dir}"
fi
shopt -u nullglob

timestamp="$(date +%Y%m%d-%H%M%S)"
eval_json="logs/eval-${suffix}-256ep-${timestamp}.json"

./.venv/bin/python scripts/eval_twmr_checkpoint.py \
  --checkpoint_path="${ckpt_dir}" \
  --playground_config_overrides="${overrides}" \
  --num_episodes=256 \
  --episode_length=1600 \
  --batch_size=32 \
  --seed=1 \
  --output_json="${eval_json}"

echo "Wrote consistency eval JSON: ${eval_json}"
