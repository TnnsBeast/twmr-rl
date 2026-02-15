#!/bin/bash
#SBATCH -J cyl-2obs-render
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 04:00:00
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
num_videos="${2:-6}"
episode_length="${3:-2200}"
terminate_on_success="${4:-false}"
rollout_camera="${5:-}"

if ! [[ "${num_videos}" =~ ^[0-9]+$ ]] || [[ "${num_videos}" -le 0 ]]; then
  echo "num_videos must be a positive integer (received: ${num_videos})"
  exit 1
fi

if ! [[ "${episode_length}" =~ ^[0-9]+$ ]] || [[ "${episode_length}" -le 0 ]]; then
  echo "episode_length must be a positive integer (received: ${episode_length})"
  exit 1
fi

if [[ "${terminate_on_success}" != "true" && "${terminate_on_success}" != "false" ]]; then
  echo "terminate_on_success must be 'true' or 'false' (received: ${terminate_on_success})"
  exit 1
fi

if [[ -z "${checkpoint_path}" ]]; then
  checkpoint_path="$(
    ls -1dt \
      logs/TransformableWheelMobileRobot-*-cyl-twoobs-rand-continue/checkpoints \
      logs/TransformableWheelMobileRobot-*-cyl-h045-long/checkpoints \
      logs/TransformableWheelMobileRobot-*-cyl-h045-overnight/checkpoints \
      2>/dev/null | head -n 1 || true
  )"
fi

if [[ -z "${checkpoint_path}" ]]; then
  echo "No checkpoint directory found. Pass checkpoint path as first argument."
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

echo "Rendering from checkpoint path: $(realpath "${checkpoint_path}")"
echo "num_videos=${num_videos}, episode_length=${episode_length}, terminate_on_success=${terminate_on_success}, rollout_camera=${rollout_camera}"

suffix="cyl-twoobs-rand-render-v${num_videos}-ep${episode_length}-tos${terminate_on_success}"
overrides="$(printf '{"xml_variant":"box_dual","enable_second_obstacle":true,"randomize_obstacles":true,"obstacle_local_windowed_reward":true,"obstacle_height":0.045,"obstacle2_height":0.045,"obstacle_half_length":0.2,"obstacle2_half_length":0.2,"randomize_obstacle_x_min":0.5,"randomize_obstacle_x_max":0.85,"randomize_obstacle_gap_min":0.2,"randomize_obstacle_gap_max":0.5,"success_x_margin":0.1,"terminate_on_success":%s}' "${terminate_on_success}")"
rollout_args=()
if [[ -n "${rollout_camera}" ]]; then
  rollout_args+=(--rollout_camera="${rollout_camera}")
fi

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
  --num_videos="${num_videos}" \
  --seed=1 \
  --suffix="${suffix}" \
  --use_wandb=false \
  --use_tb=false \
  "${rollout_args[@]}" \
  --rollout_width=640 \
  --rollout_height=480

exp_dir="$(ls -1dt logs/TransformableWheelMobileRobot-*-"${suffix}" 2>/dev/null | head -n 1 || true)"
if [[ -z "${exp_dir}" ]]; then
  echo "Could not locate render experiment directory for suffix ${suffix}."
  exit 1
fi

shopt -s nullglob
rollouts=(rollout*.mp4)
if (( ${#rollouts[@]} == 0 )); then
  echo "No rollout videos were generated."
  exit 1
fi

for rollout in "${rollouts[@]}"; do
  cp -f "${rollout}" "${exp_dir}/${rollout}"
done
shopt -u nullglob

echo "Saved ${#rollouts[@]} rollout video(s) under: ${exp_dir}"
