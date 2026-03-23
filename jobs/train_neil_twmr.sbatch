#!/bin/bash
#SBATCH -J twmr-neil
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH -t 168:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=njcf2022@mymail.pomona.edu

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"
mkdir -p logs outputs
source .venv/bin/activate

export PATH="$HOME/.pixi/envs/ffmpeg/bin:$PATH"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false

run_timestamp="$(date +%Y%m%d-%H%M%S)"
job_id="${SLURM_JOB_ID:-manual}"
run_output_dir="outputs/neil-twmr-${job_id}-${run_timestamp}"
run_start_epoch="$(date +%s)"

suffix_prefix="${NEIL_TWMR_SUFFIX_PREFIX:-neil-twmr}"
phase="${NEIL_TWMR_PHASE:-full}"
initial_checkpoint="${NEIL_TWMR_INITIAL_CHECKPOINT:-}"
max_devices_per_host="${NEIL_TWMR_MAX_DEVICES_PER_HOST:-1}"
manifest_path="logs/${suffix_prefix}-${phase}-${run_timestamp}.json"

mkdir -p \
  "${run_output_dir}/cwd-rollouts" \
  "${run_output_dir}/phase-rollouts" \
  "${run_output_dir}/manifests" \
  "${run_output_dir}/evals"

echo "[INFO] run_output_dir=${run_output_dir}"
echo "[INFO] suffix_prefix=${suffix_prefix}"
echo "[INFO] phase=${phase}"
echo "[INFO] initial_checkpoint=${initial_checkpoint:-<none>}"
echo "[INFO] max_devices_per_host=${max_devices_per_host}"

if [[ "${phase}" != "full" && "${phase}" != "flat" && "${phase}" != "simpleobstacle" && "${phase}" != "bridge" && "${phase}" != "random" ]]; then
  echo "[ERROR] Unsupported NEIL_TWMR_PHASE: ${phase}" >&2
  exit 2
fi

if [[ ("${phase}" == "simpleobstacle" || "${phase}" == "bridge" || "${phase}" == "random") && -z "${initial_checkpoint}" ]]; then
  echo "[ERROR] NEIL_TWMR_INITIAL_CHECKPOINT is required for phase=${phase}" >&2
  exit 2
fi

if [[ -n "${initial_checkpoint}" && ! -e "${initial_checkpoint}" ]]; then
  echo "[ERROR] Initial checkpoint does not exist: ${initial_checkpoint}" >&2
  exit 2
fi

cmd=(
  ./.venv/bin/python
  scripts/neil-twmr.py
  --phase="${phase}"
  --suffix_prefix="${suffix_prefix}"
  --manifest_path="${manifest_path}"
  --max_devices_per_host="${max_devices_per_host}"
)

if [[ -n "${initial_checkpoint}" ]]; then
  cmd+=(--initial_checkpoint_path="${initial_checkpoint}")
fi

cmd+=("$@")

set +e
"${cmd[@]}"
train_status=$?
set -e

shopt -s nullglob
for video in rollout*.mp4; do
  modified_epoch="$(stat -c %Y "${video}")"
  if [[ "${modified_epoch}" -ge "${run_start_epoch}" ]]; then
    mv -f "${video}" "${run_output_dir}/cwd-rollouts/"
    echo "[INFO] moved ${video} -> ${run_output_dir}/cwd-rollouts/"
  fi
done

for manifest in "${manifest_path}" logs/"${suffix_prefix}"-*.json; do
  [[ -f "${manifest}" ]] || continue
  modified_epoch="$(stat -c %Y "${manifest}")"
  if [[ "${modified_epoch}" -ge "${run_start_epoch}" ]]; then
    cp -a "${manifest}" "${run_output_dir}/manifests/"
  fi
done

for eval_json in logs/eval-*.json; do
  [[ -f "${eval_json}" ]] || continue
  modified_epoch="$(stat -c %Y "${eval_json}")"
  if [[ "${modified_epoch}" -ge "${run_start_epoch}" ]]; then
    cp -a "${eval_json}" "${run_output_dir}/evals/"
  fi
done

for video in logs/TransformableWheelMobileRobot-*/*-rollout0-title.mp4; do
  [[ -f "${video}" ]] || continue
  modified_epoch="$(stat -c %Y "${video}")"
  if [[ "${modified_epoch}" -ge "${run_start_epoch}" ]]; then
    cp -a "${video}" "${run_output_dir}/phase-rollouts/"
  fi
done
shopt -u nullglob

echo "[INFO] archived job artifacts under ${run_output_dir}"
exit "${train_status}"
