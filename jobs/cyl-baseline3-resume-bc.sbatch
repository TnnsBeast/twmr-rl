#!/bin/bash
#SBATCH -J cyl-baseline3-bc
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

stagea_ckpt_dir="$(ls -1dt logs/TransformableWheelMobileRobot-*-cyl-baseline3-stageA-flat/checkpoints 2>/dev/null | head -n 1 || true)"
if [[ -z "${stagea_ckpt_dir}" ]]; then
  echo "No cyl-baseline3 Stage A checkpoint directory found."
  exit 1
fi

echo "Resuming from Stage A checkpoints: ${stagea_ckpt_dir}"

./.venv/bin/python scripts/phase3_curriculum.py \
  --stages=stageB-box-easy,stageC-box-target \
  --initial_checkpoint_path="${stagea_ckpt_dir}" \
  --suffix_prefix=cyl-baseline3-retry \
  --num_timesteps_override=1000000 \
  --num_envs_override=512 \
  --num_eval_envs_override=64 \
  --batch_size_override=512 \
  --unroll_length_override=20 \
  --num_minibatches_override=16 \
  --num_updates_per_batch_override=8 \
  --num_evals_override=8
