#!/bin/bash
#SBATCH -J cyl-stagec-gbridge
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
  manifest_path="$(ls -1t logs/curriculum-cyl-baseline3-retry-*.json 2>/dev/null | head -n 1 || true)"
  if [[ -z "${manifest_path}" ]]; then
    echo "No cyl-baseline3-retry manifest found. Pass initial checkpoint path as first argument."
    exit 1
  fi
  echo "Using manifest: ${manifest_path}"
  initial_ckpt="$("./.venv/bin/python" - <<'PY' "${manifest_path}"
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
stage_map = {item["stage"]: item["ckpt_dir"] for item in manifest["stages"]}
print(stage_map["stageB-box-easy"])
PY
)"
fi

echo "Using initial checkpoints: ${initial_ckpt}"

./.venv/bin/python scripts/stagec_bridge_curriculum.py \
  --mode=geometry \
  --initial_checkpoint_path="${initial_ckpt}" \
  --suffix_prefix=cyl-stagec-geometry-bridge \
  --fixed_height=0.06 \
  --half_lengths=0.12,0.15,0.18,0.20 \
  --gate_index=2 \
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
