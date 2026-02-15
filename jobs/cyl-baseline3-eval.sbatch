#!/bin/bash
#SBATCH -J cyl-baseline3-eval
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

manifest_path="${1:-}"
if [[ -z "${manifest_path}" ]]; then
  manifest_path="$(ls -1t logs/curriculum-cyl-baseline3-*.json 2>/dev/null | head -n 1 || true)"
fi

if [[ -z "${manifest_path}" ]]; then
  echo "No cyl-baseline3 manifest found. Pass path explicitly as first argument."
  exit 1
fi

echo "Using manifest: ${manifest_path}"

stageb_ckpt_dir="$("./.venv/bin/python" - <<'PY' "${manifest_path}"
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
stage_map = {item["stage"]: item["ckpt_dir"] for item in manifest["stages"]}
print(stage_map["stageB-box-easy"])
PY
)"

stagec_ckpt_dir="$("./.venv/bin/python" - <<'PY' "${manifest_path}"
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
stage_map = {item["stage"]: item["ckpt_dir"] for item in manifest["stages"]}
print(stage_map["stageC-box-target"])
PY
)"

timestamp="$(date +%Y%m%d-%H%M%S)"

./.venv/bin/python scripts/eval_twmr_checkpoint.py \
  --checkpoint_path="${stageb_ckpt_dir}" \
  --playground_config_overrides='{"xml_variant":"box","obstacle_height":0.03,"obstacle_x_position":0.55,"obstacle_half_length":0.15,"success_x_margin":0.08}' \
  --num_episodes=64 \
  --batch_size=32 \
  --output_json="logs/eval-cyl-baseline3-stageB-easy-64ep-${timestamp}.json"

./.venv/bin/python scripts/eval_twmr_checkpoint.py \
  --checkpoint_path="${stagec_ckpt_dir}" \
  --playground_config_overrides='{"xml_variant":"box","obstacle_height":0.06,"obstacle_x_position":0.6,"obstacle_half_length":0.2,"success_x_margin":0.1}' \
  --num_episodes=64 \
  --batch_size=32 \
  --output_json="logs/eval-cyl-baseline3-stageC-target-64ep-${timestamp}.json"
