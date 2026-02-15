#!/bin/bash
#SBATCH -J cyl-stagec-sweep
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
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

manifest_path="${1:-}"
if [[ -z "${manifest_path}" ]]; then
  manifest_path="$(ls -1t logs/curriculum-cyl-baseline3-retry-*.json 2>/dev/null | head -n 1 || true)"
fi

if [[ -z "${manifest_path}" ]]; then
  echo "No cyl-baseline3-retry manifest found. Pass manifest path as first argument."
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

echo "Using Stage B checkpoints: ${stageb_ckpt_dir}"

timestamp="$(date +%Y%m%d-%H%M%S)"
heights=(0.04 0.045 0.05 0.055 0.06)
json_outputs=()

for h in "${heights[@]}"; do
  h_tag="${h//./p}"
  out_json="logs/eval-cyl-stagec-feasibility-h${h_tag}-64ep-${timestamp}.json"
  overrides="$(printf '{"xml_variant":"box","obstacle_height":%s,"obstacle_x_position":0.6,"obstacle_half_length":0.2,"success_x_margin":0.1}' "$h")"

  echo "Evaluating height=${h} -> ${out_json}"
  ./.venv/bin/python scripts/eval_twmr_checkpoint.py \
    --checkpoint_path="${stageb_ckpt_dir}" \
    --playground_config_overrides="${overrides}" \
    --num_episodes=64 \
    --batch_size=32 \
    --output_json="${out_json}"

  json_outputs+=("${out_json}")
done

summary_json="logs/eval-cyl-stagec-feasibility-summary-${timestamp}.json"
./.venv/bin/python - <<'PY' "${summary_json}" "${json_outputs[@]}"
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
input_paths = [Path(p) for p in sys.argv[2:]]
rows = []
for p in input_paths:
    data = json.loads(p.read_text())
    rows.append(
        {
            "file": str(p),
            "height": float(data["config_overrides"]["obstacle_height"]),
            "success_rate": float(data["success_rate"]),
            "mean_return": float(data["mean_return"]),
            "mean_x_distance": float(data["mean_x_distance"]),
        }
    )

rows.sort(key=lambda r: r["height"])
summary = {"created_from": [str(p) for p in input_paths], "rows": rows}
summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print("Sweep summary:")
for r in rows:
    print(
        f"  h={r['height']:.3f} success={r['success_rate']:.3f} "
        f"mean_return={r['mean_return']:.3f} mean_x={r['mean_x_distance']:.3f}"
    )
print(f"Wrote summary JSON to: {summary_path}")
PY
