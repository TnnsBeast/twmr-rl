#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

./.venv/bin/python sandbox/neil/train_ppo_lstm.py \
  --play \
  --load-path sandbox/neil/ppo_lstm_policy_20260210_000332.pkl \
  --control-cost 5e-4 \
  --extension-penalty 1e-4 \
  --success-bonus 30 \
  --post-success-penalty 0.02 \
  --gate-bias 0.4 \
  --gate-acc-scale 1.0 \
  --gate-gyro-scale 1.0 \
  --gate-stall-scale 0.5 \
  --gate-smooth 0.2
