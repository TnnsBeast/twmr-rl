# Neil sandbox

Simple MuJoCo viewer for the transformable wheel robot.

Run:

```bash
./.venv/bin/python sandbox/neil/view_robot.py
```

Notes:
- The default model is `packages/twmr/assets/trans_wheel_robo2_2FLAT.xml` (flat ground).
- To open a different terrain, pass `--model path/to/model.xml`.
- To disable the extension animation, pass `--no-animate`.

---

Derivative-free training (CEM) for obstacle traversal:

```bash
./.venv/bin/python sandbox/neil/train_cem_obstacle.py \
  --model packages/twmr/assets/trans_wheel_robo2_2BOX.xml
```

PPO + LSTM training (proprio-only):

```bash
./.venv/bin/python sandbox/neil/train_ppo_lstm.py
```

Walkthrough: see `sandbox/neil/README_PPO_LSTM.md`.

GPU MJX + PPO pipeline (obstacle traversal):

```bash
./.venv/bin/python sandbox/neil/gpu/train_twmr_obstacle.py
```

Notes:
- Default `--obs-mode` is `proprio` (IMU + motor sensors only).
- `--gate-mode proprio` is now the default and uses only IMU/motor signals to decide
  when to extend; no obstacle position is used.
- Default obstacle model is `sandbox/neil/models/trans_wheel_robo2_2BOX_CLY.xml`
  (with the traverse box moved to `x=1.0`).
- `--success-x` now defaults to the obstacle far edge + `0.05` when the obstacle geom
  is available.
- Retract/extend targets are auto-picked from joint limits (`--auto-retract`) and the
  extension target defaults to the opposite limit (`--extend-mode auto`).
- The initial retract pose is held for `--settle-steps` via a small PD controller
  (`--hold-retract`, `--hold-kp`, `--hold-kd`).
- For debugging in sim, `--gate-mode obstacle` and `--obs-mode compact_obstacle`
  add privileged obstacle signals.
- Obstacle gating defaults to the `traverse_box` geom; adjust with `--obstacle-geom`,
  `--obstacle-buffer`, `--obstacle-pre-buffer`, and `--obstacle-post-buffer`.
- If you trained with an older policy, pass the matching `--obs-mode` to load it.

Play the saved policy:

```bash
./.venv/bin/python sandbox/neil/train_cem_obstacle.py --play --render
```
