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

Notes:
- Default `--obs-mode` is `compact` (small feature set, easier for CEM).
- If you trained with an older policy, pass `--obs-mode full` to match it.

Play the saved policy:

```bash
./.venv/bin/python sandbox/neil/train_cem_obstacle.py --play --render
```
