# AGENTS.md

## Mission
Train and simulate a policy for a **transformable wheel mobile robot** that can traverse an obstacle (step/box) by using both wheel motion and wheel-extension joints.

This document is a working guide for contributors/agents in this repo.

## Current Repository State

### Training entrypoint
- Main script: `train_jax_ppo.py`
- Uses MuJoCo Playground + Brax PPO.
- Imports `twmr` for environment registration side effects.
- Supports config overrides via `--playground_config_overrides`.
- Includes TWMR-specific PPO defaults for `TransformableWheelMobileRobot` (still Brax PPO implementation; no algorithm fork).

### Registered custom environment
- Package env: `packages/twmr/src/twmr/twmr.py`
- Registered env name: `TransformableWheelMobileRobot`
- Current XML selection is config-driven via `xml_variant`:
  - `flat` -> `packages/twmr/assets/trans_wheel_robo2_2FLAT_CLY.xml` (default, cylinder wheels)
  - `box` -> `packages/twmr/assets/trans_wheel_robo2_2BOX_CLY.xml` (cylinder wheels)
  - `terrain` -> `packages/twmr/assets/trans_wheel_robo2_2GEN_TERR_CLY.xml` (cylinder wheels)
  - `flat_sphere` -> `packages/twmr/assets/trans_wheel_robo2_2FLAT.xml` (fallback/debug)
  - `box_sphere` -> `packages/twmr/assets/trans_wheel_robo2_2BOX.xml` (fallback/debug)
  - `terrain_sphere` -> `packages/twmr/assets/trans_wheel_robo2_2GEN_TERR.xml` (fallback/debug)
- `flat` and `box` cylinder variants were validated to load with 8 actuators.
- MJX/JAX compatibility note: cylinder wheel **visual** geoms are preserved, but wheel **collision** geoms in `*_CLY.xml` use `capsule` primitives to avoid unsupported `cylinder x box` contacts.
- Current behavior:
  - Observation: concatenated `qpos + qvel`
  - Reward: shaped with forward velocity, progress delta, survival, control cost, lateral drift penalty, tilt penalty, backward velocity/delta penalties, obstacle-local progress/backtrack/extension terms, stall penalty, failure penalty, success bonus
  - Done: failure (fall/tilt/NaN) and optional success-termination by root x threshold
  - Obstacle task params are config-driven: `obstacle_x_position`, `obstacle_half_length`, `obstacle_height`, `success_x_threshold`/`success_x_margin`
  - Success threshold now derives from obstacle geometry by default (`x + half_length + margin`) unless explicitly overridden
  - Effective minimum base height is clamped from model spawn height to avoid immediate failure on cylinder variants
  - Metrics: decomposed reward terms plus task flags (`success`, failure breakdown, upright, root pose, obstacle params)
  - Curriculum runner copies each stage's `rollout0.mp4` into that stage's log directory
  - Box XML variants now include named diagnostic cameras: `cam_obstacle_close`, `cam_obstacle_top`
  - Preferred rollout camera for box-task videos: `cam_robot_follow` (tracking view)
  - Checkpoint evaluation utility available: `scripts/eval_twmr_checkpoint.py` (reports `success_rate`, `mean_return`, `mean_x_distance`)

### Critical mismatch with end goal
- Core MDP signals now exist, but obstacle-specific task definition and curriculum are still incomplete:
  - easy obstacle is now solved (`success_rate=1.0` at Stage B),
  - target obstacle (`height=0.06`) remains unsolved (`success_rate=0.0`),
  - feasibility sweep from solved Stage B checkpoint shows a sharp cliff: success at `height<=0.045`, failure at `height>=0.05`,
  - geometry-bridge run at target height failed at the first eased stage (`height=0.06`, `half_length=0.12`, `success_rate=0.0`).
- Current bottleneck is no longer basic locomotion; it is robust transfer from easy box to harder obstacle geometry.
- `terrain` variant still references a missing mesh binary path, so only `flat` and `box` are ready now.

### Useful prior work in repo
- `sandbox/isabella/twmr_env.py` contains a basic nonzero reward/done structure.
- `sandbox/jacob/train_transform_PPO*.py` contains more advanced reward shaping and logging ideas for transformable-wheel control.

## End-State Definition
We are done when all are true:
1. `train_jax_ppo.py --env_name=TransformableWheelMobileRobot` trains on a transformable-wheel env (8-actuator model), not `wmr-spheres`.
2. Reward and done logic are task-aligned (forward progress + obstacle success + penalties/failure).
3. A trained checkpoint can be restored and used to produce deterministic rollout video(s) that show obstacle traversal.
4. Evaluation reports obstacle success metrics (not just generic return).
5. Run is reproducible from documented commands.

## Implementation Plan

### Phase 1: Correct the environment target
1. [DONE] Update `packages/twmr/src/twmr/twmr.py` config to choose XML variant (`flat`, `box`, optional `terrain`).
2. [DONE] Default to `trans_wheel_robo2_2FLAT.xml` for initial learning stability.
3. [DONE] Add path handling for XML assets and fail clearly if assets are missing.
4. [DONE] Keep env name `TransformableWheelMobileRobot` to avoid breaking training entrypoint.

**Exit criteria**
- Runtime check shows `nu == 8` and actuator names include wheel + extension actuators.

### Phase 2: Add task-relevant MDP signals
1. [DONE] Implement meaningful reward components in env step:
  - forward progress term (x velocity or x displacement),
  - control/energy penalty,
  - lateral drift/instability penalty,
  - fall penalty,
  - optional sparse success bonus for passing obstacle.
2. [DONE] Implement done conditions:
  - failure (flip/fall/invalid state),
  - success (robot COM or root x passes obstacle threshold),
  - timeout via wrapper episode length.
3. [DONE] Populate `state.metrics` with decomposed reward terms + success flag.

**Exit criteria**
- Training logs show nonzero episode rewards and metric components moving over time.

### Phase 3: Obstacle task and curriculum
1. [DONE] Add environment config keys for obstacle task parameters:
  - obstacle x-position,
  - obstacle height/size,
  - success threshold.
2. [IN PROGRESS] Train curriculum:
  - Stage A: `flat` locomotion baseline,
  - Stage B: `box` with easier height,
  - Stage C: `box` at target height,
  - Stage D (optional): randomized terrain/obstacle.
  - `cyl-baseline3-retry` achieved `success_rate=1.0` for Stage B (easy box), but Stage C target still `0.0`.
  - Stage-C feasibility sweep from solved Stage-B checkpoint: `h=0.04/0.045 -> success_rate=1.0`, `h=0.05/0.055/0.06 -> success_rate=0.0`.
3. [DONE] Resume from checkpoints between stages (`--load_checkpoint_path`).

**Exit criteria**
- Success rate increases across curriculum stages; agent crosses target obstacle reliably.

### Immediate Next Actions (Post `cyl-baseline3` + Feasibility Sweep)
1. Pause bridge escalation and switch to diagnosis/tuning:
  - Height bridge failed at `0.05` and geometry bridge failed at `0.06` even with `half_length=0.12`, so further length/height progression is not justified yet.
2. Run threshold-sensitivity diagnostics on latest `h=0.05` checkpoint:
  - evaluate with reduced `success_x_margin` values (e.g., `0.06`, `0.08`, `0.10`) to quantify whether current policy is near-threshold but not crossing.
3. Tune reward/termination for obstacle engagement before additional long runs:
  - increase incentive for final centimeters past obstacle front/top and penalize stopping just short of success threshold.
  - keep obstacle-local extension shaping active and verify extension behavior in close-up rollout.
4. Re-run only one targeted bridge stage after tuning:
  - retry `height=0.05`, `half_length=0.2` first; require nonzero success on 64-episode eval before reattempting `height=0.06`.
5. Keep promotion gates strict for compute efficiency:
  - `success_rate > 0` minimum gate at every stage and stronger gate (`>=0.3`) before advancing to harder geometry.

### Phase 4: Training config specialization for TWMR
1. [DONE] Add TWMR-specific PPO defaults in `train_jax_ppo.py` (instead of only generic dm_control defaults).
2. [IN PROGRESS] Keep initial settings conservative for fast iteration, then scale up.
3. [DONE] Log success-oriented metrics in `progress()` if present.

**Exit criteria**
- Stable training without huge resource waste; metrics include success-related values.

### Phase 5: Evaluation and simulation outputs
1. [DONE] Standardize eval command(s) and checkpoint selection (`scripts/eval_twmr_checkpoint.py`).
2. Generate deterministic rollout videos after training (`rollout*.mp4`) on obstacle variant.
3. [DONE] Add a small evaluation utility to compute:
  - success rate over N episodes,
  - mean episode return,
  - mean +x distance.

**Exit criteria**
- One command path from checkpoint to reproducible obstacle-traversal video + metrics.

## Recommended Command Path (Target Workflow)

### 0) One-command curriculum runner (recommended)
```bash
./.venv/bin/python scripts/phase3_curriculum.py
```

### 0b) Mid-scale pilot from existing Stage A checkpoint
```bash
./.venv/bin/python scripts/phase3_curriculum.py \
  --stages=stageB-box-easy,stageC-box-target \
  --initial_checkpoint_path=logs/<stageA_exp>/checkpoints \
  --num_timesteps_override=65536 \
  --num_envs_override=256 \
  --num_eval_envs_override=32 \
  --batch_size_override=256 \
  --unroll_length_override=16 \
  --num_minibatches_override=8 \
  --num_updates_per_batch_override=4 \
  --num_evals_override=3 \
  --episode_length_override=256 \
  --suffix_prefix=phase3pilot
```

### 1) Baseline train on flat
```bash
./.venv/bin/python train_jax_ppo.py \
  --env_name=TransformableWheelMobileRobot \
  --playground_config_overrides='{"xml_variant":"flat"}'
```

### 2) Fine-tune on box obstacle
```bash
./.venv/bin/python train_jax_ppo.py \
  --env_name=TransformableWheelMobileRobot \
  --playground_config_overrides='{"xml_variant":"box","obstacle_height":0.03,"obstacle_x_position":0.55,"obstacle_half_length":0.15,"success_x_margin":0.08}' \
  --load_checkpoint_path=logs/<flat_exp>/checkpoints
```

### 3) Fine-tune on target box obstacle
```bash
./.venv/bin/python train_jax_ppo.py \
  --env_name=TransformableWheelMobileRobot \
  --playground_config_overrides='{"xml_variant":"box","obstacle_height":0.06,"obstacle_x_position":0.60,"obstacle_half_length":0.20,"success_x_margin":0.10}' \
  --load_checkpoint_path=logs/<easy_box_exp>/checkpoints
```

### 4) Render policy rollouts
- Use final training output from `train_jax_ppo.py` (built-in rollout export) and `scripts/eval_twmr_checkpoint.py` for quantitative scoring.

### 5) Evaluate checkpoint metrics
```bash
./.venv/bin/python scripts/eval_twmr_checkpoint.py \
  --checkpoint_path=logs/<exp>/checkpoints \
  --playground_config_overrides='{"xml_variant":"box","obstacle_height":0.06,"obstacle_x_position":0.60,"obstacle_half_length":0.20,"success_x_margin":0.10}' \
  --num_episodes=64 \
  --batch_size=32 \
  --output_json=logs/eval-<exp>.json
```

### 6) Stage-C feasibility sweep from solved Stage-B checkpoint
```bash
sbatch jobs/cyl-stagec-feasibility-sweep.sbatch \
  logs/curriculum-cyl-baseline3-retry-<timestamp>.json
```

### 7) Stage-C height bridge with promotion gates (`0.045 -> 0.05 -> ...`)
```bash
sbatch jobs/cyl-stagec-height-bridge.sbatch \
  logs/curriculum-cyl-baseline3-retry-<timestamp>.json
```

### 7b) Resume Stage-C height bridge from saved `h=0.045` checkpoint
```bash
# Optional arg: checkpoint path; defaults to latest height-bridge h0p045 checkpoint.
sbatch jobs/cyl-stagec-height-bridge-resume.sbatch \
  logs/<height-bridge-h0p045-exp>/checkpoints
```

### 8) Stage-C geometry bridge at target height (`height=0.06`, sweep half-length)
```bash
# Optional arg: initial checkpoint path; defaults to latest cyl-baseline3-retry Stage B ckpt.
sbatch jobs/cyl-stagec-geometry-bridge.sbatch \
  logs/<bridge_or_stageB_exp>/checkpoints
```

### 9) Render `h=0.045` rollout with follow camera (no additional training)
```bash
# Optional arg: checkpoint path; defaults to latest height-bridge h0p045 checkpoint.
sbatch jobs/cyl-render-h0p045-follow.sbatch \
  logs/<height-bridge-h0p045-exp>/checkpoints
```

### 10) Long continuation training at `h=0.045` + consistency eval
```bash
# Optional arg: checkpoint path; defaults to latest height-bridge h0p045 checkpoint.
sbatch jobs/cyl-h045-continue-long.sbatch \
  logs/<height-bridge-h0p045-exp>/checkpoints
```

### Smoke validation for curriculum chain
```bash
./.venv/bin/python scripts/phase3_curriculum.py --smoke --suffix_prefix=phase3smoke
```

## Known Risks / Gaps
- `trans_wheel_robo2_2GEN_TERR.xml` currently references `meshes/terrain_height_go.bin` that is not present under `packages/twmr/assets/meshes`.
- Reward shaping can easily produce local optima (e.g., spinning/retracting without traversal); success-based terms and curriculum are required.
- Stage-C performance has a sharp feasibility cliff with current policy (`height<=0.045` solved, `height>=0.05` unsolved); promotion must be height-gated.
- Even with TWMR defaults, reward can improve without achieving task success; success-rate gating is required for checkpoint promotion.
- Rollout video writing in `train_jax_ppo.py` requires `ffmpeg` via `mediapy`; on this cluster, prefer PATH entry `~/.pixi/envs/ffmpeg/bin` (real binary) over only `~/.pixi/bin/ffmpeg` shim.
- Headless rendering can fail on Slurm if EGL is not configured before MuJoCo/OpenGL init; maintain `MUJOCO_GL=egl` and `PYOPENGL_PLATFORM=egl` in batch environments.
- Large-scale curriculum settings (`num_envs=2048`, long unrolls) can incur very long compile/start latency; use mid-scale pilot overrides first on new nodes/configurations.

## Progress Log
- 2026-02-10 20:51 PST: Completed Phase 1 Step 1 in `packages/twmr/src/twmr/twmr.py`.
  - Added `xml_variant` config with `flat` default.
  - Added xml mapping for `flat`, `box`, `terrain`.
  - Added validation errors for unsupported/missing variant files.
  - Added local asset loading so relative XML asset references are available.
  - Verified runtime load for `flat` and `box` with `nu=8`.
- 2026-02-10 20:55 PST: Phase 1 validation suite passed (4/4).
  - Verified default config uses `xml_variant=flat`.
  - Verified `flat` and `box` variants both load and complete reset+step with `nu=8`.
  - Verified invalid `xml_variant` fails with clear error message.
- 2026-02-10 21:02 PST: End-to-end tiny training smoke run passed core env wiring and training path.
  - Command used `train_jax_ppo.py` with tiny settings and `--playground_config_overrides='{\"xml_variant\":\"flat\"}'`.
  - Environment/params loaded correctly and training/inference completed.
  - Run failed only at final video write due to missing `ffmpeg` (external runtime dependency).
- 2026-02-10 21:06 PST: Installed user-level `ffmpeg` via Pixi global environment.
  - Command: `pixi global install ffmpeg --expose ffmpeg`.
  - Verified binary path: `~/.pixi/bin/ffmpeg`.
  - Verified version: `ffmpeg 8.0.1`.
- 2026-02-10 21:08 PST: Verified `mediapy` video export with real Pixi ffmpeg binary path.
  - `media.write_video` smoke test succeeds when PATH includes `~/.pixi/envs/ffmpeg/bin`.
  - Recommended for batch jobs: `export PATH=\"$HOME/.pixi/envs/ffmpeg/bin:$PATH\"`.
- 2026-02-10 21:15 PST: Re-ran `train_jax_ppo.py` tiny end-to-end smoke and confirmed rollout export works.
  - Command used `--playground_config_overrides='{\"xml_variant\":\"flat\"}'` with reduced PPO settings.
  - Training finished and printed: `Rollout video saved as 'rollout0.mp4'.`
  - Output artifact verified: `rollout0.mp4` (MP4 container) created in repo root.
- 2026-02-10 21:22 PST: Completed Phase 2 MDP implementation in `packages/twmr/src/twmr/twmr.py`.
  - Added reward config knobs: forward, survival, control cost, lateral penalty, tilt penalty, failure penalty, success bonus.
  - Added done logic for failure (`height`, `upright`, `NaN`) and success (`root_x >= success_x_threshold`) with `terminate_on_success`.
  - Added decomposed metrics keys for reward terms and task state flags.
- 2026-02-10 21:22 PST: Phase 2 validation checks passed.
  - Direct env probe (`flat`, `box`) showed nonzero reward and populated Phase 2 metric keys.
  - Tiny end-to-end `train_jax_ppo.py` smoke run completed with nonzero training reward log (`reward=-1.557`) and successful rollout export (`rollout0.mp4`).
- 2026-02-11 07:24 PST: Completed Phase 3 Step 1 obstacle-task parameterization in `packages/twmr/src/twmr/twmr.py`.
  - Added config keys: `obstacle_geom_name`, `obstacle_site_name`, `obstacle_x_position`, `obstacle_half_length`, `obstacle_height`, `success_x_threshold`, `success_x_margin`.
  - Added model-geometry overrides for obstacle x/size/height on `box` variant when config overrides are provided.
  - Added derived success threshold (`obstacle_x + obstacle_half_length + success_x_margin`) when explicit threshold is not set.
  - Added obstacle/task metrics: obstacle geometry values, success threshold, distance-to-success, obstacle-present flag.
- 2026-02-11 07:24 PST: Phase 3 initial validation checks passed.
  - Direct probes validated `flat` fallback and `box` geometry-aware thresholds, including override case (`obstacle_height=0.03`, `obstacle_x_position=0.55`, `obstacle_half_length=0.15` -> `success_x_threshold=0.78`).
  - Tiny end-to-end `box` curriculum-style smoke run completed with nonzero reward log (`reward=-1.699`) and successful rollout export (`rollout0.mp4`).
- 2026-02-11 10:06 PST: Added curriculum runner script `scripts/phase3_curriculum.py` for Stage A/B/C automation.
  - Automates checkpoint chaining: Stage B restores from Stage A checkpoints, Stage C restores from Stage B checkpoints.
  - Encodes both production defaults and `--smoke` settings.
  - Ensures ffmpeg path is injected for rollout export in each stage process.
- 2026-02-11 10:06 PST: Validated Phase 3 checkpoint curriculum chain end to end.
  - Ran `./.venv/bin/python scripts/phase3_curriculum.py --smoke --suffix_prefix=phase3smoke`.
  - Stage A completed: `TransformableWheelMobileRobot-20260211-095430-phase3smoke-stageA-flat`.
  - Stage B restored from Stage A checkpoint and completed: `TransformableWheelMobileRobot-20260211-095817-phase3smoke-stageB-box-easy`.
  - Stage C restored from Stage B checkpoint and completed: `TransformableWheelMobileRobot-20260211-100207-phase3smoke-stageC-box-target`.
- 2026-02-11 10:47 PST: Expanded `scripts/phase3_curriculum.py` controls for practical HPC iteration.
  - Added stage selection (`--stages`), optional first-stage restore (`--initial_checkpoint_path`), global PPO override flags, and manifest output.
  - Added `--dry_run` validation path for command inspection.
  - Added JSON run manifest output under `logs/curriculum-<suffix>-<timestamp>.json`.
- 2026-02-11 10:47 PST: Ran mid-scale curriculum continuation from Stage A -> Stage B -> Stage C checkpoints.
  - Stage B run completed at `TransformableWheelMobileRobot-20260211-102800-phase3pilot2-stageB-box-easy` with checkpoint `000000327680`.
  - Stage C resumed from Stage B checkpoints and completed at `TransformableWheelMobileRobot-20260211-104033-phase3pilot2c-stageC-box-target`.
  - Stage C eval reward improved during run: `0: -0.840`, `327680: -0.498`, `655360: -0.484`; rollout export succeeded.
  - Manifest saved: `logs/curriculum-phase3pilot2c-20260211-104704.json`.
- 2026-02-11 13:09 PST: Completed first full Stage A/B/C curriculum run at recommended mid-scale (`phase3run1`).
  - Command: `scripts/phase3_curriculum.py` with `num_timesteps=1,000,000`, `num_envs=512`, `num_eval_envs=64`, `batch_size=512`, `unroll_length=20`, `num_minibatches=16`, `num_updates_per_batch=8`, `num_evals=8`.
  - Stage A (`TransformableWheelMobileRobot-20260211-105718-phase3run1-stageA-flat`) showed flat reward trend near `-1.90`.
  - Stage B (`TransformableWheelMobileRobot-20260211-113444-phase3run1-stageB-box-easy`) improved strongly from `-1.823` to `-0.242`.
  - Stage C (`TransformableWheelMobileRobot-20260211-120850-phase3run1-stageC-box-target`) improved from `-0.242` to `-0.203`.
  - Checkpoint chaining worked A->B->C from latest stage checkpoints (`000011468800` restore points shown in logs).
  - Rollout export succeeded for all stages; manifest saved at `logs/curriculum-phase3run1-20260211-124345.json`.
- 2026-02-11 13:09 PST: Qualitative review of `rollout0.mp4` from full run indicates weak behavior.
  - Robot and obstacle render correctly, but policy appears largely passive with limited purposeful obstacle traversal.
  - Conclusion: reward signal is improving numerically, but objective behavior has not yet emerged; next iteration should focus on explicit success metrics + anti-stall reward shaping.
- 2026-02-11 13:45 PST: Completed `phase3run2` Stage B->C continuation with anti-stall shaping enabled.
  - Command used `scripts/phase3_curriculum.py --stages=stageB-box-easy,stageC-box-target --num_timesteps_override=131072 --num_envs_override=256 --num_eval_envs_override=32 --batch_size_override=256 --unroll_length_override=16 --num_minibatches_override=8 --num_updates_per_batch_override=4 --num_evals_override=4 --initial_checkpoint_path=<phase3run1-stageA>/checkpoints --suffix_prefix=phase3run2`.
  - Stage B (`TransformableWheelMobileRobot-20260211-132713-phase3run2-stageB-box-easy`) reward trajectory: `-2.169 -> -1.443`.
  - Stage C (`TransformableWheelMobileRobot-20260211-133625-phase3run2-stageC-box-target`) reward trajectory: `-1.402 -> -0.546`.
  - Manifest saved: `logs/curriculum-phase3run2-20260211-134527.json`.
- 2026-02-11 13:52 PST: Quantitative eval pass completed for phase3run1/phase3run2 checkpoints (64 episodes each).
  - `phase3run1` Stage B: `success_rate=0.0`, `mean_return=-0.4769`, `mean_x_distance=0.2710`.
  - `phase3run1` Stage C: `success_rate=0.0`, `mean_return=-0.4469`, `mean_x_distance=0.0378`.
  - `phase3run2` Stage B: `success_rate=0.0`, `mean_return=-0.6621`, `mean_x_distance=0.4235`.
  - `phase3run2` Stage C: `success_rate=0.0`, `mean_return=-0.4860`, `mean_x_distance=0.5358`.
  - Conclusion: anti-stall shaping improved x-distance, but still no successful obstacle completion.
- 2026-02-11 13:58 PST: Implemented Phase 4 training-script specialization and logging updates in `train_jax_ppo.py`.
  - Added TWMR-specific PPO defaults (`num_timesteps=5M`, `num_envs=1024`, `num_eval_envs=64`, `batch_size=512`, `unroll_length=20`, `num_minibatches=16`, `num_updates_per_batch=8`, `reward_scaling=10.0`, `learning_rate=1e-3`).
  - Progress callback now logs success-oriented eval metrics when present (`success`, `root_x`, `dist_to_success`).
  - Tiny smoke run passed (`TransformableWheelMobileRobot-20260211-135435-phase4-smoke`) and printed: `reward=-2.011, success=0.000, root_x=-0.000, dist_to_success=4.500`.
- 2026-02-11 14:07 PST: Switched default TWMR model variants to cylinder-wheel XMLs.
  - Copied transformable cylinder XML assets from `sandbox/jacob/TestingAndDataCollectionFunctions/Cylindrial Wheel XMLs/` into `packages/twmr/assets/`.
  - Updated `packages/twmr/src/twmr/twmr.py` mapping so `flat/box/terrain` use `*_CLY.xml`.
  - Added explicit sphere fallback variants: `flat_sphere`, `box_sphere`, `terrain_sphere`.
  - Runtime verification: `flat` and `box` now report wheel geom type `cylinder` and `nu=8`; sphere fallback variants remain available for A/B comparisons.
- 2026-02-11 14:30 PST: Resolved JAX backend failure for cylinder-wheel variants.
  - Observed failure in `train_jax_ppo.py`: `NotImplementedError: (mjtGeom.mjGEOM_CYLINDER, mjtGeom.mjGEOM_BOX) collisions not implemented.`
  - Confirmed `impl=warp` loads env but was unstable for PPO on this stack (segfault in warp collision narrowphase during training).
  - Updated `packages/twmr/assets/trans_wheel_robo2_2{FLAT,BOX,GEN_TERR}_CLY.xml`:
    - wheel visual geoms remain `type="cylinder"`,
    - wheel collision geoms changed to `type="capsule"` for MJX/JAX compatibility.
  - Re-validated env loading in JAX for `xml_variant=flat` and `xml_variant=box` with `nu=8`.
- 2026-02-11 20:09 PST: Cylinder curriculum smoke chain (`cyl-smoke2`) completed end-to-end after compatibility patch.
  - Command: `./.venv/bin/python scripts/phase3_curriculum.py --smoke --suffix_prefix=cyl-smoke2`.
  - Stage A/B/C all completed with checkpoint handoff and rollout export.
  - Stage A: `TransformableWheelMobileRobot-20260211-195700-cyl-smoke2-stageA-flat`.
  - Stage B: `TransformableWheelMobileRobot-20260211-200109-cyl-smoke2-stageB-box-easy`.
  - Stage C: `TransformableWheelMobileRobot-20260211-200521-cyl-smoke2-stageC-box-target`.
  - Manifest: `logs/curriculum-cyl-smoke2-20260211-200919.json`.
  - Smoke rewards remained near `-1.935` with `success=0.000` (expected at tiny `num_timesteps=4096`; goal was runtime validation, not policy quality).
- 2026-02-11 20:20 PST: Added Slurm batch script for reproducible cylinder mid-scale curriculum runs.
  - New file: `jobs/cyl-baseline2.sbatch`.
  - Encodes A100/8 CPU/120G/24h resource request and runs `scripts/phase3_curriculum.py` with `cyl-baseline2` mid-scale overrides.
- 2026-02-12 09:56 PST: Ran `jobs/cyl-baseline3.sbatch` (`JobID=29201`); Stage A training completed but job failed at rollout rendering.
  - Stage A experiment completed training and saved checkpoints at `logs/TransformableWheelMobileRobot-20260212-091238-cyl-baseline3-stageA-flat/checkpoints`.
  - Failure occurred during `mujoco.Renderer(...)` initialization (`DISPLAY` missing / no OpenGL platform context) while writing final rollout.
  - This was a render-path failure after checkpoint generation, not a PPO training crash.
- 2026-02-12 (post-`29201`): Hardened headless rendering + notifications in training jobs.
  - `train_jax_ppo.py` now sets EGL-safe defaults before MuJoCo init: `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`, and `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
  - Slurm scripts for follow-up runs include explicit email directives (`--mail-type=END,FAIL`, `--mail-user=njcf2022@mymail.pomona.edu`).
  - Added/updated scripts: `jobs/cyl-baseline3.sbatch`, `jobs/cyl-baseline3-resume-bc.sbatch`, `jobs/cyl-baseline3-eval.sbatch`, `jobs/cyl-stagec-feasibility-sweep.sbatch`.
- 2026-02-12 16:04 PST: Resume run (`JobID=29217`, `jobs/cyl-baseline3-resume-bc.sbatch`) completed Stage B -> Stage C.
  - Stage B experiment: `logs/TransformableWheelMobileRobot-20260212-135445-cyl-baseline3-retry-stageB-box-easy`.
  - Stage C experiment: `logs/TransformableWheelMobileRobot-20260212-150110-cyl-baseline3-retry-stageC-box-target`.
  - Curriculum manifest saved: `logs/curriculum-cyl-baseline3-retry-20260212-160435.json`.
  - Both stages produced rollouts and checkpoints; run exit code was `0`.
- 2026-02-12 18:40 PST: Eval run (`JobID=29237`, `jobs/cyl-baseline3-eval.sbatch`) completed quantitative check.
  - Stage B easy checkpoint eval (`64` episodes): `success_rate=1.0`, `mean_return=115.50`, `mean_x_distance=0.8528`.
  - Stage C target checkpoint eval (`64` episodes): `success_rate=0.0`, `mean_return=71.45`, `mean_x_distance=0.3829`.
  - Output JSONs: `logs/eval-cyl-baseline3-stageB-easy-64ep-20260212-183738.json`, `logs/eval-cyl-baseline3-stageC-target-64ep-20260212-183738.json`.
- 2026-02-12 19:07 PST: Stage-C feasibility sweep (`JobID=29238`, `jobs/cyl-stagec-feasibility-sweep.sbatch`) completed.
  - Sweep used solved Stage B checkpoint from `cyl-baseline3-retry`.
  - Results (64 episodes each): `h=0.04 -> 1.0`, `h=0.045 -> 1.0`, `h=0.05 -> 0.0`, `h=0.055 -> 0.0`, `h=0.06 -> 0.0`.
  - Summary artifact: `logs/eval-cyl-stagec-feasibility-summary-20260212-185909.json`.
  - Conclusion: sharp transfer cliff begins at approximately `obstacle_height=0.05` under current shaping/policy.
- 2026-02-12 19:20 PST: Added automated Stage-C bridge tooling and launch scripts for next development cycle.
  - New script: `scripts/stagec_bridge_curriculum.py`.
  - Supports two modes:
    - `height`: sequential height bridge with checkpoint handoff + eval gating.
    - `geometry`: fixed-height (`0.06`) half-length bridge with checkpoint handoff + eval gating.
  - New Slurm jobs:
    - `jobs/cyl-stagec-height-bridge.sbatch` (defaults to latest `cyl-baseline3-retry` Stage B checkpoint, enforces `>0` success promotion baseline via `0.015625` threshold on 64-episode eval, and applies strong gate at stage index `1` with threshold `0.3`).
    - `jobs/cyl-stagec-geometry-bridge.sbatch` (defaults to latest `cyl-baseline3-retry` Stage B checkpoint if no explicit checkpoint arg is provided, enforces `>0` success promotion baseline via `0.015625`, and applies strong gate at stage index `2` with threshold `0.3`).
  - Validation:
    - `python -m py_compile scripts/stagec_bridge_curriculum.py` passed.
    - `bash -n` syntax checks for both new Slurm scripts passed.
    - Dry run command path validated with `--dry_run` and wrote smoke manifest `logs/bridge-cyl-stagec-height-bridge-smokecheck-20260212-191945.json`.
- 2026-02-12 19:23 PST: Added obstacle-focused rollout camera support for easier behavior diagnosis near contact.
  - Added rollout flags in `train_jax_ppo.py`: `--rollout_camera`, `--rollout_width`, `--rollout_height`.
  - Added named cameras to box XML variants (`packages/twmr/assets/trans_wheel_robo2_2BOX_CLY.xml`, `packages/twmr/assets/trans_wheel_robo2_2BOX.xml`):
    - `cam_obstacle_close`
    - `cam_obstacle_top`
  - `scripts/stagec_bridge_curriculum.py` now requests `cam_obstacle_close` by default and writes higher-resolution rollouts (`960x540`).
  - Validation: loaded `xml_variant=box` and confirmed camera names are available (`ncam=2`).
- 2026-02-12 19:28 PST: Launched Stage-C height bridge run on Slurm (`JobID=29239`) and it is currently running.
  - Submit command: `sbatch jobs/cyl-stagec-height-bridge.sbatch logs/curriculum-cyl-baseline3-retry-20260212-160435.json`.
  - Current scheduler state at log time: `R` on `gpu002`.
  - Job output logs: `logs/cyl-stagec-hbridge-29239.out`, `logs/cyl-stagec-hbridge-29239.err`.
  - Expected completion artifact: `logs/bridge-cyl-stagec-height-bridge-<timestamp>.json`.
- 2026-02-12 20:32 PST: Stage-C height bridge run (`JobID=29239`) failed after Stage-0 training due to rollout framebuffer dimensions.
  - Stage-0 (`height=0.045`) training completed and saved checkpoints under `logs/TransformableWheelMobileRobot-20260212-192709-cyl-stagec-height-bridge-h0p045-l0p2/checkpoints`.
  - Failure happened during rollout rendering, not PPO optimization:
    - `ValueError: Image width 960 > framebuffer width 640`.
  - Root cause: bridge runner requested `--rollout_width=960` without corresponding MuJoCo offscreen framebuffer resize in XML.
- 2026-02-12 20:35 PST: Applied rendering robustness + resume support after `29239` failure.
  - `scripts/stagec_bridge_curriculum.py` rollout defaults changed to `640x480` for framebuffer compatibility.
  - `train_jax_ppo.py` rollout export is now best-effort (render exceptions are logged, but training/checkpoint completion is preserved).
  - Added resume launcher: `jobs/cyl-stagec-height-bridge-resume.sbatch` to continue from saved `h=0.045` checkpoint and run `0.05 -> 0.055 -> 0.06` with gates.
  - Validation:
    - `python -m py_compile train_jax_ppo.py scripts/stagec_bridge_curriculum.py` passed.
    - `bash -n jobs/cyl-stagec-height-bridge-resume.sbatch` passed.
    - Resume dry-run command path validated and wrote smoke manifest `logs/bridge-cyl-stagec-height-bridge-resume-smokecheck-20260212-203558.json`.
- 2026-02-12 20:37 PST: Launched height-bridge resume run (`JobID=29242`) from saved `h=0.045` checkpoint; currently running.
  - Submit command: `sbatch jobs/cyl-stagec-height-bridge-resume.sbatch logs/TransformableWheelMobileRobot-20260212-192709-cyl-stagec-height-bridge-h0p045-l0p2/checkpoints`.
  - Current scheduler state at log time: `R` on `gpu002`.
  - Live logs: `logs/cyl-stagec-hbridge-r-29242.out`, `logs/cyl-stagec-hbridge-r-29242.err`.
  - Stage 0 resume target confirmed in logs: `height=0.05`, `half_length=0.2`.
- 2026-02-12 21:45 PST: Height-bridge resume run (`JobID=29242`) completed; gate failed at Stage 0 (`height=0.05`).
  - Slurm state: `COMPLETED`, `ExitCode=0` (run finished cleanly with expected early-stop behavior from gating logic).
  - Stage 0 training artifact: `logs/TransformableWheelMobileRobot-20260212-203706-cyl-stagec-height-bridge-resume-h0p05-l0p2`.
  - Stage 0 rollout artifact exists: `rollout0.mp4` (obstacle-close camera, `640x480`).
  - Stage 0 eval (`64` episodes): `success_rate=0.0`, `mean_return=210.75`, `mean_x_distance=0.8776`.
  - Promotion gate result: failed (`threshold=0.3`), so stages `0.055`/`0.06` were not launched in this job.
  - Manifest: `logs/bridge-cyl-stagec-height-bridge-resume-20260212-203653.json`.
  - Decision impact: continue with Stage-C geometry bridge branch (`height=0.06`, half-length sweep) before additional long runtime scaling.
- 2026-02-13 07:49 PST: Launched Stage-C geometry bridge run (`JobID=29246`); currently running.
  - Submit command: `sbatch jobs/cyl-stagec-geometry-bridge.sbatch`.
  - Current scheduler state at log time: `R` on `gpu002`.
  - Active Stage 0 target from log: `height=0.06`, `half_length=0.12`.
  - Initial checkpoint source: `logs/TransformableWheelMobileRobot-20260212-135445-cyl-baseline3-retry-stageB-box-easy/checkpoints`.
  - Live logs: `logs/cyl-stagec-gbridge-29246.out`, `logs/cyl-stagec-gbridge-29246.err`.
- 2026-02-13 08:54 PST: Stage-C geometry bridge run (`JobID=29246`) completed; failed gate at Stage 0.
  - Slurm state: `COMPLETED`, `ExitCode=0` (run ended cleanly with expected gate stop).
  - Stage 0 training artifact: `logs/TransformableWheelMobileRobot-20260213-074927-cyl-stagec-geometry-bridge-h0p06-l0p12`.
  - Stage 0 rollout artifact exists: `rollout0.mp4` (obstacle-close camera, `640x480`).
  - Stage 0 eval (`64` episodes): `success_rate=0.0`, `mean_return=54.32`, `mean_x_distance=0.4615`.
  - Promotion gate result: failed baseline gate (`threshold=0.015625`), so stages `half_length=0.15/0.18/0.20` were not launched.
  - Manifest: `logs/bridge-cyl-stagec-geometry-bridge-20260213-074914.json`.
  - Decision impact: bridge curricula (height + geometry) are both blocked at first gate; next iteration should prioritize reward/termination tuning + threshold-sensitivity diagnostics.
- 2026-02-13 09:00 PST: Added follow-camera render path for `h=0.045` checkpoint videos.
  - Added camera `cam_robot_follow` (`mode=trackcom`) to box XML variants:
    - `packages/twmr/assets/trans_wheel_robo2_2BOX_CLY.xml`
    - `packages/twmr/assets/trans_wheel_robo2_2BOX.xml`
  - Added render-only Slurm script: `jobs/cyl-render-h0p045-follow.sbatch`.
  - Script restores the `h=0.045` checkpoint, runs `train_jax_ppo.py` with `num_timesteps=0`, and renders rollout video using `--rollout_camera=cam_robot_follow`.
  - Output video is copied into the render experiment directory as `rollout0-follow.mp4`.
- 2026-02-13 09:35 PST: Render-only follow-camera job completed successfully (`JobID=29247`).
  - Submit command used: `sbatch jobs/cyl-render-h0p045-follow.sbatch`.
  - Restored checkpoint source: `logs/TransformableWheelMobileRobot-20260212-192709-cyl-stagec-height-bridge-h0p045-l0p2/checkpoints/000011468800`.
  - Render experiment directory: `logs/TransformableWheelMobileRobot-20260213-093244-cyl-render-h0p045-follow`.
  - Output videos:
    - `logs/TransformableWheelMobileRobot-20260213-093244-cyl-render-h0p045-follow/rollout0-follow.mp4`
    - `logs/TransformableWheelMobileRobot-20260213-093244-cyl-render-h0p045-follow/rollout0.mp4`
- 2026-02-13 09:39 PST: Adjusted follow camera to wider view and re-rendered `h=0.045` checkpoint video (`JobID=29248`).
  - Updated `cam_robot_follow` offset in box XML variants from `(-0.35, -0.35, 0.18)` to `(-0.75, -0.55, 0.30)` while keeping `mode=trackcom`.
  - New render experiment directory: `logs/TransformableWheelMobileRobot-20260213-093748-cyl-render-h0p045-follow`.
  - Updated videos:
    - `logs/TransformableWheelMobileRobot-20260213-093748-cyl-render-h0p045-follow/rollout0-follow.mp4`
    - `logs/TransformableWheelMobileRobot-20260213-093748-cyl-render-h0p045-follow/rollout0.mp4`
- 2026-02-13 10:46 PST: Reverted camera to pre-follow behavior and re-rendered with default camera (`JobID=29250`).
  - Render script `jobs/cyl-render-h0p045-follow.sbatch` now uses default rollout camera again (no explicit `--rollout_camera` flag).
  - Render suffix switched to `cyl-render-h0p045-defaultcam`.
  - New output directory: `logs/TransformableWheelMobileRobot-20260213-104412-cyl-render-h0p045-defaultcam`.
  - Output videos:
    - `logs/TransformableWheelMobileRobot-20260213-104412-cyl-render-h0p045-defaultcam/rollout0-defaultcam.mp4`
    - `logs/TransformableWheelMobileRobot-20260213-104412-cyl-render-h0p045-defaultcam/rollout0.mp4`
- 2026-02-13 11:01 PST: Added tuned follow-camera profile (~10% robot frame occupancy target) and rendered new video (`JobID=29251`).
  - Updated `cam_robot_follow` in box XML variants to use explicit offset + orientation:
    - `pos="-1.000 -0.800 0.350"`
    - `quat="0.716419 0.546893 -0.262850 -0.344329"`
    - `mode="trackcom"`, `fovy=45` (default)
  - Added dedicated render script: `jobs/cyl-render-h0p045-follow10.sbatch`.
  - Render output directory: `logs/TransformableWheelMobileRobot-20260213-110000-cyl-render-h0p045-follow10`.
  - Output videos:
    - `logs/TransformableWheelMobileRobot-20260213-110000-cyl-render-h0p045-follow10/rollout0-follow10.mp4`
    - `logs/TransformableWheelMobileRobot-20260213-110000-cyl-render-h0p045-follow10/rollout0.mp4`
- 2026-02-13 11:05 PST: Locked approved follow camera as default for future rollout generation.
  - `scripts/stagec_bridge_curriculum.py` now defaults `--rollout_camera=cam_robot_follow`.
  - `jobs/cyl-render-h0p045-follow.sbatch` now renders with `cam_robot_follow` and outputs `rollout0-follow10.mp4`.
  - This camera/profile should be used going forward unless explicitly overridden.
- 2026-02-13 11:16 PST: Generated extended follow-camera rollout to continue beyond obstacle top crossing (`JobID=29252`).
  - Updated `jobs/cyl-render-h0p045-follow.sbatch` to accept optional args:
    - arg1: checkpoint path
    - arg2: `episode_length` (default `2000`)
    - arg3: `terminate_on_success` (`true|false`, default `false`)
  - Render run used: `episode_length=2000`, `terminate_on_success=false`.
  - Output directory: `logs/TransformableWheelMobileRobot-20260213-111501-cyl-render-h0p045-follow10-ep2000-tosfalse`.
  - Output videos:
    - `logs/TransformableWheelMobileRobot-20260213-111501-cyl-render-h0p045-follow10-ep2000-tosfalse/rollout0-follow10.mp4`
    - `logs/TransformableWheelMobileRobot-20260213-111501-cyl-render-h0p045-follow10-ep2000-tosfalse/rollout0.mp4`
- 2026-02-13 11:21 PST: Added long continuation training job for consistency at `h=0.045`.
  - New script: `jobs/cyl-h045-continue-long.sbatch`.
  - Behavior:
    - restores from best available `h0p045` checkpoint (or explicit arg),
    - runs `5,000,000` additional PPO timesteps on `h=0.045`,
    - renders rollout with approved `cam_robot_follow`,
  - runs post-train consistency eval over `256` episodes and writes JSON to `logs/eval-cyl-h045-long-h0p045-256ep-<timestamp>.json`.
  - Purpose: quantify whether extended training improves reliability on the current-robot target obstacle regime.
- 2026-02-13 11:24 PST: Long continuation run failed immediately (`JobID=29253`, `ExitCode=1`); patched launcher path validation.
  - Failure mode in `logs/cyl-h045-long-29253.err`: Brax restore path was invalid (`/bigdata/lab/aclarklab/twmr-rl/ `), so checkpoint load aborted before training.
  - Root cause: whitespace/empty checkpoint argument was not normalized before `--load_checkpoint_path`.
  - Fix in `jobs/cyl-h045-continue-long.sbatch`:
    - trims arg1 whitespace (`xargs`),
    - validates checkpoint directory existence,
    - validates presence of numeric checkpoint-step subdirectories before launching PPO.
  - Verified fallback now resolves to:
    - `logs/TransformableWheelMobileRobot-20260212-192709-cyl-stagec-height-bridge-h0p045-l0p2/checkpoints`.
- 2026-02-13 13:17 PST: Long continuation run completed successfully (`JobID=29254`, `ExitCode=0`).
  - Job: `cyl-h045-long` from `jobs/cyl-h045-continue-long.sbatch`.
  - Training experiment directory:
    - `logs/TransformableWheelMobileRobot-20260213-114757-cyl-h045-long`
  - Final checkpoint step:
    - `logs/TransformableWheelMobileRobot-20260213-114757-cyl-h045-long/checkpoints/000014745600`
  - Rollout videos:
    - `logs/TransformableWheelMobileRobot-20260213-114757-cyl-h045-long/rollout0-follow10.mp4`
    - `logs/TransformableWheelMobileRobot-20260213-114757-cyl-h045-long/rollout0.mp4`
  - Post-train consistency eval (256 episodes):
    - `logs/eval-cyl-h045-long-h0p045-256ep-20260213-131231.json`
    - `success_rate=0.5`, `mean_return=123.34`, `mean_x_distance=0.6701`.
  - In-training eval snapshots improved late in run (`success` reached `0.984` at final eval point), but external deterministic checkpoint eval is still only 50% success.
- 2026-02-13 19:13 PST: Ran apples-to-apples baseline eval on pre-long checkpoint for direct comparison.
  - Command: `scripts/eval_twmr_checkpoint.py` with same overrides (`h=0.045`, `x=0.6`, `half_length=0.2`, `success_x_margin=0.1`), `256` episodes, `seed=1`.
  - Output:
    - `logs/eval-cyl-h045-prelong-h0p045-256ep-20260214-compare.json`
    - `success_rate=0.0`, `mean_return=50.58`, `mean_x_distance=0.3946`.
  - Conclusion: long continuation materially improved performance vs its starting checkpoint, but reliability is still below target for "best policy on current robot".
- 2026-02-13 19:20 PST: Added overnight continuation launcher for sleep-time training.
  - New script: `jobs/cyl-h045-continue-overnight.sbatch`.
  - Purpose: run a ~10 hour continuation without manual babysitting.
  - Default behavior:
    - job name `cyl-h045-overnight`, walltime `12:00:00`,
    - continues from latest available `cyl-h045-overnight` or `cyl-h045-long` checkpoint dir (fallback to prior `h0p045` bridge checkpoint),
    - trains `30,000,000` timesteps at `h=0.045`, `half_length=0.2`, `x=0.6`,
    - renders follow camera rollout (`rollout0-follow10.mp4`),
    - runs 256-episode consistency eval and writes `logs/eval-cyl-h045-overnight-h0p045-256ep-<timestamp>.json`.
  - Args:
    - arg1 optional checkpoint directory override,
    - arg2 optional positive-integer timestep override.
  - Validation:
    - `bash -n jobs/cyl-h045-continue-overnight.sbatch` passed.
- 2026-02-13 19:21 PST: Launched overnight continuation run (`JobID=29264`) from latest long-run checkpoint.
  - Submit command:
    - `sbatch jobs/cyl-h045-continue-overnight.sbatch logs/TransformableWheelMobileRobot-20260213-114757-cyl-h045-long/checkpoints`
  - Scheduler state at launch check: `R` on `gpu002`.
  - Early log confirms expected config:
    - checkpoint source `.../cyl-h045-long/checkpoints`
    - `Timesteps: 30000000`
  - Live logs:
    - `logs/cyl-h045-overnight-29264.out`
    - `logs/cyl-h045-overnight-29264.err`
- 2026-02-13 22:35 PST: Overnight continuation run completed (`JobID=29264`, `ExitCode=0`).
  - Training experiment directory:
    - `logs/TransformableWheelMobileRobot-20260213-192145-cyl-h045-overnight`
  - Final checkpoint step:
    - `logs/TransformableWheelMobileRobot-20260213-192145-cyl-h045-overnight/checkpoints/000036044800`
  - Rollout videos:
    - `logs/TransformableWheelMobileRobot-20260213-192145-cyl-h045-overnight/rollout0-follow10.mp4`
    - `logs/TransformableWheelMobileRobot-20260213-192145-cyl-h045-overnight/rollout0.mp4`
  - Post-train consistency eval (256 episodes):
    - `logs/eval-cyl-h045-overnight-h0p045-256ep-20260213-223508.json`
    - `success_rate=0.375`, `mean_return=107.45`, `mean_x_distance=0.8984`.
  - Comparison against prior best continuation checkpoint (`29254`):
    - prior: `success_rate=0.5`, `mean_return=123.34`, `mean_x_distance=0.6701`
    - overnight: `success_rate=0.375`, `mean_return=107.45`, `mean_x_distance=0.8984`
  - Interpretation: longer continuation did not improve final reliability on the fixed 256-episode eval and likely overshot/shifted policy behavior; keep `29254` checkpoint as current best by this metric.
- 2026-02-14 07:20 PST: Implemented two-obstacle randomized traversal path to reduce timing memorization and enable robust multi-obstacle behavior.
  - Added new XML variant and asset:
    - `xml_variant="box_dual"` in `packages/twmr/src/twmr/twmr.py`
    - `packages/twmr/assets/trans_wheel_robo2_2BOX_DUAL_CLY.xml`
  - Dual obstacle model details:
    - first and second obstacles are independent slide-joint bodies (`traverse_box_slide_x`, `traverse_box2_slide_x`) so x-position can be randomized per episode.
    - second obstacle geometry/site names: `traverse_box2`, `traverse_box2_site`.
  - Environment config extensions in `packages/twmr/src/twmr/twmr.py`:
    - `enable_second_obstacle`, `obstacle2_*` keys,
    - `randomize_obstacles`,
    - `randomize_obstacle_x_min/max`,
    - `randomize_obstacle_gap_min/max`,
    - `obstacle_local_windowed_reward`.
  - Observation/input behavior:
    - obstacle slide-joint qpos/qvel are explicitly excluded from `obs` so policy input remains robot-proprioceptive (no direct obstacle position channels injected).
    - validated obs dimensional parity vs single-obstacle `box` variant (`obs_total=45` in both cases), preserving checkpoint compatibility.
  - Added Slurm launchers:
    - `jobs/cyl-twoobs-rand-continue.sbatch`
      - continues from prior checkpoint (default prefers latest `cyl-h045-long`),
      - trains on dual randomized obstacles (`xml_variant=box_dual`, `enable_second_obstacle=true`, `randomize_obstacles=true`),
      - writes 4 rollout videos and 256-episode eval JSON.
    - `jobs/cyl-twoobs-rand-render-multi.sbatch`
      - render-only from checkpoint (`num_timesteps=0`) with configurable `num_videos`,
      - outputs multiple rollout files (`rollout0.mp4`, `rollout1.mp4`, ... ) over randomized obstacle layouts.
  - Validation:
    - `python -m py_compile packages/twmr/src/twmr/twmr.py` passed.
    - `bash -n jobs/cyl-twoobs-rand-continue.sbatch` passed.
    - `bash -n jobs/cyl-twoobs-rand-render-multi.sbatch` passed.
    - MuJoCo XML load check passed for `trans_wheel_robo2_2BOX_DUAL_CLY.xml` (`nq=25`, `nv=24`, `nu=8`).
- 2026-02-14 09:24 PST: First randomized dual-obstacle training run completed (`JobID=29268`, `ExitCode=0`).
  - Job: `cyl-2obs-rand` from `jobs/cyl-twoobs-rand-continue.sbatch`.
  - Start checkpoint source:
    - `logs/TransformableWheelMobileRobot-20260213-114757-cyl-h045-long/checkpoints/000014745600`
  - Training experiment directory:
    - `logs/TransformableWheelMobileRobot-20260214-073431-cyl-twoobs-rand-continue`
  - Final checkpoint step:
    - `logs/TransformableWheelMobileRobot-20260214-073431-cyl-twoobs-rand-continue/checkpoints/000018022400`
  - Multi-rollout videos generated (randomized obstacle layouts):
    - `logs/TransformableWheelMobileRobot-20260214-073431-cyl-twoobs-rand-continue/rollout0.mp4`
    - `logs/TransformableWheelMobileRobot-20260214-073431-cyl-twoobs-rand-continue/rollout1.mp4`
    - `logs/TransformableWheelMobileRobot-20260214-073431-cyl-twoobs-rand-continue/rollout2.mp4`
    - `logs/TransformableWheelMobileRobot-20260214-073431-cyl-twoobs-rand-continue/rollout3.mp4`
  - Post-train eval JSON:
    - `logs/eval-cyl-twoobs-rand-continue-256ep-20260214-092417.json`
    - `success_rate=0.6836` over `256` episodes (`episode_length=1600`)
    - `mean_return=211.39`, `mean_x_distance=2.675`.
  - Interpretation: first dual-obstacle randomized continuation already reaches >68% success on held-out randomized episodes, indicating the randomized training setup is functional and producing robust behavior.
- 2026-02-14 09:40 PST: Improved follow-camera behavior for long randomized episodes.
  - Added dedicated long-horizon follow camera in dual XML:
    - `packages/twmr/assets/trans_wheel_robo2_2BOX_DUAL_CLY.xml`
    - camera name: `cam_robot_follow_long`
    - mode: `track` on `root` (more stable body-frame follow during long episodes than COM-follow under strong articulation changes).
  - Updated dual-obstacle Slurm jobs to use follow camera by default:
    - `jobs/cyl-twoobs-rand-continue.sbatch`
      - now defaults `rollout_camera` to `cam_robot_follow_long` (optional arg3 override).
    - `jobs/cyl-twoobs-rand-render-multi.sbatch`
      - now defaults `rollout_camera` to `cam_robot_follow_long` (optional arg5 override).
  - Validation:
    - `bash -n jobs/cyl-twoobs-rand-continue.sbatch` passed.
    - `bash -n jobs/cyl-twoobs-rand-render-multi.sbatch` passed.
    - MuJoCo camera lookup validated (`cam_robot_follow_long` present).
- 2026-02-15 09:30 PST: Multi-run randomized render job completed with long follow camera (`JobID=29273`, `ExitCode=0`).
  - Job: `cyl-2obs-render` from `jobs/cyl-twoobs-rand-render-multi.sbatch`.
  - Source checkpoint:
    - `logs/TransformableWheelMobileRobot-20260214-073431-cyl-twoobs-rand-continue/checkpoints/000018022400`
  - Render experiment directory:
    - `logs/TransformableWheelMobileRobot-20260215-092645-cyl-twoobs-rand-render-v8-ep2200-tosfalse`
  - Render settings:
    - `num_videos=8`
    - `episode_length=2200`
    - `terminate_on_success=false`
    - `rollout_camera=cam_robot_follow_long`
  - Per-run videos:
    - `rollout0.mp4` ... `rollout7.mp4` in the experiment directory above.
  - Postprocessing:
    - stitched all 8 clips into a single concatenated video:
      - `logs/TransformableWheelMobileRobot-20260215-092645-cyl-twoobs-rand-render-v8-ep2200-tosfalse/rollout-all.mp4`
- 2026-02-15 09:43 PST: Fixed non-following camera behavior in long randomized renders.
  - Issue observed: `cam_robot_follow_long` did not track the robot (camera stayed fixed while robot moved out of frame).
  - Root cause: long camera used `mode="track"` in dual XML, while the previously validated moving camera used `mode="trackcom"`.
  - Fixes applied:
    - `packages/twmr/assets/trans_wheel_robo2_2BOX_DUAL_CLY.xml`
      - changed `cam_robot_follow_long` to `mode="trackcom"`.
    - `jobs/cyl-twoobs-rand-continue.sbatch`
      - default rollout camera reverted to `cam_robot_follow`.
    - `jobs/cyl-twoobs-rand-render-multi.sbatch`
      - default rollout camera reverted to `cam_robot_follow`.
  - Validation:
    - both job scripts pass `bash -n`.
    - MuJoCo reports both follow cameras in mode `2` (`trackcom`) with root target.
- 2026-02-15 09:50 PST: Enforced static-within-episode obstacle layouts and replaced dual follow cameras with robot-mounted follow cameras.
  - Problem statement:
    - obstacle slide joints could drift after reset due contact dynamics,
    - dual-task follow camera still appeared fixed in rendered videos.
  - Environment/task fix in `packages/twmr/src/twmr/twmr.py`:
    - `reset()` now stores sampled slide-joint positions in `info` (`obstacle_slide_qpos`, `obstacle2_slide_qpos`).
    - `step()` now clamps both obstacle slide-joint qpos to those stored reset values and zeros corresponding qvel dofs, then calls `mjx.forward`.
    - Result: obstacle positions remain constant during each episode while still being randomized on each reset.
  - Camera fix in `packages/twmr/assets/trans_wheel_robo2_2BOX_DUAL_CLY.xml`:
    - removed world-mounted `cam_robot_follow` / `cam_robot_follow_long` `trackcom` cameras,
    - re-added both cameras with the same names as cameras mounted on body `root` (fixed mode).
    - Result: camera transform is now physically attached to the robot, so follow behavior does not depend on `track`/`trackcom` runtime behavior.
  - Validation:
    - `python -m py_compile packages/twmr/src/twmr/twmr.py` passed.
    - `bash -n jobs/cyl-twoobs-rand-continue.sbatch jobs/cyl-twoobs-rand-render-multi.sbatch` passed.
    - XML parse check passed for `trans_wheel_robo2_2BOX_DUAL_CLY.xml`.
    - MuJoCo camera lookup confirms follow cameras are attached to body `root` (mode `0`, fixed).
- 2026-02-15 09:52 PST: Fixed immediate render-job failure when passing explicit checkpoint step paths.
  - Failure observed:
    - `JobID=29275` failed at startup with
      - `Checkpoint directory has no saved training steps: .../checkpoints/000018022400`
  - Root cause:
    - the Slurm scripts validated only checkpoint-root directories (`.../checkpoints`) and rejected explicit step directories (`.../checkpoints/<step>`),
    - `train_jax_ppo.py` checkpoint resolver also assumed directory inputs were always checkpoint-root directories.
  - Fixes applied:
    - `jobs/cyl-twoobs-rand-render-multi.sbatch`
      - now accepts either checkpoint root directory or explicit step directory.
    - `jobs/cyl-twoobs-rand-continue.sbatch`
      - same checkpoint-path acceptance fix for continuation runs.
    - `train_jax_ppo.py`
      - if `--load_checkpoint_path` is a numeric directory name, restores directly from that step;
      - otherwise selects latest numeric subdirectory under checkpoint root;
      - emits a clear error if no numeric steps exist.
  - Validation:
    - `python -m py_compile train_jax_ppo.py` passed.
    - `bash -n jobs/cyl-twoobs-rand-render-multi.sbatch jobs/cyl-twoobs-rand-continue.sbatch` passed.
- 2026-02-15 10:44 PST: Changed follow camera behavior to remove pitch/roll tilt coupling with robot body attitude.
  - Problem:
    - body-mounted follow camera moved with robot (good) but inherited robot angular tilt (undesired).
  - Camera update in `packages/twmr/assets/trans_wheel_robo2_2BOX_DUAL_CLY.xml`:
    - moved `cam_robot_follow` and `cam_robot_follow_long` from `root` body back to worldbody,
    - set both to `mode="trackcom"` targeting `root`.
  - Expected behavior:
    - camera follows robot translation,
    - camera orientation remains stable in world frame (does not pitch/roll with robot articulation).
  - Validation:
    - XML parses successfully.
    - MuJoCo camera metadata confirms:
      - `mode=2` (`trackcom`),
      - `bodyid=0` (world-mounted),
      - `targetbodyid=root`.
    - `bash -n jobs/cyl-twoobs-rand-render-multi.sbatch` passed.
- 2026-02-15 11:06 PST: Added renderer-level stable follow mode to guarantee camera tracks robot position without inheriting robot tilt.
  - Issue:
    - XML camera mode toggles still produced non-following or inconsistent behavior in rendered videos.
  - Implementation in `train_jax_ppo.py`:
    - Added `_render_rollout_with_stable_follow(...)` helper.
    - For `--rollout_camera` values `cam_robot_follow` and `cam_robot_follow_long`:
      - compute initial framing from the named camera at frame 0,
      - render with `MjvCamera` free mode using fixed `distance`/`azimuth`/`elevation`,
      - update `lookat` to robot root position every frame.
    - Result: camera follows robot translation each frame while pitch/roll stays world-stable.
  - Validation:
    - `python -m py_compile train_jax_ppo.py` passed.
    - `bash -n jobs/cyl-twoobs-rand-render-multi.sbatch` passed.
    - Local runtime render smoke test is blocked in this environment due missing EGL display support; full validation requires Slurm GPU render job.
- 2026-02-15 13:02 PST: Reverted camera customization path back toward pre-camera-change behavior for active dual-obstacle runs.
  - `train_jax_ppo.py`:
    - removed renderer-level stable-follow helper (`_render_rollout_with_stable_follow`),
    - restored rollout rendering path to direct `eval_env.render(...)` with optional `--rollout_camera` only.
  - `jobs/cyl-twoobs-rand-render-multi.sbatch`:
    - default `rollout_camera` now empty (no explicit camera),
    - passes `--rollout_camera` only when a non-empty 5th argument is provided.
  - `jobs/cyl-twoobs-rand-continue.sbatch`:
    - default `rollout_camera` now empty (no explicit camera),
    - passes `--rollout_camera` only when a non-empty 3rd argument is provided.
  - Result:
    - active dual-obstacle jobs now use the environment’s default/free camera unless explicitly overridden at submit time.
  - Validation:
    - `python -m py_compile train_jax_ppo.py` passed.
    - `bash -n jobs/cyl-twoobs-rand-render-multi.sbatch jobs/cyl-twoobs-rand-continue.sbatch` passed.

## Engineering Rules for This Project
1. Keep environment dynamics/task logic in `packages/twmr/src/twmr/twmr.py`.
2. Keep training orchestration in `train_jax_ppo.py`.
3. Do not rely on sandbox scripts as source of truth; port only validated ideas.
4. Every major reward/termination change must include:
  - updated metrics keys,
  - at least one short training sanity run,
  - one rendered rollout for qualitative check.
5. Prefer incremental changes: first make reward nonzero and 8-actuator model active, then add obstacle success shaping.
6. All future Slurm job scripts under `jobs/` must include email notifications:
  - `#SBATCH --mail-type=END,FAIL`
  - `#SBATCH --mail-user=njcf2022@mymail.pomona.edu`
  - If a job is already running without these settings, update in-place with:
    - `scontrol update JobId=<jobid> MailType=END,FAIL MailUser=njcf2022@mymail.pomona.edu`
