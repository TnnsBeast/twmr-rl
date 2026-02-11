# DreamerV3 Integration Outline (TWM-RL)

This is a concise plan for integrating DreamerV3 into this repo for the
transformable wheel robot obstacle task.

## 1) Choose observation type
- Prefer state-based observations for speed and stability.
- Current compact state: `x, z, vx, vz, roll, pitch, 4 leg angles`.

## 2) Create a Gym-style wrapper
DreamerV3 expects:
- `reset() -> obs`
- `step(action) -> obs, reward, done, info`
- `observation_space`, `action_space`

Suggested structure:
- Move `TwmrEnv` into `sandbox/neil/twmr_env.py`
- Wrap with a Gym API that clips actions to `[-1, 1]`.

## 3) Keep dense reward shaping
Keep the existing reward components:
- Forward velocity reward
- Alive bonus
- Success bonus for clearing the obstacle
- Extension penalty to encourage retraction

## 4) Pick a DreamerV3 implementation
Use an existing implementation (JAX or TF) and point it at the Gym env.
Key config values:
- `action_dim = 8`
- `obs_dim = 10` (for compact state)

## 5) Training flow
Typical DreamerV3 workflow:
- Seed replay buffer with random actions
- Train world model + actor/critic
- Periodically evaluate and save checkpoints

## 6) Evaluation & rendering
- Headless: `mujoco.Renderer` -> MP4
- GUI: `mujoco.viewer` for interactive playback

## Next implementation steps
1. Add `sandbox/neil/twmr_env.py` (env class + Gym wrapper).
2. Add `sandbox/neil/train_dreamer.py` to wire DreamerV3 to the env.
3. Add a small eval script to render MP4s after training.
