# PPO + LSTM Training Walkthrough

This walkthrough is for `sandbox/neil/train_ppo_lstm.py`, which trains a recurrent
policy (LSTM) with PPO on the transformable wheel robot using the environment
defined in `sandbox/neil/train_cem_obstacle.py`.

**Quick Start**
```bash
./.venv/bin/python sandbox/neil/train_ppo_lstm.py
```

Obstacle-gated, compact observations (your current setup):
```bash
./.venv/bin/python sandbox/neil/train_ppo_lstm.py   --gate-mode obstacle   --obs-mode compact_obstacle   --extension-penalty 0.05   --control-cost 1e-4   --total-timesteps 600000
```

**What The Script Does**
1. Builds `TwmrEnv` from `sandbox/neil/train_cem_obstacle.py`.
2. Creates an `LSTMActorCritic` with a tanh MLP + LSTM + tanh MLP.
3. Collects a fixed rollout buffer (`--rollout-steps`, default 512).
4. Computes GAE and PPO loss.
5. Runs several PPO epochs (`--epochs`, default 5).
6. Logs progress and saves a policy at the end.

**Reward Signal (from `TwmrEnv.step`)**
Per step:
```
reward = forward + alive_bonus
       - control_cost * sum(ctrl^2)
       - extension_penalty * mean((leg_angles - target)^2)
       - post_success_penalty * mean((leg_angles - retract)^2) [optional]
       + success_bonus (once, when x >= success_x)
```
Notes:
- `forward = (x_after - x_before) / ctrl_dt`.
- `success_x` defaults to obstacle far edge + 0.05 when the obstacle geom exists.
- Your `--extension-penalty 0.05` can dominate if legs miss targets.

**Interpreting The Logs**
Example log line:
```
[PPO-LSTM] update 120/1171 |  10.25% | loss=0.165 | ep_return=23.34 | passed_obstacle=False
```
Key points:
- `ep_return` is the sum of rewards over the rollout buffer (not a single episode).
- `passed_obstacle` is `True` if any step in the rollout crossed `success_x`.
- `loss` is the mean PPO loss across the update epochs.

**Outputs**
Policies are saved with a timestamp suffix:
- Example: `sandbox/neil/ppo_lstm_policy_20260209_153045.pkl`

**Playing A Saved Policy**
Because saves are timestamped, use `--load-path`:
```bash
./.venv/bin/python sandbox/neil/train_ppo_lstm.py   --play   --load-path sandbox/neil/ppo_lstm_policy_YYYYMMDD_HHMMSS.pkl
```

**Common Knobs**
- `--obs-mode`: `proprio`, `compact`, `compact_obstacle`, `full`
- `--gate-mode`: `proprio`, `obstacle`, `none`
- `--total-timesteps`: total environment steps used for training
- `--rollout-steps`: steps per PPO update (default 512)
- `--epochs`: PPO epochs per update
- `--learning-rate`, `--clip-coef`, `--ent-coef`, `--vf-coef`
- `--extension-penalty`, `--control-cost`, `--alive-bonus`, `--success-bonus`

**Where To Modify Things**
- Model + environment mechanics: `sandbox/neil/train_cem_obstacle.py`
- Policy network: `sandbox/neil/train_ppo_lstm.py` (`LSTMActorCritic`)
- PPO loss / update: `sandbox/neil/train_ppo_lstm.py` (`ppo_loss`, `update_step`)
- Rollout loop + logging: `sandbox/neil/train_ppo_lstm.py` (main training loop)

If you want a more detailed logging breakdown (e.g., average `x`, success rate per rollout, or per-episode returns), say the word and I can add it.
