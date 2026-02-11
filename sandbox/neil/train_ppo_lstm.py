#!/usr/bin/env python3
"""PPO + LSTM training for the transformable wheel robot (proprio-only).

This script uses the TwmrEnv defined in train_cem_obstacle.py and trains a
recurrent policy (LSTM) with PPO. The observation defaults to proprioception
only (IMU + motor sensors).
"""

from __future__ import annotations

import argparse
import math
import pickle
import time
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn

try:
    from train_cem_obstacle import (
        DEFAULT_MODEL,
        TwmrEnv,
        _resolve_model_path,
        _resolve_save_path,
    )
except Exception as exc:  # pragma: no cover - import error path
    raise SystemExit(
        "Failed to import TwmrEnv from train_cem_obstacle.py. "
        "Run from the repo root or sandbox/neil directory."
    ) from exc


def gaussian_log_prob(actions, mean, log_std):
    std = jnp.exp(log_std)
    var = std**2
    return jnp.sum(
        -0.5 * (((actions - mean) ** 2) / var + 2.0 * log_std + jnp.log(2.0 * jnp.pi)),
        axis=-1,
    )


def atanh(x: jnp.ndarray) -> jnp.ndarray:
    eps = 1e-6
    x = jnp.clip(x, -1.0 + eps, 1.0 - eps)
    return 0.5 * (jnp.log1p(x) - jnp.log1p(-x))


class LSTMActorCritic(nn.Module):
    action_dim: int
    hidden_size: int = 128
    mlp_size: int = 64

    @nn.compact
    def __call__(self, carry, x):
        x = x.astype(jnp.float32)
        x = nn.tanh(nn.Dense(self.mlp_size)(x))
        carry, y = nn.LSTMCell(self.hidden_size)(carry, x)
        y = nn.tanh(nn.Dense(self.mlp_size)(y))
        mean = nn.Dense(self.action_dim)(y)
        value = nn.Dense(1)(y)
        return carry, (mean, jnp.squeeze(value, axis=-1))


def _zero_carry(hidden_size: int):
    return (
        jnp.zeros((1, hidden_size), dtype=jnp.float32),
        jnp.zeros((1, hidden_size), dtype=jnp.float32),
    )


def _compute_gae(rewards, values, dones, last_value, gamma, gae_lambda):
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(len(rewards))):
        next_value = last_value if t == len(rewards) - 1 else values[t + 1]
        mask = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_value * mask - values[t]
        last_gae = delta + gamma * gae_lambda * mask * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def rollout_policy(
    env: TwmrEnv,
    model: LSTMActorCritic,
    params: dict,
    *,
    hidden_size: int,
    seed: int = 0,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    obs = env.reset(rng)
    carry = _zero_carry(hidden_size)
    total = 0.0
    info = {"x": 0.0}
    for _ in range(env.max_steps):
        obs_jnp = jnp.asarray(obs[None, :])
        carry, (mean, _) = model.apply(params["model"], carry, obs_jnp)
        action = jnp.tanh(mean)[0]
        obs, reward, done, info = env.step(np.asarray(action, dtype=np.float32))
        total += float(reward)
        if done:
            break
    return total, float(info.get("x", 0.0))


def render_policy(
    env: TwmrEnv,
    model: LSTMActorCritic,
    params: dict,
    *,
    hidden_size: int,
    playback_speed: float = 1.0,
    follow: bool = True,
    follow_body: str = "root",
    seed: int = 0,
    debug_gate: bool = False,
    gate_threshold: float = 0.2,
    gate_print_every: int = 50,
    debug_legs: bool = False,
    leg_print_every: int = 50,
    leg_action_threshold: float = 0.2,
    debug_log_path: Path | None = None,
    debug_log_append: bool = False,
) -> None:
    import mujoco
    import mujoco.viewer  # local import to avoid glfw dependency for training

    obs = env.reset(np.random.default_rng(seed))
    carry = _zero_carry(hidden_size)

    track_body_id = -1
    if follow:
        track_body_id = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_BODY, follow_body
        )
        if track_body_id < 0:
            print(
                f"[render] follow body '{follow_body}' not found; "
                "falling back to default camera."
            )
            follow = False

    step = 0
    last_gate = 0.0
    log_fh = None
    if debug_log_path is not None:
        debug_log_path = debug_log_path.expanduser().resolve()
        debug_log_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if debug_log_append else "w"
        log_fh = open(debug_log_path, mode, encoding="utf-8")

    def _log(line: str) -> None:
        print(line)
        if log_fh is not None:
            log_fh.write(line + "\n")
            log_fh.flush()
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        if follow:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = track_body_id
        while viewer.is_running():
            obs_jnp = jnp.asarray(obs[None, :])
            carry, (mean, _) = model.apply(params["model"], carry, obs_jnp)
            action = jnp.tanh(mean)[0]
            obs, _, done, info = env.step(np.asarray(action, dtype=np.float32))
            if debug_gate:
                gate = float(info.get("extension_gate", 0.0))
                crossed = gate >= gate_threshold and last_gate < gate_threshold
                if crossed or (gate_print_every > 0 and step % gate_print_every == 0):
                    x = float(info.get("x", 0.0))
                    _log(
                        f"[gate] step={step:04d} gate={gate:.3f} x={x:.3f} crossed={crossed}"
                    )
                last_gate = gate
            if debug_legs:
                wheel_act_count = int(getattr(env, "wheel_act_count", 0))
                ext_action = np.asarray(action[wheel_act_count:], dtype=np.float32)
                ext_mag = float(np.max(np.abs(ext_action))) if ext_action.size else 0.0
                should_print = (
                    (leg_print_every > 0 and step % leg_print_every == 0)
                    or ext_mag >= leg_action_threshold
                )
                if should_print:
                    x = float(info.get("x", 0.0))
                    leg_qpos = [
                        float(env.data.qpos[idx]) for idx in env.leg_qpos_idx
                    ]
                    act_str = ", ".join(f"{a:+.3f}" for a in ext_action.tolist())
                    qpos_str = ", ".join(f"{q:+.3f}" for q in leg_qpos)
                    _log(
                        f"[legs] step={step:04d} x={x:.3f} ext_act=[{act_str}] "
                        f"leg_qpos=[{qpos_str}] ext_mag={ext_mag:.3f}"
                    )
            viewer.sync()
            speed = max(float(playback_speed), 1e-6)
            time.sleep(env.ctrl_dt / speed)
            step += 1
            if done:
                obs = env.reset(np.random.default_rng(seed))
                carry = _zero_carry(hidden_size)
                step = 0
                last_gate = 0.0
    if log_fh is not None:
        log_fh.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PPO + LSTM training (proprio-only).")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--total-timesteps", type=int, default=300_000)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--mlp-size", type=int, default=64)
    parser.add_argument("--log-std-init", type=float, default=-0.5)
    parser.add_argument("--save-path", type=Path, default=Path("sandbox/neil/ppo_lstm_policy.pkl"))
    parser.add_argument("--load-path", type=Path, default=None)

    # Environment controls (proprio by default)
    parser.add_argument(
        "--obs-mode",
        type=str,
        default="proprio",
        choices=("proprio",),
        help="Observation mode. PPO-LSTM is proprio-only (no obstacle info).",
    )
    parser.add_argument(
        "--gate-mode",
        type=str,
        default="proprio",
        choices=("proprio", "none"),
    )
    parser.add_argument("--ctrl-dt", type=float, default=0.02)
    parser.add_argument("--frame-skip", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--settle-steps", type=int, default=25)
    parser.add_argument("--reset-noise", type=float, default=0.01)
    parser.add_argument("--alive-bonus", type=float, default=0.05)
    parser.add_argument("--control-cost", type=float, default=1e-3)
    parser.add_argument("--extension-penalty", type=float, default=1e-3)
    parser.add_argument("--post-success-penalty", type=float, default=0.0)
    parser.add_argument("--success-x", type=float, default=None)
    parser.add_argument("--success-bonus", type=float, default=1.0)
    parser.add_argument("--extend-angle", type=float, default=None)
    parser.add_argument(
        "--extend-mode",
        type=str,
        default="auto",
        choices=("auto", "midpoint", "opposite"),
    )
    parser.add_argument(
        "--retract-legs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Start with extension joints set to a retracted angle.",
    )
    parser.add_argument(
        "--retract-angle",
        type=float,
        default=None,
        help=(
            "Explicit leg retraction angle (rad). Defaults to the auto-picked "
            "retract target when not provided."
        ),
    )
    parser.add_argument(
        "--auto-retract",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--gate-acc-scale", type=float, default=3.0)
    parser.add_argument("--gate-gyro-scale", type=float, default=3.0)
    parser.add_argument("--gate-stall-scale", type=float, default=2.0)
    parser.add_argument("--gate-bias", type=float, default=1.0)
    parser.add_argument("--gate-smooth", type=float, default=0.5)
    parser.add_argument(
        "--hold-retract",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--hold-kp", type=float, default=4.0)
    parser.add_argument("--hold-kd", type=float, default=0.2)
    parser.add_argument(
        "--play",
        action="store_true",
        help="Load a saved policy and run it in the MuJoCo viewer.",
    )
    parser.add_argument(
        "--playback-speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier (0.5 = half speed, 2.0 = double speed).",
    )
    parser.add_argument(
        "--follow",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Track the robot with the viewer camera during playback.",
    )
    parser.add_argument(
        "--follow-body",
        type=str,
        default="root",
        help="Body name to track when --follow is enabled.",
    )
    parser.add_argument(
        "--debug-gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print extension gate values during --play.",
    )
    parser.add_argument(
        "--gate-threshold",
        type=float,
        default=0.2,
        help="Threshold for reporting gate openings when --debug-gate is enabled.",
    )
    parser.add_argument(
        "--gate-print-every",
        type=int,
        default=50,
        help="Print gate status every N steps when --debug-gate is enabled.",
    )
    parser.add_argument(
        "--debug-legs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print extension actions and leg joint angles during --play.",
    )
    parser.add_argument(
        "--leg-print-every",
        type=int,
        default=50,
        help="Print leg debug every N steps when --debug-legs is enabled.",
    )
    parser.add_argument(
        "--leg-action-threshold",
        type=float,
        default=0.2,
        help=(
            "Print leg debug when max extension action magnitude exceeds this "
            "threshold."
        ),
    )
    parser.add_argument(
        "--debug-log",
        type=Path,
        default=None,
        help="Write debug output to this path during --play.",
    )
    parser.add_argument(
        "--debug-log-append",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Append to --debug-log instead of overwriting.",
    )
    return parser.parse_args()


def _timestamped_path(path: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if path.suffix:
        return path.with_name(f"{path.stem}_{ts}{path.suffix}")
    return path.with_name(f"{path.name}_{ts}")


def main() -> None:
    args = _parse_args()
    np_rng = np.random.default_rng(args.seed)
    key = jax.random.PRNGKey(args.seed)

    model_path = _resolve_model_path(args.model)
    save_path = _resolve_save_path(args.save_path)
    load_path = _resolve_save_path(args.load_path) if args.load_path else None

    env = TwmrEnv(
        xml_path=model_path,
        ctrl_dt=args.ctrl_dt,
        frame_skip=args.frame_skip,
        max_steps=args.max_steps,
        settle_steps=args.settle_steps,
        reset_noise=args.reset_noise,
        alive_bonus=args.alive_bonus,
        control_cost=args.control_cost,
        success_x=args.success_x,
        success_bonus=args.success_bonus,
        extension_penalty=args.extension_penalty,
        post_success_penalty=args.post_success_penalty,
        extend_angle=args.extend_angle,
        extend_mode=args.extend_mode,
        auto_retract=args.auto_retract,
        retract_legs=args.retract_legs,
        retract_angle=args.retract_angle,
        gate_mode=args.gate_mode,
        gate_acc_scale=args.gate_acc_scale,
        gate_gyro_scale=args.gate_gyro_scale,
        gate_stall_scale=args.gate_stall_scale,
        gate_bias=args.gate_bias,
        gate_smooth=args.gate_smooth,
        hold_retract=args.hold_retract,
        hold_kp=args.hold_kp,
        hold_kd=args.hold_kd,
        obs_mode=args.obs_mode,
    )

    act_dim = env.nu

    model = LSTMActorCritic(
        action_dim=act_dim, hidden_size=args.hidden_size, mlp_size=args.mlp_size
    )

    if args.play:
        play_path = load_path or save_path
        if not play_path.exists():
            raise FileNotFoundError(f"Policy file not found: {play_path}")
        with open(play_path, "rb") as f:
            params = pickle.load(f)
        print(f"[PPO-LSTM] Loaded policy from {play_path}")
        render_policy(
            env,
            model,
            params,
            hidden_size=args.hidden_size,
            playback_speed=args.playback_speed,
            follow=args.follow,
            follow_body=args.follow_body,
            seed=args.seed,
            debug_gate=args.debug_gate,
            gate_threshold=args.gate_threshold,
            gate_print_every=args.gate_print_every,
            debug_legs=args.debug_legs,
            leg_print_every=args.leg_print_every,
            leg_action_threshold=args.leg_action_threshold,
            debug_log_path=args.debug_log,
            debug_log_append=args.debug_log_append,
        )
        return

    obs = env.reset(np_rng)
    obs_dim = obs.shape[0]
    dummy_obs = jnp.zeros((1, obs_dim), dtype=jnp.float32)
    key, subkey = jax.random.split(key)
    params = {
        "model": model.init(subkey, _zero_carry(args.hidden_size), dummy_obs),
        "log_std": jnp.full((act_dim,), args.log_std_init, dtype=jnp.float32),
    }

    if load_path is not None and load_path.exists():
        with open(load_path, "rb") as f:
            params = pickle.load(f)
        print(f"[PPO-LSTM] Loaded policy from {load_path}")

    optimizer = optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        optax.adam(learning_rate=args.learning_rate),
    )
    opt_state = optimizer.init(params)

    @jax.jit
    def ppo_loss(params, batch):
        obs = batch["obs"]
        actions = batch["actions"]
        old_log_probs = batch["log_probs"]
        advantages = batch["advantages"]
        returns = batch["returns"]
        reset_masks = batch["reset_masks"]

        def step(carry, inputs):
            obs_t, reset_t = inputs
            carry = jax.tree_util.tree_map(lambda x: x * reset_t, carry)
            carry, (mean, value) = model.apply(
                params["model"], carry, obs_t[None, :]
            )
            return carry, (mean[0], value[0])

        carry0 = _zero_carry(args.hidden_size)
        _, (means, values) = jax.lax.scan(step, carry0, (obs, reset_masks))

        pre = atanh(actions)
        log_prob_pre = gaussian_log_prob(pre, means, params["log_std"])
        eps = 1e-6
        log_det = jnp.sum(jnp.log(1.0 - actions**2 + eps), axis=-1)
        log_probs = log_prob_pre - log_det

        ratio = jnp.exp(log_probs - old_log_probs)
        adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        pg_loss1 = -adv_norm * ratio
        pg_loss2 = -adv_norm * jnp.clip(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef)
        pg_loss = jnp.mean(jnp.maximum(pg_loss1, pg_loss2))

        v_loss = jnp.mean((returns - values) ** 2)

        entropy = jnp.mean(0.5 + 0.5 * jnp.log(2.0 * jnp.pi) + params["log_std"])
        loss = pg_loss + args.vf_coef * v_loss - args.ent_coef * entropy
        return loss, (pg_loss, v_loss, entropy)

    @jax.jit
    def update_step(params, opt_state, batch):
        (loss, aux), grads = jax.value_and_grad(ppo_loss, has_aux=True)(params, batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, aux

    total_steps = int(args.total_timesteps)
    rollout_steps = int(args.rollout_steps)
    num_updates = max(1, total_steps // rollout_steps)

    obs = env.reset(np_rng)
    done_prev = False

    for update in range(1, num_updates + 1):
        obs_buf = np.zeros((rollout_steps, obs_dim), dtype=np.float32)
        act_buf = np.zeros((rollout_steps, act_dim), dtype=np.float32)
        logp_buf = np.zeros((rollout_steps,), dtype=np.float32)
        val_buf = np.zeros((rollout_steps,), dtype=np.float32)
        rew_buf = np.zeros((rollout_steps,), dtype=np.float32)
        done_buf = np.zeros((rollout_steps,), dtype=np.float32)
        reset_buf = np.zeros((rollout_steps,), dtype=np.float32)

        carry = _zero_carry(args.hidden_size)
        ep_return = 0.0
        passed_obstacle = False
        max_x = -float("inf")

        for t in range(rollout_steps):
            reset_mask = 0.0 if done_prev else 1.0
            reset_buf[t] = reset_mask
            carry = jax.tree_util.tree_map(lambda x: x * reset_mask, carry)

            obs_buf[t] = obs
            obs_jnp = jnp.asarray(obs[None, :])
            carry, (mean, value) = model.apply(params["model"], carry, obs_jnp)

            key, subkey = jax.random.split(key)
            noise = jax.random.normal(subkey, shape=mean.shape)
            pre = mean + jnp.exp(params["log_std"]) * noise
            action = jnp.tanh(pre)
            log_prob_pre = gaussian_log_prob(pre, mean, params["log_std"])
            eps = 1e-6
            log_det = jnp.sum(jnp.log(1.0 - action**2 + eps), axis=-1)
            log_prob = log_prob_pre - log_det

            action_np = np.asarray(action[0], dtype=np.float32)
            obs, reward, done, info = env.step(action_np)

            act_buf[t] = action_np
            logp_buf[t] = float(log_prob[0])
            val_buf[t] = float(value[0])
            rew_buf[t] = float(reward)
            done_buf[t] = float(done)
            ep_return += reward
            passed_obstacle = passed_obstacle or bool(info.get("passed_obstacle", False))
            x_val = info.get("x")
            if x_val is not None:
                max_x = max(max_x, float(x_val))

            done_prev = done
            if done:
                obs = env.reset(np_rng)

        if done_prev:
            last_value = 0.0
        else:
            obs_jnp = jnp.asarray(obs[None, :])
            _, (_, value) = model.apply(params["model"], carry, obs_jnp)
            last_value = float(value[0])

        advantages, returns = _compute_gae(
            rew_buf, val_buf, done_buf, last_value, args.gamma, args.gae_lambda
        )

        batch = {
            "obs": jnp.asarray(obs_buf),
            "actions": jnp.asarray(act_buf),
            "log_probs": jnp.asarray(logp_buf),
            "advantages": jnp.asarray(advantages),
            "returns": jnp.asarray(returns),
            "reset_masks": jnp.asarray(reset_buf),
        }

        losses = []
        for _ in range(int(args.epochs)):
            params, opt_state, loss, aux = update_step(params, opt_state, batch)
            losses.append((float(loss), float(aux[0]), float(aux[1]), float(aux[2])))

        mean_loss = np.mean([l[0] for l in losses])
        progress = 100.0 * update / max(num_updates, 1)
        print(
            (
                f"[PPO-LSTM] update {update:03d}/{num_updates} | "
                f"{progress:6.2f}% | loss={mean_loss:.3f} | "
                f"ep_return={ep_return:.2f} | passed_obstacle={passed_obstacle} | "
                f"max_x={(max_x if math.isfinite(max_x) else float('nan')):.3f}"
            )
        )

    timestamped_save_path = _timestamped_path(save_path)
    timestamped_save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(timestamped_save_path, "wb") as f:
        pickle.dump(params, f)
    print(f"[PPO-LSTM] Saved policy to {timestamped_save_path}")


if __name__ == "__main__":
    main()
