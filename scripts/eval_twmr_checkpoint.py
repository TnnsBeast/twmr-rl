#!/usr/bin/env python3
"""Evaluate a TWMR PPO checkpoint and report task-aligned metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jp
import numpy as np
import twmr  # noqa: F401  # Ensure env registration side effects.
from brax.training import checkpoint as brax_checkpoint
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from mujoco_playground import registry


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a checkpoint for TransformableWheelMobileRobot and "
            "report success_rate, mean_return, and mean_x_distance."
        )
    )
    parser.add_argument(
        "--checkpoint_path",
        required=True,
        help=(
            "Path to a checkpoint step directory (e.g. .../checkpoints/000001234000) "
            "or a checkpoints directory that contains step subdirectories."
        ),
    )
    parser.add_argument(
        "--env_name",
        default="TransformableWheelMobileRobot",
        help="Environment name registered in mujoco_playground.",
    )
    parser.add_argument(
        "--impl",
        default="jax",
        choices=["jax", "warp"],
        help="MJX implementation for evaluation.",
    )
    parser.add_argument(
        "--playground_config_overrides",
        default="{}",
        help='JSON dict of env config overrides (e.g. \'{"xml_variant":"box"}\').',
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=64,
        help="Number of evaluation episodes.",
    )
    parser.add_argument(
        "--episode_length",
        type=int,
        default=1000,
        help="Max rollout length per episode.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Number of episodes to evaluate in parallel per chunk.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for episode initialization and policy sampling key.",
    )
    parser.add_argument(
        "--output_json",
        default=None,
        help="Optional output JSON path for metrics.",
    )
    return parser.parse_args()


def _resolve_checkpoint_step(path_str: str) -> Path:
    path = Path(path_str).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {path}")

    if path.is_dir():
        subdirs = [p for p in path.iterdir() if p.is_dir() and p.name.isdigit()]
        if subdirs:
            return max(subdirs, key=lambda p: int(p.name))
        return path
    return path


def _resolve_network_config_path(ckpt_step_path: Path) -> Path:
    # Brax versions differ on config filename.
    preferred = [
        ckpt_step_path / "config.json",
        ckpt_step_path / "ppo_network_config.json",
    ]
    for path in preferred:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find network config in checkpoint step path: {ckpt_step_path}"
    )


def _evaluate_chunk(
    eval_env: Any,
    policy: Any,
    keys: jax.Array,
    episode_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    def single_rollout(key: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        state = eval_env.reset(key)
        x_start = state.data.qpos[0]
        max_x = x_start
        done = jp.array(False)
        success = jp.array(False)
        ep_return = jp.array(0.0, dtype=jp.float32)

        def step_fn(carry: tuple[Any, ...], _unused: None) -> tuple[tuple[Any, ...], None]:
            state, key, done, success, ep_return, max_x = carry
            key, act_key = jax.random.split(key)
            action, _ = policy(state.obs, act_key)
            next_state = eval_env.step(state, action)

            active = jp.logical_not(done)
            ep_return = ep_return + active.astype(jp.float32) * next_state.reward
            succ_now = next_state.metrics["task/success"] > 0.5
            success = jp.logical_or(success, jp.logical_and(active, succ_now))
            done = jp.logical_or(done, next_state.done > 0.5)
            max_x = jp.maximum(max_x, next_state.metrics["task/root_x"])
            return (next_state, key, done, success, ep_return, max_x), None

        carry, _ = jax.lax.scan(
            step_fn,
            (state, key, done, success, ep_return, max_x),
            None,
            length=episode_length,
        )
        _, _, _, success, ep_return, max_x = carry
        x_distance = max_x - x_start
        return ep_return, success.astype(jp.float32), x_distance

    batched_rollout = jax.jit(jax.vmap(single_rollout))
    returns, successes, x_distances = batched_rollout(keys)
    return np.asarray(returns), np.asarray(successes), np.asarray(x_distances)


def _resolve_observation_size(observation_size: Any) -> Any:
    if hasattr(observation_size, "to_dict"):
        observation_size = observation_size.to_dict()
    if isinstance(observation_size, dict) and "shape" in observation_size:
        return tuple(int(x) for x in observation_size["shape"])
    return observation_size


def _build_ppo_network_from_config(net_cfg: Any) -> Any:
    kwargs = (
        net_cfg.network_factory_kwargs.to_dict()
        if hasattr(net_cfg.network_factory_kwargs, "to_dict")
        else dict(net_cfg.network_factory_kwargs)
    )
    observation_size = _resolve_observation_size(net_cfg.observation_size)

    # For array observations, obs_key should be None instead of "state".
    if not isinstance(observation_size, dict):
        if kwargs.get("policy_obs_key") == "state":
            kwargs["policy_obs_key"] = None
        if kwargs.get("value_obs_key") == "state":
            kwargs["value_obs_key"] = None

    preprocess_observations_fn = (
        running_statistics.normalize if net_cfg.normalize_observations else (lambda x, y: x)
    )
    return ppo_networks.make_ppo_networks(
        observation_size=observation_size,
        action_size=int(net_cfg.action_size),
        preprocess_observations_fn=preprocess_observations_fn,
        **kwargs,
    )


def main() -> int:
    args = _parse_args()
    env_cfg = registry.get_default_config(args.env_name)
    env_cfg["impl"] = args.impl
    config_overrides = json.loads(args.playground_config_overrides)
    eval_env = registry.load(args.env_name, config=env_cfg, config_overrides=config_overrides)

    ckpt_step_path = _resolve_checkpoint_step(args.checkpoint_path)
    params = brax_checkpoint.load(ckpt_step_path)
    net_cfg_path = _resolve_network_config_path(ckpt_step_path)
    net_cfg = brax_checkpoint.load_config(net_cfg_path)
    ppo_net = _build_ppo_network_from_config(net_cfg)
    make_policy = ppo_networks.make_inference_fn(ppo_net)
    policy = make_policy(params, deterministic=True)

    all_returns: list[np.ndarray] = []
    all_successes: list[np.ndarray] = []
    all_x_distances: list[np.ndarray] = []

    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.num_episodes)
    for start in range(0, args.num_episodes, args.batch_size):
        end = min(start + args.batch_size, args.num_episodes)
        chunk_keys = keys[start:end]
        returns, successes, x_distances = _evaluate_chunk(
            eval_env=eval_env,
            policy=policy,
            keys=chunk_keys,
            episode_length=args.episode_length,
        )
        all_returns.append(returns)
        all_successes.append(successes)
        all_x_distances.append(x_distances)

    returns = np.concatenate(all_returns, axis=0)
    successes = np.concatenate(all_successes, axis=0)
    x_distances = np.concatenate(all_x_distances, axis=0)

    result = {
        "checkpoint_step_path": str(ckpt_step_path),
        "network_config_path": str(net_cfg_path),
        "env_name": args.env_name,
        "impl": args.impl,
        "config_overrides": config_overrides,
        "num_episodes": int(args.num_episodes),
        "episode_length": int(args.episode_length),
        "success_rate": float(successes.mean()),
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std()),
        "mean_x_distance": float(x_distances.mean()),
        "std_x_distance": float(x_distances.std()),
    }

    print(json.dumps(result, indent=2))
    if args.output_json is not None:
        output_path = Path(args.output_json).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Wrote metrics JSON to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
