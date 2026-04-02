#!/usr/bin/env python3
"""Standalone Neil TWMR curriculum runner.

This script carries out the following phase logic:

  flat -> simpleobstacle -> bridge -> random

By default it runs the full curriculum. It can also run a single phase with an
explicit checkpoint handoff.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ENV_NAME = "TransformableWheelMobileRobot"
PHASE_FULL = "full"
PHASE_FLAT = "flat"
PHASE_SIMPLE = "simpleobstacle"
PHASE_BRIDGE = "bridge"
PHASE_RANDOM = "random"
ALL_PHASES = (PHASE_FULL, PHASE_FLAT, PHASE_SIMPLE, PHASE_BRIDGE, PHASE_RANDOM)


@dataclass(frozen=True)
class PpoParams:
    num_timesteps: int
    num_envs: int
    num_eval_envs: int
    batch_size: int
    unroll_length: int
    num_minibatches: int
    num_updates_per_batch: int
    num_evals: int
    episode_length: int
    seed: int
    max_devices_per_host: int | None
    rollout_camera: str | None
    rollout_width: int
    rollout_height: int


@dataclass(frozen=True)
class Scenario:
    index: int
    obstacle_x_position: float
    obstacle_height: float
    obstacle_half_length: float

    @property
    def difficulty(self) -> float:
        return self.obstacle_height * self.obstacle_half_length


@dataclass(frozen=True)
class BridgeStage:
    index: int
    height: float
    half_length: float
    overrides: dict[str, Any]
    suffix: str


@dataclass(frozen=True)
class RandomStage:
    index: int
    scenario_index: int
    obstacle_x_position: float
    obstacle_height: float
    obstacle_half_length: float
    overrides: dict[str, Any]
    suffix: str


def _parse_bool(raw: str) -> bool:
    value = raw.strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {raw!r}")


def _parse_float_list(raw: str) -> list[float]:
    values = [x.strip() for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("Expected at least one numeric value.")
    return [float(value) for value in values]


def _tag(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def _slugify_label(label: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in label)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-") or "phase"


def _runtime_env() -> dict[str, str]:
    env = os.environ.copy()
    ffmpeg_bin = Path.home() / ".pixi" / "envs" / "ffmpeg" / "bin"
    env["PATH"] = f"{ffmpeg_bin}:{env.get('PATH', '')}"
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def _run_cmd(cmd: list[str], env: dict[str, str], dry_run: bool) -> None:
    printable = " ".join(cmd)
    print(f"[CMD] {printable}", flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True, env=env)


def _merge_overrides(
    overrides: dict[str, Any],
    common_overrides: dict[str, Any],
) -> dict[str, Any]:
    if not common_overrides:
        return dict(overrides)
    merged = dict(overrides)
    merged.update(common_overrides)
    return merged


def _latest_log_with_suffix(logs_root: Path, suffix: str) -> Path:
    candidates = [path for path in logs_root.iterdir() if path.is_dir() and path.name.endswith(f"-{suffix}")]
    if not candidates:
        raise RuntimeError(f"No log directory found for suffix '{suffix}'.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _write_titled_rollout(
    *,
    source_video: Path,
    target_video: Path,
    phase_title: str,
    env: dict[str, str],
) -> None:
    escaped_title = (
        phase_title.replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", "\\'")
    )
    video_filter = (
        "drawbox=x=0:y=0:w=iw:h=52:color=black@0.55:t=fill,"
        f"drawtext=text='{escaped_title}':x=12:y=14:fontsize=22:fontcolor=white"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(source_video),
        "-vf",
        video_filter,
        "-codec:a",
        "copy",
        str(target_video),
    ]
    try:
        subprocess.run(
            cmd,
            check=True,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as err:  # pragma: no cover - best-effort annotation.
        print(
            "[WARN] Failed to render titled rollout; copying untitled video instead. "
            f"title={phase_title!r}, error={err}",
            flush=True,
        )
        shutil.copy2(source_video, target_video)


def _copy_rollout_artifacts(
    *,
    exp_dir: Path,
    phase_title: str,
    titled_name: str,
    env: dict[str, str],
) -> tuple[str, str]:
    rollout_src = Path("rollout0.mp4").resolve()
    if not rollout_src.exists():
        return "", ""

    rollout_dst = exp_dir / "rollout0.mp4"
    shutil.copy2(rollout_src, rollout_dst)
    titled_rollout_dst = exp_dir / titled_name
    _write_titled_rollout(
        source_video=rollout_dst,
        target_video=titled_rollout_dst,
        phase_title=phase_title,
        env=env,
    )
    print(f"  rollout={rollout_dst}", flush=True)
    print(f"  titled_rollout={titled_rollout_dst}", flush=True)
    return str(rollout_dst), str(titled_rollout_dst)


def _train_cmd(
    *,
    python_exec: str,
    suffix: str,
    overrides: dict[str, Any],
    params: PpoParams,
    load_checkpoint_path: Path | None,
) -> list[str]:
    cmd = [
        python_exec,
        "train_jax_ppo.py",
        f"--env_name={ENV_NAME}",
        f"--playground_config_overrides={json.dumps(overrides, separators=(',', ':'))}",
        f"--num_timesteps={params.num_timesteps}",
        f"--num_envs={params.num_envs}",
        f"--num_eval_envs={params.num_eval_envs}",
        f"--batch_size={params.batch_size}",
        f"--unroll_length={params.unroll_length}",
        f"--num_minibatches={params.num_minibatches}",
        f"--num_updates_per_batch={params.num_updates_per_batch}",
        f"--num_evals={params.num_evals}",
        f"--episode_length={params.episode_length}",
        f"--seed={params.seed}",
        f"--suffix={suffix}",
        "--use_wandb=false",
        "--use_tb=false",
    ]
    if params.max_devices_per_host is not None:
        cmd.append(f"--max_devices_per_host={params.max_devices_per_host}")
    if load_checkpoint_path is not None:
        cmd.append(f"--load_checkpoint_path={load_checkpoint_path}")
    if params.rollout_camera:
        cmd.append(f"--rollout_camera={params.rollout_camera}")
        cmd.append(f"--rollout_width={params.rollout_width}")
        cmd.append(f"--rollout_height={params.rollout_height}")
    return cmd


def _eval_cmd(
    *,
    python_exec: str,
    checkpoint_path: Path,
    overrides: dict[str, Any],
    params: PpoParams,
    eval_num_episodes: int,
    eval_batch_size: int,
    output_json: Path,
) -> list[str]:
    return [
        python_exec,
        "scripts/eval_twmr_checkpoint.py",
        f"--checkpoint_path={checkpoint_path}",
        f"--playground_config_overrides={json.dumps(overrides, separators=(',', ':'))}",
        f"--num_episodes={eval_num_episodes}",
        f"--episode_length={params.episode_length}",
        f"--batch_size={eval_batch_size}",
        f"--seed={params.seed}",
        f"--output_json={output_json}",
    ]


def _default_flat_overrides() -> dict[str, Any]:
    return {
        "xml_variant": "flat",
        "success_x_threshold": 1.0,
        "terminate_on_success": False,
        "success_bonus": 0.0,
    }


def _default_simpleobstacle_overrides() -> dict[str, Any]:
    return {
        "xml_variant": "box",
        "obstacle_height": 0.03,
        "obstacle_x_position": 0.55,
        "obstacle_half_length": 0.15,
        "success_x_margin": 0.30,
        "terminate_on_success": False,
        "success_bonus": 0.0,
        "obstacle_local_windowed_reward": True,
        "obstacle_local_activation_margin": 0.08,
        "obstacle_local_extension_reward_weight": 0.0,
        "post_success_progress_weight": 5.0,
        "stall_penalty_after_success": True,
        "stall_penalty_weight": 0.08,
        "extension_retracted_penalty_weight": 0.8,
        "extension_retracted_penalty_outside_obstacle_only": True,
    }


def _bridge_base_overrides(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "xml_variant": "box",
        "obstacle_x_position": args.bridge_obstacle_x_position,
        "success_x_margin": args.bridge_success_x_margin,
        "terminate_on_success": args.bridge_terminate_on_success,
        "success_bonus": args.bridge_success_bonus,
        "obstacle_local_windowed_reward": args.bridge_obstacle_local_windowed_reward,
        "obstacle_local_activation_margin": args.bridge_obstacle_local_activation_margin,
        "obstacle_local_extension_reward_weight": args.bridge_obstacle_local_extension_reward_weight,
        "post_success_progress_weight": args.bridge_post_success_progress_weight,
        "stall_penalty_after_success": args.bridge_stall_penalty_after_success,
        "stall_penalty_weight": args.bridge_stall_penalty_weight,
        "extension_retracted_penalty_weight": args.bridge_extension_retracted_penalty_weight,
        "extension_retracted_penalty_outside_obstacle_only": (
            args.bridge_extension_retracted_penalty_outside_obstacle_only
        ),
    }


def _random_base_overrides(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "xml_variant": "box",
        "obstacle_local_windowed_reward": args.random_obstacle_local_windowed_reward,
        "obstacle_local_activation_margin": args.random_obstacle_local_activation_margin,
        "obstacle_local_extension_reward_weight": args.random_obstacle_local_extension_reward_weight,
        "post_success_progress_weight": args.random_post_success_progress_weight,
        "stall_penalty_after_success": args.random_stall_penalty_after_success,
        "stall_penalty_weight": args.random_stall_penalty_weight,
        "extension_retracted_penalty_weight": args.random_extension_retracted_penalty_weight,
        "extension_retracted_penalty_outside_obstacle_only": (
            args.random_extension_retracted_penalty_outside_obstacle_only
        ),
        "success_x_margin": args.random_success_x_margin,
        "terminate_on_success": args.random_terminate_on_success,
        "success_bonus": args.random_success_bonus,
    }


def _flat_params(args: argparse.Namespace) -> PpoParams:
    return PpoParams(
        num_timesteps=args.flat_num_timesteps,
        num_envs=args.phase3_num_envs,
        num_eval_envs=args.phase3_num_eval_envs,
        batch_size=args.phase3_batch_size,
        unroll_length=args.phase3_unroll_length,
        num_minibatches=args.phase3_num_minibatches,
        num_updates_per_batch=args.phase3_num_updates_per_batch,
        num_evals=args.phase3_num_evals,
        episode_length=args.phase3_episode_length,
        seed=args.seed,
        max_devices_per_host=args.max_devices_per_host,
        rollout_camera=None,
        rollout_width=args.phase3_rollout_width,
        rollout_height=args.phase3_rollout_height,
    )


def _simple_params(args: argparse.Namespace) -> PpoParams:
    return PpoParams(
        num_timesteps=args.simpleobstacle_num_timesteps,
        num_envs=args.phase3_num_envs,
        num_eval_envs=args.phase3_num_eval_envs,
        batch_size=args.phase3_batch_size,
        unroll_length=args.phase3_unroll_length,
        num_minibatches=args.phase3_num_minibatches,
        num_updates_per_batch=args.phase3_num_updates_per_batch,
        num_evals=args.phase3_num_evals,
        episode_length=args.phase3_episode_length,
        seed=args.seed,
        max_devices_per_host=args.max_devices_per_host,
        rollout_camera=args.simpleobstacle_rollout_camera,
        rollout_width=args.phase3_rollout_width,
        rollout_height=args.phase3_rollout_height,
    )


def _bridge_params(args: argparse.Namespace) -> PpoParams:
    return PpoParams(
        num_timesteps=args.bridge_num_timesteps,
        num_envs=args.bridge_num_envs,
        num_eval_envs=args.bridge_num_eval_envs,
        batch_size=args.bridge_batch_size,
        unroll_length=args.bridge_unroll_length,
        num_minibatches=args.bridge_num_minibatches,
        num_updates_per_batch=args.bridge_num_updates_per_batch,
        num_evals=args.bridge_num_evals,
        episode_length=args.bridge_episode_length,
        seed=args.seed,
        max_devices_per_host=args.max_devices_per_host,
        rollout_camera=args.bridge_rollout_camera,
        rollout_width=args.bridge_rollout_width,
        rollout_height=args.bridge_rollout_height,
    )


def _random_params(args: argparse.Namespace) -> PpoParams:
    return PpoParams(
        num_timesteps=args.random_num_timesteps,
        num_envs=args.random_num_envs,
        num_eval_envs=args.random_num_eval_envs,
        batch_size=args.random_batch_size,
        unroll_length=args.random_unroll_length,
        num_minibatches=args.random_num_minibatches,
        num_updates_per_batch=args.random_num_updates_per_batch,
        num_evals=args.random_num_evals,
        episode_length=args.random_episode_length,
        seed=args.seed,
        max_devices_per_host=args.max_devices_per_host,
        rollout_camera=args.random_rollout_camera,
        rollout_width=args.random_rollout_width,
        rollout_height=args.random_rollout_height,
    )


def _train_single_stage(
    *,
    args: argparse.Namespace,
    env: dict[str, str],
    logs_root: Path,
    phase_name: str,
    suffix: str,
    overrides: dict[str, Any],
    params: PpoParams,
    load_checkpoint_path: Path | None,
    phase_title: str,
    dry_run: bool,
) -> dict[str, Any]:
    merged_overrides = _merge_overrides(overrides, args.common_overrides)
    train_cmd = _train_cmd(
        python_exec=args.python,
        suffix=suffix,
        overrides=merged_overrides,
        params=params,
        load_checkpoint_path=load_checkpoint_path,
    )
    _run_cmd(train_cmd, env=env, dry_run=dry_run)

    record: dict[str, Any] = {
        "phase": phase_name,
        "suffix": suffix,
        "restore_from": str(load_checkpoint_path) if load_checkpoint_path is not None else "",
        "train_params": asdict(params),
        "overrides": merged_overrides,
    }
    if dry_run:
        record["dry_run"] = True
        record["exp_dir"] = f"<{phase_name}_exp_dir>"
        record["ckpt_dir"] = f"<{phase_name}_ckpt_dir>"
        record["rollout"] = ""
        record["titled_rollout"] = ""
        return record

    exp_dir = _latest_log_with_suffix(logs_root, suffix)
    ckpt_dir = exp_dir / "checkpoints"
    if not ckpt_dir.exists():
        raise RuntimeError(f"Missing checkpoints directory for '{suffix}': {ckpt_dir}")

    rollout_path, titled_rollout_path = _copy_rollout_artifacts(
        exp_dir=exp_dir,
        phase_title=phase_title,
        titled_name=f"{_slugify_label(suffix)}-rollout0-title.mp4",
        env=env,
    )

    record.update(
        {
            "exp_dir": str(exp_dir),
            "ckpt_dir": str(ckpt_dir),
            "rollout": rollout_path,
            "titled_rollout": titled_rollout_path,
        }
    )
    print(f"[STAGE COMPLETE] {phase_name}", flush=True)
    print(f"  exp_dir={exp_dir}", flush=True)
    print(f"  ckpt_dir={ckpt_dir}", flush=True)
    return record


def _build_bridge_stages(args: argparse.Namespace, suffix_prefix: str) -> list[BridgeStage]:
    base_overrides = _bridge_base_overrides(args)
    stages: list[BridgeStage] = []
    for idx, height in enumerate(_parse_float_list(args.bridge_heights)):
        suffix = f"{suffix_prefix}-h{_tag(height)}-l{_tag(args.bridge_fixed_half_length)}"
        stages.append(
            BridgeStage(
                index=idx,
                height=height,
                half_length=args.bridge_fixed_half_length,
                overrides={
                    **base_overrides,
                    "obstacle_height": height,
                    "obstacle_half_length": args.bridge_fixed_half_length,
                },
                suffix=suffix,
            )
        )
    return stages


def _bridge_threshold(args: argparse.Namespace, stage_index: int) -> float:
    threshold = args.bridge_min_success_to_continue
    if args.bridge_gate_index >= 0 and stage_index == args.bridge_gate_index:
        threshold = max(threshold, args.bridge_gate_threshold)
    return threshold


def _sample_random_scenario_bank(args: argparse.Namespace) -> list[Scenario]:
    if args.random_scenario_bank_size <= 0:
        raise ValueError("--random_scenario_bank_size must be > 0.")
    if args.random_height_max < args.random_height_min:
        raise ValueError("--random_height_max must be >= --random_height_min.")
    if args.random_half_length_max < args.random_half_length_min:
        raise ValueError("--random_half_length_max must be >= --random_half_length_min.")
    if args.random_randomize_obstacle_x_max <= args.random_randomize_obstacle_x_min:
        raise ValueError(
            "--random_randomize_obstacle_x_max must be > --random_randomize_obstacle_x_min."
        )

    rng = random.Random(args.seed)
    scenarios: list[Scenario] = []
    for idx in range(args.random_scenario_bank_size):
        scenarios.append(
            Scenario(
                index=idx,
                obstacle_x_position=rng.uniform(
                    float(args.random_randomize_obstacle_x_min),
                    float(args.random_randomize_obstacle_x_max),
                ),
                obstacle_height=rng.uniform(
                    float(args.random_height_min),
                    float(args.random_height_max),
                ),
                obstacle_half_length=rng.uniform(
                    float(args.random_half_length_min),
                    float(args.random_half_length_max),
                ),
            )
        )

    if args.random_scenario_order == "easy_to_hard":
        return sorted(scenarios, key=lambda scenario: (scenario.difficulty, scenario.index))
    if args.random_scenario_order == "hard_to_easy":
        return sorted(scenarios, key=lambda scenario: (-scenario.difficulty, scenario.index))
    rng.shuffle(scenarios)
    return scenarios


def _build_random_stages(
    args: argparse.Namespace,
    scenarios: list[Scenario],
    suffix_prefix: str,
) -> list[RandomStage]:
    if args.random_num_stages <= 0:
        raise ValueError("--random_num_stages must be > 0.")

    base_overrides = _random_base_overrides(args)
    rng = random.Random(args.seed + 1337)
    sequence: list[Scenario] = []

    if args.random_stage_sampling == "ordered":
        for idx in range(args.random_num_stages):
            sequence.append(scenarios[idx % len(scenarios)])
    elif args.random_stage_sampling == "random_with_replacement":
        for _ in range(args.random_num_stages):
            sequence.append(scenarios[rng.randrange(len(scenarios))])
    else:
        while len(sequence) < args.random_num_stages:
            cycle = list(scenarios)
            rng.shuffle(cycle)
            need = args.random_num_stages - len(sequence)
            sequence.extend(cycle[:need])

    stages: list[RandomStage] = []
    for stage_index, scenario in enumerate(sequence):
        suffix = (
            f"{suffix_prefix}-s{stage_index:02d}-"
            f"b{scenario.index:03d}-"
            f"x{_tag(scenario.obstacle_x_position)}-"
            f"h{_tag(scenario.obstacle_height)}-"
            f"l{_tag(scenario.obstacle_half_length)}"
        )
        stages.append(
            RandomStage(
                index=stage_index,
                scenario_index=scenario.index,
                obstacle_x_position=scenario.obstacle_x_position,
                obstacle_height=scenario.obstacle_height,
                obstacle_half_length=scenario.obstacle_half_length,
                overrides={
                    **base_overrides,
                    "obstacle_x_position": float(scenario.obstacle_x_position),
                    "obstacle_height": float(scenario.obstacle_height),
                    "obstacle_half_length": float(scenario.obstacle_half_length),
                },
                suffix=suffix,
            )
        )
    return stages


def _scenario_bank_payload(args: argparse.Namespace, scenarios: list[Scenario]) -> dict[str, Any]:
    return {
        "created_at": dt.datetime.now().isoformat(),
        "seed": int(args.seed),
        "scenario_bank_size": int(args.random_scenario_bank_size),
        "scenario_order": str(args.random_scenario_order),
        "stage_sampling": str(args.random_stage_sampling),
        "ranges": {
            "x_min": float(args.random_randomize_obstacle_x_min),
            "x_max": float(args.random_randomize_obstacle_x_max),
            "height_min": float(args.random_height_min),
            "height_max": float(args.random_height_max),
            "half_length_min": float(args.random_half_length_min),
            "half_length_max": float(args.random_half_length_max),
        },
        "scenarios": [
            {
                "scenario_index": scenario.index,
                "obstacle_x_position": scenario.obstacle_x_position,
                "obstacle_height": scenario.obstacle_height,
                "obstacle_half_length": scenario.obstacle_half_length,
                "difficulty": scenario.difficulty,
            }
            for scenario in scenarios
        ],
    }


def _run_bridge_phase(
    *,
    args: argparse.Namespace,
    env: dict[str, str],
    logs_root: Path,
    initial_checkpoint_path: Path,
    suffix_prefix: str,
    dry_run: bool,
) -> dict[str, Any]:
    stages = _build_bridge_stages(args, suffix_prefix)
    params = _bridge_params(args)
    checkpoint_from_prev_stage = initial_checkpoint_path
    summary: list[dict[str, Any]] = []
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    stopped_early = False
    stop_reason = ""

    for stage in stages:
        print(
            f"[STAGE] phase=bridge idx={stage.index} height={stage.height:.3f} half_length={stage.half_length:.3f}",
            flush=True,
        )
        train_cmd = _train_cmd(
            python_exec=args.python,
            suffix=stage.suffix,
            overrides=_merge_overrides(stage.overrides, args.common_overrides),
            params=params,
            load_checkpoint_path=checkpoint_from_prev_stage,
        )
        _run_cmd(train_cmd, env=env, dry_run=dry_run)

        merged_overrides = _merge_overrides(stage.overrides, args.common_overrides)
        record: dict[str, Any] = {
            "stage_index": stage.index,
            "height": stage.height,
            "half_length": stage.half_length,
            "suffix": stage.suffix,
            "restore_from": str(checkpoint_from_prev_stage),
            "overrides": merged_overrides,
        }
        if dry_run:
            record["dry_run"] = True
            summary.append(record)
            checkpoint_from_prev_stage = Path(f"<bridge_stage_{stage.index:02d}_ckpt>")
            continue

        exp_dir = _latest_log_with_suffix(logs_root, stage.suffix)
        ckpt_dir = exp_dir / "checkpoints"
        if not ckpt_dir.exists():
            raise RuntimeError(f"Missing checkpoints directory for '{stage.suffix}': {ckpt_dir}")

        rollout_path, titled_rollout_path = _copy_rollout_artifacts(
            exp_dir=exp_dir,
            phase_title=f"Bridge s{stage.index:02d} h={stage.height:.3f} l={stage.half_length:.3f}",
            titled_name=f"{_slugify_label(stage.suffix)}-rollout0-title.mp4",
            env=env,
        )

        eval_json = logs_root / (
            f"eval-{suffix_prefix}-stage{stage.index:02d}-"
            f"h{_tag(stage.height)}-l{_tag(stage.half_length)}-"
            f"{args.bridge_eval_num_episodes}ep-{timestamp}.json"
        )
        eval_cmd = _eval_cmd(
            python_exec=args.python,
            checkpoint_path=ckpt_dir,
            overrides=merged_overrides,
            params=params,
            eval_num_episodes=args.bridge_eval_num_episodes,
            eval_batch_size=args.bridge_eval_batch_size,
            output_json=eval_json,
        )
        _run_cmd(eval_cmd, env=env, dry_run=False)
        eval_data = json.loads(eval_json.read_text(encoding="utf-8"))

        success_rate = float(eval_data["success_rate"])
        threshold = _bridge_threshold(args, stage.index)
        promoted = success_rate >= threshold
        print(
            f"  eval success_rate={success_rate:.3f} threshold={threshold:.3f} promoted={promoted}",
            flush=True,
        )

        record.update(
            {
                "exp_dir": str(exp_dir),
                "ckpt_dir": str(ckpt_dir),
                "eval_json": str(eval_json),
                "success_rate": success_rate,
                "mean_return": float(eval_data["mean_return"]),
                "mean_x_distance": float(eval_data["mean_x_distance"]),
                "promotion_threshold": threshold,
                "promoted": promoted,
                "rollout": rollout_path,
                "titled_rollout": titled_rollout_path,
            }
        )
        summary.append(record)

        if not promoted:
            stopped_early = True
            stop_reason = (
                f"Stage {stage.index} gate failed: success_rate={success_rate:.3f} "
                f"< threshold={threshold:.3f}"
            )
            print(f"[STOP] {stop_reason}", flush=True)
            break

        checkpoint_from_prev_stage = ckpt_dir

    selected_stage = _select_stage_record(summary, args.bridge_selection_strategy)
    return {
        "phase": PHASE_BRIDGE,
        "initial_checkpoint_path": str(initial_checkpoint_path),
        "train_params": asdict(params),
        "bridge_heights": _parse_float_list(args.bridge_heights),
        "bridge_fixed_half_length": args.bridge_fixed_half_length,
        "min_success_to_continue": args.bridge_min_success_to_continue,
        "gate_index": args.bridge_gate_index,
        "gate_threshold": args.bridge_gate_threshold,
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "selection_strategy": args.bridge_selection_strategy,
        "selected_stage": selected_stage,
        "selected_checkpoint_path": selected_stage.get("ckpt_dir", "") if selected_stage else "",
        "stages": summary,
    }


def _run_random_phase(
    *,
    args: argparse.Namespace,
    env: dict[str, str],
    logs_root: Path,
    initial_checkpoint_path: Path,
    suffix_prefix: str,
    dry_run: bool,
) -> dict[str, Any]:
    scenario_bank = _sample_random_scenario_bank(args)
    if args.random_scenario_bank_path:
        bank_path = Path(args.random_scenario_bank_path).resolve()
        bank_path.parent.mkdir(parents=True, exist_ok=True)
        bank_path.write_text(
            json.dumps(_scenario_bank_payload(args, scenario_bank), indent=2),
            encoding="utf-8",
        )
        print(f"[INFO] Wrote scenario bank: {bank_path}", flush=True)

    stages = _build_random_stages(args, scenario_bank, suffix_prefix)
    params = _random_params(args)
    checkpoint_from_prev_stage = initial_checkpoint_path
    summary: list[dict[str, Any]] = []
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    stopped_early = False
    stop_reason = ""

    for stage in stages:
        print(
            (
                f"[STAGE] phase=random idx={stage.index} scenario={stage.scenario_index} "
                f"x={stage.obstacle_x_position:.3f} h={stage.obstacle_height:.3f} "
                f"l={stage.obstacle_half_length:.3f}"
            ),
            flush=True,
        )
        train_cmd = _train_cmd(
            python_exec=args.python,
            suffix=stage.suffix,
            overrides=_merge_overrides(stage.overrides, args.common_overrides),
            params=params,
            load_checkpoint_path=checkpoint_from_prev_stage,
        )
        _run_cmd(train_cmd, env=env, dry_run=dry_run)

        merged_overrides = _merge_overrides(stage.overrides, args.common_overrides)
        record: dict[str, Any] = {
            "stage_index": stage.index,
            "scenario_index": stage.scenario_index,
            "obstacle_x_position": stage.obstacle_x_position,
            "obstacle_height": stage.obstacle_height,
            "obstacle_half_length": stage.obstacle_half_length,
            "suffix": stage.suffix,
            "restore_from": str(checkpoint_from_prev_stage),
            "overrides": merged_overrides,
        }
        if dry_run:
            record["dry_run"] = True
            summary.append(record)
            checkpoint_from_prev_stage = Path(f"<random_stage_{stage.index:02d}_ckpt>")
            continue

        exp_dir = _latest_log_with_suffix(logs_root, stage.suffix)
        ckpt_dir = exp_dir / "checkpoints"
        if not ckpt_dir.exists():
            raise RuntimeError(f"Missing checkpoints directory for '{stage.suffix}': {ckpt_dir}")

        rollout_path, titled_rollout_path = _copy_rollout_artifacts(
            exp_dir=exp_dir,
            phase_title=(
                f"Random s{stage.index:02d} b{stage.scenario_index:03d} "
                f"x={stage.obstacle_x_position:.3f} h={stage.obstacle_height:.3f} "
                f"l={stage.obstacle_half_length:.3f}"
            ),
            titled_name=f"{_slugify_label(stage.suffix)}-rollout0-title.mp4",
            env=env,
        )

        eval_json = logs_root / (
            f"eval-{suffix_prefix}-stage{stage.index:02d}-"
            f"b{stage.scenario_index:03d}-"
            f"x{_tag(stage.obstacle_x_position)}-"
            f"h{_tag(stage.obstacle_height)}-"
            f"l{_tag(stage.obstacle_half_length)}-"
            f"{args.random_eval_num_episodes}ep-{timestamp}.json"
        )
        eval_cmd = _eval_cmd(
            python_exec=args.python,
            checkpoint_path=ckpt_dir,
            overrides=merged_overrides,
            params=params,
            eval_num_episodes=args.random_eval_num_episodes,
            eval_batch_size=args.random_eval_batch_size,
            output_json=eval_json,
        )
        _run_cmd(eval_cmd, env=env, dry_run=False)
        eval_data = json.loads(eval_json.read_text(encoding="utf-8"))

        success_rate = float(eval_data["success_rate"])
        promoted = success_rate >= args.random_min_success_to_continue
        print(
            (
                f"  eval success_rate={success_rate:.3f} "
                f"threshold={args.random_min_success_to_continue:.3f} promoted={promoted}"
            ),
            flush=True,
        )

        record.update(
            {
                "exp_dir": str(exp_dir),
                "ckpt_dir": str(ckpt_dir),
                "eval_json": str(eval_json),
                "success_rate": success_rate,
                "mean_return": float(eval_data["mean_return"]),
                "mean_x_distance": float(eval_data["mean_x_distance"]),
                "promotion_threshold": args.random_min_success_to_continue,
                "promoted": promoted,
                "rollout": rollout_path,
                "titled_rollout": titled_rollout_path,
            }
        )
        summary.append(record)

        if not promoted:
            stopped_early = True
            stop_reason = (
                f"Stage {stage.index} gate failed: success_rate={success_rate:.3f} "
                f"< threshold={args.random_min_success_to_continue:.3f}"
            )
            print(f"[STOP] {stop_reason}", flush=True)
            break

        checkpoint_from_prev_stage = ckpt_dir

    selected_stage = _select_stage_record(summary, args.random_selection_strategy)
    return {
        "phase": PHASE_RANDOM,
        "initial_checkpoint_path": str(initial_checkpoint_path),
        "train_params": asdict(params),
        "num_stages": args.random_num_stages,
        "scenario_bank_size": args.random_scenario_bank_size,
        "scenario_order": args.random_scenario_order,
        "stage_sampling": args.random_stage_sampling,
        "height_min": args.random_height_min,
        "height_max": args.random_height_max,
        "half_length_min": args.random_half_length_min,
        "half_length_max": args.random_half_length_max,
        "randomize_obstacle_x_min": args.random_randomize_obstacle_x_min,
        "randomize_obstacle_x_max": args.random_randomize_obstacle_x_max,
        "min_success_to_continue": args.random_min_success_to_continue,
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "selection_strategy": args.random_selection_strategy,
        "selected_stage": selected_stage,
        "selected_checkpoint_path": selected_stage.get("ckpt_dir", "") if selected_stage else "",
        "scenario_bank": _scenario_bank_payload(args, scenario_bank)["scenarios"],
        "stages": summary,
    }


def _select_stage_record(stages: list[dict[str, Any]], strategy: str) -> dict[str, Any] | None:
    if not stages:
        return None
    if strategy == "best_success":
        return max(
            stages,
            key=lambda stage: (
                float(stage.get("success_rate", float("-inf"))),
                int(stage.get("stage_index", -1)),
            ),
        )
    if strategy == "last_promoted":
        promoted = [stage for stage in stages if bool(stage.get("promoted"))]
        if promoted:
            return promoted[-1]
        return None
    raise ValueError(f"Unsupported selection strategy: {strategy}")


def _require_initial_checkpoint(args: argparse.Namespace) -> Path:
    if args.initial_checkpoint_path is None:
        raise ValueError(f"--initial_checkpoint_path is required for --phase={args.phase}.")
    checkpoint_path = Path(args.initial_checkpoint_path).expanduser().resolve()
    if not args.dry_run and not checkpoint_path.exists():
        raise FileNotFoundError(f"Initial checkpoint path not found: {checkpoint_path}")
    return checkpoint_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone Neil TWMR curriculum runner. Defaults are copied from "
            "the successful flat/simpleobstacle/bridge/random jobs."
        )
    )
    parser.add_argument("--python", default="./.venv/bin/python")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--max_devices_per_host",
        type=int,
        default=1,
        help=(
            "Cap Brax training to a single local device by default. This keeps "
            "the standalone Neil run on the same training path as the successful "
            "single-GPU flat/simpleobstacle/bridge runs."
        ),
    )
    parser.add_argument("--suffix_prefix", default="neil-twmr")
    parser.add_argument("--phase", choices=ALL_PHASES, default=PHASE_FULL)
    parser.add_argument(
        "--initial_checkpoint_path",
        default=None,
        help="Required for simpleobstacle, bridge, and random single-phase runs.",
    )
    parser.add_argument("--manifest_path", default=None)
    parser.add_argument(
        "--common_overrides_json",
        default="{}",
        help=(
            "JSON dict of environment config overrides merged into every curriculum "
            "stage after the stage-specific overrides are constructed."
        ),
    )
    parser.add_argument("--dry_run", action="store_true")

    parser.add_argument("--flat_num_timesteps", type=int, default=2_000_000)
    parser.add_argument("--simpleobstacle_num_timesteps", type=int, default=10_000_000)
    parser.add_argument("--phase3_num_envs", type=int, default=2048)
    parser.add_argument("--phase3_num_eval_envs", type=int, default=128)
    parser.add_argument("--phase3_batch_size", type=int, default=1024)
    parser.add_argument("--phase3_unroll_length", type=int, default=30)
    parser.add_argument("--phase3_num_minibatches", type=int, default=32)
    parser.add_argument("--phase3_num_updates_per_batch", type=int, default=16)
    parser.add_argument("--phase3_num_evals", type=int, default=10)
    parser.add_argument("--phase3_episode_length", type=int, default=1000)
    parser.add_argument("--simpleobstacle_rollout_camera", default="cam_robot_follow")
    parser.add_argument("--phase3_rollout_width", type=int, default=640)
    parser.add_argument("--phase3_rollout_height", type=int, default=480)

    parser.add_argument("--bridge_obstacle_x_position", type=float, default=0.60)
    parser.add_argument("--bridge_success_x_margin", type=float, default=0.10)
    parser.add_argument("--bridge_terminate_on_success", type=_parse_bool, default=False)
    parser.add_argument("--bridge_success_bonus", type=float, default=0.0)
    parser.add_argument("--bridge_obstacle_local_windowed_reward", type=_parse_bool, default=True)
    parser.add_argument("--bridge_obstacle_local_activation_margin", type=float, default=0.08)
    parser.add_argument("--bridge_obstacle_local_extension_reward_weight", type=float, default=0.0)
    parser.add_argument("--bridge_post_success_progress_weight", type=float, default=5.0)
    parser.add_argument("--bridge_stall_penalty_after_success", type=_parse_bool, default=True)
    parser.add_argument("--bridge_stall_penalty_weight", type=float, default=0.08)
    parser.add_argument("--bridge_extension_retracted_penalty_weight", type=float, default=0.8)
    parser.add_argument(
        "--bridge_extension_retracted_penalty_outside_obstacle_only",
        type=_parse_bool,
        default=True,
    )
    parser.add_argument(
        "--bridge_heights",
        default="0.03,0.035,0.04,0.045,0.05,0.055,0.06,0.065",
    )
    parser.add_argument("--bridge_fixed_half_length", type=float, default=0.20)
    parser.add_argument("--bridge_gate_index", type=int, default=-1)
    parser.add_argument("--bridge_gate_threshold", type=float, default=0.3)
    parser.add_argument("--bridge_min_success_to_continue", type=float, default=0.5)
    parser.add_argument("--bridge_selection_strategy", choices=["best_success", "last_promoted"], default="last_promoted")
    parser.add_argument("--bridge_num_timesteps", type=int, default=4_000_000)
    parser.add_argument("--bridge_num_envs", type=int, default=2048)
    parser.add_argument("--bridge_num_eval_envs", type=int, default=128)
    parser.add_argument("--bridge_batch_size", type=int, default=1024)
    parser.add_argument("--bridge_unroll_length", type=int, default=30)
    parser.add_argument("--bridge_num_minibatches", type=int, default=32)
    parser.add_argument("--bridge_num_updates_per_batch", type=int, default=16)
    parser.add_argument("--bridge_num_evals", type=int, default=10)
    parser.add_argument("--bridge_episode_length", type=int, default=1000)
    parser.add_argument("--bridge_rollout_camera", default="cam_robot_follow")
    parser.add_argument("--bridge_rollout_width", type=int, default=640)
    parser.add_argument("--bridge_rollout_height", type=int, default=480)
    parser.add_argument("--bridge_eval_num_episodes", type=int, default=64)
    parser.add_argument("--bridge_eval_batch_size", type=int, default=32)

    parser.add_argument("--random_num_stages", type=int, default=6)
    parser.add_argument("--random_height_min", type=float, default=0.03)
    parser.add_argument("--random_height_max", type=float, default=0.065)
    parser.add_argument("--random_half_length_min", type=float, default=0.05)
    parser.add_argument("--random_half_length_max", type=float, default=0.50)
    parser.add_argument("--random_randomize_obstacle_x_min", type=float, default=0.45)
    parser.add_argument("--random_randomize_obstacle_x_max", type=float, default=0.85)
    parser.add_argument("--random_scenario_bank_size", type=int, default=100)
    parser.add_argument(
        "--random_scenario_order",
        choices=["random", "easy_to_hard", "hard_to_easy"],
        default="random",
    )
    parser.add_argument(
        "--random_stage_sampling",
        choices=["ordered", "random_without_replacement", "random_with_replacement"],
        default="random_without_replacement",
    )
    parser.add_argument("--random_scenario_bank_path", default=None)
    parser.add_argument("--random_success_x_margin", type=float, default=0.10)
    parser.add_argument("--random_terminate_on_success", type=_parse_bool, default=False)
    parser.add_argument("--random_success_bonus", type=float, default=0.0)
    parser.add_argument("--random_obstacle_local_windowed_reward", type=_parse_bool, default=True)
    parser.add_argument("--random_obstacle_local_activation_margin", type=float, default=0.08)
    parser.add_argument("--random_obstacle_local_extension_reward_weight", type=float, default=0.0)
    parser.add_argument("--random_post_success_progress_weight", type=float, default=5.0)
    parser.add_argument("--random_stall_penalty_after_success", type=_parse_bool, default=True)
    parser.add_argument("--random_stall_penalty_weight", type=float, default=0.08)
    parser.add_argument("--random_extension_retracted_penalty_weight", type=float, default=0.8)
    parser.add_argument(
        "--random_extension_retracted_penalty_outside_obstacle_only",
        type=_parse_bool,
        default=True,
    )
    parser.add_argument("--random_min_success_to_continue", type=float, default=0.5)
    parser.add_argument("--random_selection_strategy", choices=["best_success", "last_promoted"], default="last_promoted")
    parser.add_argument("--random_num_timesteps", type=int, default=2_000_000)
    parser.add_argument("--random_num_envs", type=int, default=2048)
    parser.add_argument("--random_num_eval_envs", type=int, default=128)
    parser.add_argument("--random_batch_size", type=int, default=1024)
    parser.add_argument("--random_unroll_length", type=int, default=30)
    parser.add_argument("--random_num_minibatches", type=int, default=32)
    parser.add_argument("--random_num_updates_per_batch", type=int, default=16)
    parser.add_argument("--random_num_evals", type=int, default=10)
    parser.add_argument("--random_episode_length", type=int, default=1000)
    parser.add_argument("--random_rollout_camera", default="cam_robot_follow")
    parser.add_argument("--random_rollout_width", type=int, default=640)
    parser.add_argument("--random_rollout_height", type=int, default=480)
    parser.add_argument("--random_eval_num_episodes", type=int, default=64)
    parser.add_argument("--random_eval_batch_size", type=int, default=32)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    common_overrides = json.loads(args.common_overrides_json)
    if not isinstance(common_overrides, dict):
        raise ValueError("--common_overrides_json must decode to a JSON object.")
    args.common_overrides = common_overrides
    logs_root = Path("logs").resolve()
    logs_root.mkdir(parents=True, exist_ok=True)
    env = _runtime_env()
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")

    if args.phase == PHASE_FLAT:
        result = _train_single_stage(
            args=args,
            env=env,
            logs_root=logs_root,
            phase_name=PHASE_FLAT,
            suffix=f"{args.suffix_prefix}-{PHASE_FLAT}",
            overrides=_default_flat_overrides(),
            params=_flat_params(args),
            load_checkpoint_path=None,
            phase_title="Flat",
            dry_run=args.dry_run,
        )
        manifest = {
            "created_at": dt.datetime.now().isoformat(),
            "phase": PHASE_FLAT,
            "common_overrides": args.common_overrides,
            "result": result,
        }
    elif args.phase == PHASE_SIMPLE:
        initial_checkpoint = _require_initial_checkpoint(args)
        result = _train_single_stage(
            args=args,
            env=env,
            logs_root=logs_root,
            phase_name=PHASE_SIMPLE,
            suffix=f"{args.suffix_prefix}-{PHASE_SIMPLE}",
            overrides=_default_simpleobstacle_overrides(),
            params=_simple_params(args),
            load_checkpoint_path=initial_checkpoint,
            phase_title="SimpleObstacle",
            dry_run=args.dry_run,
        )
        manifest = {
            "created_at": dt.datetime.now().isoformat(),
            "phase": PHASE_SIMPLE,
            "initial_checkpoint_path": str(initial_checkpoint),
            "common_overrides": args.common_overrides,
            "result": result,
        }
    elif args.phase == PHASE_BRIDGE:
        initial_checkpoint = _require_initial_checkpoint(args)
        bridge_result = _run_bridge_phase(
            args=args,
            env=env,
            logs_root=logs_root,
            initial_checkpoint_path=initial_checkpoint,
            suffix_prefix=f"{args.suffix_prefix}-{PHASE_BRIDGE}",
            dry_run=args.dry_run,
        )
        manifest = {
            "created_at": dt.datetime.now().isoformat(),
            "phase": PHASE_BRIDGE,
            "common_overrides": args.common_overrides,
            "result": bridge_result,
        }
    elif args.phase == PHASE_RANDOM:
        initial_checkpoint = _require_initial_checkpoint(args)
        random_result = _run_random_phase(
            args=args,
            env=env,
            logs_root=logs_root,
            initial_checkpoint_path=initial_checkpoint,
            suffix_prefix=f"{args.suffix_prefix}-{PHASE_RANDOM}",
            dry_run=args.dry_run,
        )
        manifest = {
            "created_at": dt.datetime.now().isoformat(),
            "phase": PHASE_RANDOM,
            "common_overrides": args.common_overrides,
            "result": random_result,
        }
    else:
        flat_result = _train_single_stage(
            args=args,
            env=env,
            logs_root=logs_root,
            phase_name=PHASE_FLAT,
            suffix=f"{args.suffix_prefix}-{PHASE_FLAT}",
            overrides=_default_flat_overrides(),
            params=_flat_params(args),
            load_checkpoint_path=None,
            phase_title="Flat",
            dry_run=args.dry_run,
        )
        flat_ckpt = (
            Path("<flat_ckpt_dir>")
            if args.dry_run
            else Path(str(flat_result["ckpt_dir"])).resolve()
        )

        simple_result = _train_single_stage(
            args=args,
            env=env,
            logs_root=logs_root,
            phase_name=PHASE_SIMPLE,
            suffix=f"{args.suffix_prefix}-{PHASE_SIMPLE}",
            overrides=_default_simpleobstacle_overrides(),
            params=_simple_params(args),
            load_checkpoint_path=flat_ckpt,
            phase_title="SimpleObstacle",
            dry_run=args.dry_run,
        )
        simple_ckpt = (
            Path("<simpleobstacle_ckpt_dir>")
            if args.dry_run
            else Path(str(simple_result["ckpt_dir"])).resolve()
        )

        bridge_result = _run_bridge_phase(
            args=args,
            env=env,
            logs_root=logs_root,
            initial_checkpoint_path=simple_ckpt,
            suffix_prefix=f"{args.suffix_prefix}-{PHASE_BRIDGE}",
            dry_run=args.dry_run,
        )
        bridge_ckpt = (
            Path("<bridge_ckpt_dir>")
            if args.dry_run
            else Path(bridge_result["selected_checkpoint_path"]).resolve()
            if bridge_result.get("selected_checkpoint_path")
            else simple_ckpt
        )

        random_result = _run_random_phase(
            args=args,
            env=env,
            logs_root=logs_root,
            initial_checkpoint_path=bridge_ckpt,
            suffix_prefix=f"{args.suffix_prefix}-{PHASE_RANDOM}",
            dry_run=args.dry_run,
        )

        manifest = {
            "created_at": dt.datetime.now().isoformat(),
            "phase": PHASE_FULL,
            "pipeline": [PHASE_FLAT, PHASE_SIMPLE, PHASE_BRIDGE, PHASE_RANDOM],
            "notes": {
                "bridge_start_for_random_selection_strategy": args.bridge_selection_strategy,
                "final_random_selection_strategy": args.random_selection_strategy,
                "training_defaults_source": (
                    "Copied from the successful individual flat/simpleobstacle/bridge/random jobs."
                ),
            },
            "common_overrides": args.common_overrides,
            "flat": flat_result,
            "simpleobstacle": simple_result,
            "bridge": bridge_result,
            "random": random_result,
        }

    manifest_path = (
        Path(args.manifest_path).resolve()
        if args.manifest_path
        else logs_root / f"{args.suffix_prefix}-{args.phase}-{timestamp}.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[DONE] Wrote manifest: {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
