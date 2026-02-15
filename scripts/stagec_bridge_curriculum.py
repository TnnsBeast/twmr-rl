#!/usr/bin/env python3
"""Run Stage-C bridge curricula with checkpoint handoff and eval gating."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrainParams:
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
    rollout_camera: str | None
    rollout_width: int
    rollout_height: int


@dataclass(frozen=True)
class BridgeStage:
    index: int
    height: float
    half_length: float
    overrides: dict[str, Any]
    suffix: str


def _parse_float_list(raw: str) -> list[float]:
    values = [x.strip() for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("Expected at least one numeric value.")
    return [float(v) for v in values]


def _tag(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a Stage-C bridge curriculum (height or geometry) with "
            "automatic checkpoint handoff and evaluation-based promotion gates."
        )
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["height", "geometry"],
        help="Bridge mode: vary obstacle height or obstacle half-length.",
    )
    parser.add_argument(
        "--python",
        default="./.venv/bin/python",
        help="Python executable used to run train/eval scripts.",
    )
    parser.add_argument(
        "--initial_checkpoint_path",
        required=True,
        help="Checkpoint directory/path used to restore the first bridge stage.",
    )
    parser.add_argument(
        "--suffix_prefix",
        default="stagec-bridge",
        help="Prefix used in experiment suffix for each stage.",
    )
    parser.add_argument(
        "--obstacle_x_position",
        type=float,
        default=0.60,
        help="Obstacle x-position for all bridge stages.",
    )
    parser.add_argument(
        "--success_x_margin",
        type=float,
        default=0.10,
        help="Success margin for all bridge stages.",
    )
    parser.add_argument(
        "--heights",
        default="0.045,0.05,0.055,0.06",
        help="Comma-separated heights for --mode=height.",
    )
    parser.add_argument(
        "--half_lengths",
        default="0.12,0.15,0.18,0.20",
        help="Comma-separated half-lengths for --mode=geometry.",
    )
    parser.add_argument(
        "--fixed_height",
        type=float,
        default=0.06,
        help="Fixed height for --mode=geometry.",
    )
    parser.add_argument(
        "--fixed_half_length",
        type=float,
        default=0.20,
        help="Fixed half-length for --mode=height.",
    )
    parser.add_argument(
        "--gate_index",
        type=int,
        default=-1,
        help=(
            "0-based stage index for strong promotion gate. "
            "Set to -1 to disable strong gate."
        ),
    )
    parser.add_argument(
        "--gate_threshold",
        type=float,
        default=0.3,
        help="Success-rate threshold for the strong promotion gate stage.",
    )
    parser.add_argument(
        "--min_success_to_continue",
        type=float,
        default=0.0,
        help="Baseline success-rate threshold required to continue to next stage.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for all stages and evaluation.",
    )
    parser.add_argument(
        "--num_timesteps",
        type=int,
        default=1_000_000,
        help="PPO timesteps per stage.",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=512,
        help="Number of parallel train environments.",
    )
    parser.add_argument(
        "--num_eval_envs",
        type=int,
        default=64,
        help="Number of eval environments during PPO train loop.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="PPO batch size.",
    )
    parser.add_argument(
        "--unroll_length",
        type=int,
        default=20,
        help="PPO unroll length.",
    )
    parser.add_argument(
        "--num_minibatches",
        type=int,
        default=16,
        help="PPO minibatches.",
    )
    parser.add_argument(
        "--num_updates_per_batch",
        type=int,
        default=8,
        help="PPO updates per batch.",
    )
    parser.add_argument(
        "--num_evals",
        type=int,
        default=8,
        help="PPO eval intervals.",
    )
    parser.add_argument(
        "--episode_length",
        type=int,
        default=1000,
        help="Episode length for train and checkpoint eval.",
    )
    parser.add_argument(
        "--rollout_camera",
        default="cam_robot_follow",
        help="Rollout camera name/id passed to train_jax_ppo.py.",
    )
    parser.add_argument(
        "--rollout_width",
        type=int,
        default=640,
        help="Rollout video width passed to train_jax_ppo.py.",
    )
    parser.add_argument(
        "--rollout_height",
        type=int,
        default=480,
        help="Rollout video height passed to train_jax_ppo.py.",
    )
    parser.add_argument(
        "--eval_num_episodes",
        type=int,
        default=64,
        help="Number of episodes for checkpoint promotion eval.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=32,
        help="Parallel batch size for checkpoint promotion eval.",
    )
    parser.add_argument(
        "--manifest_path",
        default=None,
        help="Optional output JSON path for stage summary.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing.",
    )
    return parser.parse_args()


def _build_stages(args: argparse.Namespace) -> list[BridgeStage]:
    if args.mode == "height":
        heights = _parse_float_list(args.heights)
        return [
            BridgeStage(
                index=i,
                height=h,
                half_length=args.fixed_half_length,
                overrides={
                    "xml_variant": "box",
                    "obstacle_height": h,
                    "obstacle_x_position": args.obstacle_x_position,
                    "obstacle_half_length": args.fixed_half_length,
                    "success_x_margin": args.success_x_margin,
                },
                suffix=(
                    f"{args.suffix_prefix}-h{_tag(h)}-l{_tag(args.fixed_half_length)}"
                ),
            )
            for i, h in enumerate(heights)
        ]

    half_lengths = _parse_float_list(args.half_lengths)
    return [
        BridgeStage(
            index=i,
            height=args.fixed_height,
            half_length=half_length,
            overrides={
                "xml_variant": "box",
                "obstacle_height": args.fixed_height,
                "obstacle_x_position": args.obstacle_x_position,
                "obstacle_half_length": half_length,
                "success_x_margin": args.success_x_margin,
            },
            suffix=f"{args.suffix_prefix}-h{_tag(args.fixed_height)}-l{_tag(half_length)}",
        )
        for i, half_length in enumerate(half_lengths)
    ]


def _latest_log_with_suffix(logs_root: Path, suffix: str) -> Path:
    candidates = [
        p for p in logs_root.iterdir() if p.is_dir() and p.name.endswith(f"-{suffix}")
    ]
    if not candidates:
        raise RuntimeError(f"No log directory found for suffix '{suffix}'")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _train_cmd(
    python_exec: str,
    stage: BridgeStage,
    params: TrainParams,
    load_checkpoint_path: Path,
) -> list[str]:
    cmd = [
        python_exec,
        "train_jax_ppo.py",
        "--env_name=TransformableWheelMobileRobot",
        f"--playground_config_overrides={json.dumps(stage.overrides, separators=(',', ':'))}",
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
        f"--suffix={stage.suffix}",
        "--use_wandb=false",
        "--use_tb=false",
        f"--load_checkpoint_path={load_checkpoint_path}",
    ]
    if params.rollout_camera:
        cmd.append(f"--rollout_camera={params.rollout_camera}")
    cmd.append(f"--rollout_width={params.rollout_width}")
    cmd.append(f"--rollout_height={params.rollout_height}")
    return cmd


def _eval_cmd(
    python_exec: str,
    checkpoint_path: Path,
    stage: BridgeStage,
    params: TrainParams,
    eval_num_episodes: int,
    eval_batch_size: int,
    output_json: Path,
) -> list[str]:
    return [
        python_exec,
        "scripts/eval_twmr_checkpoint.py",
        f"--checkpoint_path={checkpoint_path}",
        f"--playground_config_overrides={json.dumps(stage.overrides, separators=(',', ':'))}",
        f"--num_episodes={eval_num_episodes}",
        f"--episode_length={params.episode_length}",
        f"--batch_size={eval_batch_size}",
        f"--seed={params.seed}",
        f"--output_json={output_json}",
    ]


def _run_cmd(cmd: list[str], env: dict[str, str], dry_run: bool) -> None:
    printable = " ".join(cmd)
    print(f"[CMD] {printable}", flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True, env=env)


def _promotion_threshold(args: argparse.Namespace, stage_index: int) -> float:
    threshold = args.min_success_to_continue
    if args.gate_index >= 0 and stage_index == args.gate_index:
        threshold = max(threshold, args.gate_threshold)
    return threshold


def main() -> int:
    args = _parse_args()
    logs_root = Path("logs").resolve()
    logs_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    ffmpeg_bin = Path.home() / ".pixi" / "envs" / "ffmpeg" / "bin"
    env["PATH"] = f"{ffmpeg_bin}:{env.get('PATH', '')}"
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")

    stages = _build_stages(args)
    train_params = TrainParams(
        num_timesteps=args.num_timesteps,
        num_envs=args.num_envs,
        num_eval_envs=args.num_eval_envs,
        batch_size=args.batch_size,
        unroll_length=args.unroll_length,
        num_minibatches=args.num_minibatches,
        num_updates_per_batch=args.num_updates_per_batch,
        num_evals=args.num_evals,
        episode_length=args.episode_length,
        seed=args.seed,
        rollout_camera=args.rollout_camera,
        rollout_width=args.rollout_width,
        rollout_height=args.rollout_height,
    )

    checkpoint_from_prev_stage = Path(args.initial_checkpoint_path).resolve()
    if not args.dry_run and not checkpoint_from_prev_stage.exists():
        raise FileNotFoundError(
            f"Initial checkpoint path not found: {checkpoint_from_prev_stage}"
        )

    summary: list[dict[str, Any]] = []
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    stopped_early = False
    stop_reason = ""

    for stage in stages:
        print(
            (
                f"[STAGE] idx={stage.index} mode={args.mode} "
                f"height={stage.height:.3f} half_length={stage.half_length:.3f}"
            ),
            flush=True,
        )

        train_cmd = _train_cmd(
            python_exec=args.python,
            stage=stage,
            params=train_params,
            load_checkpoint_path=checkpoint_from_prev_stage,
        )
        _run_cmd(train_cmd, env=env, dry_run=args.dry_run)

        if args.dry_run:
            summary.append(
                {
                    "stage_index": stage.index,
                    "height": stage.height,
                    "half_length": stage.half_length,
                    "suffix": stage.suffix,
                    "restore_from": str(checkpoint_from_prev_stage),
                    "dry_run": True,
                }
            )
            continue

        exp_dir = _latest_log_with_suffix(logs_root, stage.suffix)
        ckpt_dir = exp_dir / "checkpoints"
        if not ckpt_dir.exists():
            raise RuntimeError(f"Missing checkpoints directory for {stage.suffix}: {ckpt_dir}")

        rollout_src = Path("rollout0.mp4").resolve()
        if rollout_src.exists():
            rollout_dst = exp_dir / "rollout0.mp4"
            shutil.copy2(rollout_src, rollout_dst)
            print(f"  rollout={rollout_dst}", flush=True)

        eval_json = logs_root / (
            f"eval-{args.suffix_prefix}-stage{stage.index:02d}-"
            f"h{_tag(stage.height)}-l{_tag(stage.half_length)}-"
            f"{args.eval_num_episodes}ep-{timestamp}.json"
        )
        eval_cmd = _eval_cmd(
            python_exec=args.python,
            checkpoint_path=ckpt_dir,
            stage=stage,
            params=train_params,
            eval_num_episodes=args.eval_num_episodes,
            eval_batch_size=args.eval_batch_size,
            output_json=eval_json,
        )
        _run_cmd(eval_cmd, env=env, dry_run=False)
        eval_data = json.loads(eval_json.read_text(encoding="utf-8"))

        success_rate = float(eval_data["success_rate"])
        threshold = _promotion_threshold(args, stage.index)
        promoted = success_rate >= threshold
        print(
            (
                f"  eval success_rate={success_rate:.3f} "
                f"threshold={threshold:.3f} promoted={promoted}"
            ),
            flush=True,
        )

        summary.append(
            {
                "stage_index": stage.index,
                "height": stage.height,
                "half_length": stage.half_length,
                "suffix": stage.suffix,
                "restore_from": str(checkpoint_from_prev_stage),
                "exp_dir": str(exp_dir),
                "ckpt_dir": str(ckpt_dir),
                "eval_json": str(eval_json),
                "success_rate": success_rate,
                "mean_return": float(eval_data["mean_return"]),
                "mean_x_distance": float(eval_data["mean_x_distance"]),
                "promotion_threshold": threshold,
                "promoted": promoted,
            }
        )

        if not promoted:
            stopped_early = True
            stop_reason = (
                f"Stage {stage.index} gate failed: success_rate={success_rate:.3f} "
                f"< threshold={threshold:.3f}"
            )
            print(f"[STOP] {stop_reason}", flush=True)
            break

        checkpoint_from_prev_stage = ckpt_dir

    manifest_path = Path(args.manifest_path).resolve() if args.manifest_path else None
    if manifest_path is None:
        manifest_path = logs_root / f"bridge-{args.suffix_prefix}-{timestamp}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": dt.datetime.now().isoformat(),
        "mode": args.mode,
        "initial_checkpoint_path": str(Path(args.initial_checkpoint_path).resolve()),
        "train_params": vars(train_params),
        "obstacle_x_position": args.obstacle_x_position,
        "success_x_margin": args.success_x_margin,
        "min_success_to_continue": args.min_success_to_continue,
        "gate_index": args.gate_index,
        "gate_threshold": args.gate_threshold,
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "stages": summary,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[DONE] Wrote manifest: {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
