#!/usr/bin/env python3
"""Run Phase 3 TWMR curriculum with automatic checkpoint handoff."""

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

STAGE_A = "stageA-flat"
STAGE_B = "stageB-box-easy"
STAGE_C = "stageC-box-target"
ALL_STAGE_NAMES = (STAGE_A, STAGE_B, STAGE_C)


@dataclass(frozen=True)
class Stage:
    name: str
    overrides: dict[str, Any]
    num_timesteps: int
    num_envs: int
    num_eval_envs: int
    batch_size: int
    unroll_length: int
    num_minibatches: int
    num_updates_per_batch: int
    num_evals: int
    episode_length: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 3 curriculum (flat -> easy box -> target box)."
    )
    parser.add_argument(
        "--python",
        default="./.venv/bin/python",
        help="Python executable used to run train_jax_ppo.py",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for all stages",
    )
    parser.add_argument(
        "--suffix_prefix",
        default="phase3",
        help="Prefix used in experiment suffix for each stage",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use tiny settings for fast validation.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing.",
    )
    parser.add_argument(
        "--skip_stage_c",
        action="store_true",
        help="Skip target box stage.",
    )
    parser.add_argument(
        "--stages",
        default=",".join(ALL_STAGE_NAMES),
        help=(
            "Comma-separated stage names to run in canonical order. "
            f"Valid: {', '.join(ALL_STAGE_NAMES)}"
        ),
    )
    parser.add_argument(
        "--initial_checkpoint_path",
        default=None,
        help="Optional checkpoint directory/path to restore for the first selected stage.",
    )
    parser.add_argument("--num_timesteps_override", type=int, default=None)
    parser.add_argument("--num_envs_override", type=int, default=None)
    parser.add_argument("--num_eval_envs_override", type=int, default=None)
    parser.add_argument("--batch_size_override", type=int, default=None)
    parser.add_argument("--unroll_length_override", type=int, default=None)
    parser.add_argument("--num_minibatches_override", type=int, default=None)
    parser.add_argument("--num_updates_per_batch_override", type=int, default=None)
    parser.add_argument("--num_evals_override", type=int, default=None)
    parser.add_argument("--episode_length_override", type=int, default=None)
    parser.add_argument(
        "--manifest_path",
        default=None,
        help="Optional output JSON file for curriculum stage summary.",
    )
    return parser.parse_args()


def _build_stages(smoke: bool) -> list[Stage]:
    if smoke:
        return [
            Stage(
                name=STAGE_A,
                overrides={"xml_variant": "flat"},
                num_timesteps=4096,
                num_envs=64,
                num_eval_envs=8,
                batch_size=64,
                unroll_length=8,
                num_minibatches=4,
                num_updates_per_batch=2,
                num_evals=1,
                episode_length=64,
            ),
            Stage(
                name=STAGE_B,
                overrides={
                    "xml_variant": "box",
                    "obstacle_height": 0.03,
                    "obstacle_x_position": 0.55,
                    "obstacle_half_length": 0.15,
                    "success_x_margin": 0.08,
                },
                num_timesteps=4096,
                num_envs=64,
                num_eval_envs=8,
                batch_size=64,
                unroll_length=8,
                num_minibatches=4,
                num_updates_per_batch=2,
                num_evals=1,
                episode_length=64,
            ),
            Stage(
                name=STAGE_C,
                overrides={
                    "xml_variant": "box",
                    "obstacle_height": 0.06,
                    "obstacle_x_position": 0.60,
                    "obstacle_half_length": 0.20,
                    "success_x_margin": 0.10,
                },
                num_timesteps=4096,
                num_envs=64,
                num_eval_envs=8,
                batch_size=64,
                unroll_length=8,
                num_minibatches=4,
                num_updates_per_batch=2,
                num_evals=1,
                episode_length=64,
            ),
        ]

    return [
        Stage(
            name=STAGE_A,
            overrides={"xml_variant": "flat"},
            num_timesteps=5_000_000,
            num_envs=2048,
            num_eval_envs=128,
            batch_size=1024,
            unroll_length=30,
            num_minibatches=32,
            num_updates_per_batch=16,
            num_evals=10,
            episode_length=1000,
        ),
        Stage(
            name=STAGE_B,
            overrides={
                "xml_variant": "box",
                "obstacle_height": 0.03,
                "obstacle_x_position": 0.55,
                "obstacle_half_length": 0.15,
                "success_x_margin": 0.08,
            },
            num_timesteps=5_000_000,
            num_envs=2048,
            num_eval_envs=128,
            batch_size=1024,
            unroll_length=30,
            num_minibatches=32,
            num_updates_per_batch=16,
            num_evals=10,
            episode_length=1000,
        ),
        Stage(
            name=STAGE_C,
            overrides={
                "xml_variant": "box",
                "obstacle_height": 0.06,
                "obstacle_x_position": 0.60,
                "obstacle_half_length": 0.20,
                "success_x_margin": 0.10,
            },
            num_timesteps=5_000_000,
            num_envs=2048,
            num_eval_envs=128,
            batch_size=1024,
            unroll_length=30,
            num_minibatches=32,
            num_updates_per_batch=16,
            num_evals=10,
            episode_length=1000,
        ),
    ]


def _apply_global_overrides(stage: Stage, args: argparse.Namespace) -> Stage:
    return Stage(
        name=stage.name,
        overrides=stage.overrides,
        num_timesteps=args.num_timesteps_override
        if args.num_timesteps_override is not None
        else stage.num_timesteps,
        num_envs=args.num_envs_override
        if args.num_envs_override is not None
        else stage.num_envs,
        num_eval_envs=args.num_eval_envs_override
        if args.num_eval_envs_override is not None
        else stage.num_eval_envs,
        batch_size=args.batch_size_override
        if args.batch_size_override is not None
        else stage.batch_size,
        unroll_length=args.unroll_length_override
        if args.unroll_length_override is not None
        else stage.unroll_length,
        num_minibatches=args.num_minibatches_override
        if args.num_minibatches_override is not None
        else stage.num_minibatches,
        num_updates_per_batch=args.num_updates_per_batch_override
        if args.num_updates_per_batch_override is not None
        else stage.num_updates_per_batch,
        num_evals=args.num_evals_override
        if args.num_evals_override is not None
        else stage.num_evals,
        episode_length=args.episode_length_override
        if args.episode_length_override is not None
        else stage.episode_length,
    )


def _select_stages(stages: list[Stage], args: argparse.Namespace) -> list[Stage]:
    requested = [name.strip() for name in args.stages.split(",") if name.strip()]
    invalid = [name for name in requested if name not in ALL_STAGE_NAMES]
    if invalid:
        allowed = ", ".join(ALL_STAGE_NAMES)
        raise ValueError(f"Invalid stage names: {invalid}. Allowed: {allowed}")

    requested_set = set(requested)
    selected = [stage for stage in stages if stage.name in requested_set]
    if args.skip_stage_c:
        selected = [stage for stage in selected if stage.name != STAGE_C]

    if not selected:
        raise ValueError("No stages selected. Update --stages / --skip_stage_c.")
    return selected


def _latest_log_with_suffix(logs_root: Path, suffix: str) -> Path:
    candidates = [p for p in logs_root.iterdir() if p.is_dir() and p.name.endswith(f"-{suffix}")]
    if not candidates:
        raise RuntimeError(f"No log directory found for suffix '{suffix}'")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _stage_command(
    python_exec: str,
    stage: Stage,
    suffix: str,
    seed: int,
    load_checkpoint_path: Path | None,
) -> list[str]:
    cmd = [
        python_exec,
        "train_jax_ppo.py",
        "--env_name=TransformableWheelMobileRobot",
        f"--playground_config_overrides={json.dumps(stage.overrides, separators=(',', ':'))}",
        f"--num_timesteps={stage.num_timesteps}",
        f"--num_envs={stage.num_envs}",
        f"--num_eval_envs={stage.num_eval_envs}",
        f"--batch_size={stage.batch_size}",
        f"--unroll_length={stage.unroll_length}",
        f"--num_minibatches={stage.num_minibatches}",
        f"--num_updates_per_batch={stage.num_updates_per_batch}",
        f"--num_evals={stage.num_evals}",
        f"--episode_length={stage.episode_length}",
        f"--seed={seed}",
        f"--suffix={suffix}",
        "--use_wandb=false",
        "--use_tb=false",
    ]
    if load_checkpoint_path is not None:
        cmd.append(f"--load_checkpoint_path={load_checkpoint_path}")
    return cmd


def _run_cmd(cmd: list[str], env: dict[str, str], dry_run: bool) -> None:
    printable = " ".join(cmd)
    print(f"[CMD] {printable}", flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True, env=env)


def main() -> int:
    args = _parse_args()
    logs_root = Path("logs").resolve()
    logs_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    ffmpeg_bin = Path.home() / ".pixi" / "envs" / "ffmpeg" / "bin"
    env["PATH"] = f"{ffmpeg_bin}:{env.get('PATH', '')}"

    stages = _build_stages(smoke=args.smoke)
    stages = _select_stages(stages, args)
    stages = [_apply_global_overrides(stage, args) for stage in stages]

    checkpoint_from_prev_stage: Path | None = (
        Path(args.initial_checkpoint_path).resolve()
        if args.initial_checkpoint_path is not None
        else None
    )
    summary: list[dict[str, str | int | dict[str, Any]]] = []

    for stage in stages:
        suffix = f"{args.suffix_prefix}-{stage.name}"
        cmd = _stage_command(
            python_exec=args.python,
            stage=stage,
            suffix=suffix,
            seed=args.seed,
            load_checkpoint_path=checkpoint_from_prev_stage,
        )
        _run_cmd(cmd, env=env, dry_run=args.dry_run)

        if args.dry_run:
            continue

        exp_dir = _latest_log_with_suffix(logs_root, suffix)
        ckpt_dir = exp_dir / "checkpoints"
        if not ckpt_dir.exists():
            raise RuntimeError(f"Missing checkpoints directory after stage '{stage.name}': {ckpt_dir}")
        rollout_src = Path("rollout0.mp4").resolve()
        if rollout_src.exists():
            rollout_dst = exp_dir / "rollout0.mp4"
            shutil.copy2(rollout_src, rollout_dst)
            print(f"  rollout={rollout_dst}", flush=True)

        print(f"[STAGE COMPLETE] {stage.name}", flush=True)
        print(f"  exp_dir={exp_dir}", flush=True)
        print(f"  ckpt_dir={ckpt_dir}", flush=True)

        summary.append(
            {
                "stage": stage.name,
                "exp_dir": str(exp_dir),
                "ckpt_dir": str(ckpt_dir),
                "restore_from": str(checkpoint_from_prev_stage)
                if checkpoint_from_prev_stage is not None
                else "",
                "num_timesteps": stage.num_timesteps,
                "num_envs": stage.num_envs,
                "num_eval_envs": stage.num_eval_envs,
                "batch_size": stage.batch_size,
                "unroll_length": stage.unroll_length,
                "num_minibatches": stage.num_minibatches,
                "num_updates_per_batch": stage.num_updates_per_batch,
                "num_evals": stage.num_evals,
                "episode_length": stage.episode_length,
                "overrides": stage.overrides,
            }
        )
        checkpoint_from_prev_stage = ckpt_dir

    if args.dry_run:
        print("[DONE] Dry run complete.", flush=True)
        return 0

    manifest_path = Path(args.manifest_path).resolve() if args.manifest_path else None
    if manifest_path is None:
        ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        manifest_path = logs_root / f"curriculum-{args.suffix_prefix}-{ts}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": dt.datetime.now().isoformat(),
        "seed": args.seed,
        "smoke": args.smoke,
        "suffix_prefix": args.suffix_prefix,
        "stages": summary,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("[DONE] Curriculum complete.", flush=True)
    for item in summary:
        print(f"  {item['stage']}:", flush=True)
        print(f"    exp_dir={item['exp_dir']}", flush=True)
        print(f"    ckpt_dir={item['ckpt_dir']}", flush=True)
    print(f"  manifest={manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
