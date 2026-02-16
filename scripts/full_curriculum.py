#!/usr/bin/env python3
"""Run the full TWMR training pipeline through rollout rendering."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any

STAGE_B = "stageB-box-easy"
STAGE_C = "stageC-box-target"
ENV_NAME = "TransformableWheelMobileRobot"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full TWMR pipeline: Stage A/B/C curriculum, optional Stage-C "
            "bridge, dual-obstacle training, checkpoint evaluation, and rollout rendering."
        )
    )
    parser.add_argument(
        "--python",
        default="./.venv/bin/python",
        help="Python executable used to run train/eval scripts.",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument(
        "--suffix_prefix",
        default="multiobs-full",
        help="Prefix used to name all generated runs and manifests.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing.",
    )
    parser.add_argument(
        "--manifest_path",
        default=None,
        help="Optional output path for the top-level full-curriculum manifest.",
    )

    # Stage A/B/C curriculum settings (scripts/phase3_curriculum.py)
    parser.add_argument(
        "--phase3_initial_checkpoint_path",
        default=None,
        help="Optional checkpoint to restore for the first selected Stage A/B/C run.",
    )
    parser.add_argument("--phase3_num_timesteps", type=int, default=1_000_000)
    parser.add_argument("--phase3_num_envs", type=int, default=512)
    parser.add_argument("--phase3_num_eval_envs", type=int, default=64)
    parser.add_argument("--phase3_batch_size", type=int, default=512)
    parser.add_argument("--phase3_unroll_length", type=int, default=20)
    parser.add_argument("--phase3_num_minibatches", type=int, default=16)
    parser.add_argument("--phase3_num_updates_per_batch", type=int, default=8)
    parser.add_argument("--phase3_num_evals", type=int, default=8)
    parser.add_argument("--phase3_episode_length", type=int, default=1000)
    parser.add_argument(
        "--phase3_manifest_path",
        default=None,
        help="Optional explicit path for the Stage A/B/C manifest JSON.",
    )

    # Stage-C bridge settings (scripts/stagec_bridge_curriculum.py)
    parser.add_argument(
        "--skip_bridge",
        action="store_true",
        help="Skip Stage-C bridge and continue from Stage C checkpoint directly.",
    )
    parser.add_argument(
        "--bridge_mode",
        choices=["height", "geometry"],
        default="height",
        help="Bridge mode to run if bridge is enabled.",
    )
    parser.add_argument(
        "--bridge_initial_stage",
        choices=[STAGE_B, STAGE_C],
        default=STAGE_B,
        help="Which Stage A/B/C checkpoint to use as bridge starting point.",
    )
    parser.add_argument("--bridge_obstacle_x_position", type=float, default=0.60)
    parser.add_argument("--bridge_success_x_margin", type=float, default=0.10)
    parser.add_argument("--bridge_heights", default="0.045,0.05,0.055,0.06")
    parser.add_argument("--bridge_half_lengths", default="0.12,0.15,0.18,0.20")
    parser.add_argument("--bridge_fixed_height", type=float, default=0.06)
    parser.add_argument("--bridge_fixed_half_length", type=float, default=0.20)
    parser.add_argument("--bridge_gate_index", type=int, default=1)
    parser.add_argument("--bridge_gate_threshold", type=float, default=0.3)
    parser.add_argument("--bridge_min_success_to_continue", type=float, default=0.015625)
    parser.add_argument("--bridge_num_timesteps", type=int, default=1_500_000)
    parser.add_argument("--bridge_num_envs", type=int, default=512)
    parser.add_argument("--bridge_num_eval_envs", type=int, default=64)
    parser.add_argument("--bridge_batch_size", type=int, default=512)
    parser.add_argument("--bridge_unroll_length", type=int, default=20)
    parser.add_argument("--bridge_num_minibatches", type=int, default=16)
    parser.add_argument("--bridge_num_updates_per_batch", type=int, default=8)
    parser.add_argument("--bridge_num_evals", type=int, default=8)
    parser.add_argument("--bridge_episode_length", type=int, default=1000)
    parser.add_argument("--bridge_eval_num_episodes", type=int, default=64)
    parser.add_argument("--bridge_eval_batch_size", type=int, default=32)
    parser.add_argument("--bridge_rollout_camera", default="cam_obstacle_close")
    parser.add_argument("--bridge_rollout_width", type=int, default=640)
    parser.add_argument("--bridge_rollout_height", type=int, default=480)
    parser.add_argument(
        "--bridge_manifest_path",
        default=None,
        help="Optional explicit path for bridge manifest JSON.",
    )

    # Dual-obstacle training settings (train_jax_ppo.py)
    parser.add_argument(
        "--dual_start_checkpoint_path",
        default=None,
        help=(
            "Optional explicit checkpoint to start dual-obstacle training from. "
            "If set, bridge output selection is ignored."
        ),
    )
    parser.add_argument("--dual_num_timesteps", type=int, default=10_000_000)
    parser.add_argument("--dual_num_envs", type=int, default=512)
    parser.add_argument("--dual_num_eval_envs", type=int, default=64)
    parser.add_argument("--dual_batch_size", type=int, default=512)
    parser.add_argument("--dual_unroll_length", type=int, default=20)
    parser.add_argument("--dual_num_minibatches", type=int, default=16)
    parser.add_argument("--dual_num_updates_per_batch", type=int, default=8)
    parser.add_argument("--dual_num_evals", type=int, default=12)
    parser.add_argument("--dual_episode_length", type=int, default=1600)
    parser.add_argument("--dual_num_videos", type=int, default=4)
    parser.add_argument("--dual_rollout_camera", default="cam_robot_follow")
    parser.add_argument("--dual_rollout_width", type=int, default=640)
    parser.add_argument("--dual_rollout_height", type=int, default=480)
    parser.add_argument("--dual_suffix", default=None, help="Optional suffix for dual training run.")
    parser.add_argument("--dual_obstacle_height", type=float, default=0.045)
    parser.add_argument("--dual_obstacle2_height", type=float, default=0.045)
    parser.add_argument("--dual_obstacle_half_length", type=float, default=0.2)
    parser.add_argument("--dual_obstacle2_half_length", type=float, default=0.2)
    parser.add_argument("--dual_randomize_obstacle_x_min", type=float, default=0.5)
    parser.add_argument("--dual_randomize_obstacle_x_max", type=float, default=0.85)
    parser.add_argument("--dual_randomize_obstacle_gap_min", type=float, default=0.2)
    parser.add_argument("--dual_randomize_obstacle_gap_max", type=float, default=0.5)
    parser.add_argument("--dual_success_x_margin", type=float, default=0.1)

    # Evaluation settings (scripts/eval_twmr_checkpoint.py)
    parser.add_argument("--skip_eval", action="store_true", help="Skip post-training checkpoint eval.")
    parser.add_argument("--eval_num_episodes", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--eval_output_json", default=None)

    # Rendering settings (train_jax_ppo.py with --num_timesteps=0)
    parser.add_argument("--skip_render", action="store_true", help="Skip deterministic rollout rendering.")
    parser.add_argument("--render_num_videos", type=int, default=6)
    parser.add_argument("--render_episode_length", type=int, default=2200)
    parser.add_argument("--render_num_envs", type=int, default=64)
    parser.add_argument("--render_num_eval_envs", type=int, default=8)
    parser.add_argument("--render_batch_size", type=int, default=64)
    parser.add_argument("--render_unroll_length", type=int, default=8)
    parser.add_argument("--render_num_minibatches", type=int, default=4)
    parser.add_argument("--render_num_updates_per_batch", type=int, default=2)
    parser.add_argument("--render_num_evals", type=int, default=1)
    parser.add_argument("--render_rollout_camera", default="cam_robot_follow_long")
    parser.add_argument("--render_rollout_width", type=int, default=640)
    parser.add_argument("--render_rollout_height", type=int, default=480)
    parser.add_argument("--render_suffix", default=None, help="Optional suffix for render-only run.")
    parser.add_argument(
        "--render_terminate_on_success",
        action="store_true",
        help="Terminate episodes immediately on success during render run.",
    )

    return parser.parse_args()


def _runtime_env() -> dict[str, str]:
    env = os.environ.copy()
    ffmpeg_bin = Path.home() / ".pixi" / "envs" / "ffmpeg" / "bin"
    env["PATH"] = f"{ffmpeg_bin}:{env.get('PATH', '')}"
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    return env


def _compact_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":"))


def _run_cmd(cmd: list[str], env: dict[str, str], dry_run: bool) -> None:
    print(f"[CMD] {shlex.join(cmd)}", flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True, env=env)


def _latest_log_with_suffix(logs_root: Path, suffix: str) -> Path:
    candidates = [p for p in logs_root.iterdir() if p.is_dir() and p.name.endswith(f"-{suffix}")]
    if not candidates:
        raise RuntimeError(f"No log directory found for suffix '{suffix}'.")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _copy_rollouts_into(exp_dir: Path) -> list[str]:
    copied: list[str] = []
    for rollout in sorted(Path(".").glob("rollout*.mp4")):
        target = exp_dir / rollout.name
        shutil.copy2(rollout, target)
        copied.append(str(target))
    return copied


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_stage_ckpt(manifest: dict[str, Any], stage_name: str) -> Path:
    for stage in manifest.get("stages", []):
        if stage.get("stage") == stage_name:
            ckpt = stage.get("ckpt_dir")
            if not ckpt:
                break
            return Path(str(ckpt)).resolve()
    raise RuntimeError(f"Could not find checkpoint for stage '{stage_name}' in manifest.")


def _choose_bridge_ckpt(bridge_manifest: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    stages = bridge_manifest.get("stages", [])
    if not stages:
        raise RuntimeError("Bridge manifest contains no stage records.")
    promoted = [stage for stage in stages if bool(stage.get("promoted"))]
    chosen = promoted[-1] if promoted else stages[-1]
    ckpt_dir = chosen.get("ckpt_dir")
    if not ckpt_dir:
        raise RuntimeError("Selected bridge stage has no ckpt_dir.")
    return Path(str(ckpt_dir)).resolve(), chosen


def _dual_overrides(args: argparse.Namespace, *, terminate_on_success: bool | None = None) -> dict[str, Any]:
    overrides: dict[str, Any] = {
        "xml_variant": "box_dual",
        "enable_second_obstacle": True,
        "randomize_obstacles": True,
        "obstacle_local_windowed_reward": True,
        "obstacle_height": args.dual_obstacle_height,
        "obstacle2_height": args.dual_obstacle2_height,
        "obstacle_half_length": args.dual_obstacle_half_length,
        "obstacle2_half_length": args.dual_obstacle2_half_length,
        "randomize_obstacle_x_min": args.dual_randomize_obstacle_x_min,
        "randomize_obstacle_x_max": args.dual_randomize_obstacle_x_max,
        "randomize_obstacle_gap_min": args.dual_randomize_obstacle_gap_min,
        "randomize_obstacle_gap_max": args.dual_randomize_obstacle_gap_max,
        "success_x_margin": args.dual_success_x_margin,
    }
    if terminate_on_success is not None:
        overrides["terminate_on_success"] = terminate_on_success
    return overrides


def main() -> int:
    args = _parse_args()
    logs_root = Path("logs").resolve()
    logs_root.mkdir(parents=True, exist_ok=True)
    env = _runtime_env()

    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    phase3_suffix_prefix = f"{args.suffix_prefix}-phase3"
    bridge_suffix_prefix = f"{args.suffix_prefix}-bridge-{args.bridge_mode}"
    dual_suffix = args.dual_suffix or f"{args.suffix_prefix}-dual-rand"
    render_suffix = args.render_suffix or f"{args.suffix_prefix}-dual-render"

    phase3_manifest_path = (
        Path(args.phase3_manifest_path).resolve()
        if args.phase3_manifest_path is not None
        else logs_root / f"full-curriculum-{args.suffix_prefix}-phase3-{timestamp}.json"
    )
    bridge_manifest_path = (
        Path(args.bridge_manifest_path).resolve()
        if args.bridge_manifest_path is not None
        else logs_root / f"full-curriculum-{args.suffix_prefix}-bridge-{timestamp}.json"
    )
    full_manifest_path = (
        Path(args.manifest_path).resolve()
        if args.manifest_path is not None
        else logs_root / f"full-curriculum-{args.suffix_prefix}-{timestamp}.json"
    )
    eval_output_json = (
        Path(args.eval_output_json).resolve()
        if args.eval_output_json is not None
        else logs_root / f"eval-{dual_suffix}-{timestamp}.json"
    )

    # 1) Stage A/B/C curriculum
    phase3_cmd = [
        args.python,
        "scripts/phase3_curriculum.py",
        f"--seed={args.seed}",
        f"--suffix_prefix={phase3_suffix_prefix}",
        f"--num_timesteps_override={args.phase3_num_timesteps}",
        f"--num_envs_override={args.phase3_num_envs}",
        f"--num_eval_envs_override={args.phase3_num_eval_envs}",
        f"--batch_size_override={args.phase3_batch_size}",
        f"--unroll_length_override={args.phase3_unroll_length}",
        f"--num_minibatches_override={args.phase3_num_minibatches}",
        f"--num_updates_per_batch_override={args.phase3_num_updates_per_batch}",
        f"--num_evals_override={args.phase3_num_evals}",
        f"--episode_length_override={args.phase3_episode_length}",
        f"--manifest_path={phase3_manifest_path}",
    ]
    if args.phase3_initial_checkpoint_path is not None:
        phase3_cmd.append(f"--initial_checkpoint_path={Path(args.phase3_initial_checkpoint_path).resolve()}")
    _run_cmd(phase3_cmd, env=env, dry_run=args.dry_run)

    if args.dry_run:
        stageb_ckpt = Path("<stageB_ckpt_dir>")
        stagec_ckpt = Path("<stageC_ckpt_dir>")
    else:
        phase3_manifest = _read_json(phase3_manifest_path)
        stageb_ckpt = _find_stage_ckpt(phase3_manifest, STAGE_B)
        stagec_ckpt = _find_stage_ckpt(phase3_manifest, STAGE_C)

    # Determine dual start checkpoint.
    if args.dual_start_checkpoint_path is not None:
        dual_start_ckpt = Path(args.dual_start_checkpoint_path).resolve()
        bridge_chosen_stage: dict[str, Any] | None = None
        bridge_manifest_for_summary: dict[str, Any] | None = None
    elif args.skip_bridge:
        dual_start_ckpt = stagec_ckpt
        bridge_chosen_stage = None
        bridge_manifest_for_summary = None
    else:
        bridge_initial_ckpt = stageb_ckpt if args.bridge_initial_stage == STAGE_B else stagec_ckpt
        bridge_cmd = [
            args.python,
            "scripts/stagec_bridge_curriculum.py",
            f"--mode={args.bridge_mode}",
            f"--initial_checkpoint_path={bridge_initial_ckpt}",
            f"--suffix_prefix={bridge_suffix_prefix}",
            f"--obstacle_x_position={args.bridge_obstacle_x_position}",
            f"--success_x_margin={args.bridge_success_x_margin}",
            f"--heights={args.bridge_heights}",
            f"--half_lengths={args.bridge_half_lengths}",
            f"--fixed_height={args.bridge_fixed_height}",
            f"--fixed_half_length={args.bridge_fixed_half_length}",
            f"--gate_index={args.bridge_gate_index}",
            f"--gate_threshold={args.bridge_gate_threshold}",
            f"--min_success_to_continue={args.bridge_min_success_to_continue}",
            f"--seed={args.seed}",
            f"--num_timesteps={args.bridge_num_timesteps}",
            f"--num_envs={args.bridge_num_envs}",
            f"--num_eval_envs={args.bridge_num_eval_envs}",
            f"--batch_size={args.bridge_batch_size}",
            f"--unroll_length={args.bridge_unroll_length}",
            f"--num_minibatches={args.bridge_num_minibatches}",
            f"--num_updates_per_batch={args.bridge_num_updates_per_batch}",
            f"--num_evals={args.bridge_num_evals}",
            f"--episode_length={args.bridge_episode_length}",
            f"--rollout_camera={args.bridge_rollout_camera}",
            f"--rollout_width={args.bridge_rollout_width}",
            f"--rollout_height={args.bridge_rollout_height}",
            f"--eval_num_episodes={args.bridge_eval_num_episodes}",
            f"--eval_batch_size={args.bridge_eval_batch_size}",
            f"--manifest_path={bridge_manifest_path}",
        ]
        _run_cmd(bridge_cmd, env=env, dry_run=args.dry_run)
        if args.dry_run:
            dual_start_ckpt = Path("<bridge_ckpt_dir>")
            bridge_chosen_stage = None
            bridge_manifest_for_summary = None
        else:
            bridge_manifest_for_summary = _read_json(bridge_manifest_path)
            dual_start_ckpt, bridge_chosen_stage = _choose_bridge_ckpt(bridge_manifest_for_summary)

    # 2) Dual-obstacle training
    dual_overrides = _dual_overrides(args)
    dual_cmd = [
        args.python,
        "train_jax_ppo.py",
        f"--env_name={ENV_NAME}",
        f"--playground_config_overrides={_compact_json(dual_overrides)}",
        f"--load_checkpoint_path={dual_start_ckpt}",
        f"--num_timesteps={args.dual_num_timesteps}",
        f"--num_envs={args.dual_num_envs}",
        f"--num_eval_envs={args.dual_num_eval_envs}",
        f"--batch_size={args.dual_batch_size}",
        f"--unroll_length={args.dual_unroll_length}",
        f"--num_minibatches={args.dual_num_minibatches}",
        f"--num_updates_per_batch={args.dual_num_updates_per_batch}",
        f"--num_evals={args.dual_num_evals}",
        f"--episode_length={args.dual_episode_length}",
        f"--num_videos={args.dual_num_videos}",
        f"--seed={args.seed}",
        f"--suffix={dual_suffix}",
        "--use_wandb=false",
        "--use_tb=false",
        f"--rollout_camera={args.dual_rollout_camera}",
        f"--rollout_width={args.dual_rollout_width}",
        f"--rollout_height={args.dual_rollout_height}",
    ]
    _run_cmd(dual_cmd, env=env, dry_run=args.dry_run)

    if args.dry_run:
        dual_exp_dir = Path(f"<logs/*-{dual_suffix}>")
        dual_ckpt_dir = Path("<dual_ckpt_dir>")
        dual_rollouts_copied: list[str] = []
    else:
        dual_exp_dir = _latest_log_with_suffix(logs_root, dual_suffix)
        dual_ckpt_dir = dual_exp_dir / "checkpoints"
        if not dual_ckpt_dir.exists():
            raise RuntimeError(f"Dual-obstacle checkpoint directory missing: {dual_ckpt_dir}")
        dual_rollouts_copied = _copy_rollouts_into(dual_exp_dir)

    # 3) Post-training evaluation
    eval_data: dict[str, Any] | None = None
    if not args.skip_eval:
        eval_cmd = [
            args.python,
            "scripts/eval_twmr_checkpoint.py",
            f"--checkpoint_path={dual_ckpt_dir}",
            f"--playground_config_overrides={_compact_json(dual_overrides)}",
            f"--num_episodes={args.eval_num_episodes}",
            f"--episode_length={args.dual_episode_length}",
            f"--batch_size={args.eval_batch_size}",
            f"--seed={args.seed}",
            f"--output_json={eval_output_json}",
        ]
        _run_cmd(eval_cmd, env=env, dry_run=args.dry_run)
        if not args.dry_run:
            eval_data = _read_json(eval_output_json)

    # 4) Deterministic render rollouts
    render_exp_dir: Path | None = None
    render_rollouts_copied: list[str] = []
    if not args.skip_render:
        render_overrides = _dual_overrides(
            args,
            terminate_on_success=args.render_terminate_on_success,
        )
        render_cmd = [
            args.python,
            "train_jax_ppo.py",
            f"--env_name={ENV_NAME}",
            f"--playground_config_overrides={_compact_json(render_overrides)}",
            f"--load_checkpoint_path={dual_ckpt_dir}",
            "--num_timesteps=0",
            f"--num_envs={args.render_num_envs}",
            f"--num_eval_envs={args.render_num_eval_envs}",
            f"--batch_size={args.render_batch_size}",
            f"--unroll_length={args.render_unroll_length}",
            f"--num_minibatches={args.render_num_minibatches}",
            f"--num_updates_per_batch={args.render_num_updates_per_batch}",
            f"--num_evals={args.render_num_evals}",
            f"--episode_length={args.render_episode_length}",
            f"--num_videos={args.render_num_videos}",
            f"--seed={args.seed}",
            f"--suffix={render_suffix}",
            "--use_wandb=false",
            "--use_tb=false",
            f"--rollout_camera={args.render_rollout_camera}",
            f"--rollout_width={args.render_rollout_width}",
            f"--rollout_height={args.render_rollout_height}",
        ]
        _run_cmd(render_cmd, env=env, dry_run=args.dry_run)
        if not args.dry_run:
            render_exp_dir = _latest_log_with_suffix(logs_root, render_suffix)
            render_rollouts_copied = _copy_rollouts_into(render_exp_dir)

    full_manifest = {
        "created_at": dt.datetime.now().isoformat(),
        "dry_run": args.dry_run,
        "seed": args.seed,
        "suffix_prefix": args.suffix_prefix,
        "phase3": {
            "manifest_path": str(phase3_manifest_path),
            "suffix_prefix": phase3_suffix_prefix,
            "stageB_ckpt_dir": str(stageb_ckpt),
            "stageC_ckpt_dir": str(stagec_ckpt),
        },
        "bridge": {
            "enabled": not args.skip_bridge and args.dual_start_checkpoint_path is None,
            "manifest_path": str(bridge_manifest_path),
            "mode": args.bridge_mode,
            "initial_stage": args.bridge_initial_stage,
            "selected_stage": bridge_chosen_stage,
            "manifest": bridge_manifest_for_summary,
        },
        "dual_train": {
            "suffix": dual_suffix,
            "start_checkpoint_path": str(dual_start_ckpt),
            "exp_dir": str(dual_exp_dir),
            "ckpt_dir": str(dual_ckpt_dir),
            "overrides": dual_overrides,
            "rollouts_copied": dual_rollouts_copied,
        },
        "eval": {
            "enabled": not args.skip_eval,
            "output_json": str(eval_output_json),
            "result": eval_data,
        },
        "render": {
            "enabled": not args.skip_render,
            "suffix": render_suffix,
            "exp_dir": str(render_exp_dir) if render_exp_dir is not None else "",
            "rollouts_copied": render_rollouts_copied,
        },
    }

    full_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    full_manifest_path.write_text(json.dumps(full_manifest, indent=2), encoding="utf-8")

    print("[DONE] Full curriculum pipeline complete.", flush=True)
    print(f"  full_manifest={full_manifest_path}", flush=True)
    print(f"  dual_ckpt_dir={dual_ckpt_dir}", flush=True)
    if not args.skip_eval:
        print(f"  eval_json={eval_output_json}", flush=True)
    if not args.skip_render:
        print(f"  render_exp_dir={render_exp_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
