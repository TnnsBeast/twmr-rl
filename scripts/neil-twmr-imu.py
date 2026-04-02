#!/usr/bin/env python3
"""Run the Neil TWMR curriculum with a hardware-faithful proprioceptive interface.

This wrapper keeps the existing curriculum logic in `scripts/neil-twmr.py` and
injects a common set of environment overrides for every stage:

- Observation mode: IMU + motor-encoder proprioception
- Action mode: desired wheel/extension velocities mapped to torques by a
  proportional controller inside the environment

Any additional curriculum flags are passed through unchanged to
`scripts/neil-twmr.py`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Wrapper around scripts/neil-twmr.py that enables the hardware-faithful "
            "IMU + encoder observation path and velocity-setpoint action path."
        )
    )
    parser.add_argument(
        "--imu_observation_mode",
        default="imu_encoder",
        help="Observation mode to inject into all curriculum stages.",
    )
    parser.add_argument(
        "--imu_action_mode",
        default="velocity_setpoint",
        help="Action mode to inject into all curriculum stages.",
    )
    parser.add_argument("--imu_wheel_velocity_limit", type=float, default=20.0)
    parser.add_argument("--imu_extension_velocity_limit", type=float, default=10.0)
    parser.add_argument("--imu_wheel_velocity_kp", type=float, default=1.0)
    parser.add_argument("--imu_extension_velocity_kp", type=float, default=1.0)
    parser.add_argument("--imu_wheel_encoder_noise_std", type=float, default=0.0)
    parser.add_argument("--imu_extension_encoder_noise_std", type=float, default=0.0)
    parser.add_argument("--imu_extension_position_noise_std", type=float, default=0.0)
    parser.add_argument("--imu_acc_noise_std", type=float, default=0.0)
    parser.add_argument("--imu_gyro_noise_std", type=float, default=0.0)
    parser.add_argument("--imu_acc_bias_std", type=float, default=0.0)
    parser.add_argument("--imu_gyro_bias_std", type=float, default=0.0)
    return parser.parse_known_args()


def _has_flag(args: list[str], flag: str) -> bool:
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in args)


def _common_overrides(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "observation_mode": args.imu_observation_mode,
        "action_mode": args.imu_action_mode,
        "wheel_velocity_limit": float(args.imu_wheel_velocity_limit),
        "extension_velocity_limit": float(args.imu_extension_velocity_limit),
        "wheel_velocity_kp": float(args.imu_wheel_velocity_kp),
        "extension_velocity_kp": float(args.imu_extension_velocity_kp),
        "wheel_encoder_noise_std": float(args.imu_wheel_encoder_noise_std),
        "extension_encoder_noise_std": float(args.imu_extension_encoder_noise_std),
        "extension_position_noise_std": float(args.imu_extension_position_noise_std),
        "imu_acc_noise_std": float(args.imu_acc_noise_std),
        "imu_gyro_noise_std": float(args.imu_gyro_noise_std),
        "imu_acc_bias_std": float(args.imu_acc_bias_std),
        "imu_gyro_bias_std": float(args.imu_gyro_bias_std),
    }


def main() -> int:
    args, passthrough = _parse_args()
    if _has_flag(passthrough, "--common_overrides_json"):
        raise ValueError(
            "Pass IMU/hardware-faithful overrides via the --imu_* flags on this wrapper, "
            "not via --common_overrides_json."
        )

    repo_root = Path(__file__).resolve().parent.parent
    base_script = repo_root / "scripts" / "neil-twmr.py"
    cmd = [
        sys.executable,
        str(base_script),
        f"--common_overrides_json={json.dumps(_common_overrides(args), separators=(',', ':'))}",
    ]
    if not _has_flag(passthrough, "--suffix_prefix"):
        cmd.append("--suffix_prefix=neil-twmr-imu")
    cmd.extend(passthrough)
    print("[INFO] Launching IMU curriculum wrapper with overrides:", flush=True)
    print(json.dumps(_common_overrides(args), indent=2), flush=True)
    subprocess.run(cmd, check=True, cwd=repo_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
