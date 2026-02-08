#!/usr/bin/env python3
"""Simple MuJoCo viewer for the transformable wheel robot."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np

try:
    import mujoco
    import mujoco.viewer
except Exception as exc:  # pragma: no cover - import error path
    raise SystemExit(
        "Failed to import 'mujoco'. Install deps with: pixi run uv sync"
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = REPO_ROOT / "packages" / "twmr" / "assets" / "trans_wheel_robo2_2FLAT.xml"

EXTENSION_ACTUATORS = (
    "front_left_wheel_0_extension_joint_ctrl",
    "front_right_wheel_0_extension_joint_ctrl",
    "rear_left_wheel_0_extension_joint_ctrl",
    "rear_right_wheel_0_extension_joint_ctrl",
)


def _require_actuator(model: "mujoco.MjModel", name: str) -> int:
    act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if act_id < 0:
        raise ValueError(f"Actuator '{name}' not found in model.")
    return int(act_id)


def _actuator_joint_indices(model: "mujoco.MjModel", act_id: int) -> tuple[int, int]:
    joint_id = int(model.actuator_trnid[act_id, 0])
    qpos_adr = int(model.jnt_qposadr[joint_id])
    qvel_adr = int(model.jnt_dofadr[joint_id])
    return qpos_adr, qvel_adr


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch a MuJoCo viewer for the transformable wheel robot.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Path to a MuJoCo XML model.",
    )
    parser.add_argument(
        "--animate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Animate the extendable wheel joints with a slow sinusoid.",
    )
    parser.add_argument(
        "--freq",
        type=float,
        default=0.2,
        help="Extension oscillation frequency in Hz (only if --animate).",
    )
    parser.add_argument(
        "--amp",
        type=float,
        default=0.5,
        help="Extension oscillation amplitude in radians (only if --animate).",
    )
    parser.add_argument(
        "--bias",
        type=float,
        default=0.7,
        help="Extension oscillation bias in radians (only if --animate).",
    )
    parser.add_argument(
        "--kp",
        type=float,
        default=3.0,
        help="Proportional gain for extension joint control.",
    )
    parser.add_argument(
        "--kd",
        type=float,
        default=0.1,
        help="Derivative gain for extension joint control.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    model_path = args.model.expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = mujoco.MjModel.from_xml_path(model_path.as_posix())
    data = mujoco.MjData(model)

    extension_act_ids = [_require_actuator(model, name) for name in EXTENSION_ACTUATORS]
    extension_indices = [
        _actuator_joint_indices(model, act_id) for act_id in extension_act_ids
    ]

    start_time = time.time()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            data.ctrl[:] = 0.0

            if args.animate:
                t = time.time() - start_time
                target = args.bias + args.amp * math.sin(2.0 * math.pi * args.freq * t)
                for act_id, (qpos_adr, qvel_adr) in zip(
                    extension_act_ids, extension_indices, strict=True
                ):
                    qpos = float(data.qpos[qpos_adr])
                    qvel = float(data.qvel[qvel_adr])
                    torque = args.kp * (target - qpos) - args.kd * qvel
                    lo, hi = model.actuator_ctrlrange[act_id]
                    data.ctrl[act_id] = np.clip(torque, lo, hi)

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
