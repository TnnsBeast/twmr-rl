#!/usr/bin/env python3
"""Train a linear policy with CEM to traverse the obstacle.

Key difference vs PPO scripts in this repo:
- This uses a derivative-free, population-based optimizer (CEM),
  not gradient-based policy optimization.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path
from typing import Tuple

import numpy as np

try:
    import mujoco
except Exception as exc:  # pragma: no cover - import error path
    raise SystemExit(
        "Failed to import 'mujoco'. Install deps with: pixi run uv sync"
    ) from exc


WHEEL_ACTUATOR_NAMES = (
    "front_left_wheel_joint_ctrl",
    "front_right_wheel_joint_ctrl",
    "rear_left_wheel_joint_ctrl",
    "rear_right_wheel_joint_ctrl",
)

LEG_ACTUATOR_NAMES = (
    "front_left_wheel_0_extension_joint_ctrl",
    "front_right_wheel_0_extension_joint_ctrl",
    "rear_left_wheel_0_extension_joint_ctrl",
    "rear_right_wheel_0_extension_joint_ctrl",
)

ACTUATOR_NAMES = WHEEL_ACTUATOR_NAMES + LEG_ACTUATOR_NAMES

WHEEL_JOINT_NAMES = (
    "front_left_wheel_joint",
    "front_right_wheel_joint",
    "rear_left_wheel_joint",
    "rear_right_wheel_joint",
)


def _find_repo_root() -> Path:
    def _scan(start: Path) -> Path | None:
        for p in [start, *start.parents]:
            if (p / "pixi.toml").exists() and (p / "packages").is_dir():
                return p
            if (p / "pyproject.toml").exists() and (p / "packages").is_dir():
                return p
            if (p / ".git").exists():
                return p
        return None

    here = Path(__file__).resolve()
    root = _scan(here)
    if root is not None:
        return root
    cwd = Path.cwd().resolve()
    root = _scan(cwd)
    if root is not None:
        return root
    return here.parent


def _resolve_model_path(path: Path) -> Path:
    if path.exists():
        return path.resolve()
    if not path.is_absolute():
        candidate = (REPO_ROOT / path).resolve()
        if candidate.exists():
            return candidate
    # Last resort: look for packages/twmr/assets relative to cwd.
    candidate = (Path.cwd() / "packages" / "twmr" / "assets" / path.name).resolve()
    if candidate.exists():
        return candidate
    return path.resolve()


def _resolve_save_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


REPO_ROOT = _find_repo_root()
DEFAULT_MODEL = (
    REPO_ROOT
    / "sandbox"
    / "neil"
    / "models"
    / "trans_wheel_robo2_2BOX_CLY.xml"
)
DEFAULT_SAVE = REPO_ROOT / "sandbox" / "neil" / "cem_policy_box.npz"


class TwmrEnv:
    def __init__(
        self,
        xml_path: Path,
        ctrl_dt: float = 0.02,
        frame_skip: int | None = None,
        max_steps: int = 1000,
        settle_steps: int = 25,
        min_z: float = 0.02,
        reset_noise: float = 0.01,
        alive_bonus: float = 0.05,
        control_cost: float = 1e-3,
        success_x: float | None = None,
        success_bonus: float = 1.0,
        extension_penalty: float = 1e-3,
        post_success_penalty: float = 0.0,
        extend_angle: float | None = None,
        extend_mode: str = "auto",
        auto_retract: bool = True,
        obstacle_geom: str | None = "traverse_box",
        obstacle_buffer: float = 0.05,
        obstacle_pre_buffer: float | None = None,
        obstacle_post_buffer: float | None = None,
        obstacle_smooth: float = 0.05,
        gate_mode: str = "proprio",
        gate_acc_scale: float = 3.0,
        gate_gyro_scale: float = 3.0,
        gate_stall_scale: float = 2.0,
        gate_bias: float = 1.0,
        gate_smooth: float = 0.5,
        hold_retract: bool = True,
        hold_kp: float = 4.0,
        hold_kd: float = 0.2,
        obs_mode: str = "proprio",
        start_on_ground: bool = True,
        ground_clearance: float = 1e-3,
        retract_legs: bool = True,
        retract_angle: float | None = None,
    ) -> None:
        xml_path = xml_path.expanduser().resolve()
        if not xml_path.exists():
            raise FileNotFoundError(f"Model not found: {xml_path}")

        self.model = mujoco.MjModel.from_xml_path(xml_path.as_posix())
        self.data = mujoco.MjData(self.model)
        self.max_steps = int(max_steps)
        self.settle_steps = int(settle_steps)
        self.min_z = float(min_z)
        self.reset_noise = float(reset_noise)
        self.alive_bonus = float(alive_bonus)
        self.control_cost = float(control_cost)
        self.success_x = None
        self.success_bonus = float(success_bonus)
        self.extension_penalty = float(extension_penalty)
        self.post_success_penalty = float(post_success_penalty)
        self.extend_angle = extend_angle
        self.extend_mode = extend_mode
        self.auto_retract = bool(auto_retract)
        self.obstacle_geom = obstacle_geom
        self.obstacle_buffer = float(obstacle_buffer)
        self.obstacle_pre_buffer = (
            float(obstacle_pre_buffer)
            if obstacle_pre_buffer is not None
            else self.obstacle_buffer
        )
        self.obstacle_post_buffer = (
            float(obstacle_post_buffer)
            if obstacle_post_buffer is not None
            else self.obstacle_buffer
        )
        self.obstacle_smooth = max(0.0, float(obstacle_smooth))
        self.gate_mode = gate_mode
        self.gate_acc_scale = float(gate_acc_scale)
        self.gate_gyro_scale = float(gate_gyro_scale)
        self.gate_stall_scale = float(gate_stall_scale)
        self.gate_bias = float(gate_bias)
        self.gate_smooth = max(1e-6, float(gate_smooth))
        self.hold_retract = bool(hold_retract)
        self.hold_kp = float(hold_kp)
        self.hold_kd = float(hold_kd)
        self.obs_mode = obs_mode
        self.start_on_ground = bool(start_on_ground)
        self.ground_clearance = float(ground_clearance)
        self.retract_legs = bool(retract_legs)
        self.retract_angle = retract_angle
        self.step_count = 0
        self.passed_obstacle = False
        self.obstacle_center = None
        self.obstacle_half_width = None
        self.obstacle_x_min = None
        self.obstacle_x_max = None
        self.acc_slice = None
        self.gyro_slice = None
        self.retract_is_max = None

        self.sim_dt = float(self.model.opt.timestep)
        if frame_skip is None:
            frame_skip = max(1, int(round(ctrl_dt / self.sim_dt)))
        self.frame_skip = int(frame_skip)
        self.ctrl_dt = self.frame_skip * self.sim_dt

        self.free_qpos_adr = None
        for j in range(self.model.njnt):
            if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                self.free_qpos_adr = int(self.model.jnt_qposadr[j])
                break
        if self.free_qpos_adr is None:
            raise RuntimeError("No free joint found; expected a floating base.")

        self.x_index = self.free_qpos_adr + 0
        self.z_index = self.free_qpos_adr + 2

        self.root_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "root"
        )
        if self.root_body_id < 0:
            self.root_body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "chassis"
            )
        self._body_in_root_subtree = None
        if self.root_body_id >= 0:
            in_subtree = np.zeros((self.model.nbody,), dtype=bool)
            for bid in range(self.model.nbody):
                cur = int(bid)
                while True:
                    if cur == self.root_body_id:
                        in_subtree[bid] = True
                        break
                    parent = int(self.model.body_parentid[cur])
                    if parent == cur:
                        break
                    cur = parent
            self._body_in_root_subtree = in_subtree

        self.actuator_ids = []
        for name in ACTUATOR_NAMES:
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid < 0:
                raise RuntimeError(f"Actuator '{name}' not found in model.")
            self.actuator_ids.append(int(aid))

        self.actuator_ids = np.asarray(self.actuator_ids, dtype=np.int32)
        self.wheel_act_count = len(WHEEL_ACTUATOR_NAMES)
        self.nu = len(self.actuator_ids)
        if self.nu != 8:
            raise RuntimeError(f"Expected 8 actuators, got {self.nu}.")
        self.extension_actuator_ids = self.actuator_ids[self.wheel_act_count :].tolist()

        ctrlrange = np.asarray(self.model.actuator_ctrlrange, dtype=np.float32)
        self.ctrl_mid = (ctrlrange[:, 0] + ctrlrange[:, 1]) * 0.5
        self.ctrl_half = (ctrlrange[:, 1] - ctrlrange[:, 0]) * 0.5

        self.leg_qpos_idx = []
        self.leg_joint_ids = []
        self.leg_qvel_idx = []
        for name in (
            "front_left_wheel_0_extension_joint",
            "front_right_wheel_0_extension_joint",
            "rear_left_wheel_0_extension_joint",
            "rear_right_wheel_0_extension_joint",
        ):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise RuntimeError(f"Joint '{name}' not found in model.")
            self.leg_joint_ids.append(int(jid))
            self.leg_qpos_idx.append(int(self.model.jnt_qposadr[jid]))
            self.leg_qvel_idx.append(int(self.model.jnt_dofadr[jid]))

        if self.obs_mode not in {"full", "compact", "compact_obstacle", "proprio"}:
            raise ValueError(
                "obs_mode must be 'full', 'compact', 'compact_obstacle', or 'proprio'"
            )
        if self.gate_mode not in {"proprio", "obstacle", "none"}:
            raise ValueError("gate_mode must be 'proprio', 'obstacle', or 'none'")
        if self.extend_mode not in {"auto", "midpoint", "opposite"}:
            raise ValueError("extend_mode must be 'auto', 'midpoint', or 'opposite'")

        if self.obs_mode == "full":
            self.obs_dim = int(self.model.nq + self.model.nv)
        elif self.obs_mode == "compact":
            self.obs_dim = 10
        elif self.obs_mode == "compact_obstacle":
            self.obs_dim = 12
        else:
            # wheel qpos/qvel (4+4), leg qpos/qvel (4+4), IMU acc/gyro (3+3)
            self.obs_dim = 22
        self.retract_targets = self._compute_retract_targets()
        self.extend_targets = self._compute_extend_targets()
        self._resolve_obstacle_geom()
        self._set_success_x(success_x)
        self._resolve_imu_sensors()
        self._resolve_wheel_joints()

    def _compute_retract_targets(self) -> np.ndarray:
        targets = np.zeros((len(self.leg_joint_ids),), dtype=np.float32)
        if self.retract_angle is not None:
            targets[:] = float(self.retract_angle)
            return targets
        if self.auto_retract and self.leg_joint_ids:
            lows = []
            highs = []
            for jid in self.leg_joint_ids:
                if self.model.jnt_limited[jid]:
                    lo, hi = self.model.jnt_range[jid]
                    lows.append(float(lo))
                    highs.append(float(hi))
                else:
                    lows.append(0.0)
                    highs.append(0.0)
            z_low = self._evaluate_min_z(lows)
            z_high = self._evaluate_min_z(highs)
            if z_high > z_low:
                targets[:] = np.asarray(highs, dtype=np.float32)
                self.retract_is_max = True
            else:
                targets[:] = np.asarray(lows, dtype=np.float32)
                self.retract_is_max = False
            return targets
        for i, jid in enumerate(self.leg_joint_ids):
            if self.model.jnt_limited[jid]:
                targets[i] = float(self.model.jnt_range[jid, 0])
            else:
                targets[i] = 0.0
        return targets

    def _compute_extend_targets(self) -> np.ndarray:
        targets = np.zeros((len(self.leg_joint_ids),), dtype=np.float32)
        if self.extend_angle is not None:
            targets[:] = float(self.extend_angle)
            return targets
        use_opposite = False
        if self.extend_mode == "opposite":
            use_opposite = True
        elif self.extend_mode == "auto" and self.retract_is_max is not None:
            use_opposite = True
        if use_opposite:
            for i, jid in enumerate(self.leg_joint_ids):
                if self.model.jnt_limited[jid]:
                    lo, hi = self.model.jnt_range[jid]
                    targets[i] = float(lo if self.retract_is_max else hi)
                else:
                    targets[i] = 0.0
            return targets

        for i, jid in enumerate(self.leg_joint_ids):
            if self.model.jnt_limited[jid]:
                lo, hi = self.model.jnt_range[jid]
                targets[i] = float(0.5 * (lo + hi))
            else:
                targets[i] = 0.0
        return targets

    def _evaluate_min_z(self, targets: list[float]) -> float:
        if not targets:
            return self._approx_min_geom_z()
        qpos_backup = self.data.qpos.copy()
        qvel_backup = self.data.qvel.copy()
        for qpos_idx, target in zip(self.leg_qpos_idx, targets, strict=True):
            self.data.qpos[qpos_idx] = float(target)
        mujoco.mj_forward(self.model, self.data)
        z_min = self._approx_min_geom_z()
        self.data.qpos[:] = qpos_backup
        self.data.qvel[:] = qvel_backup
        mujoco.mj_forward(self.model, self.data)
        return float(z_min)

    def _resolve_wheel_joints(self) -> None:
        self.wheel_joint_ids = []
        self.wheel_qpos_idx = []
        self.wheel_qvel_idx = []
        for name in WHEEL_JOINT_NAMES:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise RuntimeError(f"Wheel joint '{name}' not found in model.")
            self.wheel_joint_ids.append(int(jid))
            self.wheel_qpos_idx.append(int(self.model.jnt_qposadr[jid]))
            self.wheel_qvel_idx.append(int(self.model.jnt_dofadr[jid]))

    def _resolve_imu_sensors(self) -> None:
        try:
            acc_sid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SENSOR, "root_acc"
            )
            gyro_sid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SENSOR, "root_gyro"
            )
        except Exception:
            acc_sid = -1
            gyro_sid = -1
        if acc_sid >= 0:
            acc_adr = int(self.model.sensor_adr[acc_sid])
            acc_dim = int(self.model.sensor_dim[acc_sid])
            if acc_dim == 3:
                self.acc_slice = slice(acc_adr, acc_adr + 3)
        if gyro_sid >= 0:
            gyro_adr = int(self.model.sensor_adr[gyro_sid])
            gyro_dim = int(self.model.sensor_dim[gyro_sid])
            if gyro_dim == 3:
                self.gyro_slice = slice(gyro_adr, gyro_adr + 3)

    def _resolve_obstacle_geom(self) -> None:
        self.obstacle_center = None
        self.obstacle_half_width = None
        self.obstacle_x_min = None
        self.obstacle_x_max = None
        if not self.obstacle_geom:
            return
        geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, self.obstacle_geom
        )
        if geom_id < 0:
            return
        mujoco.mj_forward(self.model, self.data)
        pos = self.data.geom_xpos[geom_id]
        size = self.model.geom_size[geom_id]
        geom_type = int(self.model.geom_type[geom_id])
        if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            half_x = float(size[0])
        else:
            half_x = float(size[0])
        self.obstacle_center = float(pos[0])
        self.obstacle_half_width = float(half_x)
        self.obstacle_x_min = (
            self.obstacle_center - self.obstacle_half_width - self.obstacle_pre_buffer
        )
        self.obstacle_x_max = (
            self.obstacle_center + self.obstacle_half_width + self.obstacle_post_buffer
        )
        if self.obstacle_x_max <= self.obstacle_x_min:
            self.obstacle_x_min = None
            self.obstacle_x_max = None

    def _set_success_x(self, success_x: float | None) -> None:
        if success_x is None:
            if self.obstacle_x_max is not None:
                self.success_x = float(self.obstacle_x_max + 0.05)
            else:
                self.success_x = 0.85
        else:
            self.success_x = float(success_x)

    def _obstacle_gate(self, x: float) -> float:
        if self.obstacle_x_min is None or self.obstacle_x_max is None:
            return 0.0
        if self.obstacle_smooth <= 1e-6:
            return 1.0 if self.obstacle_x_min <= x <= self.obstacle_x_max else 0.0
        smooth = self.obstacle_smooth
        left = 0.5 * (math.tanh((x - self.obstacle_x_min) / smooth) + 1.0)
        right = 0.5 * (math.tanh((self.obstacle_x_max - x) / smooth) + 1.0)
        return float(left * right)

    def _read_imu(self) -> tuple[np.ndarray, np.ndarray]:
        acc = np.zeros((3,), dtype=np.float32)
        gyro = np.zeros((3,), dtype=np.float32)
        if self.acc_slice is not None:
            acc = np.asarray(self.data.sensordata[self.acc_slice], dtype=np.float32)
        if self.gyro_slice is not None:
            gyro = np.asarray(self.data.sensordata[self.gyro_slice], dtype=np.float32)
        return acc, gyro

    def _proprio_gate(self, ctrl_wheels: np.ndarray) -> float:
        acc, gyro = self._read_imu()
        acc_mag = float(np.linalg.norm(acc))
        gyro_mag = float(np.linalg.norm(gyro))
        g_mag = float(np.linalg.norm(self.model.opt.gravity))
        acc_dev = abs(acc_mag - g_mag)

        wheel_speed = np.asarray(
            [self.data.qvel[idx] for idx in self.wheel_qvel_idx], dtype=np.float32
        )
        wheel_speed_mean = float(np.mean(np.abs(wheel_speed)))
        wheel_torque_mean = float(np.mean(np.abs(ctrl_wheels)))
        stall = wheel_torque_mean / (wheel_speed_mean + 1e-3)

        score = (
            acc_dev / max(self.gate_acc_scale, 1e-6)
            + gyro_mag / max(self.gate_gyro_scale, 1e-6)
            + stall / max(self.gate_stall_scale, 1e-6)
            - self.gate_bias
        )
        gate = 0.5 * (math.tanh(score / self.gate_smooth) + 1.0)
        return float(np.clip(gate, 0.0, 1.0))

    def _apply_leg_retraction(self) -> None:
        if not self.retract_legs:
            return
        for qpos_idx, target in zip(self.leg_qpos_idx, self.retract_targets, strict=True):
            self.data.qpos[qpos_idx] = float(target)

    def _approx_min_geom_z(self) -> float:
        mujoco.mj_forward(self.model, self.data)
        z_min = np.inf
        for gid in range(self.model.ngeom):
            if self.model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_PLANE:
                continue
            if self._body_in_root_subtree is not None:
                body_id = int(self.model.geom_bodyid[gid])
                if not self._body_in_root_subtree[body_id]:
                    continue
            zc = float(self.data.geom_xpos[gid][2])
            size = self.model.geom_size[gid]
            geom_type = self.model.geom_type[gid]
            if geom_type in (
                mujoco.mjtGeom.mjGEOM_SPHERE,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                mujoco.mjtGeom.mjGEOM_CYLINDER,
            ):
                radius = float(size[0])
            elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                radius = float(size[2])
            else:
                radius = float(size[2]) if len(size) >= 3 else float(size[0])
            z_min = min(z_min, zc - radius)
        return z_min

    def _shift_root_to_ground(self) -> None:
        if not self.start_on_ground:
            return
        z_min = self._approx_min_geom_z()
        delta = self.ground_clearance - z_min
        self.data.qpos[self.z_index] += delta
        mujoco.mj_forward(self.model, self.data)

    def reset(self, rng: np.random.Generator) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        self._apply_leg_retraction()
        self._shift_root_to_ground()
        if self.reset_noise > 0:
            noise_qpos = rng.normal(0.0, self.reset_noise, size=self.model.nq)
            noise_qvel = rng.normal(0.0, self.reset_noise, size=self.model.nv)
            if self.free_qpos_adr is not None:
                noise_qpos[self.free_qpos_adr : self.free_qpos_adr + 7] = 0.0
            self.data.qpos[:] = self.data.qpos + noise_qpos
            self.data.qvel[:] = self.data.qvel + noise_qvel
            mujoco.mj_forward(self.model, self.data)
            self._apply_leg_retraction()
            self._shift_root_to_ground()

        # Let the robot settle onto the ground before starting control.
        if self.settle_steps > 0:
            for _ in range(self.settle_steps):
                self.data.ctrl[:] = 0.0
                if self.hold_retract:
                    for i, act_id in enumerate(self.extension_actuator_ids):
                        qpos_idx = self.leg_qpos_idx[i]
                        qvel_idx = self.leg_qvel_idx[i]
                        target = float(self.retract_targets[i])
                        qpos = float(self.data.qpos[qpos_idx])
                        qvel = float(self.data.qvel[qvel_idx])
                        torque = self.hold_kp * (target - qpos) - self.hold_kd * qvel
                        lo, hi = self.model.actuator_ctrlrange[act_id]
                        self.data.ctrl[act_id] = np.clip(torque, lo, hi)
                mujoco.mj_step(self.model, self.data)

        self.step_count = 0
        self.passed_obstacle = False
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        if self.obs_mode == "full":
            return np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32)

        qpos = self.data.qpos
        qvel = self.data.qvel

        if self.obs_mode == "proprio":
            wheel_qpos = [float(qpos[idx]) for idx in self.wheel_qpos_idx]
            wheel_qvel = [float(qvel[idx]) for idx in self.wheel_qvel_idx]
            leg_qpos = [float(qpos[idx]) for idx in self.leg_qpos_idx]
            leg_qvel = [float(qvel[idx]) for idx in self.leg_qvel_idx]
            acc, gyro = self._read_imu()
            obs = np.array(
                [*wheel_qpos, *wheel_qvel, *leg_qpos, *leg_qvel, *acc, *gyro],
                dtype=np.float32,
            )
            return obs

        x = float(qpos[self.x_index])
        z = float(qpos[self.z_index])
        vx = float(qvel[0])
        vz = float(qvel[2])

        qw, qx, qy, qz = qpos[self.free_qpos_adr + 3 : self.free_qpos_adr + 7]
        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        sinp = 2.0 * (qw * qy - qz * qx)
        sinp = max(-1.0, min(1.0, sinp))
        pitch = math.asin(sinp)

        leg_angles = [float(qpos[idx]) for idx in self.leg_qpos_idx]

        if self.obs_mode == "compact":
            obs = np.array([x, z, vx, vz, roll, pitch, *leg_angles], dtype=np.float32)
            return obs

        gate = self._obstacle_gate(x)
        dist_center = 0.0
        if self.obstacle_center is not None and self.obstacle_half_width is not None:
            denom = self.obstacle_half_width + max(
                self.obstacle_pre_buffer, self.obstacle_post_buffer
            )
            denom = max(denom, 1e-6)
            dist_center = (x - self.obstacle_center) / denom

        obs = np.array(
            [x, z, vx, vz, roll, pitch, *leg_angles, gate, dist_center],
            dtype=np.float32,
        )
        return obs

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        ctrl = self.ctrl_mid[self.actuator_ids] + self.ctrl_half[self.actuator_ids] * action

        x_before = float(self.data.qpos[self.x_index])

        ctrl_full = np.zeros((self.model.nu,), dtype=np.float32)
        ctrl_full[self.actuator_ids] = ctrl
        self.data.ctrl[:] = ctrl_full

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1

        x_after = float(self.data.qpos[self.x_index])
        z_after = float(self.data.qpos[self.z_index])

        forward = (x_after - x_before) / self.ctrl_dt
        reward = forward + self.alive_bonus

        reward -= self.control_cost * float(np.sum(np.square(ctrl)))

        if self.gate_mode == "obstacle":
            gate = self._obstacle_gate(x_after)
        elif self.gate_mode == "proprio":
            ctrl_wheels = ctrl[: self.wheel_act_count]
            gate = self._proprio_gate(ctrl_wheels)
        elif self.gate_mode == "none":
            gate = 0.0
        else:
            raise ValueError(f"Unknown gate_mode '{self.gate_mode}'")
        if self.extension_penalty > 0.0 or (
            self.post_success_penalty > 0.0 and self.passed_obstacle
        ):
            leg_angles = np.asarray(self.data.qpos[self.leg_qpos_idx], dtype=np.float32)
        if self.extension_penalty > 0.0:
            target = self.retract_targets + gate * (
                self.extend_targets - self.retract_targets
            )
            dev = leg_angles - target
            reward -= self.extension_penalty * float(np.mean(np.square(dev)))
        if self.post_success_penalty > 0.0 and self.passed_obstacle:
            dev = leg_angles - self.retract_targets
            reward -= self.post_success_penalty * float(np.mean(np.square(dev)))

        if (not self.passed_obstacle) and x_after >= self.success_x:
            reward += self.success_bonus
            self.passed_obstacle = True

        done = False
        if z_after < self.min_z or np.isnan(self.data.qpos).any() or np.isnan(self.data.qvel).any():
            done = True
        if self.step_count >= self.max_steps:
            done = True

        info = {
            "x": x_after,
            "z": z_after,
            "passed_obstacle": self.passed_obstacle,
            "extension_gate": gate,
        }
        return self._get_obs(), float(reward), bool(done), info


class LinearPolicy:
    def __init__(self, obs_dim: int, act_dim: int) -> None:
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.param_dim = self.act_dim * self.obs_dim + self.act_dim

    def unpack(self, params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        w_end = self.act_dim * self.obs_dim
        w = params[:w_end].reshape(self.act_dim, self.obs_dim)
        b = params[w_end:].reshape(self.act_dim)
        return w, b

    def act(self, obs: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.tanh(w @ obs + b)


def rollout(
    env: TwmrEnv,
    policy: LinearPolicy,
    params: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    w, b = policy.unpack(params)
    obs = env.reset(rng)
    total = 0.0
    info = {"x": 0.0}
    for _ in range(env.max_steps):
        action = policy.act(obs, w, b)
        obs, reward, done, info = env.step(action)
        total += reward
        if done:
            break
    return total, float(info.get("x", 0.0))


def render_policy(
    env: TwmrEnv,
    policy: LinearPolicy,
    params: np.ndarray,
    *,
    follow: bool = True,
    follow_body: str = "root",
    playback_speed: float = 1.0,
) -> None:
    import mujoco.viewer  # local import to avoid glfw dependency for training

    w, b = policy.unpack(params)
    obs = env.reset(np.random.default_rng(0))

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

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        if follow:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = track_body_id
        while viewer.is_running():
            action = policy.act(obs, w, b)
            obs, _, done, _ = env.step(action)
            viewer.sync()
            speed = max(float(playback_speed), 1e-6)
            time.sleep(env.ctrl_dt / speed)
            if done:
                obs = env.reset(np.random.default_rng(0))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CEM training for obstacle traversal with the TMWR robot.",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--pop-size", type=int, default=64)
    parser.add_argument("--elite-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ctrl-dt", type=float, default=0.02)
    parser.add_argument("--frame-skip", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--settle-steps", type=int, default=25)
    parser.add_argument("--min-z", type=float, default=0.02)
    parser.add_argument("--reset-noise", type=float, default=0.01)
    parser.add_argument(
        "--start-on-ground",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shift the robot down so the lowest geom touches the ground plane.",
    )
    parser.add_argument(
        "--ground-clearance",
        type=float,
        default=1e-3,
        help="Clearance above ground after start-on-ground shift.",
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
            "Explicit leg extension angle (rad). Defaults to the joint minimum "
            "when not provided."
        ),
    )
    parser.add_argument("--init-std", type=float, default=0.5)
    parser.add_argument("--min-std", type=float, default=0.05)
    parser.add_argument("--alive-bonus", type=float, default=0.05)
    parser.add_argument("--control-cost", type=float, default=1e-3)
    parser.add_argument(
        "--success-x",
        type=float,
        default=None,
        help=(
            "X position for success bonus. Defaults to obstacle far edge + 0.05 "
            "when available."
        ),
    )
    parser.add_argument("--success-bonus", type=float, default=1.0)
    parser.add_argument(
        "--extension-penalty",
        type=float,
        default=1e-3,
        help=(
            "Penalty on squared extension angle away from the target pose "
            "(retracted away from obstacle, extended near obstacle)."
        ),
    )
    parser.add_argument(
        "--post-success-penalty",
        type=float,
        default=0.0,
        help="Additional extension penalty applied after clearing the obstacle.",
    )
    parser.add_argument(
        "--extend-angle",
        type=float,
        default=None,
        help=(
            "Explicit extension angle (rad). Defaults to the joint midpoint when "
            "not provided."
        ),
    )
    parser.add_argument(
        "--extend-mode",
        type=str,
        default="auto",
        choices=("auto", "midpoint", "opposite"),
        help=(
            "How to choose the default extension target when --extend-angle is not set."
        ),
    )
    parser.add_argument(
        "--auto-retract",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Auto-pick retract joint limit based on which yields a smaller footprint."
        ),
    )
    parser.add_argument(
        "--obstacle-geom",
        type=str,
        default="traverse_box",
        help=(
            "Geom name for obstacle gating. Set to an empty string to disable "
            "obstacle-based gating."
        ),
    )
    parser.add_argument(
        "--obstacle-buffer",
        type=float,
        default=0.05,
        help="Extra distance before/after the obstacle to allow extension (meters).",
    )
    parser.add_argument(
        "--obstacle-pre-buffer",
        type=float,
        default=None,
        help="Override distance before the obstacle for gating (meters).",
    )
    parser.add_argument(
        "--obstacle-post-buffer",
        type=float,
        default=None,
        help="Override distance after the obstacle for gating (meters).",
    )
    parser.add_argument(
        "--obstacle-smooth",
        type=float,
        default=0.05,
        help="Smoothing distance for obstacle gating (meters).",
    )
    parser.add_argument(
        "--gate-mode",
        type=str,
        default="proprio",
        choices=("proprio", "obstacle", "none"),
        help="How to gate extension targets (proprioceptive, obstacle, or none).",
    )
    parser.add_argument(
        "--gate-acc-scale",
        type=float,
        default=3.0,
        help="IMU accel deviation scale for proprio gating (m/s^2).",
    )
    parser.add_argument(
        "--gate-gyro-scale",
        type=float,
        default=3.0,
        help="IMU gyro magnitude scale for proprio gating (rad/s).",
    )
    parser.add_argument(
        "--gate-stall-scale",
        type=float,
        default=2.0,
        help="Wheel torque/speed scale for proprio gating.",
    )
    parser.add_argument(
        "--gate-bias",
        type=float,
        default=1.0,
        help="Bias term for proprio gate score (higher keeps gate closed).",
    )
    parser.add_argument(
        "--gate-smooth",
        type=float,
        default=0.5,
        help="Smoothness for proprio gate sigmoid.",
    )
    parser.add_argument(
        "--hold-retract",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Hold extension joints at the retract target during settle steps.",
    )
    parser.add_argument(
        "--hold-kp",
        type=float,
        default=4.0,
        help="Proportional gain for retract hold during settle.",
    )
    parser.add_argument(
        "--hold-kd",
        type=float,
        default=0.2,
        help="Derivative gain for retract hold during settle.",
    )
    parser.add_argument(
        "--obs-mode",
        type=str,
        default="proprio",
        choices=("proprio", "compact", "compact_obstacle", "full"),
        help="Observation set: proprio, compact, compact_obstacle, or full qpos/qvel.",
    )
    parser.add_argument("--save-path", type=Path, default=DEFAULT_SAVE)
    parser.add_argument("--load-path", type=Path, default=None)
    parser.add_argument("--play", action="store_true")
    parser.add_argument("--render", action="store_true")
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
        "--playback-speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier (0.5 = half speed, 2.0 = double speed).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rng = np.random.default_rng(args.seed)

    model_path = _resolve_model_path(args.model)
    save_path = _resolve_save_path(args.save_path)

    env = TwmrEnv(
        xml_path=model_path,
        ctrl_dt=args.ctrl_dt,
        frame_skip=args.frame_skip,
        max_steps=args.max_steps,
        settle_steps=args.settle_steps,
        min_z=args.min_z,
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
        obstacle_geom=args.obstacle_geom if args.obstacle_geom else None,
        obstacle_buffer=args.obstacle_buffer,
        obstacle_pre_buffer=args.obstacle_pre_buffer,
        obstacle_post_buffer=args.obstacle_post_buffer,
        obstacle_smooth=args.obstacle_smooth,
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
        start_on_ground=args.start_on_ground,
        ground_clearance=args.ground_clearance,
        retract_legs=args.retract_legs,
        retract_angle=args.retract_angle,
    )
    policy = LinearPolicy(env.obs_dim, env.nu)

    if args.play:
        load_path = _resolve_save_path(args.load_path or save_path)
        if not load_path.exists():
            fallback_paths = [
                REPO_ROOT / "sandbox" / "neil" / "cem_policy_box.npz",
                Path.cwd() / "sandbox" / "neil" / "cem_policy_box.npz",
                REPO_ROOT / "sandbox" / "neil" / "sandbox" / "neil" / "cem_policy_box.npz",
            ]
            for candidate in fallback_paths:
                if candidate.exists():
                    load_path = candidate
                    break
        if not load_path.exists():
            raise FileNotFoundError(f"Policy file not found: {load_path}")
        data = np.load(load_path, allow_pickle=False)
        saved_obs_dim = int(data["obs_dim"]) if "obs_dim" in data else None
        if saved_obs_dim is not None and saved_obs_dim != env.obs_dim:
            raise RuntimeError(
                f"Policy obs_dim={saved_obs_dim} does not match env obs_dim={env.obs_dim}. "
                "Try re-training or pass --obs-mode full."
            )
        params = data["best_params"]
        if args.render:
            render_policy(
                env,
                policy,
                params,
                follow=args.follow,
                follow_body=args.follow_body,
                playback_speed=args.playback_speed,
            )
        else:
            ret, x_final = rollout(env, policy, params, rng)
            print(f"Return: {ret:.2f} | x_final={x_final:.3f}")
        return

    pop_size = int(args.pop_size)
    elite_size = max(1, int(math.ceil(pop_size * args.elite_frac)))

    mean = np.zeros((policy.param_dim,), dtype=np.float32)
    std = np.ones((policy.param_dim,), dtype=np.float32) * float(args.init_std)

    best_return = -np.inf
    best_params = mean.copy()
    best_x = -np.inf

    total_iters = int(args.iters)
    for it in range(1, total_iters + 1):
        samples = rng.normal(mean, std, size=(pop_size, policy.param_dim)).astype(np.float32)
        returns = np.zeros((pop_size,), dtype=np.float32)
        x_finals = np.zeros((pop_size,), dtype=np.float32)

        for i in range(pop_size):
            ret, x_final = rollout(env, policy, samples[i], rng)
            returns[i] = ret
            x_finals[i] = x_final

        elite_idx = np.argsort(returns)[-elite_size:]
        elites = samples[elite_idx]

        mean = elites.mean(axis=0)
        std = elites.std(axis=0) + float(args.min_std)

        iter_best = float(returns[elite_idx[-1]])
        iter_x = float(x_finals[elite_idx[-1]])
        if iter_best > best_return:
            best_return = iter_best
            best_params = elites[-1].copy()
            best_x = iter_x

        progress = 100.0 * it / max(total_iters, 1)
        print(
            f"[CEM] iter {it:03d} | "
            f"{progress:6.2f}% | "
            f"mean={returns.mean():.2f} "
            f"best={iter_best:.2f} "
            f"x_best={iter_x:.3f} "
            f"std={std.mean():.3f}"
        )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        save_path,
        best_params=best_params,
        best_return=best_return,
        best_x=best_x,
        obs_dim=env.obs_dim,
        act_dim=env.nu,
        model_path=str(model_path),
    )

    print(f"Saved best policy to {save_path}")

    if args.render:
        render_policy(env, policy, best_params)


if __name__ == "__main__":
    main()
