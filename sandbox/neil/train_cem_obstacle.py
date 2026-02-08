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


ACTUATOR_NAMES = (
    "front_left_wheel_joint_ctrl",
    "front_right_wheel_joint_ctrl",
    "rear_left_wheel_joint_ctrl",
    "rear_right_wheel_joint_ctrl",
    "front_left_wheel_0_extension_joint_ctrl",
    "front_right_wheel_0_extension_joint_ctrl",
    "rear_left_wheel_0_extension_joint_ctrl",
    "rear_right_wheel_0_extension_joint_ctrl",
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
    / "jacob"
    / "TestingAndDataCollectionFunctions"
    / "Cylindrial Wheel XMLs"
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
        success_x: float = 0.85,
        success_bonus: float = 1.0,
        obs_mode: str = "compact",
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
        self.success_x = float(success_x)
        self.success_bonus = float(success_bonus)
        self.obs_mode = obs_mode
        self.start_on_ground = bool(start_on_ground)
        self.ground_clearance = float(ground_clearance)
        self.retract_legs = bool(retract_legs)
        self.retract_angle = retract_angle
        self.step_count = 0
        self.passed_obstacle = False

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
        self.nu = len(self.actuator_ids)
        if self.nu != 8:
            raise RuntimeError(f"Expected 8 actuators, got {self.nu}.")

        ctrlrange = np.asarray(self.model.actuator_ctrlrange, dtype=np.float32)
        self.ctrl_mid = (ctrlrange[:, 0] + ctrlrange[:, 1]) * 0.5
        self.ctrl_half = (ctrlrange[:, 1] - ctrlrange[:, 0]) * 0.5

        self.leg_qpos_idx = []
        self.leg_joint_ids = []
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

        if self.obs_mode not in {"full", "compact"}:
            raise ValueError("obs_mode must be 'full' or 'compact'")

        self.obs_dim = int(self.model.nq + self.model.nv) if self.obs_mode == "full" else 10

    def _apply_leg_retraction(self) -> None:
        if not self.retract_legs:
            return
        if self.retract_angle is None:
            for jid, qpos_idx in zip(self.leg_joint_ids, self.leg_qpos_idx, strict=True):
                if self.model.jnt_limited[jid]:
                    angle = float(self.model.jnt_range[jid, 0])
                else:
                    angle = 0.0
                self.data.qpos[qpos_idx] = angle
        else:
            angle = float(self.retract_angle)
            for qpos_idx in self.leg_qpos_idx:
                self.data.qpos[qpos_idx] = angle

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
            self.data.ctrl[:] = 0.0
            for _ in range(self.settle_steps):
                mujoco.mj_step(self.model, self.data)

        self.step_count = 0
        self.passed_obstacle = False
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        if self.obs_mode == "full":
            return np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32)

        qpos = self.data.qpos
        qvel = self.data.qvel

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

        obs = np.array([x, z, vx, vz, roll, pitch, *leg_angles], dtype=np.float32)
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
    parser.add_argument("--success-x", type=float, default=0.85)
    parser.add_argument("--success-bonus", type=float, default=1.0)
    parser.add_argument(
        "--obs-mode",
        type=str,
        default="compact",
        choices=("compact", "full"),
        help="Observation set: compact features or full qpos/qvel.",
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

    for it in range(1, int(args.iters) + 1):
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

        print(
            f"[CEM] iter {it:03d} | "
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
