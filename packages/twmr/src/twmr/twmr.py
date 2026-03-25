from pathlib import Path
from typing import Any

import jax
import jax.numpy as jp
from jax import Array as JaxArray
from ml_collections import config_dict
from mujoco import MjModel, mjx  # type: ignore
from mujoco.mjx import Model as MjxModel
from mujoco_playground import MjxEnv, State, dm_control_suite
from mujoco_playground._src import mjx_env
from mujoco_playground._src.dm_control_suite import common

# Current status:
# - During individual phase training, the robot was able to pass a height of
#   0.065, but the first run of the full curriculum only reached 0.045.
# - The robot learns to extend its legs on top of an obstacle, even if the
#   obstacle is very long.
# - The robot does not learn to retract its legs completely after passing an
#   obstacle.
#
# All of the above should be fixable with proper reward function tuning.

ConfigOverridesDict = dict[str, str | int | float | bool | list | None]
_ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "assets"
_XML_BY_VARIANT = {
    # Default task variants now use cylinder-wheel models.
    "flat": "trans_wheel_robo2_2FLAT_CLY.xml",
    "box": "trans_wheel_robo2_2BOX_CLY.xml",
    "box_dual": "trans_wheel_robo2_2BOX_DUAL_CLY.xml",
    "terrain": "trans_wheel_robo2_2GEN_TERR_CLY.xml",
    # Keep explicit sphere-wheel variants for controlled comparisons/debugging.
    "flat_sphere": "trans_wheel_robo2_2FLAT.xml",
    "box_sphere": "trans_wheel_robo2_2BOX.xml",
    "terrain_sphere": "trans_wheel_robo2_2GEN_TERR.xml",
}


def _resolve_xml_path(xml_variant: str) -> Path:
    xml_name = _XML_BY_VARIANT.get(xml_variant)
    if xml_name is None:
        allowed = ", ".join(sorted(_XML_BY_VARIANT))
        raise ValueError(
            f"Unsupported xml_variant '{xml_variant}'. Choose one of: {allowed}."
        )

    xml_path = _ASSETS_DIR / xml_name
    if not xml_path.exists():
        raise FileNotFoundError(
            f"XML file for xml_variant '{xml_variant}' was not found: {xml_path}"
        )

    return xml_path


def _load_model_assets(assets_dir: Path) -> dict[str, Any]:
    # Include playground assets plus all local assets, preserving relative paths
    # for XML references like "meshes/foo.bin".
    model_assets = dict(common.get_assets())
    for file_path in assets_dir.rglob("*"):
        if file_path.is_file():
            rel_path = file_path.relative_to(assets_dir).as_posix()
            model_assets[rel_path] = file_path.read_bytes()
    return model_assets


# def default_vision_config() -> config_dict.ConfigDict:
#     return config_dict.create(
#         gpu_id=0,
#         render_batch_size=512,
#         render_width=64,
#         render_height=64,
#         enable_geom_groups=[0, 1, 2],
#         use_rasterizer=False,
#         history=3,
#     )


# TODO: check all of these default values
def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,  # 50 hz control
        sim_dt=0.01,
        episode_length=1000,
        action_repeat=1,  # TODO: should this be a ratio of ctrl_dt / sim_dt?
        xml_variant="flat",  # flat, box, terrain (cylinder defaults), *_sphere fallback
        vision=False,
        # vision_config=default_vision_config(),
        impl="warp",  # TODO: cartpole uses jax
        nconmax=100,  # allow collisions
        njmax=500,  # allow complex joints
        forward_reward_weight=1.0,
        progress_delta_reward_weight=5.0,
        survival_reward=0.02,
        control_cost_weight=0.001,
        lateral_velocity_cost_weight=0.05,
        tilt_cost_weight=0.1,
        backward_velocity_cost_weight=0.5,
        backward_delta_cost_weight=5.0,
        stall_penalty_weight=0.02,
        stall_penalty_after_success=False,
        stall_delta_x_threshold=5e-4,
        failure_penalty=2.0,
        success_bonus=15.0,
        post_success_progress_weight=0.0,
        min_base_height=0.02,
        min_upright=0.2,
        obstacle_geom_name="traverse_box",
        obstacle_slide_joint_name="traverse_box_slide_x",
        obstacle_site_name="traverse_box_site",
        obstacle_x_position=None,
        obstacle_half_length=None,
        obstacle_height=None,
        enable_second_obstacle=False,
        obstacle2_geom_name="traverse_box2",
        obstacle2_slide_joint_name="traverse_box2_slide_x",
        obstacle2_site_name="traverse_box2_site",
        obstacle2_x_position=None,
        obstacle2_half_length=None,
        obstacle2_height=None,
        randomize_obstacles=False,
        randomize_obstacle_x_min=0.45,
        randomize_obstacle_x_max=0.85,
        randomize_obstacle_gap_min=0.25,
        randomize_obstacle_gap_max=0.55,
        randomize_obstacle_height_min=None,
        randomize_obstacle_height_max=None,
        randomize_obstacle_half_length_min=None,
        randomize_obstacle_half_length_max=None,
        # Diagnostics/perf controls for randomized obstacle execution path.
        # Keep defaults matching current behavior.
        apply_episode_geometry_in_step=True,
        lock_obstacle_layout_during_episode=True,
        forward_after_layout_lock=True,
        obstacle_local_windowed_reward=False,
        obstacle_local_activation_margin=0.2,
        obstacle_local_progress_weight=8.0,
        obstacle_local_backtrack_cost_weight=12.0,
        obstacle_local_extension_reward_weight=0.25,
        obstacle_extension_peak_fraction=0.25,
        extension_retracted_penalty_weight=0.0,
        extension_retracted_penalty_outside_obstacle_only=True,
        extension_retracted_penalty_on_obstacle_top=True,
        reset_extensions_retracted=True,
        forward_only_when_no_obstacle=True,
        freeze_extensions_when_no_obstacle=True,
        success_x_threshold=None,
        success_x_margin=0.1,
        terminate_on_success=True,
    )


class TransformableWheelMobileRobot(MjxEnv):
    def __init__(
        self,
        # Task specific config
        config: config_dict.ConfigDict = default_config(),
        config_overrides: ConfigOverridesDict | None = None,
    ):
        super().__init__(config, config_overrides)

        xml_path = _resolve_xml_path(str(self._config.xml_variant))
        self._xml_path = xml_path.as_posix()
        model_xml = xml_path.read_text(encoding="utf-8")
        self._model_assets = _load_model_assets(_ASSETS_DIR)
        self._mj_model: MjModel = MjModel.from_xml_string(model_xml, self._model_assets)
        self._mj_model.opt.timestep = self.sim_dt
        self._configure_task_geometry()
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)  # type: ignore
        self._post_init()

        # TODO: figure out vision with the madrona batch renderer

        # TODO: what does this do for us exactly?
        # self._root_body_id = self._mj_model.body("root").id

    def _maybe_get_geom_id(self, geom_name: str) -> int | None:
        try:
            return self._mj_model.geom(geom_name).id
        except KeyError:
            return None

    def _maybe_get_site_id(self, site_name: str) -> int | None:
        try:
            return self._mj_model.site(site_name).id
        except KeyError:
            return None

    def _maybe_get_joint_id(self, joint_name: str) -> int | None:
        try:
            return self._mj_model.joint(joint_name).id
        except KeyError:
            return None

    def _resolve_obstacle_center_x(self, geom_id: int, slide_qpos_adr: int | None) -> float:
        body_id = int(self._mj_model.geom_bodyid[geom_id])
        center_x = float(self._mj_model.body_pos[body_id, 0] + self._mj_model.geom_pos[geom_id, 0])
        return center_x

    def _set_obstacle_center_x(
        self,
        *,
        geom_id: int,
        site_id: int | None,
        slide_qpos_adr: int | None,
        target_x: float,
    ) -> float:
        if slide_qpos_adr is not None:
            # For slide joints, keep qpos0 at the XML reference pose and apply
            # obstacle placement only through runtime qpos values. Mutating
            # qpos0 causes MuJoCo replay/rendering to cancel the slide offset.
            return float(target_x)

        body_id = int(self._mj_model.geom_bodyid[geom_id])
        local_x = float(target_x) - float(self._mj_model.body_pos[body_id, 0])
        self._mj_model.geom_pos[geom_id, 0] = local_x
        if site_id is not None:
            self._mj_model.site_pos[site_id, 0] = local_x
        return float(target_x)

    def _parse_randomization_range(
        self,
        *,
        min_value: float | None,
        max_value: float | None,
        range_name: str,
    ) -> tuple[float, float] | None:
        if min_value is None and max_value is None:
            return None
        if min_value is None or max_value is None:
            raise ValueError(f"{range_name}_min and {range_name}_max must be set together.")
        parsed_min = float(min_value)
        parsed_max = float(max_value)
        if parsed_min <= 0.0 or parsed_max <= 0.0:
            raise ValueError(f"{range_name}_min and {range_name}_max must both be > 0.")
        if parsed_max < parsed_min:
            raise ValueError(f"{range_name}_max must be >= {range_name}_min.")
        return parsed_min, parsed_max

    def _build_episode_model(
        self,
        obstacle_half_length: JaxArray,
        obstacle_height: JaxArray,
    ) -> MjxModel:
        if not self._randomize_obstacle_geometry or self._obstacle_geom_id is None:
            return self.mjx_model

        model = self.mjx_model
        target_half_length = jp.maximum(
            jp.asarray(obstacle_half_length, dtype=jp.float32),
            jp.array(1e-3, dtype=jp.float32),
        )
        target_half_height = jp.maximum(
            0.5 * jp.asarray(obstacle_height, dtype=jp.float32),
            jp.array(1e-3, dtype=jp.float32),
        )
        geom_size = model.geom_size.at[self._obstacle_geom_id, 0].set(target_half_length)
        geom_size = geom_size.at[self._obstacle_geom_id, 2].set(target_half_height)
        geom_pos = model.geom_pos.at[self._obstacle_geom_id, 2].set(target_half_height)
        model = model.replace(geom_size=geom_size, geom_pos=geom_pos)

        if self._obstacle_site_id is not None:
            site_pos = model.site_pos.at[self._obstacle_site_id, 2].set(target_half_height)
            model = model.replace(site_pos=site_pos)

        return model

    def _configure_task_geometry(self) -> None:
        fallback_obstacle_x = 0.6
        fallback_obstacle_half_length = 0.2
        fallback_obstacle_height = 0.06

        obstacle_geom_name = str(self._config.obstacle_geom_name)
        obstacle_site_name = str(self._config.obstacle_site_name)
        obstacle_slide_joint_name = str(self._config.obstacle_slide_joint_name)
        self._obstacle_geom_id = self._maybe_get_geom_id(obstacle_geom_name)
        obstacle_site_id = self._maybe_get_site_id(obstacle_site_name)
        self._obstacle_site_id = obstacle_site_id
        obstacle_slide_joint_id = self._maybe_get_joint_id(obstacle_slide_joint_name)
        self._obstacle_slide_qpos_adr = (
            int(self._mj_model.jnt_qposadr[obstacle_slide_joint_id])
            if obstacle_slide_joint_id is not None
            else None
        )
        self._obstacle_slide_dof_adr = (
            int(self._mj_model.jnt_dofadr[obstacle_slide_joint_id])
            if obstacle_slide_joint_id is not None
            else None
        )
        self._obstacle2_hidden_slide_qpos = None

        if self._obstacle_geom_id is not None:
            obstacle_body_id = int(self._mj_model.geom_bodyid[self._obstacle_geom_id])
            self._obstacle_slide_base_center_x = float(
                self._mj_model.body_pos[obstacle_body_id, 0]
                + self._mj_model.geom_pos[self._obstacle_geom_id, 0]
            )
            if self._config.obstacle_x_position is not None:
                target_x = float(self._config.obstacle_x_position)
                self._obstacle_x_position = self._set_obstacle_center_x(
                    geom_id=self._obstacle_geom_id,
                    site_id=obstacle_site_id,
                    slide_qpos_adr=self._obstacle_slide_qpos_adr,
                    target_x=target_x,
                )
                if self._obstacle_slide_qpos_adr is None:
                    self._obstacle_slide_base_center_x = self._obstacle_x_position

            if self._config.obstacle_half_length is not None:
                self._mj_model.geom_size[self._obstacle_geom_id, 0] = max(
                    float(self._config.obstacle_half_length), 1e-3
                )

            if self._config.obstacle_height is not None:
                half_height = max(0.5 * float(self._config.obstacle_height), 1e-3)
                self._mj_model.geom_size[self._obstacle_geom_id, 2] = half_height
                # Keep the obstacle base on the ground.
                self._mj_model.geom_pos[self._obstacle_geom_id, 2] = half_height
                if obstacle_site_id is not None:
                    self._mj_model.site_pos[obstacle_site_id, 2] = half_height

            if self._config.obstacle_x_position is None:
                self._obstacle_x_position = self._resolve_obstacle_center_x(
                    self._obstacle_geom_id, self._obstacle_slide_qpos_adr
                )
            self._obstacle_half_length = float(
                self._mj_model.geom_size[self._obstacle_geom_id, 0]
            )
            self._obstacle_height = float(
                self._mj_model.geom_pos[self._obstacle_geom_id, 2]
                + self._mj_model.geom_size[self._obstacle_geom_id, 2]
            )
        else:
            self._obstacle_slide_base_center_x = fallback_obstacle_x
            self._obstacle_x_position = (
                float(self._config.obstacle_x_position)
                if self._config.obstacle_x_position is not None
                else fallback_obstacle_x
            )
            self._obstacle_half_length = (
                max(float(self._config.obstacle_half_length), 1e-3)
                if self._config.obstacle_half_length is not None
                else fallback_obstacle_half_length
            )
            self._obstacle_height = (
                max(float(self._config.obstacle_height), 1e-3)
                if self._config.obstacle_height is not None
                else fallback_obstacle_height
            )

        self._randomize_obstacle_height_range = self._parse_randomization_range(
            min_value=(
                self._config.randomize_obstacle_height_min  # type: ignore[arg-type]
            ),
            max_value=(
                self._config.randomize_obstacle_height_max  # type: ignore[arg-type]
            ),
            range_name="randomize_obstacle_height",
        )
        self._randomize_obstacle_half_length_range = self._parse_randomization_range(
            min_value=(
                self._config.randomize_obstacle_half_length_min  # type: ignore[arg-type]
            ),
            max_value=(
                self._config.randomize_obstacle_half_length_max  # type: ignore[arg-type]
            ),
            range_name="randomize_obstacle_half_length",
        )
        self._randomize_obstacle_geometry = (
            self._randomize_obstacle_height_range is not None
            or self._randomize_obstacle_half_length_range is not None
        )
        if self._randomize_obstacle_geometry and self._obstacle_geom_id is None:
            raise ValueError(
                "Obstacle geometry randomization requires a valid obstacle geom "
                f"'{obstacle_geom_name}'."
            )

        self._has_second_obstacle = bool(self._config.enable_second_obstacle)
        obstacle2_geom_name = str(self._config.obstacle2_geom_name)
        obstacle2_site_name = str(self._config.obstacle2_site_name)
        obstacle2_slide_joint_name = str(self._config.obstacle2_slide_joint_name)
        obstacle2_geom_id = self._maybe_get_geom_id(obstacle2_geom_name)
        obstacle2_site_id = self._maybe_get_site_id(obstacle2_site_name)
        obstacle2_slide_joint_id = self._maybe_get_joint_id(obstacle2_slide_joint_name)

        self._obstacle2_slide_qpos_adr = (
            int(self._mj_model.jnt_qposadr[obstacle2_slide_joint_id])
            if obstacle2_slide_joint_id is not None
            else None
        )
        self._obstacle2_slide_dof_adr = (
            int(self._mj_model.jnt_dofadr[obstacle2_slide_joint_id])
            if obstacle2_slide_joint_id is not None
            else None
        )

        if self._has_second_obstacle:
            if obstacle2_geom_id is None:
                raise ValueError(
                    "enable_second_obstacle=true requires obstacle2 geom. "
                    f"Missing geom '{obstacle2_geom_name}'."
                )
            self._obstacle2_geom_id = obstacle2_geom_id
            obstacle2_body_id = int(self._mj_model.geom_bodyid[obstacle2_geom_id])
            self._obstacle2_slide_base_center_x = float(
                self._mj_model.body_pos[obstacle2_body_id, 0]
                + self._mj_model.geom_pos[obstacle2_geom_id, 0]
            )

            if self._config.obstacle2_x_position is not None:
                target_x2 = float(self._config.obstacle2_x_position)
                self._obstacle2_x_position = self._set_obstacle_center_x(
                    geom_id=obstacle2_geom_id,
                    site_id=obstacle2_site_id,
                    slide_qpos_adr=self._obstacle2_slide_qpos_adr,
                    target_x=target_x2,
                )
                if self._obstacle2_slide_qpos_adr is None:
                    self._obstacle2_slide_base_center_x = self._obstacle2_x_position

            if self._config.obstacle2_half_length is not None:
                self._mj_model.geom_size[obstacle2_geom_id, 0] = max(
                    float(self._config.obstacle2_half_length), 1e-3
                )

            if self._config.obstacle2_height is not None:
                half_height2 = max(0.5 * float(self._config.obstacle2_height), 1e-3)
                self._mj_model.geom_size[obstacle2_geom_id, 2] = half_height2
                self._mj_model.geom_pos[obstacle2_geom_id, 2] = half_height2
                if obstacle2_site_id is not None:
                    self._mj_model.site_pos[obstacle2_site_id, 2] = half_height2

            if self._config.obstacle2_x_position is None:
                self._obstacle2_x_position = self._resolve_obstacle_center_x(
                    obstacle2_geom_id, self._obstacle2_slide_qpos_adr
                )
            self._obstacle2_half_length = float(self._mj_model.geom_size[obstacle2_geom_id, 0])
            self._obstacle2_height = float(
                self._mj_model.geom_pos[obstacle2_geom_id, 2]
                + self._mj_model.geom_size[obstacle2_geom_id, 2]
            )
        else:
            self._obstacle2_geom_id = None
            self._obstacle2_slide_base_center_x = -5.0
            self._obstacle2_x_position = -5.0
            self._obstacle2_half_length = self._obstacle_half_length
            self._obstacle2_height = self._obstacle_height

            # Keep optional second obstacle geometry out of the way if present.
            if obstacle2_geom_id is not None:
                obstacle2_body_id = int(self._mj_model.geom_bodyid[obstacle2_geom_id])
                obstacle2_base_x = float(
                    self._mj_model.body_pos[obstacle2_body_id, 0]
                    + self._mj_model.geom_pos[obstacle2_geom_id, 0]
                )
                if self._obstacle2_slide_qpos_adr is not None:
                    self._obstacle2_hidden_slide_qpos = -5.0 - obstacle2_base_x
                else:
                    local_x2 = -5.0 - float(self._mj_model.body_pos[obstacle2_body_id, 0])
                    self._mj_model.geom_pos[obstacle2_geom_id, 0] = local_x2
                    if obstacle2_site_id is not None:
                        self._mj_model.site_pos[obstacle2_site_id, 0] = local_x2

        if self._config.randomize_obstacles:
            if self._obstacle_slide_qpos_adr is None:
                raise ValueError(
                    "randomize_obstacles=true requires obstacle slide joint "
                    f"'{obstacle_slide_joint_name}'."
                )
            if self._has_second_obstacle and self._obstacle2_slide_qpos_adr is None:
                raise ValueError(
                    "randomize_obstacles with enable_second_obstacle=true requires obstacle2 "
                    f"slide joint '{obstacle2_slide_joint_name}'."
                )
            if float(self._config.randomize_obstacle_x_min) >= float(
                self._config.randomize_obstacle_x_max
            ):
                raise ValueError(
                    "randomize_obstacle_x_min must be < randomize_obstacle_x_max."
                )
            if float(self._config.randomize_obstacle_gap_min) >= float(
                self._config.randomize_obstacle_gap_max
            ):
                raise ValueError(
                    "randomize_obstacle_gap_min must be < randomize_obstacle_gap_max."
                )

        if self._config.success_x_threshold is not None:
            self._success_x_threshold = float(self._config.success_x_threshold)
        else:
            success_anchor_x = (
                self._obstacle2_x_position + self._obstacle2_half_length
                if self._has_second_obstacle
                else self._obstacle_x_position + self._obstacle_half_length
            )
            self._success_x_threshold = success_anchor_x + float(self._config.success_x_margin)
        self._obstacle_front_x = self._obstacle_x_position - self._obstacle_half_length
        self._obstacle_back_x = self._obstacle_x_position + self._obstacle_half_length
        self._obstacle2_front_x = self._obstacle2_x_position - self._obstacle2_half_length
        self._obstacle2_back_x = self._obstacle2_x_position + self._obstacle2_half_length
        self._obstacle_local_start_x = (
            self._obstacle_front_x - float(self._config.obstacle_local_activation_margin)
        )
        self._obstacle_extension_peak_fraction = float(self._config.obstacle_extension_peak_fraction)
        if not 0.0 <= self._obstacle_extension_peak_fraction <= 1.0:
            raise ValueError("obstacle_extension_peak_fraction must be within [0, 1].")

    def _post_init(self) -> None:
        try:
            root_joint_id = self._mj_model.joint("root").id
            self._root_qpos_adr = int(self._mj_model.jnt_qposadr[root_joint_id])
            self._root_qvel_adr = int(self._mj_model.jnt_dofadr[root_joint_id])
            self._root_body_id = self._mj_model.body("root").id
        except Exception as err:  # pragma: no cover - defensive init guard
            raise ValueError(
                "Could not resolve required root joint/body indices in the TWMR model."
            ) from err
        self._extension_actuator_ids = tuple(
            i for i in range(self._mj_model.nu) if "_extension_" in self._mj_model.actuator(i).name
        )
        self._extension_actuator_idx = jp.array(self._extension_actuator_ids, dtype=jp.int32)
        self._has_extension_actuators = bool(self._extension_actuator_ids)
        self._extension_retract_ctrl = jp.array(
            [float(self._mj_model.actuator_ctrlrange[i, 0]) for i in self._extension_actuator_ids],
            dtype=jp.float32,
        )
        # Cache actuated extension joint qpos indices and "fully retracted"
        # setpoints so episode reset can enforce a consistent starting geometry.
        extension_joint_qpos_adrs: list[int] = []
        extension_joint_dof_adrs: list[int] = []
        extension_joint_retracted_qpos: list[float] = []
        extension_joint_range_span: list[float] = []
        seen_extension_joint_ids: set[int] = set()
        for actuator_id in self._extension_actuator_ids:
            joint_id = int(self._mj_model.actuator_trnid[actuator_id, 0])
            if joint_id < 0 or joint_id in seen_extension_joint_ids:
                continue
            seen_extension_joint_ids.add(joint_id)
            qpos_adr = int(self._mj_model.jnt_qposadr[joint_id])
            dof_adr = int(self._mj_model.jnt_dofadr[joint_id])
            if bool(self._mj_model.jnt_limited[joint_id]):
                retracted_qpos = float(self._mj_model.jnt_range[joint_id, 0])
                range_span = max(
                    float(self._mj_model.jnt_range[joint_id, 1]) - retracted_qpos,
                    1e-6,
                )
            else:
                retracted_qpos = float(self._mj_model.qpos0[qpos_adr])
                range_span = 1.0
            extension_joint_qpos_adrs.append(qpos_adr)
            extension_joint_dof_adrs.append(dof_adr)
            extension_joint_retracted_qpos.append(retracted_qpos)
            extension_joint_range_span.append(range_span)
        self._extension_joint_qpos_adrs = tuple(extension_joint_qpos_adrs)
        self._extension_joint_dof_adrs = tuple(extension_joint_dof_adrs)
        self._extension_joint_retracted_qpos = tuple(extension_joint_retracted_qpos)
        self._extension_joint_qpos_idx = jp.array(self._extension_joint_qpos_adrs, dtype=jp.int32)
        self._extension_joint_retracted_qpos_arr = jp.array(
            self._extension_joint_retracted_qpos, dtype=jp.float32
        )
        self._extension_joint_range_span_arr = jp.array(extension_joint_range_span, dtype=jp.float32)
        nominal_root_height = float(self._mj_model.body(self._root_body_id).pos[2])
        # Keep failure height below nominal spawn height so episodes do not terminate immediately.
        self._effective_min_base_height = min(
            float(self._config.min_base_height),
            max(0.65 * nominal_root_height, 0.01),
        )

        # Keep policy inputs robot-proprioceptive only by excluding obstacle slide
        # joints (used for randomized layout) from the observation vector.
        excluded_qpos = {
            adr
            for adr in (self._obstacle_slide_qpos_adr, self._obstacle2_slide_qpos_adr)
            if adr is not None
        }
        excluded_qvel = {
            adr
            for adr in (self._obstacle_slide_dof_adr, self._obstacle2_slide_dof_adr)
            if adr is not None
        }
        obs_qpos_indices = [i for i in range(self.mjx_model.nq) if i not in excluded_qpos]
        obs_qvel_indices = [i for i in range(self.mjx_model.nv) if i not in excluded_qvel]
        self._obs_qpos_idx = jp.array(obs_qpos_indices, dtype=jp.int32)
        self._obs_qvel_idx = jp.array(obs_qvel_indices, dtype=jp.int32)

    def reset(self, rng: JaxArray) -> State:
        # TODO: randomize initial state (qpos, qvel)
        # qpos = qpos.at[2].set(0.2)
        # qpos = qpos + 0.01 * jax.random.normal(rng_init, qpos.shape)

        # Initially reset to the original position
        # qpos = jp.zeros(self.mjx_model.nq)
        # qvel = jp.zeros(self.mjx_model.nv)

        obstacle_x = jp.array(self._obstacle_x_position, dtype=jp.float32)
        obstacle2_x = jp.array(self._obstacle2_x_position, dtype=jp.float32)
        obstacle_half_length = jp.array(self._obstacle_half_length, dtype=jp.float32)
        obstacle_height = jp.array(self._obstacle_height, dtype=jp.float32)

        if self._randomize_obstacle_height_range is not None:
            rng, obstacle_height_rng = jax.random.split(rng)
            obstacle_height = jax.random.uniform(
                obstacle_height_rng,
                shape=(),
                minval=self._randomize_obstacle_height_range[0],
                maxval=self._randomize_obstacle_height_range[1],
            ).astype(jp.float32)

        if self._randomize_obstacle_half_length_range is not None:
            rng, obstacle_half_length_rng = jax.random.split(rng)
            obstacle_half_length = jax.random.uniform(
                obstacle_half_length_rng,
                shape=(),
                minval=self._randomize_obstacle_half_length_range[0],
                maxval=self._randomize_obstacle_half_length_range[1],
            ).astype(jp.float32)

        model = self._build_episode_model(
            obstacle_half_length=obstacle_half_length,
            obstacle_height=obstacle_height,
        )
        data = mjx.make_data(
            model,
            # qpos=qpos,
            # qvel=qvel,
            impl=model.impl.value,
            nconmax=self._config.nconmax,  # type: ignore
            njmax=self._config.njmax,  # type: ignore
        )

        if self._config.randomize_obstacles:
            rng, x_rng = jax.random.split(rng)
            obstacle_x = jax.random.uniform(
                x_rng,
                shape=(),
                minval=float(self._config.randomize_obstacle_x_min),  # type: ignore[arg-type]
                maxval=float(self._config.randomize_obstacle_x_max),  # type: ignore[arg-type]
            ).astype(jp.float32)
            if self._has_second_obstacle:
                rng, gap_rng = jax.random.split(rng)
                gap = jax.random.uniform(
                    gap_rng,
                    shape=(),
                    minval=float(self._config.randomize_obstacle_gap_min),  # type: ignore[arg-type]
                    maxval=float(self._config.randomize_obstacle_gap_max),  # type: ignore[arg-type]
                ).astype(jp.float32)
                obstacle2_x = (
                    obstacle_x
                    + jp.array(
                        self._obstacle2_half_length, dtype=jp.float32
                    )
                    + obstacle_half_length
                    + gap
                )

        qpos = data.qpos
        if bool(self._config.reset_extensions_retracted) and self._extension_joint_qpos_adrs:  # type: ignore
            for qpos_adr, retracted_qpos in zip(
                self._extension_joint_qpos_adrs, self._extension_joint_retracted_qpos
            ):
                qpos = qpos.at[qpos_adr].set(jp.array(retracted_qpos, dtype=jp.float32))
        if self._obstacle_slide_qpos_adr is not None:
            qpos = qpos.at[self._obstacle_slide_qpos_adr].set(
                obstacle_x - jp.array(self._obstacle_slide_base_center_x, dtype=jp.float32)
            )
        if self._has_second_obstacle and self._obstacle2_slide_qpos_adr is not None:
            qpos = qpos.at[self._obstacle2_slide_qpos_adr].set(
                obstacle2_x - jp.array(self._obstacle2_slide_base_center_x, dtype=jp.float32)
            )
        elif self._obstacle2_hidden_slide_qpos is not None and self._obstacle2_slide_qpos_adr is not None:
            qpos = qpos.at[self._obstacle2_slide_qpos_adr].set(
                jp.array(self._obstacle2_hidden_slide_qpos, dtype=jp.float32)
            )

        obstacle_slide_qpos = (
            qpos[self._obstacle_slide_qpos_adr]
            if self._obstacle_slide_qpos_adr is not None
            else jp.array(0.0, dtype=jp.float32)
        )
        obstacle2_slide_qpos = (
            qpos[self._obstacle2_slide_qpos_adr]
            if self._obstacle2_slide_qpos_adr is not None
            else jp.array(0.0, dtype=jp.float32)
        )

        data = data.replace(qpos=qpos)
        data = mjx.forward(model, data)

        metrics = self._empty_metrics()
        root_x0 = data.qpos[self._root_qpos_adr + 0]
        obstacle2_half_length = jp.array(self._obstacle2_half_length, dtype=jp.float32)
        obstacle_front_x = obstacle_x - obstacle_half_length
        obstacle_back_x = obstacle_x + obstacle_half_length
        obstacle2_front_x = obstacle2_x - obstacle2_half_length
        obstacle2_back_x = obstacle2_x + obstacle2_half_length

        no_obstacles_task = self._obstacle_geom_id is None and not self._has_second_obstacle
        if self._config.success_x_threshold is not None:
            success_x_threshold = jp.array(float(self._config.success_x_threshold), dtype=jp.float32)
        elif self._has_second_obstacle:
            success_x_threshold = obstacle2_back_x + jp.array(
                float(self._config.success_x_margin), dtype=jp.float32
            )
        elif no_obstacles_task:
            # Flat/no-obstacle variants should not terminate on synthetic
            # obstacle-derived success criteria.
            success_x_threshold = root_x0 + jp.array(1_000_000.0, dtype=jp.float32)
        else:
            success_x_threshold = obstacle_back_x + jp.array(
                float(self._config.success_x_margin), dtype=jp.float32
            )

        info = {
            "rng": rng,
            "prev_root_x": root_x0,
            "obstacle_x_position": obstacle_x,
            "obstacle_front_x": obstacle_front_x,
            "obstacle_back_x": obstacle_back_x,
            "obstacle2_x_position": obstacle2_x,
            "obstacle2_front_x": obstacle2_front_x,
            "obstacle2_back_x": obstacle2_back_x,
            "success_x_threshold": success_x_threshold,
            "obstacle_local_start_x": obstacle_front_x
            - jp.array(float(self._config.obstacle_local_activation_margin), dtype=jp.float32),
            "obstacle_slide_qpos": obstacle_slide_qpos,
            "obstacle2_slide_qpos": obstacle2_slide_qpos,
            "episode_obstacle_height": obstacle_height,
            "episode_obstacle_half_length": obstacle_half_length,
        }
        if self._randomize_obstacle_geometry:
            info["episode_geom_size"] = model.geom_size
            info["episode_geom_pos"] = model.geom_pos
            if self._obstacle_site_id is not None:
                info["episode_site_pos"] = model.site_pos

        obs = self._get_obs(data, info)

        return mjx_env.State(
            data=data,
            obs=obs,
            reward=jp.array(0.0),
            done=jp.array(0.0),
            metrics=metrics,
            info=info,
        )

    def step(self, state: State, action: JaxArray) -> State:
        no_obstacles_task = self._obstacle_geom_id is None and not self._has_second_obstacle
        if (
            no_obstacles_task
            and bool(self._config.freeze_extensions_when_no_obstacle)  # type: ignore
            and self._has_extension_actuators
        ):
            action = action.at[self._extension_actuator_idx].set(self._extension_retract_ctrl)

        episode_obstacle_height = state.info.get(
            "episode_obstacle_height", jp.array(self._obstacle_height, dtype=jp.float32)
        )
        episode_obstacle_half_length = state.info.get(
            "episode_obstacle_half_length",
            jp.array(self._obstacle_half_length, dtype=jp.float32),
        )
        model = self.mjx_model
        if self._randomize_obstacle_geometry and bool(
            self._config.apply_episode_geometry_in_step
        ):  # type: ignore
            episode_geom_size = state.info.get("episode_geom_size", self.mjx_model.geom_size)
            episode_geom_pos = state.info.get("episode_geom_pos", self.mjx_model.geom_pos)
            if self._obstacle_site_id is not None:
                episode_site_pos = state.info.get("episode_site_pos", self.mjx_model.site_pos)
                model = model.replace(
                    geom_size=episode_geom_size,
                    geom_pos=episode_geom_pos,
                    site_pos=episode_site_pos,
                )
            else:
                model = model.replace(
                    geom_size=episode_geom_size,
                    geom_pos=episode_geom_pos,
                )
        data = mjx_env.step(model, state.data, action, self.n_substeps)

        # Keep randomized obstacle layouts fixed during the episode.
        obstacle_layout_locked = False
        extension_state_locked = False
        qpos = data.qpos
        qvel = data.qvel
        lock_obstacle_layout = bool(self._config.lock_obstacle_layout_during_episode)  # type: ignore
        if lock_obstacle_layout and self._obstacle_slide_qpos_adr is not None:
            locked_qpos = state.info.get(
                "obstacle_slide_qpos", qpos[self._obstacle_slide_qpos_adr]
            )
            qpos = qpos.at[self._obstacle_slide_qpos_adr].set(locked_qpos)
            obstacle_layout_locked = True
            if self._obstacle_slide_dof_adr is not None:
                qvel = qvel.at[self._obstacle_slide_dof_adr].set(0.0)

        if lock_obstacle_layout and self._obstacle2_slide_qpos_adr is not None:
            locked_qpos2 = state.info.get(
                "obstacle2_slide_qpos", qpos[self._obstacle2_slide_qpos_adr]
            )
            qpos = qpos.at[self._obstacle2_slide_qpos_adr].set(locked_qpos2)
            obstacle_layout_locked = True
            if self._obstacle2_slide_dof_adr is not None:
                qvel = qvel.at[self._obstacle2_slide_dof_adr].set(0.0)

        if (
            no_obstacles_task
            and bool(self._config.freeze_extensions_when_no_obstacle)  # type: ignore
            and self._extension_joint_qpos_adrs
        ):
            for qpos_adr, dof_adr, retracted_qpos in zip(
                self._extension_joint_qpos_adrs,
                self._extension_joint_dof_adrs,
                self._extension_joint_retracted_qpos,
            ):
                qpos = qpos.at[qpos_adr].set(jp.array(retracted_qpos, dtype=jp.float32))
                qvel = qvel.at[dof_adr].set(jp.array(0.0, dtype=jp.float32))
            extension_state_locked = True

        if obstacle_layout_locked or extension_state_locked:
            data = data.replace(qpos=qpos, qvel=qvel)
            if bool(self._config.forward_after_layout_lock):  # type: ignore
                data = mjx.forward(model, data)

        root_x = data.qpos[self._root_qpos_adr + 0]
        root_y = data.qpos[self._root_qpos_adr + 1]
        root_z = data.qpos[self._root_qpos_adr + 2]
        root_quat = data.qpos[self._root_qpos_adr + 3 : self._root_qpos_adr + 7]
        root_vx = data.qvel[self._root_qvel_adr + 0]
        root_vy = data.qvel[self._root_qvel_adr + 1]
        upright = self._compute_upright(root_quat)
        prev_root_x = state.info.get("prev_root_x", root_x)
        success_x_threshold = state.info.get(
            "success_x_threshold", jp.array(self._success_x_threshold, dtype=jp.float32)
        )
        obstacle_x_position = state.info.get(
            "obstacle_x_position", jp.array(self._obstacle_x_position, dtype=jp.float32)
        )
        obstacle_front_x = state.info.get(
            "obstacle_front_x", jp.array(self._obstacle_front_x, dtype=jp.float32)
        )
        obstacle_back_x = state.info.get(
            "obstacle_back_x", jp.array(self._obstacle_back_x, dtype=jp.float32)
        )
        obstacle2_x_position = state.info.get(
            "obstacle2_x_position", jp.array(self._obstacle2_x_position, dtype=jp.float32)
        )
        obstacle2_front_x = state.info.get(
            "obstacle2_front_x", jp.array(self._obstacle2_front_x, dtype=jp.float32)
        )
        obstacle2_back_x = state.info.get(
            "obstacle2_back_x", jp.array(self._obstacle2_back_x, dtype=jp.float32)
        )
        delta_x = root_x - prev_root_x
        distance_to_success = success_x_threshold - root_x
        backward_vx = jp.maximum(-root_vx, 0.0)
        forward_delta_x = jp.maximum(delta_x, 0.0)
        backward_delta_x = jp.maximum(-delta_x, 0.0)

        if no_obstacles_task:
            obstacle_local_active = jp.array(False)
            obstacle_extension_active = jp.array(False)
            obstacle_top_active = jp.array(False)
        elif self._config.obstacle_local_windowed_reward:
            margin = jp.array(float(self._config.obstacle_local_activation_margin), dtype=jp.float32)
            peak_fraction = jp.array(self._obstacle_extension_peak_fraction, dtype=jp.float32)
            obstacle_extension_peak_x_1 = obstacle_front_x + peak_fraction * (
                obstacle_back_x - obstacle_front_x
            )
            obstacle_extension_active_1 = (root_x >= (obstacle_front_x - margin)) & (
                root_x <= obstacle_extension_peak_x_1
            )
            obstacle_top_active_1 = (root_x > obstacle_extension_peak_x_1) & (
                root_x <= (obstacle_back_x + margin)
            )
            obstacle_local_active_1 = (root_x >= (obstacle_front_x - margin)) & (
                root_x <= (obstacle_back_x + margin)
            )
            if self._has_second_obstacle:
                obstacle_extension_peak_x_2 = obstacle2_front_x + peak_fraction * (
                    obstacle2_back_x - obstacle2_front_x
                )
                obstacle_extension_active_2 = (root_x >= (obstacle2_front_x - margin)) & (
                    root_x <= obstacle_extension_peak_x_2
                )
                obstacle_top_active_2 = (root_x > obstacle_extension_peak_x_2) & (
                    root_x <= (obstacle2_back_x + margin)
                )
                obstacle_local_active_2 = (root_x >= (obstacle2_front_x - margin)) & (
                    root_x <= (obstacle2_back_x + margin)
                )
            else:
                obstacle_extension_active_2 = jp.array(False)
                obstacle_top_active_2 = jp.array(False)
                obstacle_local_active_2 = jp.array(False)
            obstacle_extension_active = obstacle_extension_active_1 | obstacle_extension_active_2
            obstacle_top_active = obstacle_top_active_1 | obstacle_top_active_2
            obstacle_local_active = obstacle_local_active_1 | obstacle_local_active_2
        else:
            obstacle_local_start_x = state.info.get(
                "obstacle_local_start_x", jp.array(self._obstacle_local_start_x, dtype=jp.float32)
            )
            obstacle_local_active = (
                (root_x >= obstacle_local_start_x)
                & (root_x <= success_x_threshold)
                & (distance_to_success > 0.0)
            )
            obstacle_extension_active = obstacle_local_active
            obstacle_top_active = jp.array(False)

        obstacle_local_active_f = obstacle_local_active.astype(jp.float32)
        obstacle_extension_active_f = obstacle_extension_active.astype(jp.float32)
        obstacle_top_active_f = obstacle_top_active.astype(jp.float32)
        if self._has_extension_actuators:
            extension_action_magnitude = jp.mean(jp.abs(action[self._extension_actuator_idx]))
        else:
            extension_action_magnitude = jp.array(0.0, dtype=jp.float32)
        if self._extension_joint_qpos_adrs:
            extension_joint_qpos = data.qpos[self._extension_joint_qpos_idx]
            extension_from_retracted = jp.maximum(
                extension_joint_qpos - self._extension_joint_retracted_qpos_arr,
                0.0,
            )
            extension_normalized = extension_from_retracted / self._extension_joint_range_span_arr
            extension_retracted_error = jp.mean(jp.square(extension_normalized))
        else:
            extension_retracted_error = jp.array(0.0, dtype=jp.float32)

        forward_reward = self._config.forward_reward_weight * root_vx  # type: ignore
        progress_delta_reward = self._config.progress_delta_reward_weight * delta_x  # type: ignore
        survival_reward = jp.array(self._config.survival_reward, dtype=jp.float32)  # type: ignore
        control_penalty = self._config.control_cost_weight * jp.sum(jp.square(action))  # type: ignore
        lateral_penalty = self._config.lateral_velocity_cost_weight * jp.square(root_vy)  # type: ignore
        tilt_penalty = self._config.tilt_cost_weight * jp.maximum(0.0, 1.0 - upright)  # type: ignore
        backward_velocity_penalty = self._config.backward_velocity_cost_weight * backward_vx  # type: ignore
        backward_delta_penalty = self._config.backward_delta_cost_weight * backward_delta_x  # type: ignore
        obstacle_local_progress_bonus = (
            self._config.obstacle_local_progress_weight * obstacle_local_active_f * forward_delta_x  # type: ignore
        )
        obstacle_local_backtrack_penalty = (
            self._config.obstacle_local_backtrack_cost_weight * obstacle_local_active_f * backward_delta_x  # type: ignore
        )
        obstacle_local_extension_bonus = (
            self._config.obstacle_local_extension_reward_weight  # type: ignore
            * obstacle_extension_active_f
            * extension_action_magnitude
        )
        extension_retracted_penalty = (
            jp.array(self._config.extension_retracted_penalty_weight, dtype=jp.float32)  # type: ignore
            * extension_retracted_error
        )
        if bool(self._config.extension_retracted_penalty_outside_obstacle_only):  # type: ignore
            penalty_active_f = 1.0 - obstacle_local_active_f
            if bool(self._config.extension_retracted_penalty_on_obstacle_top):  # type: ignore
                penalty_active_f = jp.maximum(penalty_active_f, obstacle_top_active_f)
            extension_retracted_penalty = extension_retracted_penalty * penalty_active_f
        if no_obstacles_task and bool(self._config.forward_only_when_no_obstacle):  # type: ignore
            survival_reward = jp.array(0.0, dtype=jp.float32)
            obstacle_local_progress_bonus = jp.array(0.0, dtype=jp.float32)
            obstacle_local_backtrack_penalty = jp.array(0.0, dtype=jp.float32)
            obstacle_local_extension_bonus = jp.array(0.0, dtype=jp.float32)
            extension_retracted_penalty = jp.array(0.0, dtype=jp.float32)

        failure_height = root_z < self._effective_min_base_height
        failure_tilt = upright < self._config.min_upright  # type: ignore
        failure_nan = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        failure = failure_height | failure_tilt | failure_nan

        no_obstacle_success_disabled = (
            no_obstacles_task
            and bool(self._config.forward_only_when_no_obstacle)  # type: ignore
            and self._config.success_x_threshold is None
        )
        if no_obstacle_success_disabled:
            success = jp.array(False)
        else:
            success = root_x >= success_x_threshold
        terminate_on_success = jp.array(self._config.terminate_on_success, dtype=jp.bool_)  # type: ignore

        failure_penalty = self._config.failure_penalty * failure.astype(jp.float32)  # type: ignore
        success_bonus = self._config.success_bonus * success.astype(jp.float32)  # type: ignore
        post_success_progress_bonus = (
            self._config.post_success_progress_weight  # type: ignore
            * success.astype(jp.float32)
            * forward_delta_x
        )
        stall = jp.abs(delta_x) < self._config.stall_delta_x_threshold  # type: ignore
        if bool(self._config.stall_penalty_after_success):  # type: ignore
            stall_active = stall
        else:
            stall_active = jp.logical_and(stall, ~success)
        stall_penalty = (
            self._config.stall_penalty_weight * stall_active.astype(jp.float32)  # type: ignore
        )

        reward = (
            forward_reward
            + progress_delta_reward
            + survival_reward
            + obstacle_local_progress_bonus
            + obstacle_local_extension_bonus
            + post_success_progress_bonus
            - control_penalty
            - lateral_penalty
            - tilt_penalty
            - backward_velocity_penalty
            - backward_delta_penalty
            - obstacle_local_backtrack_penalty
            - extension_retracted_penalty
            - stall_penalty
            - failure_penalty
            + success_bonus
        )

        obs = self._get_obs(data, state.info)
        done = (failure | (success & terminate_on_success)).astype(jp.float32)

        metrics = state.metrics.copy()
        metrics["reward/forward_vel"] = forward_reward
        metrics["reward/progress_delta"] = progress_delta_reward
        metrics["reward/survival"] = survival_reward
        metrics["reward/control_penalty"] = -control_penalty
        metrics["reward/lateral_penalty"] = -lateral_penalty
        metrics["reward/tilt_penalty"] = -tilt_penalty
        metrics["reward/backward_velocity_penalty"] = -backward_velocity_penalty
        metrics["reward/backward_delta_penalty"] = -backward_delta_penalty
        metrics["reward/obstacle_local_progress_bonus"] = obstacle_local_progress_bonus
        metrics["reward/obstacle_local_backtrack_penalty"] = -obstacle_local_backtrack_penalty
        metrics["reward/obstacle_local_extension_bonus"] = obstacle_local_extension_bonus
        metrics["reward/extension_retracted_penalty"] = -extension_retracted_penalty
        metrics["reward/stall_penalty"] = -stall_penalty
        metrics["reward/failure_penalty"] = -failure_penalty
        metrics["reward/success_bonus"] = success_bonus
        metrics["reward/post_success_progress_bonus"] = post_success_progress_bonus
        metrics["reward/total"] = reward
        metrics["task/root_x"] = root_x
        metrics["task/root_y"] = root_y
        metrics["task/root_z"] = root_z
        metrics["task/upright"] = upright
        metrics["task/obstacle_x_position"] = obstacle_x_position
        metrics["task/obstacle_half_length"] = jp.asarray(
            episode_obstacle_half_length, dtype=jp.float32
        )
        metrics["task/obstacle_height"] = jp.asarray(episode_obstacle_height, dtype=jp.float32)
        metrics["task/obstacle2_x_position"] = obstacle2_x_position
        metrics["task/obstacle2_half_length"] = jp.array(
            self._obstacle2_half_length, dtype=jp.float32
        )
        metrics["task/obstacle2_height"] = jp.array(
            self._obstacle2_height, dtype=jp.float32
        )
        metrics["task/success_x_threshold"] = success_x_threshold
        metrics["task/delta_x"] = delta_x
        metrics["task/distance_to_success"] = distance_to_success
        metrics["task/obstacle_local_active"] = obstacle_local_active_f
        metrics["task/obstacle_extension_active"] = obstacle_extension_active_f
        metrics["task/obstacle_top_active"] = obstacle_top_active_f
        metrics["task/extension_action_magnitude"] = extension_action_magnitude
        metrics["task/extension_retracted_error"] = extension_retracted_error
        metrics["task/min_base_height"] = jp.array(
            self._effective_min_base_height, dtype=jp.float32
        )
        metrics["task/obstacle_present"] = jp.array(
            1.0 if self._obstacle_geom_id is not None else 0.0, dtype=jp.float32
        )
        metrics["task/obstacle2_present"] = jp.array(
            1.0 if self._has_second_obstacle else 0.0, dtype=jp.float32
        )
        metrics["task/stall"] = stall.astype(jp.float32)
        metrics["task/success"] = success.astype(jp.float32)
        metrics["task/failure"] = failure.astype(jp.float32)
        metrics["task/failure_height"] = failure_height.astype(jp.float32)
        metrics["task/failure_tilt"] = failure_tilt.astype(jp.float32)
        metrics["task/failure_nan"] = failure_nan.astype(jp.float32)
        metrics["task/done"] = done

        info = state.info.copy()
        info["prev_root_x"] = root_x

        return mjx_env.State(
            data=data,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
            info=info,
        )

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> JaxArray:
        # TODO: center of mass dynamics
        qpos = data.qpos[self._obs_qpos_idx]
        qvel = data.qvel[self._obs_qvel_idx]
        return jp.concatenate([qpos, qvel])

    def _compute_upright(self, root_quat: JaxArray) -> JaxArray:
        # MuJoCo quaternions are [w, x, y, z]. R[2,2] measures world-aligned up.
        quat_x = root_quat[1]
        quat_y = root_quat[2]
        return 1.0 - 2.0 * (quat_x * quat_x + quat_y * quat_y)

    def _empty_metrics(self) -> dict[str, JaxArray]:
        zero = jp.array(0.0, dtype=jp.float32)
        return {
            "reward/forward_vel": zero,
            "reward/progress_delta": zero,
            "reward/survival": zero,
            "reward/control_penalty": zero,
            "reward/lateral_penalty": zero,
            "reward/tilt_penalty": zero,
            "reward/backward_velocity_penalty": zero,
            "reward/backward_delta_penalty": zero,
            "reward/obstacle_local_progress_bonus": zero,
            "reward/obstacle_local_backtrack_penalty": zero,
            "reward/obstacle_local_extension_bonus": zero,
            "reward/extension_retracted_penalty": zero,
            "reward/stall_penalty": zero,
            "reward/failure_penalty": zero,
            "reward/success_bonus": zero,
            "reward/post_success_progress_bonus": zero,
            "reward/total": zero,
            "task/root_x": zero,
            "task/root_y": zero,
            "task/root_z": zero,
            "task/upright": zero,
            "task/obstacle_x_position": zero,
            "task/obstacle_half_length": zero,
            "task/obstacle_height": zero,
            "task/obstacle2_x_position": zero,
            "task/obstacle2_half_length": zero,
            "task/obstacle2_height": zero,
            "task/success_x_threshold": zero,
            "task/delta_x": zero,
            "task/distance_to_success": zero,
            "task/obstacle_local_active": zero,
            "task/obstacle_extension_active": zero,
            "task/obstacle_top_active": zero,
            "task/extension_action_magnitude": zero,
            "task/extension_retracted_error": zero,
            "task/min_base_height": zero,
            "task/obstacle_present": zero,
            "task/obstacle2_present": zero,
            "task/stall": zero,
            "task/success": zero,
            "task/failure": zero,
            "task/failure_height": zero,
            "task/failure_tilt": zero,
            "task/failure_nan": zero,
            "task/done": zero,
        }

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self.mjx_model.nu

    @property
    def mj_model(self) -> MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> MjxModel:
        return self._mjx_model


dm_control_suite.register_environment(
    env_name="TransformableWheelMobileRobot",
    env_class=TransformableWheelMobileRobot,
    cfg_class=default_config,
)
