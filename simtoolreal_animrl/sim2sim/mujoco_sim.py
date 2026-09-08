"""MuJoCo physics backend matching the AnimRL UR5e + right-DG5F scene."""

from __future__ import annotations

import math
import re
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import mujoco
import mujoco.viewer
import numpy as np

from .constants import (
    ACTION_DIM,
    FINGERTIP_BODY_NAMES,
    JOINT_NAMES,
    WRIST_BODY_NAME,
)
from .observation import normalize_canonical_quaternion


@dataclass(frozen=True)
class MujocoSceneConfig:
    repo_root: Path
    robot_urdf_path: Path
    robot_position_world: np.ndarray
    robot_orientation_world_xyzw: np.ndarray
    object_size_m: np.ndarray
    object_mass_kg: float
    object_inertia_kg_m2: np.ndarray
    object_friction: float
    table_size_m: np.ndarray
    table_surface_below_robot_base_m: float
    table_friction: float
    floor_friction: float
    robot_friction: float
    fingertip_friction: float
    reference_ghost_offset_world: np.ndarray
    reference_ghost_color: np.ndarray
    arm_kp: float = 300.0
    arm_kv: float = 20.0
    hand_kp: float = 5.0
    hand_kv: float = 0.25
    sim_dt: float = 1.0 / 600.0
    enable_viewer: bool = True
    enable_reference_ghost: bool = True

    @classmethod
    def from_saved_config(
        cls,
        repo_root: Path,
        env_cfg: Mapping[str, Any],
        *,
        sim_dt: float = 1.0 / 600.0,
        enable_viewer: bool = True,
        arm_kp: float = 300.0,
        arm_kv: float = 20.0,
        hand_kp: float = 5.0,
        hand_kv: float = 0.25,
        enable_reference_ghost: bool = True,
    ) -> "MujocoSceneConfig":
        asset = env_cfg["asset"]
        init_state = env_cfg["init_state"]
        object_cfg = env_cfg["object"]
        table = env_cfg["table"]
        terrain = env_cfg["terrain"]
        robot_path = Path(asset["file"])
        if not robot_path.is_absolute():
            robot_path = Path(repo_root) / robot_path
        config = cls(
            repo_root=Path(repo_root).resolve(),
            robot_urdf_path=robot_path.resolve(),
            robot_position_world=np.asarray(init_state["pos"], dtype=np.float64),
            robot_orientation_world_xyzw=np.asarray(
                init_state["rot"], dtype=np.float64
            ),
            object_size_m=np.asarray(object_cfg["size_m"], dtype=np.float64),
            object_mass_kg=float(object_cfg["mass_kg"]),
            object_inertia_kg_m2=np.asarray(
                object_cfg["inertia_kg_m2"], dtype=np.float64
            ),
            object_friction=float(object_cfg["friction"]),
            table_size_m=np.asarray(table["size_m"], dtype=np.float64),
            table_surface_below_robot_base_m=float(
                table["surface_below_robot_base_m"]
            ),
            table_friction=float(table["friction"]),
            floor_friction=float(terrain["dynamic_friction"]),
            robot_friction=float(asset["friction"]),
            fingertip_friction=float(asset["fingertip_friction"]),
            reference_ghost_offset_world=np.asarray(
                env_cfg["viewer"]["reference_ghost_offset"], dtype=np.float64
            ),
            reference_ghost_color=np.asarray(
                env_cfg["viewer"]["reference_ghost_color"], dtype=np.float64
            ),
            arm_kp=float(arm_kp),
            arm_kv=float(arm_kv),
            hand_kp=float(hand_kp),
            hand_kv=float(hand_kv),
            sim_dt=float(sim_dt),
            enable_viewer=bool(enable_viewer),
            enable_reference_ghost=bool(enable_reference_ghost),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.robot_urdf_path.is_file():
            raise FileNotFoundError("Robot URDF not found: {}".format(self.robot_urdf_path))
        for name, values, shape in (
            ("robot_position_world", self.robot_position_world, (3,)),
            ("robot_orientation_world_xyzw", self.robot_orientation_world_xyzw, (4,)),
            ("object_size_m", self.object_size_m, (3,)),
            ("object_inertia_kg_m2", self.object_inertia_kg_m2, (3,)),
            ("table_size_m", self.table_size_m, (3,)),
            ("reference_ghost_offset_world", self.reference_ghost_offset_world, (3,)),
            ("reference_ghost_color", self.reference_ghost_color, (3,)),
        ):
            if values.shape != shape or not np.all(np.isfinite(values)):
                raise ValueError("{} must be finite with shape {}".format(name, shape))
        if np.any(self.object_size_m <= 0.0) or np.any(self.table_size_m <= 0.0):
            raise ValueError("Object and table dimensions must be positive")
        if self.object_mass_kg <= 0.0 or self.sim_dt <= 0.0:
            raise ValueError("Object mass and simulation timestep must be positive")
        if np.any(self.reference_ghost_color < 0.0) or np.any(
            self.reference_ghost_color > 1.0
        ):
            raise ValueError("Reference-ghost color must lie in [0, 1]")
        for name in ("arm_kp", "arm_kv", "hand_kp", "hand_kv"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0.0:
                raise ValueError("{} must be finite and positive".format(name))


class AnimRLMujocoSim:
    """One physical MuJoCo environment with the training robot and cuboid."""

    def __init__(self, config: MujocoSceneConfig) -> None:
        self.config = config
        self._urdf_joint_limits = self._read_urdf_joint_limits()
        self._tmp_dir = tempfile.TemporaryDirectory(prefix="animrl_mujoco_")
        self.viewer = None
        self._init_scene()

    def _read_urdf_joint_limits(self) -> dict[str, tuple[float, float, float]]:
        root = ET.parse(str(self.config.robot_urdf_path)).getroot()
        limits = {}
        for joint in root.findall("joint"):
            limit = joint.find("limit")
            name = joint.get("name")
            if name in JOINT_NAMES and limit is not None:
                limits[name] = (
                    float(limit.get("lower")),
                    float(limit.get("upper")),
                    float(limit.get("effort")),
                )
        missing = sorted(set(JOINT_NAMES) - set(limits))
        if missing:
            raise ValueError("URDF is missing joint limits for {}".format(missing))
        return limits

    def _make_mujoco_compatible_urdf(self) -> Path:
        text = self.config.robot_urdf_path.read_text(encoding="utf-8")
        if "<mujoco>" not in text:
            text = re.sub(
                r'(<robot\s+name="[^"]+">)',
                r'\1\n  <mujoco><compiler strippath="false"/></mujoco>',
                text,
                count=1,
            )

        def absolute_mesh_path(match: re.Match[str]) -> str:
            filename = match.group(1)
            if filename.startswith("urdf/"):
                path = self.config.repo_root / "assets" / filename
            else:
                path = self.config.robot_urdf_path.parent / filename
            return 'filename="{}"'.format(path.resolve())

        text = re.sub(r'filename="([^"]+)"', absolute_mesh_path, text)
        # MuJoCo warns once for every anonymous visual/collision geometry.
        # URDF permits names here, so assign stable unique names in the
        # temporary copy without touching the source asset.
        geometry_counts = {"visual": 0, "collision": 0}

        def name_geometry(match: re.Match[str]) -> str:
            kind = match.group(1)
            index = geometry_counts[kind]
            geometry_counts[kind] += 1
            return '<{} name="{}_{}">'.format(kind, kind, index)

        text = re.sub(r"<(visual|collision)>", name_geometry, text)
        output = Path(self._tmp_dir.name) / "ur5e_right_dg5f_mujoco.urdf"
        output.write_text(text, encoding="utf-8")
        return output

    def _init_scene(self) -> None:
        robot_path = self._make_mujoco_compatible_urdf()
        spec = mujoco.MjSpec()
        spec.from_file(str(robot_path))
        spec.discardvisual = False
        if self.config.enable_reference_ghost:
            self._attach_reference_ghost(spec, robot_path)
        self._add_world(spec)
        self._add_position_actuators(spec)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)

        self.model.opt.timestep = float(self.config.sim_dt)
        self.model.opt.gravity[:] = np.asarray((0.0, 0.0, -9.81))
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        self.model.opt.iterations = 50
        self.model.opt.ls_iterations = 50

        self._resolve_indices()
        self._place_and_configure_robot()
        self._configure_collision_masks()
        self._validate_object_inertia()
        mujoco.mj_forward(self.model, self.data)

        if self.config.enable_viewer:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            ghost_camera_offset = (
                0.5 * self.config.reference_ghost_offset_world
                if self.config.enable_reference_ghost
                else np.zeros(3, dtype=np.float64)
            )
            self.viewer.cam.lookat[:] = self.config.robot_position_world + (
                ghost_camera_offset
            ) + np.asarray((0.0, -0.05, 0.10))
            self.viewer.cam.distance = 2.2
            # Face the robot from the workspace/front side. The previous
            # 220-degree view showed the back of the arm and hand.
            self.viewer.cam.azimuth = 40.0
            self.viewer.cam.elevation = -20.0
            self.viewer.sync()

    @staticmethod
    def _attach_reference_ghost(spec: mujoco.MjSpec, robot_path: Path) -> None:
        ghost_spec = mujoco.MjSpec()
        ghost_spec.from_file(str(robot_path))
        ghost_frame = spec.worldbody.add_frame()
        ghost_frame.name = "reference_ghost_mount"
        ghost_frame.attach_body(
            ghost_spec.worldbody.first_body(), "ghost_", ""
        )

    def _add_world(self, spec: mujoco.MjSpec) -> None:
        floor = spec.worldbody.add_geom()
        floor.name = "floor"
        floor.type = mujoco.mjtGeom.mjGEOM_PLANE
        floor.size = np.asarray((1.5, 1.5, 0.05))
        floor.rgba = np.asarray((0.20, 0.25, 0.28, 1.0))
        floor.friction = np.asarray((self.config.floor_friction, 0.005, 0.0001))

        table = spec.worldbody.add_body()
        table.name = "table"
        table.pos = np.asarray(
            (
                0.0,
                0.0,
                self.config.robot_position_world[2]
                - self.config.table_surface_below_robot_base_m
                - self.config.table_size_m[2] / 2.0,
            )
        )
        table_geom = table.add_geom()
        table_geom.name = "table_geom"
        table_geom.type = mujoco.mjtGeom.mjGEOM_BOX
        table_geom.size = self.config.table_size_m / 2.0
        table_geom.rgba = np.asarray((0.82, 0.56, 0.35, 1.0))
        table_geom.friction = np.asarray(
            (self.config.table_friction, 0.005, 0.0001)
        )
        self._set_low_bounce_contact(table_geom)

        cube = spec.worldbody.add_body()
        cube.name = "cube"
        cube_joint = cube.add_joint()
        cube_joint.name = "cube_free_joint"
        cube_joint.type = mujoco.mjtJoint.mjJNT_FREE
        cube_geom = cube.add_geom()
        cube_geom.name = "cube_geom"
        cube_geom.type = mujoco.mjtGeom.mjGEOM_BOX
        cube_geom.size = self.config.object_size_m / 2.0
        cube_geom.density = self.config.object_mass_kg / float(
            np.prod(self.config.object_size_m)
        )
        cube_geom.rgba = np.asarray((0.78, 0.78, 0.82, 1.0))
        cube_geom.friction = np.asarray(
            (self.config.object_friction, 0.005, 0.0001)
        )
        self._set_low_bounce_contact(cube_geom)

        light = spec.worldbody.add_light()
        light.name = "key_light"
        light.pos = np.asarray((0.0, -1.0, 1.5))
        light.dir = np.asarray((0.0, 0.5, -1.0))
        light.directional = True

    @staticmethod
    def _set_low_bounce_contact(geom) -> None:
        geom.solref = np.asarray((0.012, 1.4))
        geom.solimp = np.asarray((0.95, 0.99, 0.002, 0.5, 2.0))

    def _add_position_actuators(self, spec: mujoco.MjSpec) -> None:
        for index, joint_name in enumerate(JOINT_NAMES):
            actuator = spec.add_actuator()
            actuator.name = "{}_pos".format(joint_name)
            actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
            actuator.target = joint_name
            lower, upper, effort = self._urdf_joint_limits[joint_name]
            # PhysX's position drive is effectively held at a hard mechanical
            # stop. MuJoCo's soft joint constraint can otherwise overshoot by
            # several tenths of a radian in one control frame, so constrain the
            # actuator command while retaining the raw, unclipped target in the
            # policy observation.
            actuator.ctrllimited = True
            actuator.ctrlrange = np.asarray((lower, upper))
            actuator.forcelimited = True
            actuator.forcerange = np.asarray((-effort, effort))
            kp = self.config.arm_kp if index < 6 else self.config.hand_kp
            kv = self.config.arm_kv if index < 6 else self.config.hand_kv
            actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            actuator.gainprm[0] = kp
            actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE
            actuator.biasprm[1] = -kp
            actuator.biasprm[2] = -kv

    def _resolve_indices(self) -> None:
        missing_joints = [
            name for name in JOINT_NAMES if self.model.joint(name).id < 0
        ]
        if missing_joints:
            raise RuntimeError("MuJoCo model is missing joints: {}".format(missing_joints))
        self._joint_qpos_adrs = np.asarray(
            [self.model.joint(name).qposadr[0] for name in JOINT_NAMES],
            dtype=np.int32,
        )
        self._joint_dof_adrs = np.asarray(
            [self.model.joint(name).dofadr[0] for name in JOINT_NAMES],
            dtype=np.int32,
        )
        self._actuator_ids = np.asarray(
            [self.model.actuator("{}_pos".format(name)).id for name in JOINT_NAMES],
            dtype=np.int32,
        )
        self._base_body_id = self.model.body("base_link").id
        self._wrist_body_id = self.model.body(WRIST_BODY_NAME).id
        self._fingertip_body_ids = np.asarray(
            [self.model.body(name).id for name in FINGERTIP_BODY_NAMES],
            dtype=np.int32,
        )
        self._cube_body_id = self.model.body("cube").id
        cube_joint = self.model.joint("cube_free_joint")
        self._cube_qpos_adr = int(cube_joint.qposadr[0])
        self._cube_dof_adr = int(cube_joint.dofadr[0])
        if self.config.enable_reference_ghost:
            self._ghost_base_body_id = self.model.body("ghost_base_link").id
            self._ghost_joint_qpos_adrs = np.asarray(
                [
                    self.model.joint("ghost_{}".format(name)).qposadr[0]
                    for name in JOINT_NAMES
                ],
                dtype=np.int32,
            )
            self._ghost_joint_dof_adrs = np.asarray(
                [
                    self.model.joint("ghost_{}".format(name)).dofadr[0]
                    for name in JOINT_NAMES
                ],
                dtype=np.int32,
            )
        else:
            self._ghost_base_body_id = None
            self._ghost_joint_qpos_adrs = np.empty(0, dtype=np.int32)
            self._ghost_joint_dof_adrs = np.empty(0, dtype=np.int32)
        self.joint_lower_limits = np.asarray(
            [self.model.jnt_range[self.model.joint(name).id, 0] for name in JOINT_NAMES],
            dtype=np.float64,
        )
        self.joint_upper_limits = np.asarray(
            [self.model.jnt_range[self.model.joint(name).id, 1] for name in JOINT_NAMES],
            dtype=np.float64,
        )
        if np.any(self.joint_lower_limits >= self.joint_upper_limits):
            raise RuntimeError("MuJoCo imported invalid robot joint limits")

    def _place_and_configure_robot(self) -> None:
        self.model.body_pos[self._base_body_id] = self.config.robot_position_world
        quat_xyzw = normalize_canonical_quaternion(
            self.config.robot_orientation_world_xyzw
        )
        self.model.body_quat[self._base_body_id] = quat_xyzw[[3, 0, 1, 2]]
        robot_body_ids = self._robot_body_ids()
        self.model.body_gravcomp[robot_body_ids] = 1.0
        self.model.dof_damping[self._joint_dof_adrs] = 0.0
        if self.config.enable_reference_ghost:
            self.model.body_pos[self._ghost_base_body_id] = (
                self.config.robot_position_world
                + self.config.reference_ghost_offset_world
            )
            self.model.body_quat[self._ghost_base_body_id] = quat_xyzw[
                [3, 0, 1, 2]
            ]
            self.model.dof_damping[self._ghost_joint_dof_adrs] = 0.0
            ghost_rgba = np.concatenate(
                (self.config.reference_ghost_color, np.asarray((0.72,)))
            )
            for geom_id in range(self.model.ngeom):
                body_id = int(self.model.geom_bodyid[geom_id])
                if self._body_is_descendant_of(
                    body_id, self._ghost_base_body_id
                ):
                    self.model.geom_rgba[geom_id] = ghost_rgba

    def _robot_body_ids(self) -> np.ndarray:
        excluded = {"world", "table", "cube"}
        return np.asarray(
            [
                body_id
                for body_id in range(1, self.model.nbody)
                if self.model.body(body_id).name not in excluded
            ],
            dtype=np.int32,
        )

    def _body_is_descendant_of(self, body_id: int, ancestor_id: int) -> bool:
        current = int(body_id)
        while current != 0:
            if current == ancestor_id:
                return True
            current = int(self.model.body_parentid[current])
        return False

    def _configure_collision_masks(self) -> None:
        # Match MotionImitationEnv's external-contact graph. The robot has no
        # self collisions, no robot shape can touch the table, and the arm up
        # through wrist_2_link is filtered against the cube. The wrist_3/hand
        # assembly can touch the cube, and the cube can touch the table. The
        # ground remains visual only in this elevated tabletop task.
        cube_bit, hand_bit, table_bit = 1, 2, 4
        cube_geom_id = self.model.geom("cube_geom").id
        table_geom_id = self.model.geom("table_geom").id
        floor_geom_id = self.model.geom("floor").id
        fingertip_ids = set(int(value) for value in self._fingertip_body_ids)

        for geom_id in range(self.model.ngeom):
            body_id = int(self.model.geom_bodyid[geom_id])
            if geom_id == cube_geom_id:
                self.model.geom_contype[geom_id] = cube_bit
                self.model.geom_conaffinity[geom_id] = hand_bit | table_bit
            elif geom_id == table_geom_id:
                self.model.geom_contype[geom_id] = table_bit
                self.model.geom_conaffinity[geom_id] = cube_bit
            elif geom_id == floor_geom_id:
                self.model.geom_contype[geom_id] = 0
                self.model.geom_conaffinity[geom_id] = 0
            elif self._body_is_descendant_of(body_id, self._wrist_body_id):
                self.model.geom_contype[geom_id] = hand_bit
                self.model.geom_conaffinity[geom_id] = cube_bit
                is_fingertip = any(
                    self._body_is_descendant_of(body_id, fingertip_id)
                    for fingertip_id in fingertip_ids
                )
                friction = (
                    self.config.fingertip_friction
                    if is_fingertip
                    else self.config.robot_friction
                )
                self.model.geom_friction[geom_id, 0] = friction
            else:
                self.model.geom_contype[geom_id] = 0
                self.model.geom_conaffinity[geom_id] = 0
                self.model.geom_friction[geom_id, 0] = self.config.robot_friction

    def _robot_cube_contact_distances(self) -> np.ndarray:
        cube_geom_id = self.model.geom("cube_geom").id
        distances = []
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if cube_geom_id not in (geom1, geom2):
                continue
            other_geom = geom2 if geom1 == cube_geom_id else geom1
            other_body = int(self.model.geom_bodyid[other_geom])
            if self._body_is_descendant_of(other_body, self._wrist_body_id):
                distances.append(float(contact.dist))
        return np.asarray(distances, dtype=np.float64)

    def settle_robot_cube_contacts(self, duration_sec: float) -> dict[str, float]:
        """Relax an RSI grasp while preserving the recorded cuboid pose.

        The demonstration intentionally contains a small contact preload. A
        direct dynamic reset makes MuJoCo resolve that preload as an impulse.
        During this warm start the cube pose and the six arm joints are
        restored before every physics step while only the fingers are free to
        establish a solver-consistent equilibrium. The cube is released at
        rest afterward.

        If the reset has no robot/cube contact, no settling steps are run. This
        keeps an RSI before the grasp identical to the ordinary reset path.
        """
        duration_sec = float(duration_sec)
        if not np.isfinite(duration_sec) or duration_sec < 0.0:
            raise ValueError("Contact-settling duration must be finite and non-negative")
        initial_distances = self._robot_cube_contact_distances()
        initial_count = int(initial_distances.size)
        initial_minimum = (
            float(initial_distances.min()) if initial_count else 0.0
        )
        if duration_sec == 0.0 or initial_count == 0:
            return {
                "steps": 0,
                "contacts_before": initial_count,
                "contacts_after": initial_count,
                "minimum_distance_before_m": initial_minimum,
                "minimum_distance_after_m": initial_minimum,
                "max_joint_displacement_rad": 0.0,
            }

        cube_qpos = self.data.qpos[
            self._cube_qpos_adr : self._cube_qpos_adr + 7
        ].copy()
        arm_qpos = self.data.qpos[self._joint_qpos_adrs[:6]].copy()
        joint_positions = self.data.qpos[self._joint_qpos_adrs].copy()
        steps = max(1, int(math.ceil(duration_sec / self.config.sim_dt)))
        for _ in range(steps):
            self.data.qpos[
                self._cube_qpos_adr : self._cube_qpos_adr + 7
            ] = cube_qpos
            self.data.qvel[
                self._cube_dof_adr : self._cube_dof_adr + 6
            ] = 0.0
            self.data.qpos[self._joint_qpos_adrs[:6]] = arm_qpos
            self.data.qvel[self._joint_dof_adrs[:6]] = 0.0
            mujoco.mj_forward(self.model, self.data)
            mujoco.mj_step(self.model, self.data)

        self.data.qpos[
            self._cube_qpos_adr : self._cube_qpos_adr + 7
        ] = cube_qpos
        self.data.qvel[self._cube_dof_adr : self._cube_dof_adr + 6] = 0.0
        self.data.qpos[self._joint_qpos_adrs[:6]] = arm_qpos
        self.data.qvel[self._joint_dof_adrs[:6]] = 0.0
        mujoco.mj_forward(self.model, self.data)
        final_distances = self._robot_cube_contact_distances()
        final_count = int(final_distances.size)
        final_minimum = float(final_distances.min()) if final_count else 0.0
        max_displacement = float(
            np.max(
                np.abs(
                    self.data.qpos[self._joint_qpos_adrs] - joint_positions
                )
            )
        )
        if self.viewer is not None:
            self.viewer.sync()
        return {
            "steps": steps,
            "contacts_before": initial_count,
            "contacts_after": final_count,
            "minimum_distance_before_m": initial_minimum,
            "minimum_distance_after_m": final_minimum,
            "max_joint_displacement_rad": max_displacement,
        }

    def _validate_object_inertia(self) -> None:
        actual_mass = float(self.model.body_mass[self._cube_body_id])
        actual_inertia = self.model.body_inertia[self._cube_body_id]
        if not np.isclose(actual_mass, self.config.object_mass_kg, rtol=1e-5):
            raise RuntimeError(
                "Compiled cube mass {} does not match {}".format(
                    actual_mass, self.config.object_mass_kg
                )
            )
        expected_sorted = np.sort(self.config.object_inertia_kg_m2)
        if not np.allclose(np.sort(actual_inertia), expected_sorted, rtol=2e-4, atol=1e-9):
            raise RuntimeError(
                "Compiled cube inertia {} does not match {}".format(
                    actual_inertia.tolist(), self.config.object_inertia_kg_m2.tolist()
                )
            )

    def reference_cube_state_to_world(
        self,
        cube_pose_xyzw: np.ndarray,
        linear_velocity: np.ndarray,
        angular_velocity: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pose = np.asarray(cube_pose_xyzw, dtype=np.float64)
        if pose.shape != (7,):
            raise ValueError("cube_pose_xyzw must have shape (7,)")
        axis_sign = np.asarray((-1.0, -1.0, 1.0))
        position = self.config.robot_position_world + pose[:3] * axis_sign
        x, y, z, w = normalize_canonical_quaternion(pose[3:7])
        orientation_xyzw = normalize_canonical_quaternion(
            np.asarray((-y, x, w, -z))
        )
        return (
            position,
            orientation_xyzw,
            np.asarray(linear_velocity, dtype=np.float64) * axis_sign,
            np.asarray(angular_velocity, dtype=np.float64) * axis_sign,
        )

    def reset(
        self,
        joint_positions: np.ndarray,
        joint_velocities: np.ndarray,
        cube_pose_xyzw: np.ndarray,
        cube_linear_velocity: np.ndarray,
        cube_angular_velocity: np.ndarray,
    ) -> None:
        q = np.asarray(joint_positions, dtype=np.float64)
        dq = np.asarray(joint_velocities, dtype=np.float64)
        if q.shape != (ACTION_DIM,) or dq.shape != (ACTION_DIM,):
            raise ValueError("Robot reset state must have shape (26,)")
        self.data.qpos[self._joint_qpos_adrs] = q
        self.data.qvel[self._joint_dof_adrs] = dq
        position, orientation_xyzw, linear, angular = self.reference_cube_state_to_world(
            cube_pose_xyzw, cube_linear_velocity, cube_angular_velocity
        )
        address = self._cube_qpos_adr
        self.data.qpos[address : address + 3] = position
        self.data.qpos[address + 3 : address + 7] = orientation_xyzw[[3, 0, 1, 2]]
        dof = self._cube_dof_adr
        self.data.qvel[dof : dof + 3] = linear
        self.data.qvel[dof + 3 : dof + 6] = angular
        self.set_position_targets(q)
        mujoco.mj_forward(self.model, self.data)
        if self.viewer is not None:
            self.viewer.sync()

    def set_position_targets(self, targets: np.ndarray) -> None:
        targets = np.asarray(targets, dtype=np.float64)
        if targets.shape != (ACTION_DIM,):
            raise ValueError("Position targets must have shape (26,)")
        self.data.ctrl[self._actuator_ids] = targets

    def set_reference_ghost(self, joint_positions: np.ndarray) -> None:
        """Place the visual-only green robot at one reference sample."""
        if not self.config.enable_reference_ghost:
            return
        positions = np.asarray(joint_positions, dtype=np.float64)
        if positions.shape != (ACTION_DIM,):
            raise ValueError("Reference-ghost positions must have shape (26,)")
        self.data.qpos[self._ghost_joint_qpos_adrs] = positions
        self.data.qvel[self._ghost_joint_dof_adrs] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def sync_viewer(self) -> None:
        if self.viewer is not None:
            self.viewer.sync()

    def step_for(self, duration_sec: float) -> None:
        steps = max(1, int(round(float(duration_sec) / self.config.sim_dt)))
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
        if self.viewer is not None:
            self.viewer.sync()

    @staticmethod
    def _xyzw(quaternion_wxyz: np.ndarray) -> np.ndarray:
        return normalize_canonical_quaternion(
            np.asarray(quaternion_wxyz, dtype=np.float64)[[1, 2, 3, 0]]
        )

    def get_state(self) -> dict[str, np.ndarray]:
        fingertip_positions = self.data.xpos[self._fingertip_body_ids].copy()
        fingertip_orientations = np.stack(
            [self._xyzw(self.data.xquat[body_id]) for body_id in self._fingertip_body_ids]
        )
        return {
            "joint_positions": self.data.qpos[self._joint_qpos_adrs].copy(),
            "joint_velocities": self.data.qvel[self._joint_dof_adrs].copy(),
            "robot_position_world": self.data.xpos[self._base_body_id].copy(),
            "robot_orientation_world_xyzw": self._xyzw(
                self.data.xquat[self._base_body_id]
            ),
            "wrist_position_world": self.data.xpos[self._wrist_body_id].copy(),
            "wrist_orientation_world_xyzw": self._xyzw(
                self.data.xquat[self._wrist_body_id]
            ),
            "fingertip_body_positions_world": fingertip_positions,
            "fingertip_body_orientations_world_xyzw": fingertip_orientations,
            "cube_position_world": self.data.xpos[self._cube_body_id].copy(),
            "cube_orientation_world_xyzw": self._xyzw(
                self.data.xquat[self._cube_body_id]
            ),
            "cube_linear_velocity_world": self.data.qvel[
                self._cube_dof_adr : self._cube_dof_adr + 3
            ].copy(),
            "cube_angular_velocity_world": self.data.qvel[
                self._cube_dof_adr + 3 : self._cube_dof_adr + 6
            ].copy(),
        }

    def viewer_is_running(self) -> bool:
        return self.viewer is None or self.viewer.is_running()

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        self._tmp_dir.cleanup()

    def __enter__(self) -> "AnimRLMujocoSim":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
