"""Minimal vectorized UR5e + DG5F discrete motion-imitation environment."""

import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

# Isaac Gym must be imported before torch.
from isaacgym import gymapi, gymtorch, gymutil
import torch

from simtoolreal_animrl import ROOT_DIR
from simtoolreal_animrl.envs.controller import (
    ARM_JOINT_NAMES,
    HAND_JOINT_NAMES,
    HAND_PD_DAMPING,
    HAND_PD_STIFFNESS,
    JOINT_NAMES,
    configure_asset_wrist_collision_filters,
    configure_pd_properties,
    validate_joint_order,
)
from simtoolreal_animrl.envs.contact import (
    fingertip_contact_diagnostics,
    fingertip_force_norms,
)
from simtoolreal_animrl.envs.demonstration import JointDemonstration60Hz
from simtoolreal_animrl.envs.proximity import fingertip_cuboid_proximity
from simtoolreal_animrl.envs.rsi import resolve_rsi_settings, sample_rsi_indices


# The fixed wrist -> mount -> base -> palm chain is collapsed while loading the
# robot asset.  These constants reconstruct the exact rl_dg_palm URDF frame
# from the surviving wrist_3_link rigid body (xyzw quaternion convention).
PALM_PARENT_BODY_NAME = "wrist_3_link"
PALM_POSITION_IN_WRIST = (0.0, 0.0, 0.0738)
# Fixed 60-degree rotation of the hand/palm frame relative to wrist_3_link,
# introduced by the ur5e_dg5f_mount joint. In xyzw form this is
# (0, 0, sin(60 deg / 2), cos(60 deg / 2)).
PALM_ORIENTATION_IN_WRIST = (0.0, 0.0, 0.5, 0.8660254037844386)
FINGERTIP_BODY_NAMES_BY_SEMANTIC_NAME = {
    "thumb": "rl_dg_1_4",
    "index": "rl_dg_2_4",
    "middle": "rl_dg_3_4",
    "ring": "rl_dg_4_4",
    "pinky": "rl_dg_5_4",
}
FINGERTIP_BODY_NAMES = tuple(FINGERTIP_BODY_NAMES_BY_SEMANTIC_NAME.values())
FINGERTIP_OFFSETS = (
    # Exact origins of the fixed rj_dg_<finger>_tip joints in the respective
    # rl_dg_<finger>_4 frames. The tip bodies themselves are collapsed.
    (0.0, 0.0363, 0.0),
    (0.0, 0.0, 0.0255),
    (0.0, 0.0, 0.0255),
    (0.0, 0.0, 0.0255),
    (0.0, 0.0, 0.0363),
)


def _quat_conjugate(quaternion: torch.Tensor) -> torch.Tensor:
    return torch.cat((-quaternion[..., :3], quaternion[..., 3:4]), dim=-1)


def _quat_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Hamilton product for xyzw quaternions."""
    left_xyz, left_w = left[..., :3], left[..., 3:4]
    right_xyz, right_w = right[..., :3], right[..., 3:4]
    xyz = (
        left_w * right_xyz
        + right_w * left_xyz
        + torch.cross(left_xyz, right_xyz, dim=-1)
    )
    w = left_w * right_w - (left_xyz * right_xyz).sum(dim=-1, keepdim=True)
    return torch.cat((xyz, w), dim=-1)


def _quat_rotate(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate a vector by a unit xyzw quaternion without constructing matrices."""
    quaternion_xyz = quaternion[..., :3]
    uv = torch.cross(quaternion_xyz, vector, dim=-1)
    uuv = torch.cross(quaternion_xyz, uv, dim=-1)
    return vector + 2.0 * (quaternion[..., 3:4] * uv + uuv)


def _quat_rotate_inverse(
    quaternion: torch.Tensor, vector: torch.Tensor
) -> torch.Tensor:
    return _quat_rotate(_quat_conjugate(quaternion), vector)


def _normalize_canonical_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    quaternion = torch.nn.functional.normalize(quaternion, dim=-1)
    return torch.where(quaternion[..., 3:4] < 0.0, -quaternion, quaternion)


class MotionImitationEnv:
    """AnimRL-compatible environment API without an RL algorithm dependency."""

    JOINT_NAMES = JOINT_NAMES
    FINGERTIP_NAMES = tuple(FINGERTIP_BODY_NAMES_BY_SEMANTIC_NAME)

    def __init__(
        self,
        cfg,
        sim_device: str = "cuda:0",
        headless: bool = True,
        num_envs_override: Optional[int] = None,
    ) -> None:
        self.cfg = cfg
        self.headless = headless
        self.sim_device = sim_device
        if num_envs_override is not None:
            if num_envs_override <= 0:
                raise ValueError("num_envs_override must be positive")
            self.cfg.env.num_envs = int(num_envs_override)

        self.num_envs = int(self.cfg.env.num_envs)
        self.num_obs = int(self.cfg.env.num_observations)
        self.num_privileged_obs = self.cfg.env.num_privileged_obs
        self.num_actions = int(self.cfg.env.num_actions)
        self.max_episode_length = int(self.cfg.env.episode_length)
        self.dt = float(self.cfg.sim.dt) * int(self.cfg.control.decimation)
        if self.num_actions != len(JOINT_NAMES):
            raise ValueError(
                "The policy drives every joint and requires exactly {} "
                "actions".format(len(JOINT_NAMES))
            )
        if self.cfg.control.action_parameterization != "animrl_residual":
            raise ValueError("Only the AnimRL residual action contract is supported")
        self.action_scale = float(self.cfg.control.scale_joint_target)
        self.hand_action_scale = float(self.cfg.control.scale_hand_joint_target)
        self.action_target_clip = float(self.cfg.control.clip_joint_target)
        if (
            self.action_scale <= 0.0
            or self.hand_action_scale <= 0.0
            or self.action_target_clip <= 0.0
        ):
            raise ValueError("AnimRL action scales and target clip must be positive")
        self.contact_enabled = bool(self.cfg.contact.enabled)
        self.contact_collection = int(self.cfg.contact.collection)
        self.contact_force_threshold_n = float(
            self.cfg.contact.force_threshold_n
        )
        self.contact_reward_per_finger = float(
            self.cfg.contact.reward_per_finger
        )
        self.contact_fingertip_names = tuple(
            str(name).strip().lower() for name in self.cfg.contact.fingertip_names
        )
        if self.contact_collection not in (1, 2):
            raise ValueError("contact.collection must be 1 or 2")
        if (
            not math.isfinite(self.contact_force_threshold_n)
            or self.contact_force_threshold_n <= 0.0
        ):
            raise ValueError("contact.force_threshold_n must be finite and positive")
        if (
            not math.isfinite(self.contact_reward_per_finger)
            or self.contact_reward_per_finger < 0.0
        ):
            raise ValueError(
                "contact.reward_per_finger must be finite and non-negative"
            )
        if not self.contact_fingertip_names:
            raise ValueError("contact.fingertip_names must not be empty")
        if len(set(self.contact_fingertip_names)) != len(
            self.contact_fingertip_names
        ):
            raise ValueError("contact.fingertip_names must not contain duplicates")
        unknown_contact_fingers = set(self.contact_fingertip_names).difference(
            FINGERTIP_BODY_NAMES_BY_SEMANTIC_NAME
        )
        if unknown_contact_fingers:
            raise ValueError(
                "Unknown contact fingertip names: {}".format(
                    sorted(unknown_contact_fingers)
                )
            )
        self.proximity_fingertip_names = tuple(
            str(name).strip().lower()
            for name in self.cfg.rewards.fingertip_object_distance_names
        )
        self.proximity_std_m = float(
            self.cfg.rewards.fingertip_object_distance_std_m
        )
        self.proximity_weight = float(
            self.cfg.rewards.fingertip_object_distance_weight
        )
        if not self.proximity_fingertip_names:
            raise ValueError(
                "rewards.fingertip_object_distance_names must not be empty"
            )
        if len(set(self.proximity_fingertip_names)) != len(
            self.proximity_fingertip_names
        ):
            raise ValueError(
                "rewards.fingertip_object_distance_names must not contain duplicates"
            )
        unknown_proximity_fingers = set(
            self.proximity_fingertip_names
        ).difference(FINGERTIP_BODY_NAMES_BY_SEMANTIC_NAME)
        if unknown_proximity_fingers:
            raise ValueError(
                "Unknown proximity fingertip names: {}".format(
                    sorted(unknown_proximity_fingers)
                )
            )
        if not math.isfinite(self.proximity_std_m) or self.proximity_std_m <= 0.0:
            raise ValueError(
                "rewards.fingertip_object_distance_std_m must be finite and positive"
            )
        if not math.isfinite(self.proximity_weight) or self.proximity_weight < 0.0:
            raise ValueError(
                "rewards.fingertip_object_distance_weight must be finite and non-negative"
            )
        if not math.isclose(
            self.dt,
            1.0 / float(self.cfg.motion.frequency_hz),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "Control dt {:.9g} does not match the {} Hz demonstration".format(
                    self.dt, self.cfg.motion.frequency_hz
                )
            )

        device_type, self.sim_device_id = gymutil.parse_device_str(sim_device)
        use_gpu_pipeline = bool(self.cfg.sim.use_gpu_pipeline)
        if device_type == "cuda" and use_gpu_pipeline:
            self.device = torch.device(sim_device)
        else:
            self.device = torch.device("cpu")
        self.training_camera_enabled = bool(
            getattr(self.cfg.viewer, "training_camera_enabled", False)
        )
        self.training_camera_env_index = int(
            getattr(self.cfg.viewer, "training_camera_env_index", 0)
        )
        self.training_camera_width = int(
            getattr(self.cfg.viewer, "training_camera_width", 640)
        )
        self.training_camera_height = int(
            getattr(self.cfg.viewer, "training_camera_height", 480)
        )
        if not 0 <= self.training_camera_env_index < self.num_envs:
            raise ValueError("viewer.training_camera_env_index is out of range")
        if self.training_camera_width <= 0 or self.training_camera_height <= 0:
            raise ValueError("Training-camera dimensions must be positive")
        # Headless camera sensors still require a graphics context. Preserve
        # graphics_device=-1 exactly when recording is disabled so compute-only
        # servers follow the original path without graphics initialization.
        graphics_required = not headless or self.training_camera_enabled
        self.graphics_device_id = self.sim_device_id if graphics_required else -1

        torch.manual_seed(int(self.cfg.seed))
        np.random.seed(int(self.cfg.seed))

        demo_path = ROOT_DIR / self.cfg.motion.file
        self.reference = JointDemonstration60Hz.load(
            demo_path,
            device=self.device,
            expected_hz=float(self.cfg.motion.frequency_hz),
        )
        if self.reference.sample_count < 2:
            raise ValueError(
                "A demonstration needs at least two samples, got {}".format(
                    self.reference.sample_count
                )
            )
        (
            self.rsi_distribution,
            self.rsi_max_start_index,
            self.rsi_pregrasp_start_index,
            self.rsi_early_probability,
        ) = resolve_rsi_settings(self.cfg.env, self.reference.last_index)

        # Purely a visual benchmark, and it doubles the simulated bodies, so
        # only an explicit opt-in builds it. evaluate.py ties it to --viewer;
        # training never sets it. It stays available headless so it can be
        # exercised by tests.
        self.reference_ghost_enabled = bool(
            getattr(self.cfg.viewer, "reference_ghost", False)
        )
        # Only the robot and optional ghost have DOFs. The physical cube and
        # fixed table are root-state actors and therefore do not change the
        # layout of the global DOF tensor.
        self.actors_per_env = 2 if self.reference_ghost_enabled else 1
        self.total_actors_per_env = 4 if self.reference_ghost_enabled else 3
        # getattr keeps a config object that predates the field usable; note
        # that a saved config.json from before it was added carries no value to
        # restore, so replaying such a run picks up whatever the class default
        # is now rather than the self-collision it actually trained with.
        self.self_collision_enabled = bool(
            getattr(self.cfg.asset, "self_collision", True)
        )

        self.gym = gymapi.acquire_gym()
        self.sim = self._create_sim()
        self._add_ground_plane()
        self.robot_asset = self._load_robot_asset()
        self._create_object_assets()
        self._create_envs()
        self.training_camera_handle = None
        if self.training_camera_enabled:
            self._create_training_camera()
        self._resolve_observation_body_indices()
        self.gym.prepare_sim(self.sim)
        self.viewer = None
        if not self.headless:
            self._create_viewer()
        self._acquire_tensors()
        self._allocate_buffers()
        self.reset()

    def _create_sim(self):
        params = gymapi.SimParams()
        params.dt = float(self.cfg.sim.dt)
        params.substeps = int(self.cfg.sim.substeps)
        params.up_axis = gymapi.UP_AXIS_Z
        params.gravity = gymapi.Vec3(*[float(v) for v in self.cfg.sim.gravity])
        params.use_gpu_pipeline = bool(self.cfg.sim.use_gpu_pipeline)

        physx = self.cfg.sim.physx
        params.physx.use_gpu = bool(physx.use_gpu)
        params.physx.num_threads = int(physx.num_threads)
        params.physx.solver_type = int(physx.solver_type)
        params.physx.num_position_iterations = int(physx.num_position_iterations)
        params.physx.num_velocity_iterations = int(physx.num_velocity_iterations)
        params.physx.contact_offset = float(physx.contact_offset)
        params.physx.rest_offset = float(physx.rest_offset)
        params.physx.bounce_threshold_velocity = float(
            physx.bounce_threshold_velocity
        )
        params.physx.max_depenetration_velocity = float(
            physx.max_depenetration_velocity
        )
        params.physx.max_gpu_contact_pairs = int(physx.max_gpu_contact_pairs)
        params.physx.default_buffer_size_multiplier = float(
            physx.default_buffer_size_multiplier
        )
        # Contact reporting is a measurable GPU cost at 4096 environments.
        # Preserve the old zero-overhead CC_NEVER path unless the optional
        # fingertip-contact feature is explicitly enabled.
        contact_collection = self.contact_collection if self.contact_enabled else 0
        params.physx.contact_collection = gymapi.ContactCollection(
            contact_collection
        )

        sim = self.gym.create_sim(
            self.sim_device_id,
            self.graphics_device_id,
            gymapi.SIM_PHYSX,
            params,
        )
        if sim is None:
            raise RuntimeError("Isaac Gym failed to create the PhysX simulation")
        return sim

    def _add_ground_plane(self) -> None:
        plane = gymapi.PlaneParams()
        plane.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane.static_friction = float(self.cfg.terrain.static_friction)
        plane.dynamic_friction = float(self.cfg.terrain.dynamic_friction)
        plane.restitution = float(self.cfg.terrain.restitution)
        self.gym.add_ground(self.sim, plane)

    def _load_robot_asset(self):
        asset_path = ROOT_DIR / self.cfg.asset.file
        options = gymapi.AssetOptions()
        options.fix_base_link = bool(self.cfg.asset.fix_base_link)
        options.disable_gravity = bool(self.cfg.asset.disable_gravity)
        options.collapse_fixed_joints = bool(self.cfg.asset.collapse_fixed_joints)
        options.flip_visual_attachments = bool(self.cfg.asset.flip_visual_attachments)
        options.thickness = float(self.cfg.asset.thickness)
        options.angular_damping = float(self.cfg.asset.angular_damping)
        options.linear_damping = float(self.cfg.asset.linear_damping)
        options.use_physx_armature = bool(self.cfg.asset.use_physx_armature)
        options.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)

        asset = self.gym.load_asset(
            self.sim,
            str(ROOT_DIR),
            str(asset_path.relative_to(ROOT_DIR)),
            options,
        )
        if asset is None:
            raise RuntimeError("Failed to load robot asset: {}".format(asset_path))

        self.demo_to_asset = validate_joint_order(self.gym, asset)
        self.pd_properties = configure_pd_properties(
            self.gym, asset, self.demo_to_asset
        )
        self.collision_filter_bits = configure_asset_wrist_collision_filters(
            self.gym, asset
        )
        self._configure_robot_contact_properties(asset)

        lower_asset = np.asarray(self.pd_properties["lower"], dtype=np.float32)
        upper_asset = np.asarray(self.pd_properties["upper"], dtype=np.float32)
        lower_demo = lower_asset[self.demo_to_asset]
        upper_demo = upper_asset[self.demo_to_asset]
        if np.any(~np.isfinite(lower_demo)) or np.any(~np.isfinite(upper_demo)):
            raise ValueError("Robot position limits must be finite")
        if np.any(lower_demo >= upper_demo):
            raise ValueError("Robot contains an invalid position interval")
        self.joint_lower_limits = torch.as_tensor(
            lower_demo, dtype=torch.float32, device=self.device
        )
        self.joint_upper_limits = torch.as_tensor(
            upper_demo, dtype=torch.float32, device=self.device
        )
        self.arm_lower_limits = self.joint_lower_limits[: len(ARM_JOINT_NAMES)]
        self.arm_upper_limits = self.joint_upper_limits[: len(ARM_JOINT_NAMES)]
        default_arm = torch.as_tensor(
            self.cfg.init_state.default_arm_joint_angles,
            dtype=torch.float32,
            device=self.device,
        )
        default_hand = torch.as_tensor(
            self.cfg.init_state.default_hand_joint_angles,
            dtype=torch.float32,
            device=self.device,
        )
        if default_arm.shape != (len(ARM_JOINT_NAMES),):
            raise ValueError("Default arm pose must contain exactly 6 angles")
        if default_hand.shape != (len(HAND_JOINT_NAMES),):
            raise ValueError("Default hand pose must contain exactly 20 angles")
        self.default_positions = torch.cat((default_arm, default_hand))
        self.default_arm_positions = self.default_positions[: len(ARM_JOINT_NAMES)]
        if torch.any(self.default_positions < self.joint_lower_limits) or torch.any(
            self.default_positions > self.joint_upper_limits
        ):
            raise ValueError("Default pose exceeds the URDF position limits")
        # One scale per joint, so the arm and hand residuals keep their own
        # resolution while the action stays a single flat vector.
        self.action_scales = torch.cat(
            (
                torch.full(
                    (len(ARM_JOINT_NAMES),),
                    self.action_scale,
                    dtype=torch.float32,
                    device=self.device,
                ),
                torch.full(
                    (len(HAND_JOINT_NAMES),),
                    self.hand_action_scale,
                    dtype=torch.float32,
                    device=self.device,
                ),
            )
        )
        self.demo_to_asset_tensor = torch.as_tensor(
            self.demo_to_asset, dtype=torch.long, device=self.device
        )

        q = self.reference.q
        if torch.any(q < self.joint_lower_limits - 1e-6) or torch.any(
            q > self.joint_upper_limits + 1e-6
        ):
            raise ValueError("The demonstration exceeds the robot position limits")
        return asset

    def _configure_robot_contact_properties(self, asset) -> None:
        """Configure materials and external robot collision-filter bits.

        Every UR5e body, including ``wrist_3_link``, is filtered against the
        cube. With fixed joints collapsed, wrist_3 also owns the static DG5F
        mount/palm shapes, so those are necessarily filtered with it. The
        articulated ``rl_dg_*`` finger bodies remain able to contact the cube.
        """
        body_names = tuple(self.gym.get_asset_rigid_body_names(asset))
        shape_ranges = self.gym.get_asset_rigid_body_shape_indices(asset)
        shape_properties = self.gym.get_asset_rigid_shape_properties(asset)

        used_bits = 0
        for properties in shape_properties:
            used_bits |= int(properties.filter)
        def allocate_filter_bit(label):
            nonlocal used_bits
            filter_bit = 1
            while used_bits & filter_bit:
                filter_bit <<= 1
            if filter_bit >= (1 << 31):
                raise RuntimeError(
                    "No collision-filter bit remains for {}".format(label)
                )
            used_bits |= filter_bit
            return filter_bit

        self.robot_table_collision_filter_bit = allocate_filter_bit(
            "robot/table"
        )
        self.arm_cube_collision_filter_bit = allocate_filter_bit("arm/cube")

        arm_body_names = []
        hand_body_names = []
        for body_name, shape_range in zip(body_names, shape_ranges):
            is_hand = body_name.startswith("rl_dg_")
            (hand_body_names if is_hand else arm_body_names).append(body_name)
            for shape_index in range(
                shape_range.start, shape_range.start + shape_range.count
            ):
                properties = shape_properties[shape_index]
                properties.friction = float(self.cfg.asset.friction)
                properties.restitution = float(self.cfg.asset.restitution)
                # Every robot shape filters the table. UR5e shapes share the
                # cube bit; articulated DG5F finger shapes do not.
                properties.filter |= self.robot_table_collision_filter_bit
                if not is_hand:
                    properties.filter |= self.arm_cube_collision_filter_bit
        if not arm_body_names or not hand_body_names:
            raise RuntimeError(
                "Could not split robot collision bodies into arm and hand"
            )
        self.arm_collision_body_names = tuple(arm_body_names)
        self.hand_collision_body_names = tuple(hand_body_names)

        fingertip_names = {
            "rl_dg_{}_4".format(finger) for finger in range(1, 6)
        }
        for body_name, shape_range in zip(body_names, shape_ranges):
            if body_name not in fingertip_names:
                continue
            for shape_index in range(
                shape_range.start, shape_range.start + shape_range.count
            ):
                shape_properties[shape_index].friction = float(
                    self.cfg.asset.fingertip_friction
                )

        self.gym.set_asset_rigid_shape_properties(asset, shape_properties)

    def _create_object_assets(self) -> None:
        cube_options = gymapi.AssetOptions()
        cube_options.disable_gravity = False
        cube_options.fix_base_link = False
        self.cube_asset = self.gym.create_box(
            self.sim,
            *[float(v) for v in self.cfg.object.size_m],
            cube_options,
        )
        if self.cube_asset is None:
            raise RuntimeError("Isaac Gym failed to create the cuboid asset")
        cube_shapes = self.gym.get_asset_rigid_shape_properties(self.cube_asset)
        for properties in cube_shapes:
            # Shared only by UR5e shapes. The table and articulated fingers
            # lack this bit, so cube-table and cube-finger contacts stay active.
            properties.filter = self.arm_cube_collision_filter_bit
            properties.friction = float(self.cfg.object.friction)
            properties.restitution = float(self.cfg.object.restitution)
        self.gym.set_asset_rigid_shape_properties(self.cube_asset, cube_shapes)

        table_options = gymapi.AssetOptions()
        table_options.disable_gravity = True
        table_options.fix_base_link = True
        self.table_asset = self.gym.create_box(
            self.sim,
            *[float(v) for v in self.cfg.table.size_m],
            table_options,
        )
        if self.table_asset is None:
            raise RuntimeError("Isaac Gym failed to create the table asset")
        table_shapes = self.gym.get_asset_rigid_shape_properties(self.table_asset)
        for properties in table_shapes:
            properties.filter = self.robot_table_collision_filter_bit
            properties.friction = float(self.cfg.table.friction)
            properties.restitution = float(self.cfg.table.restitution)
        self.gym.set_asset_rigid_shape_properties(self.table_asset, table_shapes)

    def _cube_pose_ur_base_to_world(self, pose_xyzw: np.ndarray) -> np.ndarray:
        pose_xyzw = np.asarray(pose_xyzw, dtype=np.float64)
        if pose_xyzw.shape != (7,) or not np.all(np.isfinite(pose_xyzw)):
            raise ValueError("Expected one finite cube pose with shape (7,)")
        world = np.empty(7, dtype=np.float64)
        world[:3] = np.asarray(self.cfg.init_state.pos, dtype=np.float64) + (
            pose_xyzw[:3] * np.asarray([-1.0, -1.0, 1.0])
        )
        # q_world = q_z(pi) * q_ur, in xyzw order.
        x, y, z, w = pose_xyzw[3:7]
        world[3:7] = (-y, x, w, -z)
        world[3:7] /= np.linalg.norm(world[3:7])
        return world

    @staticmethod
    def _pose_array_to_transform(pose_xyzw: np.ndarray) -> gymapi.Transform:
        transform = gymapi.Transform()
        transform.p = gymapi.Vec3(*[float(v) for v in pose_xyzw[:3]])
        transform.r = gymapi.Quat(*[float(v) for v in pose_xyzw[3:7]])
        return transform

    def _set_cube_body_properties(self, env, actor: int) -> None:
        properties = self.gym.get_actor_rigid_body_properties(env, actor)
        properties[0].mass = float(self.cfg.object.mass_kg)
        inertia = [float(v) for v in self.cfg.object.inertia_kg_m2]
        properties[0].inertia.x = gymapi.Vec3(inertia[0], 0.0, 0.0)
        properties[0].inertia.y = gymapi.Vec3(0.0, inertia[1], 0.0)
        properties[0].inertia.z = gymapi.Vec3(0.0, 0.0, inertia[2])
        self.gym.set_actor_rigid_body_properties(env, actor, properties, False)

    def _create_envs(self) -> None:
        spacing = float(self.cfg.env.env_spacing)
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        per_row = int(math.ceil(math.sqrt(self.num_envs)))
        body_count = self.gym.get_asset_rigid_body_count(self.robot_asset)
        shape_count = self.gym.get_asset_rigid_shape_count(self.robot_asset)

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(*[float(v) for v in self.cfg.init_state.pos])
        pose.r = gymapi.Quat(*[float(v) for v in self.cfg.init_state.rot])
        cube_pose_array = self._cube_pose_ur_base_to_world(
            self.reference.cube_pose[0].detach().cpu().numpy()
        )
        cube_pose = self._pose_array_to_transform(cube_pose_array)
        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(
            0.0,
            0.0,
            float(self.cfg.init_state.pos[2])
            - float(self.cfg.table.surface_below_robot_base_m)
            - float(self.cfg.table.size_m[2]) / 2.0,
        )

        ghost_pose = None
        if self.reference_ghost_enabled:
            offset = [float(v) for v in self.cfg.viewer.reference_ghost_offset]
            ghost_pose = gymapi.Transform()
            ghost_pose.p = gymapi.Vec3(
                pose.p.x + offset[0], pose.p.y + offset[1], pose.p.z + offset[2]
            )
            ghost_pose.r = pose.r

        self.envs = []
        self.robot_handles = []
        self.ghost_handles = []
        self.cube_handles = []
        self.table_handles = []
        actor_indices = []
        ghost_actor_indices = []
        cube_actor_indices = []
        table_actor_indices = []
        for env_index in range(self.num_envs):
            env = self.gym.create_env(self.sim, lower, upper, per_row)
            if env is None:
                raise RuntimeError("Failed to create environment {}".format(env_index))
            # Keep only articulated robots in the aggregate. Putting the cube
            # in an aggregate with self-collision disabled would also suppress
            # the robot-cube contacts that this environment needs.
            aggregate_body_count = body_count
            aggregate_shape_count = shape_count
            if self.reference_ghost_enabled:
                aggregate_body_count *= 2
                aggregate_shape_count *= 2
            # The aggregate's last flag is what actually governs self-collision.
            # create_actor's own filter argument below never reaches the shapes:
            # they keep the filter bits the asset gave them.
            self.gym.begin_aggregate(
                env,
                aggregate_body_count,
                aggregate_shape_count,
                self.self_collision_enabled,
            )
            actor = self.gym.create_actor(
                env,
                self.robot_asset,
                pose,
                "robot",
                env_index,
                -1,
                0,
            )
            if actor < 0:
                raise RuntimeError("Failed to create robot actor {}".format(env_index))
            self.gym.set_actor_dof_properties(env, actor, self.pd_properties)
            if self.reference_ghost_enabled:
                # A collision group of its own keeps the ghost from touching the
                # policy robot, the ground, or itself, so it can never perturb
                # the run it is meant to illustrate.
                ghost = self.gym.create_actor(
                    env,
                    self.robot_asset,
                    ghost_pose,
                    "reference_ghost",
                    self.num_envs + env_index,
                    -1,
                    0,
                )
                if ghost < 0:
                    raise RuntimeError(
                        "Failed to create ghost actor {}".format(env_index)
                    )
                self.gym.set_actor_dof_properties(env, ghost, self.pd_properties)
                self._paint_ghost(env, ghost)
                self.ghost_handles.append(ghost)
                ghost_actor_indices.append(
                    self.gym.get_actor_index(env, ghost, gymapi.DOMAIN_SIM)
                )
            self.gym.end_aggregate(env)

            cube = self.gym.create_actor(
                env,
                self.cube_asset,
                cube_pose,
                "cube",
                env_index,
                -1,
                0,
            )
            if cube < 0:
                raise RuntimeError("Failed to create cuboid actor {}".format(env_index))
            self._set_cube_body_properties(env, cube)
            self.gym.set_rigid_body_color(
                env,
                cube,
                0,
                gymapi.MESH_VISUAL,
                gymapi.Vec3(*[float(v) for v in self.cfg.object.color]),
            )

            table = self.gym.create_actor(
                env,
                self.table_asset,
                table_pose,
                "table",
                env_index,
                -1,
                0,
            )
            if table < 0:
                raise RuntimeError("Failed to create table actor {}".format(env_index))
            self.gym.set_rigid_body_color(
                env,
                table,
                0,
                gymapi.MESH_VISUAL,
                gymapi.Vec3(*[float(v) for v in self.cfg.table.color]),
            )

            self.envs.append(env)
            self.robot_handles.append(actor)
            self.cube_handles.append(cube)
            self.table_handles.append(table)
            actor_indices.append(
                self.gym.get_actor_index(env, actor, gymapi.DOMAIN_SIM)
            )
            cube_actor_indices.append(
                self.gym.get_actor_index(env, cube, gymapi.DOMAIN_SIM)
            )
            table_actor_indices.append(
                self.gym.get_actor_index(env, table, gymapi.DOMAIN_SIM)
            )

        self.actor_indices = torch.as_tensor(
            actor_indices, dtype=torch.int32, device=self.device
        )
        self.ghost_actor_indices = torch.as_tensor(
            ghost_actor_indices, dtype=torch.int32, device=self.device
        )
        self.cube_actor_indices = torch.as_tensor(
            cube_actor_indices, dtype=torch.int32, device=self.device
        )
        self.table_actor_indices = torch.as_tensor(
            table_actor_indices, dtype=torch.int32, device=self.device
        )

    def _resolve_observation_body_indices(self) -> None:
        """Resolve the surviving rigid bodies used by the 114D observation."""
        env = self.envs[0]
        actor = self.robot_handles[0]

        def body_index(name: str) -> int:
            index = self.gym.find_actor_rigid_body_index(
                env, actor, name, gymapi.DOMAIN_ENV
            )
            if index < 0:
                raise ValueError("Robot rigid body {!r} was not found".format(name))
            return int(index)

        self.wrist_body_index = body_index(PALM_PARENT_BODY_NAME)
        self.fingertip_body_indices = torch.as_tensor(
            [body_index(name) for name in FINGERTIP_BODY_NAMES],
            dtype=torch.long,
            device=self.device,
        )
        self.contact_fingertip_body_indices = torch.as_tensor(
            [
                body_index(FINGERTIP_BODY_NAMES_BY_SEMANTIC_NAME[name])
                for name in self.contact_fingertip_names
            ],
            dtype=torch.long,
            device=self.device,
        )
        self.proximity_fingertip_indices = torch.as_tensor(
            [
                FINGERTIP_BODY_NAMES.index(
                    FINGERTIP_BODY_NAMES_BY_SEMANTIC_NAME[name]
                )
                for name in self.proximity_fingertip_names
            ],
            dtype=torch.long,
            device=self.device,
        )

    def _paint_ghost(self, env, ghost) -> None:
        color = gymapi.Vec3(
            *[float(v) for v in self.cfg.viewer.reference_ghost_color]
        )
        for body_index in range(self.gym.get_actor_rigid_body_count(env, ghost)):
            self.gym.set_rigid_body_color(
                env, ghost, body_index, gymapi.MESH_VISUAL, color
            )

    def _create_viewer(self) -> None:
        camera_properties = gymapi.CameraProperties()
        self.viewer = self.gym.create_viewer(self.sim, camera_properties)
        if self.viewer is None:
            raise RuntimeError("Isaac Gym failed to create the viewer")

        env_origin = self.gym.get_env_origin(self.envs[0])
        camera_position = gymapi.Vec3(
            env_origin.x + float(self.cfg.viewer.camera_position[0]),
            env_origin.y + float(self.cfg.viewer.camera_position[1]),
            env_origin.z + float(self.cfg.viewer.camera_position[2]),
        )
        camera_lookat = gymapi.Vec3(
            env_origin.x + float(self.cfg.viewer.camera_lookat[0]),
            env_origin.y + float(self.cfg.viewer.camera_lookat[1]),
            env_origin.z + float(self.cfg.viewer.camera_lookat[2]),
        )
        self.gym.viewer_camera_look_at(
            self.viewer, None, camera_position, camera_lookat
        )

    def _create_training_camera(self) -> None:
        """Create one off-screen sensor aimed at a real training environment."""
        camera_properties = gymapi.CameraProperties()
        camera_properties.width = self.training_camera_width
        camera_properties.height = self.training_camera_height
        camera_properties.enable_tensors = False
        env = self.envs[self.training_camera_env_index]
        handle = self.gym.create_camera_sensor(env, camera_properties)
        if handle < 0:
            raise RuntimeError("Isaac Gym failed to create the training camera")
        origin = self.gym.get_env_origin(env)
        position = gymapi.Vec3(
            origin.x + float(self.cfg.viewer.camera_position[0]),
            origin.y + float(self.cfg.viewer.camera_position[1]),
            origin.z + float(self.cfg.viewer.camera_position[2]),
        )
        lookat = gymapi.Vec3(
            origin.x + float(self.cfg.viewer.camera_lookat[0]),
            origin.y + float(self.cfg.viewer.camera_lookat[1]),
            origin.z + float(self.cfg.viewer.camera_lookat[2]),
        )
        self.gym.set_camera_location(handle, env, position, lookat)
        self.training_camera_handle = handle

    def capture_training_camera_frame(self) -> np.ndarray:
        """Render and return one RGB frame from the selected training env."""
        if not self.training_camera_enabled or self.training_camera_handle is None:
            raise RuntimeError("Training-camera capture is not enabled")
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        env = self.envs[self.training_camera_env_index]
        color = self.gym.get_camera_image(
            self.sim,
            env,
            self.training_camera_handle,
            gymapi.IMAGE_COLOR,
        )
        rgba = np.asarray(color, dtype=np.uint8)
        expected = self.training_camera_height * self.training_camera_width * 4
        if rgba.size != expected:
            raise RuntimeError(
                "Camera returned {} values, expected {}".format(
                    rgba.size, expected
                )
            )
        rgba = rgba.reshape(
            self.training_camera_height, self.training_camera_width, 4
        )
        return np.ascontiguousarray(rgba[:, :, :3])

    def viewer_closed(self) -> bool:
        return self.viewer is not None and self.gym.query_viewer_has_closed(
            self.viewer
        )

    def render(self, sync_frame_time: bool = True) -> None:
        if self.viewer is None or self.viewer_closed():
            return
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)
        if sync_frame_time:
            self.gym.sync_frame_time(self.sim)

    def _acquire_tensors(self) -> None:
        dof_state_raw = self.gym.acquire_dof_state_tensor(self.sim)
        # The simulation buffers cover every actor, so they are kept whole for
        # the Isaac Gym setters, which require the full contiguous tensor, and
        # sliced for everything else. The policy robot is always actor 0, so
        # its slice is the leading block and the rest of the environment sees
        # exactly the same shapes whether or not the ghost exists.
        dof_count = len(JOINT_NAMES)
        self.dof_state_all = gymtorch.wrap_tensor(dof_state_raw).view(
            self.num_envs, self.actors_per_env * dof_count, 2
        )
        self.dof_state = self.dof_state_all[:, :dof_count]
        self.dof_position_asset = self.dof_state[..., 0]
        self.dof_velocity_asset = self.dof_state[..., 1]
        self.position_targets_all = torch.zeros(
            (self.num_envs, self.actors_per_env * dof_count),
            dtype=torch.float32,
            device=self.device,
        )
        self.position_targets_asset = self.position_targets_all[:, :dof_count]
        # The ghost is driven by the same position drive as the policy robot,
        # fed the reference pose instead of the policy target. That keeps it in
        # the one target write the step already performs: an extra per-step
        # DOF-state write conflicts with the GPU pipeline and stalls the step.
        self.ghost_dof_state = (
            self.dof_state_all[:, dof_count:]
            if self.reference_ghost_enabled
            else None
        )
        self.ghost_position_targets = (
            self.position_targets_all[:, dof_count:]
            if self.reference_ghost_enabled
            else None
        )
        self.all_env_ids = torch.arange(
            self.num_envs, device=self.device, dtype=torch.long
        )
        root_state_raw = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_state_all = gymtorch.wrap_tensor(root_state_raw).view(-1, 13)
        rigid_body_state_raw = self.gym.acquire_rigid_body_state_tensor(self.sim)
        rigid_bodies_per_env = self.gym.get_env_rigid_body_count(self.envs[0])
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state_raw).view(
            self.num_envs, rigid_bodies_per_env, 13
        )
        self.net_contact_forces = None
        if self.contact_enabled:
            net_contact_force_raw = self.gym.acquire_net_contact_force_tensor(
                self.sim
            )
            self.net_contact_forces = gymtorch.wrap_tensor(
                net_contact_force_raw
            ).view(self.num_envs, rigid_bodies_per_env, 3)
        self.palm_position_in_wrist = torch.tensor(
            PALM_POSITION_IN_WRIST, dtype=torch.float32, device=self.device
        ).expand(self.num_envs, -1)
        self.palm_orientation_in_wrist = torch.tensor(
            PALM_ORIENTATION_IN_WRIST, dtype=torch.float32, device=self.device
        ).expand(self.num_envs, -1)
        self.fingertip_offsets = torch.tensor(
            FINGERTIP_OFFSETS, dtype=torch.float32, device=self.device
        ).unsqueeze(0).expand(self.num_envs, -1, -1)
        self.world_axis_sign = torch.tensor(
            [-1.0, -1.0, 1.0], dtype=torch.float32, device=self.device
        )
        self.robot_base_position = torch.tensor(
            self.cfg.init_state.pos, dtype=torch.float32, device=self.device
        )
        self.object_half_extents = torch.tensor(
            self.cfg.object.size_m, dtype=torch.float32, device=self.device
        ) / 2.0
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        if self.contact_enabled:
            self.gym.refresh_net_contact_force_tensor(self.sim)

    def _allocate_buffers(self) -> None:
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_obs), dtype=torch.float32, device=self.device
        )
        self.critic_obs_buf = None
        self.rew_buf = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.reset_buf = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.time_out_buf = torch.zeros_like(self.reset_buf)
        self.episode_length_buf = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.reference_index = torch.zeros_like(self.episode_length_buf)
        self.arm_violation_steps = torch.zeros_like(self.episode_length_buf)
        self.hand_violation_steps = torch.zeros_like(self.episode_length_buf)
        self.object_violation_steps = torch.zeros_like(self.episode_length_buf)
        self.arm_violation = torch.zeros_like(self.reset_buf)
        self.hand_violation = torch.zeros_like(self.reset_buf)
        self.object_violation = torch.zeros_like(self.reset_buf)
        # A primitive box actor is rooted at its centre of mass. Keep the
        # initial and running peak world-z per episode outside episode_sums:
        # these are extrema, not quantities that should be time-averaged.
        self.episode_initial_object_com_height_m = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.episode_peak_object_com_height_m = torch.zeros_like(
            self.episode_initial_object_com_height_m
        )
        self.actions = torch.zeros(
            (self.num_envs, self.num_actions), dtype=torch.float32, device=self.device
        )
        # a_{t-1} for the action-rate regularization term.
        self.previous_actions = torch.zeros_like(self.actions)
        self.episode_sums = {
            name: torch.zeros(
                self.num_envs, dtype=torch.float32, device=self.device
            )
            for name in (
                "reward",
                "position_reward",
                "velocity_reward",
                "action_rate_reward",
                "hand_position_reward",
                "hand_velocity_reward",
                "hand_action_rate_reward",
                "object_position_reward",
                "object_orientation_reward",
                "fingertip_object_distance_reward",
                "fingertip_object_distance_m",
                "rms_position_error",
                "rms_velocity_error",
                "rms_action_rate",
                "rms_hand_position_error",
                "rms_hand_velocity_error",
                "rms_hand_action_rate",
                "object_position_error_m",
                "object_orientation_error_rad",
                "fingertip_contact_reward",
                "fingertip_contact_fraction",
                "fingertip_contact_force_n",
            )
        }
        self.extras = {}

    @property
    def q(self) -> torch.Tensor:
        return self.dof_position_asset[:, self.demo_to_asset_tensor]

    @property
    def dq(self) -> torch.Tensor:
        return self.dof_velocity_asset[:, self.demo_to_asset_tensor]

    @property
    def previous_targets(self) -> torch.Tensor:
        """Most recently applied position targets, in demonstration order."""
        return self.position_targets_asset[:, self.demo_to_asset_tensor]

    @property
    def cube_root_state(self) -> torch.Tensor:
        return self.root_state_all[self.cube_actor_indices.long()]

    @property
    def cube_position(self) -> torch.Tensor:
        return self.cube_root_state[:, 0:3]

    @property
    def cube_orientation(self) -> torch.Tensor:
        return self.cube_root_state[:, 3:7]

    @property
    def cube_linear_velocity(self) -> torch.Tensor:
        return self.cube_root_state[:, 7:10]

    @property
    def cube_angular_velocity(self) -> torch.Tensor:
        return self.cube_root_state[:, 10:13]

    @property
    def robot_root_state(self) -> torch.Tensor:
        return self.root_state_all[self.actor_indices.long()]

    def _fingertip_positions_world(self) -> torch.Tensor:
        fingertip_states = self.rigid_body_state[:, self.fingertip_body_indices]
        fingertip_orientations = _normalize_canonical_quaternion(
            fingertip_states[..., 3:7]
        )
        return fingertip_states[..., 0:3] + _quat_rotate(
            fingertip_orientations, self.fingertip_offsets
        )

    def _task_space_observation_components(self) -> Tuple[torch.Tensor, ...]:
        """Return the seven object/task-space blocks appended to the old 79D.

        All cube-relative vectors are expressed in the palm frame.  Palm pose
        is expressed in the robot actor-base frame.  Relative linear velocity
        includes the rotating-frame transport term, so it is the derivative of
        the palm-frame cube displacement rather than only a velocity difference.
        """
        wrist = self.rigid_body_state[:, self.wrist_body_index]
        wrist_position = wrist[:, 0:3]
        wrist_orientation = _normalize_canonical_quaternion(wrist[:, 3:7])
        wrist_linear_velocity = wrist[:, 7:10]
        wrist_angular_velocity = wrist[:, 10:13]

        palm_offset_world = _quat_rotate(
            wrist_orientation, self.palm_position_in_wrist
        )
        palm_position_world = wrist_position + palm_offset_world
        palm_orientation_world = _normalize_canonical_quaternion(
            _quat_multiply(
                wrist_orientation, self.palm_orientation_in_wrist
            )
        )
        palm_linear_velocity_world = wrist_linear_velocity + torch.cross(
            wrist_angular_velocity, palm_offset_world, dim=-1
        )
        palm_angular_velocity_world = wrist_angular_velocity

        robot_root = self.robot_root_state
        robot_position_world = robot_root[:, 0:3]
        robot_orientation_world = _normalize_canonical_quaternion(
            robot_root[:, 3:7]
        )
        palm_position_robot = _quat_rotate_inverse(
            robot_orientation_world, palm_position_world - robot_position_world
        )
        palm_orientation_robot = _normalize_canonical_quaternion(
            _quat_multiply(
                _quat_conjugate(robot_orientation_world), palm_orientation_world
            )
        )

        cube_displacement_world = self.cube_position - palm_position_world
        cube_center_palm = _quat_rotate_inverse(
            palm_orientation_world, cube_displacement_world
        )
        cube_orientation_palm = _normalize_canonical_quaternion(
            _quat_multiply(
                _quat_conjugate(palm_orientation_world),
                _normalize_canonical_quaternion(self.cube_orientation),
            )
        )

        fingertip_positions_world = self._fingertip_positions_world()
        cube_center_fingertips_palm = _quat_rotate_inverse(
            palm_orientation_world.unsqueeze(1).expand(-1, 5, -1),
            self.cube_position.unsqueeze(1) - fingertip_positions_world,
        ).reshape(self.num_envs, 15)

        cube_linear_velocity_palm = _quat_rotate_inverse(
            palm_orientation_world,
            self.cube_linear_velocity
            - palm_linear_velocity_world
            - torch.cross(
                palm_angular_velocity_world, cube_displacement_world, dim=-1
            ),
        )
        cube_angular_velocity_palm = _quat_rotate_inverse(
            palm_orientation_world,
            self.cube_angular_velocity - palm_angular_velocity_world,
        )
        return (
            palm_position_robot,
            palm_orientation_robot,
            cube_center_palm,
            cube_orientation_palm,
            cube_center_fingertips_palm,
            cube_linear_velocity_palm,
            cube_angular_velocity_palm,
        )

    @property
    def arm_q(self) -> torch.Tensor:
        return self.q[:, : len(ARM_JOINT_NAMES)]

    @property
    def arm_dq(self) -> torch.Tensor:
        return self.dq[:, : len(ARM_JOINT_NAMES)]

    @property
    def hand_q(self) -> torch.Tensor:
        return self.q[:, len(ARM_JOINT_NAMES):]

    @property
    def hand_dq(self) -> torch.Tensor:
        return self.dq[:, len(ARM_JOINT_NAMES):]

    @property
    def previous_arm_targets(self) -> torch.Tensor:
        return self.previous_targets[:, : len(ARM_JOINT_NAMES)]

    @property
    def previous_hand_targets(self) -> torch.Tensor:
        return self.previous_targets[:, len(ARM_JOINT_NAMES):]

    def _write_demo_order_to_asset(
        self, destination: torch.Tensor, values: torch.Tensor
    ) -> None:
        destination[:, self.demo_to_asset_tensor] = values

    def _cube_reference_root_states(self, sample) -> torch.Tensor:
        """Convert recorded UR-base object state to Isaac Gym world state."""
        position = (
            self.robot_base_position
            + sample.cube_pose[:, :3] * self.world_axis_sign
        )
        quaternion_ur = sample.cube_pose[:, 3:7]
        x, y, z, w = quaternion_ur.unbind(dim=1)
        quaternion_world = torch.stack((-y, x, w, -z), dim=1)
        quaternion_world = torch.nn.functional.normalize(
            quaternion_world, dim=1
        )
        return torch.cat(
            (
                position,
                quaternion_world,
                sample.cube_linear_velocity * self.world_axis_sign,
                sample.cube_angular_velocity * self.world_axis_sign,
            ),
            dim=1,
        )

    def _reset_cube_from_reference(self, env_ids: torch.Tensor, sample) -> None:
        """Reset selected physical cubes to their matching RSI object states.

        Pose, linear velocity, and angular velocity come from the exact same
        reference samples used to reset the corresponding robot DOF states.
        The indexed root-state upload leaves all non-reset environments and
        every fixed table untouched.
        """
        cube_actor_ids = self.cube_actor_indices[env_ids].contiguous()
        self.root_state_all[cube_actor_ids.long()] = (
            self._cube_reference_root_states(sample)
        )
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state_all),
            gymtorch.unwrap_tensor(cube_actor_ids),
            cube_actor_ids.numel(),
        )

    def scale_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Apply AnimRL's unbounded residual-action target mapping."""
        residual = (actions * self.action_scales).clamp(
            -self.action_target_clip, self.action_target_clip
        )
        return self.default_positions + residual

    def normalize_positions(self, positions: torch.Tensor) -> torch.Tensor:
        """Normalize physical joint positions only for the observation vector."""
        return (
            2.0
            * (positions - self.joint_lower_limits)
            / (self.joint_upper_limits - self.joint_lower_limits)
            - 1.0
        ).clamp(-1.0, 1.0)

    def normalize_arm_positions(self, positions: torch.Tensor) -> torch.Tensor:
        return (
            2.0
            * (positions - self.arm_lower_limits)
            / (self.arm_upper_limits - self.arm_lower_limits)
            - 1.0
        ).clamp(-1.0, 1.0)

    def positions_to_actions(self, positions: torch.Tensor) -> torch.Tensor:
        """Invert the AnimRL residual mapping for ideal reference playback."""
        return (positions - self.default_positions) / self.action_scales

    def next_reference_action(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the ideal 26-joint action and the complete next pose."""
        next_indices = (self.reference_index + 1).clamp(
            max=self.reference.last_index
        )
        target = self.reference.sample(next_indices).q
        return self.positions_to_actions(target), target

    def reset_idx(
        self,
        env_ids: torch.Tensor,
        reference_indices: Optional[torch.Tensor] = None,
    ) -> None:
        if env_ids.numel() == 0:
            return
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        if reference_indices is None:
            reference_indices = sample_rsi_indices(
                env_ids.numel(),
                self.device,
                self.rsi_distribution,
                self.rsi_max_start_index,
                self.rsi_pregrasp_start_index,
                self.rsi_early_probability,
            )
        else:
            reference_indices = reference_indices.to(
                device=self.device, dtype=torch.long
            )
            if reference_indices.ndim == 0:
                reference_indices = reference_indices.repeat(env_ids.numel())
            if reference_indices.shape != (env_ids.numel(),):
                raise ValueError("reference_indices has the wrong shape")
            # Explicit indices remain available across the complete motion for
            # diagnostics and the viewer. Only automatically sampled training
            # and evaluation resets are capped at rsi_max_start_index.
            max_start = self.reference.last_index - 1
            if torch.any(reference_indices < 0) or torch.any(
                reference_indices > max_start
            ):
                raise ValueError(
                    "RSI indices must lie in [0, {}]".format(max_start)
                )

        sample = self.reference.sample(reference_indices)
        self.reference_index[env_ids] = reference_indices
        self.episode_length_buf[env_ids] = 0
        self.arm_violation_steps[env_ids] = 0
        self.hand_violation_steps[env_ids] = 0
        self.object_violation_steps[env_ids] = 0
        self.reset_buf[env_ids] = False
        self.time_out_buf[env_ids] = False
        if hasattr(self, "episode_sums"):
            for values in self.episode_sums.values():
                values[env_ids] = 0.0
        if hasattr(self, "previous_actions"):
            # Seed a_{t-1} with the action that reproduces the RSI pose instead
            # of zero, so the first step of an episode is not charged an
            # action-rate penalty for the reset discontinuity.
            reset_action = self.positions_to_actions(sample.q)
            self.actions[env_ids] = reset_action
            self.previous_actions[env_ids] = reset_action

        state_subset = self.dof_state[env_ids]
        state_subset[:, self.demo_to_asset_tensor, 0] = sample.q
        state_subset[:, self.demo_to_asset_tensor, 1] = sample.dq
        self.dof_state[env_ids] = state_subset
        self.position_targets_asset[
            env_ids.unsqueeze(1), self.demo_to_asset_tensor.unsqueeze(0)
        ] = sample.q

        self._write_ghost_state(env_ids, sample.q, sample.dq)
        if self.reference_ghost_enabled:
            self.ghost_position_targets[
                env_ids.unsqueeze(1), self.demo_to_asset_tensor.unsqueeze(0)
            ] = sample.q
        actor_ids = self.actor_indices[env_ids]
        if self.reference_ghost_enabled:
            actor_ids = torch.cat((actor_ids, self.ghost_actor_indices[env_ids]))
        self._upload_dof_state(actor_ids)
        self._reset_cube_from_reference(env_ids, sample)
        reset_object_height = self.cube_position[env_ids, 2]
        self.episode_initial_object_com_height_m[env_ids] = reset_object_height
        self.episode_peak_object_com_height_m[env_ids] = reset_object_height
        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(self.position_targets_all)
        )
        # The new observation contains link-space kinematics. Refresh here,
        # rather than relying on a caller to do it, so an auto-reset and the
        # evaluator's direct reset_idx() both return the newly reset palm and
        # fingertip poses on their very first observation.
        self.gym.refresh_rigid_body_state_tensor(self.sim)

    def _write_ghost_state(self, env_ids, q, dq) -> None:
        """Fill the ghost rows of the DOF-state buffer; the caller uploads.

        Its drive is disabled and gravity is off, so writing the state is the
        only thing that moves it: the ghost replays the demonstration with no
        tracking error of its own, which is what makes it a benchmark.
        """
        if not self.reference_ghost_enabled:
            return
        subset = self.ghost_dof_state[env_ids]
        subset[:, self.demo_to_asset_tensor, 0] = q
        subset[:, self.demo_to_asset_tensor, 1] = dq
        self.ghost_dof_state[env_ids] = subset

    def _upload_dof_state(self, actor_ids) -> None:
        """Push DOF state for the given actors.

        Isaac Gym keeps only the last indexed DOF-state write of a frame, so
        the robot and the ghost have to travel in one call: a second call for
        the ghost silently discards the robot reset.
        """
        actor_ids = actor_ids.contiguous()
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state_all),
            gymtorch.unwrap_tensor(actor_ids),
            actor_ids.numel(),
        )

    def reset(self, reference_index: Optional[int] = None) -> torch.Tensor:
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        indices = None
        if reference_index is not None:
            indices = torch.full(
                (self.num_envs,),
                int(reference_index),
                dtype=torch.long,
                device=self.device,
            )
        self.reset_idx(env_ids, indices)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.compute_observations()
        return self.obs_buf

    def compute_observations(self) -> None:
        phase = (
            self.reference_index.float() / float(self.reference.last_index)
        ).unsqueeze(1)
        self.obs_buf.copy_(
            torch.cat(
                (
                    self.normalize_positions(self.q),
                    self.previous_targets,
                    self.dq,
                    phase,
                    *self._task_space_observation_components(),
                ),
                dim=1,
            )
        )
        if self.obs_buf.shape != (self.num_envs, self.num_obs):
            raise RuntimeError("Observation shape does not match configuration")

    def _compute_reward_and_errors(self) -> Dict[str, torch.Tensor]:
        reference = self.reference.sample(self.reference_index)
        reference_arm_q = reference.q[:, : len(ARM_JOINT_NAMES)]
        reference_arm_dq = reference.dq[:, : len(ARM_JOINT_NAMES)]
        reference_hand_q = reference.q[:, len(ARM_JOINT_NAMES):]
        reference_hand_dq = reference.dq[:, len(ARM_JOINT_NAMES):]
        arm_q_error = self.arm_q - reference_arm_q
        arm_dq_error = self.arm_dq - reference_arm_dq
        hand_q_error = self.hand_q - reference_hand_q
        hand_dq_error = self.hand_dq - reference_hand_dq
        action_rate = self.actions - self.previous_actions
        arm_action_rate = action_rate[:, : len(ARM_JOINT_NAMES)]
        hand_action_rate = action_rate[:, len(ARM_JOINT_NAMES):]

        # Arm and hand keep separate Gaussians: averaging one MSE over all 26
        # joints would let the 20 hand joints outvote the 6 arm joints in a
        # single term and dilute the gradient each block needs.
        rewards_cfg = self.cfg.rewards
        position_mse = arm_q_error.square().mean(dim=1)
        velocity_mse = arm_dq_error.square().mean(dim=1)
        action_rate_mse = arm_action_rate.square().mean(dim=1)
        hand_position_mse = hand_q_error.square().mean(dim=1)
        hand_velocity_mse = hand_dq_error.square().mean(dim=1)
        hand_action_rate_mse = hand_action_rate.square().mean(dim=1)

        reference_cube_root_state = self._cube_reference_root_states(reference)
        object_position_error = (
            self.cube_position - reference_cube_root_state[:, 0:3]
        )
        object_position_error_m = torch.linalg.vector_norm(
            object_position_error, dim=1
        )
        object_com_height_m = self.cube_position[:, 2]
        object_com_lift_m = (
            object_com_height_m - self.episode_initial_object_com_height_m
        )
        # q and -q represent the same rotation, hence abs(dot). The resulting
        # angle is the shortest geodesic rotation between the two orientations.
        actual_cube_orientation = _normalize_canonical_quaternion(
            self.cube_orientation
        )
        reference_cube_orientation = _normalize_canonical_quaternion(
            reference_cube_root_state[:, 3:7]
        )
        object_orientation_dot = (
            actual_cube_orientation * reference_cube_orientation
        ).sum(dim=1).abs().clamp(max=1.0)
        object_orientation_error_rad = 2.0 * torch.acos(
            object_orientation_dot
        )

        gaussian = lambda mse, std: torch.exp(-mse / (2.0 * float(std) ** 2))
        position_reward = gaussian(position_mse, rewards_cfg.position_arm_std_rad)
        velocity_reward = gaussian(
            velocity_mse, rewards_cfg.velocity_arm_std_rad_per_s
        )
        action_rate_reward = gaussian(
            action_rate_mse, rewards_cfg.action_rate_arm_std
        )
        hand_position_reward = gaussian(
            hand_position_mse, rewards_cfg.position_hand_std_rad
        )
        hand_velocity_reward = gaussian(
            hand_velocity_mse, rewards_cfg.velocity_hand_std_rad_per_s
        )
        hand_action_rate_reward = gaussian(
            hand_action_rate_mse, rewards_cfg.action_rate_hand_std
        )
        object_position_reward = gaussian(
            object_position_error_m.square(),
            rewards_cfg.object_position_std_m,
        )
        object_orientation_reward = gaussian(
            object_orientation_error_rad.square(),
            rewards_cfg.object_orientation_std_rad,
        )
        selected_fingertips_world = self._fingertip_positions_world()[
            :, self.proximity_fingertip_indices
        ]
        cube_orientation_expanded = actual_cube_orientation.unsqueeze(1).expand(
            -1, selected_fingertips_world.shape[1], -1
        )
        selected_fingertips_cube = _quat_rotate_inverse(
            cube_orientation_expanded,
            selected_fingertips_world - self.cube_position.unsqueeze(1),
        )
        proximity_active = (
            self.reference_index >= self.rsi_pregrasp_start_index
        )
        (
            fingertip_object_distance_reward,
            fingertip_object_distance_m,
            fingertip_object_distance_per_finger_m,
        ) = fingertip_cuboid_proximity(
            selected_fingertips_cube,
            self.object_half_extents,
            self.proximity_std_m,
            proximity_active,
        )
        if self.contact_enabled:
            (
                fingertip_contact_reward,
                fingertip_contact_fraction,
                mean_fingertip_contact_force_n,
            ) = fingertip_contact_diagnostics(
                self.net_contact_forces,
                self.contact_fingertip_body_indices,
                self.contact_force_threshold_n,
            )
            # Every fingertip, not only the reward's selection: the mean above
            # hides which finger carries the load, and a diagnostic plot needs
            # the ring and pinky too.
            fingertip_force_n = fingertip_force_norms(
                self.net_contact_forces, self.fingertip_body_indices
            )
        else:
            fingertip_contact_reward = torch.zeros_like(position_reward)
            fingertip_contact_fraction = torch.zeros_like(position_reward)
            mean_fingertip_contact_force_n = torch.zeros_like(position_reward)
            fingertip_force_n = position_reward.new_zeros(
                (self.num_envs, len(FINGERTIP_BODY_NAMES))
            )
        self.rew_buf.copy_(
            float(rewards_cfg.position_arm_weight) * position_reward
            + float(rewards_cfg.velocity_arm_weight) * velocity_reward
            + float(rewards_cfg.action_rate_arm_weight) * action_rate_reward
            + float(rewards_cfg.position_hand_weight) * hand_position_reward
            + float(rewards_cfg.velocity_hand_weight) * hand_velocity_reward
            + float(rewards_cfg.action_rate_hand_weight)
            * hand_action_rate_reward
            + float(rewards_cfg.object_position_weight)
            * object_position_reward
            + float(rewards_cfg.object_orientation_weight)
            * object_orientation_reward
            + self.proximity_weight * fingertip_object_distance_reward
            + self.contact_reward_per_finger * fingertip_contact_reward
        )
        return {
            "q_error": arm_q_error,
            "dq_error": arm_dq_error,
            "hand_q_error": hand_q_error,
            "hand_dq_error": hand_dq_error,
            "position_mse": position_mse,
            "velocity_mse": velocity_mse,
            "action_rate_mse": action_rate_mse,
            "hand_position_mse": hand_position_mse,
            "hand_velocity_mse": hand_velocity_mse,
            "hand_action_rate_mse": hand_action_rate_mse,
            "position_reward": position_reward,
            "velocity_reward": velocity_reward,
            "action_rate_reward": action_rate_reward,
            "hand_position_reward": hand_position_reward,
            "hand_velocity_reward": hand_velocity_reward,
            "hand_action_rate_reward": hand_action_rate_reward,
            "object_position_error_m": object_position_error_m,
            "object_com_height_m": object_com_height_m,
            "object_com_lift_m": object_com_lift_m,
            "object_orientation_error_rad": object_orientation_error_rad,
            "object_position_reward": object_position_reward,
            "object_orientation_reward": object_orientation_reward,
            "fingertip_object_distance_reward": (
                fingertip_object_distance_reward
            ),
            "fingertip_object_distance_m": fingertip_object_distance_m,
            "fingertip_object_distance_per_finger_m": (
                fingertip_object_distance_per_finger_m
            ),
            # The Gaussian is gated off before the pre-grasp window, so a zero
            # reward there means "not yet active", not "far from the cube".
            "proximity_active": proximity_active.to(
                dtype=fingertip_object_distance_m.dtype
            ),
            "fingertip_contact_reward": fingertip_contact_reward,
            "fingertip_contact_fraction": fingertip_contact_fraction,
            "mean_fingertip_contact_force_n": mean_fingertip_contact_force_n,
            "fingertip_force_n": fingertip_force_n,
        }

    def threshold_violation(self, q_error: torch.Tensor) -> torch.Tensor:
        return q_error.abs().amax(dim=1) > float(
            self.cfg.termination.arm_position_threshold_rad
        )

    def hand_threshold_violation(self, hand_q_error: torch.Tensor) -> torch.Tensor:
        return hand_q_error.abs().amax(dim=1) > float(
            self.cfg.termination.hand_position_threshold_rad
        )

    def object_threshold_violation(
        self, object_position_error_m: torch.Tensor
    ) -> torch.Tensor:
        return object_position_error_m > float(
            self.cfg.termination.object_position_threshold_m
        )

    def _compute_termination(
        self,
        q_error: torch.Tensor,
        hand_q_error: torch.Tensor,
        object_position_error_m: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Arm, hand and object each keep an independent grace counter. Returning
        # inside one threshold clears only that source's count, regardless of
        # what the other tracking errors are doing.
        if bool(self.cfg.termination.enabled):
            self.arm_violation = self.threshold_violation(q_error)
            self.hand_violation = self.hand_threshold_violation(hand_q_error)
            if bool(self.cfg.termination.object_position_enabled):
                self.object_violation = self.object_threshold_violation(
                    object_position_error_m
                )
            else:
                self.object_violation = torch.zeros_like(self.reset_buf)
            grace = int(self.cfg.termination.grace_steps)
            self.arm_violation_steps.copy_(
                torch.where(
                    self.arm_violation,
                    self.arm_violation_steps + 1,
                    torch.zeros_like(self.arm_violation_steps),
                )
            )
            self.hand_violation_steps.copy_(
                torch.where(
                    self.hand_violation,
                    self.hand_violation_steps + 1,
                    torch.zeros_like(self.hand_violation_steps),
                )
            )
            self.object_violation_steps.copy_(
                torch.where(
                    self.object_violation,
                    self.object_violation_steps + 1,
                    torch.zeros_like(self.object_violation_steps),
                )
            )
            early = (self.arm_violation_steps >= grace) | (
                self.hand_violation_steps >= grace
            ) | (
                self.object_violation_steps >= grace
            )
        else:
            early = torch.zeros_like(self.reset_buf)
            self.arm_violation = torch.zeros_like(self.reset_buf)
            self.hand_violation = torch.zeros_like(self.reset_buf)
            self.object_violation = torch.zeros_like(self.reset_buf)

        reference_end = self.reference_index >= self.reference.last_index
        # AnimRL classifies both the configured horizon and reaching phase 1 as
        # timeouts (rather than task failures).
        timeout = (
            self.episode_length_buf >= self.max_episode_length
        ) | reference_end
        done = early | timeout
        return done, early, timeout

    def _accumulate_episode_metrics(
        self, metrics: Dict[str, torch.Tensor]
    ) -> None:
        self.episode_peak_object_com_height_m.copy_(
            torch.maximum(
                self.episode_peak_object_com_height_m,
                metrics["object_com_height_m"],
            )
        )
        self.episode_sums["reward"] += self.rew_buf
        self.episode_sums["position_reward"] += metrics["position_reward"]
        self.episode_sums["velocity_reward"] += metrics["velocity_reward"]
        self.episode_sums["action_rate_reward"] += metrics["action_rate_reward"]
        self.episode_sums["hand_position_reward"] += metrics["hand_position_reward"]
        self.episode_sums["hand_velocity_reward"] += metrics["hand_velocity_reward"]
        self.episode_sums["hand_action_rate_reward"] += metrics[
            "hand_action_rate_reward"
        ]
        self.episode_sums["object_position_reward"] += metrics[
            "object_position_reward"
        ]
        self.episode_sums["object_orientation_reward"] += metrics[
            "object_orientation_reward"
        ]
        self.episode_sums["fingertip_object_distance_reward"] += metrics[
            "fingertip_object_distance_reward"
        ]
        self.episode_sums["fingertip_object_distance_m"] += metrics[
            "fingertip_object_distance_m"
        ]
        self.episode_sums["object_position_error_m"] += metrics[
            "object_position_error_m"
        ]
        self.episode_sums["object_orientation_error_rad"] += metrics[
            "object_orientation_error_rad"
        ]
        self.episode_sums["fingertip_contact_reward"] += metrics[
            "fingertip_contact_reward"
        ]
        self.episode_sums["fingertip_contact_fraction"] += metrics[
            "fingertip_contact_fraction"
        ]
        self.episode_sums["fingertip_contact_force_n"] += metrics[
            "mean_fingertip_contact_force_n"
        ]
        self.episode_sums["rms_hand_position_error"] += metrics[
            "hand_position_mse"
        ].sqrt()
        self.episode_sums["rms_hand_velocity_error"] += metrics[
            "hand_velocity_mse"
        ].sqrt()
        self.episode_sums["rms_hand_action_rate"] += metrics[
            "hand_action_rate_mse"
        ].sqrt()
        self.episode_sums["rms_position_error"] += metrics[
            "position_mse"
        ].sqrt()
        self.episode_sums["rms_velocity_error"] += metrics[
            "velocity_mse"
        ].sqrt()
        self.episode_sums["rms_action_rate"] += metrics[
            "action_rate_mse"
        ].sqrt()

    def _build_episode_summary(
        self,
        done: torch.Tensor,
        early: torch.Tensor,
        horizon_timeout: torch.Tensor,
        reference_end: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Aggregate completed episodes in the format consumed by AnimRL PPO."""
        lengths = self.episode_length_buf[done].float().clamp_min(1.0)
        summary = {
            "return": self.episode_sums["reward"][done].mean(),
            "length": lengths.mean(),
            "early_termination_fraction": early[done].float().mean(),
            # Of the episodes that failed, which block was over threshold. The
            # two can both be true on the same step, so they need not sum to 1.
            "arm_failure_fraction": (early & self.arm_violation)[done]
            .float()
            .mean(),
            "hand_failure_fraction": (early & self.hand_violation)[done]
            .float()
            .mean(),
            "object_failure_fraction": (early & self.object_violation)[done]
            .float()
            .mean(),
            "horizon_fraction": horizon_timeout[done].float().mean(),
            "reference_end_fraction": reference_end[done].float().mean(),
            "completed_episodes": done.sum().to(dtype=torch.float32),
            "mean_peak_object_com_height_m": (
                self.episode_peak_object_com_height_m[done].mean()
            ),
            "max_peak_object_com_height_m": (
                self.episode_peak_object_com_height_m[done].max()
            ),
            "mean_peak_object_com_lift_m": (
                (
                    self.episode_peak_object_com_height_m
                    - self.episode_initial_object_com_height_m
                )[done].mean()
            ),
            "max_peak_object_com_lift_m": (
                (
                    self.episode_peak_object_com_height_m
                    - self.episode_initial_object_com_height_m
                )[done].max()
            ),
        }
        for name, values in self.episode_sums.items():
            summary["mean_{}".format(name)] = (values[done] / lengths).mean()
        return summary

    def step(self, actions: torch.Tensor):
        if actions.shape != (self.num_envs, self.num_actions):
            raise ValueError(
                "Actions have shape {}, expected {}".format(
                    tuple(actions.shape), (self.num_envs, self.num_actions)
                )
            )
        # Pre-Physics Step: clamp and scale the actions, then write them to the robot DOF position targets.
        # Capture a_{t-1} before a_t overwrites it, so the action-rate
        # regularization computed post-physics sees the correct pair.
        self.previous_actions.copy_(self.actions)
        # Copy into the owned buffer rather than rebinding: PPO collects inside
        # torch.inference_mode(), and an inference tensor cannot be updated
        # in place later by reset_idx().
        self.actions.copy_(actions.to(device=self.device, dtype=torch.float32))
        complete_target_q = self.scale_actions(self.actions)
        next_indices = (self.reference_index + 1).clamp(
            max=self.reference.last_index
        )
        next_reference = self.reference.sample(next_indices)
        self._write_demo_order_to_asset(
            self.position_targets_asset, complete_target_q
        )
        # The ghost is commanded to the same reference sample the policy robot
        # is chasing, so the two are directly comparable in every frame.
        if self.reference_ghost_enabled:
            self._write_demo_order_to_asset(
                self.ghost_position_targets, next_reference.q
            )
        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(self.position_targets_all)
        )

        # Physics Step
        for _ in range(int(self.cfg.control.decimation)):
            self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        if self.contact_enabled:
            self.gym.refresh_net_contact_force_tensor(self.sim)
        self.render(sync_frame_time=True)

        # Post-Physics Step: update the reference index, compute rewards and termination, and reset completed environments.
        self.reference_index.add_(1).clamp_(max=self.reference.last_index)
        self.episode_length_buf += 1
        metrics = self._compute_reward_and_errors()
        self._accumulate_episode_metrics(metrics)
        done, early, timeout = self._compute_termination(
            metrics["q_error"],
            metrics["hand_q_error"],
            metrics["object_position_error_m"],
        )
        self.reset_buf.copy_(done)
        self.time_out_buf.copy_(timeout)
        reference_end = self.reference_index >= self.reference.last_index
        horizon_timeout = self.episode_length_buf >= self.max_episode_length

        # Clone all pre-reset diagnostics because training-style auto-reset below
        # immediately changes state/reference buffers for completed environments.
        extras = {
            "time_outs": timeout.clone(),
            "horizon_time_outs": horizon_timeout.clone(),
            "reference_end": reference_end.clone(),
            "early_termination": early.clone(),
            # Which block is over its threshold this step. An episode can end
            # on either, so the two are reported separately to tell an arm
            # failure from a hand failure.
            "arm_threshold_violation": self.arm_violation.clone(),
            "hand_threshold_violation": self.hand_violation.clone(),
            "object_threshold_violation": self.object_violation.clone(),
            "reference_index": self.reference_index.clone(),
            "max_abs_position_error": metrics["q_error"].abs().amax(dim=1),
            "max_abs_arm_position_error": metrics["q_error"].abs().amax(dim=1),
            "max_abs_hand_position_error": metrics["hand_q_error"].abs().amax(
                dim=1
            ),
            "worst_joint_index": metrics["q_error"].abs().argmax(dim=1),
            "rms_position_error": metrics["position_mse"].sqrt(),
            "rms_velocity_error": metrics["velocity_mse"].sqrt(),
            "rms_action_rate": metrics["action_rate_mse"].sqrt(),
            "rms_hand_position_error": metrics["hand_position_mse"].sqrt(),
            "rms_hand_velocity_error": metrics["hand_velocity_mse"].sqrt(),
            "rms_hand_action_rate": metrics["hand_action_rate_mse"].sqrt(),
            "position_reward": metrics["position_reward"],
            "velocity_reward": metrics["velocity_reward"],
            "action_rate_reward": metrics["action_rate_reward"],
            "hand_position_reward": metrics["hand_position_reward"],
            "hand_velocity_reward": metrics["hand_velocity_reward"],
            "hand_action_rate_reward": metrics["hand_action_rate_reward"],
            "object_position_reward": metrics["object_position_reward"],
            "object_orientation_reward": metrics["object_orientation_reward"],
            "fingertip_object_distance_reward": metrics[
                "fingertip_object_distance_reward"
            ],
            "fingertip_object_distance_m": metrics[
                "fingertip_object_distance_m"
            ],
            "object_position_error_m": metrics["object_position_error_m"],
            # Clone pre-reset COM diagnostics so the terminating sample is not
            # replaced by the next episode's RSI pose.
            "object_com_height_m": metrics["object_com_height_m"].clone(),
            "object_com_lift_m": metrics["object_com_lift_m"].clone(),
            "object_orientation_error_rad": metrics[
                "object_orientation_error_rad"
            ],
            "fingertip_contact_reward": metrics["fingertip_contact_reward"],
            "fingertip_contact_fraction": metrics[
                "fingertip_contact_fraction"
            ],
            "mean_fingertip_contact_force_n": metrics[
                "mean_fingertip_contact_force_n"
            ],
            "fingertip_force_n": metrics["fingertip_force_n"],
            "fingertip_object_distance_per_finger_m": metrics[
                "fingertip_object_distance_per_finger_m"
            ],
            "proximity_active": metrics["proximity_active"],
        }
        rewards = self.rew_buf.clone()
        dones = done.clone()

        reset_ids = done.nonzero(as_tuple=False).flatten()
        if reset_ids.numel() > 0:
            extras["episode"] = self._build_episode_summary(
                done, early, horizon_timeout, reference_end
            )
            self.reset_idx(reset_ids)
        self.compute_observations()
        self.extras = extras
        return self.obs_buf, self.critic_obs_buf, rewards, dones, extras

    def get_observations(self) -> torch.Tensor:
        return self.obs_buf

    def get_privileged_observations(self):
        return self.critic_obs_buf

    def pd_gain_summary(self) -> Dict[str, np.ndarray]:
        stiffness = np.asarray(self.pd_properties["stiffness"], dtype=np.float64)
        damping = np.asarray(self.pd_properties["damping"], dtype=np.float64)
        return {
            "arm_stiffness": stiffness[self.demo_to_asset[:len(ARM_JOINT_NAMES)]],
            "arm_damping": damping[self.demo_to_asset[:len(ARM_JOINT_NAMES)]],
            "hand_stiffness": stiffness[self.demo_to_asset[len(ARM_JOINT_NAMES):]],
            "hand_damping": damping[self.demo_to_asset[len(ARM_JOINT_NAMES):]],
        }

    def close(self) -> None:
        if getattr(self, "viewer", None) is not None:
            self.gym.destroy_viewer(self.viewer)
            self.viewer = None
        if getattr(self, "sim", None) is not None:
            self.gym.destroy_sim(self.sim)
            self.sim = None
