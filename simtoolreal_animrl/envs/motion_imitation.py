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
from simtoolreal_animrl.envs.demonstration import JointDemonstration60Hz


class MotionImitationEnv:
    """AnimRL-compatible environment API without an RL algorithm dependency."""

    JOINT_NAMES = JOINT_NAMES

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
        self.graphics_device_id = -1 if headless else self.sim_device_id

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

        # Purely a visual benchmark, and it doubles the simulated bodies, so
        # only an explicit opt-in builds it. evaluate.py ties it to --viewer;
        # training never sets it. It stays available headless so it can be
        # exercised by tests.
        self.reference_ghost_enabled = bool(
            getattr(self.cfg.viewer, "reference_ghost", False)
        )
        self.actors_per_env = 2 if self.reference_ghost_enabled else 1
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
        self._create_envs()
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
        params.physx.contact_collection = gymapi.ContactCollection(
            int(physx.contact_collection)
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

        ghost_pose = None
        if self.reference_ghost_enabled:
            offset = [float(v) for v in self.cfg.viewer.reference_ghost_offset]
            ghost_pose = gymapi.Transform()
            ghost_pose.p = gymapi.Vec3(
                pose.p.x + offset[0], pose.p.y + offset[1], pose.p.z + offset[2]
            )
            ghost_pose.r = pose.r
            body_count *= 2
            shape_count *= 2

        self.envs = []
        self.robot_handles = []
        self.ghost_handles = []
        actor_indices = []
        ghost_actor_indices = []
        for env_index in range(self.num_envs):
            env = self.gym.create_env(self.sim, lower, upper, per_row)
            if env is None:
                raise RuntimeError("Failed to create environment {}".format(env_index))
            # The aggregate's last flag is what actually governs self-collision.
            # create_actor's own filter argument below never reaches the shapes:
            # they keep the filter bits the asset gave them.
            self.gym.begin_aggregate(
                env, body_count, shape_count, self.self_collision_enabled
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
            self.envs.append(env)
            self.robot_handles.append(actor)
            actor_indices.append(
                self.gym.get_actor_index(env, actor, gymapi.DOMAIN_SIM)
            )

        self.actor_indices = torch.as_tensor(
            actor_indices, dtype=torch.int32, device=self.device
        )
        self.ghost_actor_indices = torch.as_tensor(
            ghost_actor_indices, dtype=torch.int32, device=self.device
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
        self.gym.refresh_dof_state_tensor(self.sim)

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
        self.arm_violation = torch.zeros_like(self.reset_buf)
        self.hand_violation = torch.zeros_like(self.reset_buf)
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
                "rms_position_error",
                "rms_velocity_error",
                "rms_action_rate",
                "rms_hand_position_error",
                "rms_hand_velocity_error",
                "rms_hand_action_rate",
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
        # Match AnimRL Cartwheel RSI: sample uniformly over the complete motion,
        # even when fewer than max_episode_length reference steps remain. The
        # episode then terminates naturally when it reaches the final sample.
        max_start = self.reference.last_index - 1
        if reference_indices is None:
            if self.cfg.env.reference_init_distribution != "uniform":
                raise ValueError("Only uniform RSI is supported")
            reference_indices = torch.randint(
                low=0,
                high=max_start + 1,
                size=(env_ids.numel(),),
                device=self.device,
            )
        else:
            reference_indices = reference_indices.to(
                device=self.device, dtype=torch.long
            )
            if reference_indices.ndim == 0:
                reference_indices = reference_indices.repeat(env_ids.numel())
            if reference_indices.shape != (env_ids.numel(),):
                raise ValueError("reference_indices has the wrong shape")
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
        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(self.position_targets_all)
        )

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
        self.rew_buf.copy_(
            float(rewards_cfg.position_arm_weight) * position_reward
            + float(rewards_cfg.velocity_arm_weight) * velocity_reward
            + float(rewards_cfg.action_rate_arm_weight) * action_rate_reward
            + float(rewards_cfg.position_hand_weight) * hand_position_reward
            + float(rewards_cfg.velocity_hand_weight) * hand_velocity_reward
            + float(rewards_cfg.action_rate_hand_weight)
            * hand_action_rate_reward
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
        }

    def threshold_violation(self, q_error: torch.Tensor) -> torch.Tensor:
        return q_error.abs().amax(dim=1) > float(
            self.cfg.termination.arm_position_threshold_rad
        )

    def hand_threshold_violation(self, hand_q_error: torch.Tensor) -> torch.Tensor:
        return hand_q_error.abs().amax(dim=1) > float(
            self.cfg.termination.hand_position_threshold_rad
        )

    def _compute_termination(
        self, q_error: torch.Tensor, hand_q_error: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Either block drifting too far ends the episode. Each keeps its own
        # grace counter, so a block that returns inside its threshold clears its
        # count regardless of what the other block is doing.
        if bool(self.cfg.termination.enabled):
            self.arm_violation = self.threshold_violation(q_error)
            self.hand_violation = self.hand_threshold_violation(hand_q_error)
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
            early = (self.arm_violation_steps >= grace) | (
                self.hand_violation_steps >= grace
            )
        else:
            early = torch.zeros_like(self.reset_buf)
            self.arm_violation = torch.zeros_like(self.reset_buf)
            self.hand_violation = torch.zeros_like(self.reset_buf)

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
        self.episode_sums["reward"] += self.rew_buf
        self.episode_sums["position_reward"] += metrics["position_reward"]
        self.episode_sums["velocity_reward"] += metrics["velocity_reward"]
        self.episode_sums["action_rate_reward"] += metrics["action_rate_reward"]
        self.episode_sums["hand_position_reward"] += metrics["hand_position_reward"]
        self.episode_sums["hand_velocity_reward"] += metrics["hand_velocity_reward"]
        self.episode_sums["hand_action_rate_reward"] += metrics[
            "hand_action_rate_reward"
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
            "horizon_fraction": horizon_timeout[done].float().mean(),
            "reference_end_fraction": reference_end[done].float().mean(),
            "completed_episodes": done.sum().to(dtype=torch.float32),
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
        self.render(sync_frame_time=True)

        # Post-Physics Step: update the reference index, compute rewards and termination, and reset completed environments.
        self.reference_index.add_(1).clamp_(max=self.reference.last_index)
        self.episode_length_buf += 1
        metrics = self._compute_reward_and_errors()
        self._accumulate_episode_metrics(metrics)
        done, early, timeout = self._compute_termination(
            metrics["q_error"], metrics["hand_q_error"]
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
