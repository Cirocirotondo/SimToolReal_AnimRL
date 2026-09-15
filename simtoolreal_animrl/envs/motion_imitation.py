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
from simtoolreal_animrl.envs.adaptive_sigma import AdaptiveSigma
from simtoolreal_animrl.envs.operational_space import (
    damped_least_squares_step,
    saturate_direction_preserving,
    transfer_jacobian,
)
from simtoolreal_animrl.envs.rsi_noise import perturb_reference_pose
from simtoolreal_animrl.envs.sensing import (
    ActionDelay,
    add_observation_noise,
    sample_position_bias,
)
from simtoolreal_animrl.envs.disturbance import sample_impulses
from simtoolreal_animrl.envs.domain_randomization import DomainRandomization
from simtoolreal_animrl.envs.contact import (
    fingertip_contact_diagnostics,
    fingertip_force_norms,
    fingertip_force_observation,
    fingertip_force_observation_dim,
    select_fingertip_forces,
)
from simtoolreal_animrl.envs.demonstration import JointDemonstration60Hz
from simtoolreal_animrl.envs.object_assist import (
    assist_scale_at,
    object_assist_wrench,
    object_reward_gate,
    resolve_object_assist_settings,
)
from simtoolreal_animrl.envs.proximity import fingertip_cuboid_proximity
from simtoolreal_animrl.envs.rsi import resolve_rsi_settings, sample_rsi_indices
from simtoolreal_animrl.envs.cuboid_symmetry import (
    apply_cuboid_symmetry,
    canonicalize_cuboid_orientation,
    cuboid_rotation_symmetries,
    symmetry_invariant_orientation_error,
)
from simtoolreal_animrl.envs.keypoints import (
    hand_keypoints,
    keypoint_gaussian,
    keypoint_tracking_error,
    keypoints_in_object_frame,
    split_palm_and_fingertips,
)
from simtoolreal_animrl.envs.rotations import (
    normalize_canonical_quaternion as _normalize_canonical_quaternion,
    quat_conjugate as _quat_conjugate,
    quat_multiply as _quat_multiply,
    quat_rotate as _quat_rotate,
    quat_rotate_inverse as _quat_rotate_inverse,
    quat_to_rotation_6d,
)
from simtoolreal_animrl.envs.transform_bank import (
    TransformBank,
    nearest_transform_indices,
)


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
        # Built here rather than in _create_envs because the critic's width is
        # computed before the environments exist, and both need the same draw.
        self.domain_randomization = DomainRandomization(
            getattr(self.cfg, "domain_randomization", object()),
            self.num_envs,
            seed=int(getattr(self.cfg, "seed", 0) or 0),
        )
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
        if self.cfg.control.action_parameterization != "operational_space_arm":
            raise ValueError(
                "Only the operational-space arm contract is supported. The arm's "
                "six actions are an end-effector twist [dx, dy, dz, wx, wy, wz]; "
                "the joint-space 'animrl_residual' arm path was removed, so a "
                "config or checkpoint from before that change cannot be run"
            )
        self.hand_action_scale = float(self.cfg.control.scale_hand_joint_target)
        self.action_target_clip = float(self.cfg.control.clip_joint_target)
        self.arm_translation_speed = float(
            self.cfg.control.arm_translation_speed_m_per_s
        )
        self.arm_rotation_speed = float(self.cfg.control.arm_rotation_speed_rad_per_s)
        self.ik_damping = float(self.cfg.control.ik_damping)
        self.ik_max_joint_delta = float(self.cfg.control.ik_max_joint_delta_rad)
        if (
            self.hand_action_scale <= 0.0
            or self.action_target_clip <= 0.0
            or self.arm_translation_speed <= 0.0
            or self.arm_rotation_speed <= 0.0
            or self.ik_damping <= 0.0
            or self.ik_max_joint_delta <= 0.0
        ):
            raise ValueError("Action scales, clips and IK parameters must be positive")
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
        # Gated separately from contact.enabled: acquiring the force tensor for
        # the observation must not switch a reward term on behind the caller.
        self.contact_reward_enabled = bool(
            getattr(self.cfg.contact, "reward_enabled", False)
        )
        if self.contact_reward_enabled and not self.contact_enabled:
            raise ValueError(
                "contact.reward_enabled needs contact.enabled, which is what "
                "acquires the force tensor the shaping reads"
            )
        self.contact_observation_enabled = bool(
            getattr(self.cfg.contact, "observe_fingertip_forces", False)
        )
        self.contact_shaping_weight = (
            self.contact_reward_per_finger if self.contact_reward_enabled else 0.0
        )
        self.contact_observation_force_scale_n = float(
            getattr(self.cfg.contact, "observation_force_scale_n", 10.0)
        )
        self.contact_observation_clip = float(
            getattr(self.cfg.contact, "observation_clip", 5.0)
        )
        if self.contact_observation_enabled:
            if not self.contact_enabled:
                raise ValueError(
                    "contact.observe_fingertip_forces needs contact.enabled: "
                    "that is what turns on PhysX contact reporting and "
                    "acquires the force tensor the observation reads"
                )
            if (
                not math.isfinite(self.contact_observation_force_scale_n)
                or self.contact_observation_force_scale_n <= 0.0
            ):
                raise ValueError(
                    "contact.observation_force_scale_n must be finite and "
                    "positive"
                )
            if (
                not math.isfinite(self.contact_observation_clip)
                or self.contact_observation_clip <= 0.0
            ):
                raise ValueError(
                    "contact.observation_clip must be finite and positive"
                )
        # The fingertip-force block widens the observation vector, so
        # env.num_observations stays the base width every existing run used and
        # the policy is built from the total below.
        rewards_cfg_init = self.cfg.rewards
        self.adaptive_sigmas = {}
        if bool(getattr(rewards_cfg_init, "adaptive_sigma_enabled", False)):
            target = float(rewards_cfg_init.adaptive_sigma_target_reward)
            decay = float(rewards_cfg_init.adaptive_sigma_decay)
            for name, initial, floor in (
                (
                    "position_hand",
                    rewards_cfg_init.position_hand_std_rad,
                    rewards_cfg_init.adaptive_sigma_position_hand_floor,
                ),
                (
                    "ee_action_rate",
                    rewards_cfg_init.ee_action_rate_std,
                    rewards_cfg_init.adaptive_sigma_ee_action_rate_floor,
                ),
                (
                    "hand_action_rate",
                    rewards_cfg_init.hand_action_rate_std,
                    rewards_cfg_init.adaptive_sigma_hand_action_rate_floor,
                ),
            ):
                self.adaptive_sigmas[name] = AdaptiveSigma(
                    initial=initial,
                    floor=floor,
                    target_reward=target,
                    decay=decay,
                    slack=float(
                        getattr(rewards_cfg_init, "adaptive_sigma_slack", 1.5)
                    ),
                )
        self.contact_observation_dim = fingertip_force_observation_dim(
            self.cfg.contact
        )
        self.num_obs += self.contact_observation_dim
        # The critic may read the fingertip forces even when the actor cannot.
        # Its width is the actor's plus those forces, and it is left at None
        # when the feature is off so PPO keeps its symmetric path.
        self.critic_force_observation_dim = (
            3 * len(self.cfg.contact.fingertip_names)
            if bool(
                getattr(
                    self.cfg.contact, "critic_observes_fingertip_forces", False
                )
            )
            else 0
        )
        # Randomisation multipliers the critic may also read. Scratch runs only:
        # this widens the critic input, so no existing value network fits.
        self.critic_parameter_dim = self.domain_randomization.privileged_dim
        # The tensor itself is built in _allocate_buffers, once self.device
        # exists; only the width is needed this early.
        self.critic_parameter_table = None
        if self.critic_force_observation_dim or self.critic_parameter_dim:
            if self.critic_force_observation_dim and not bool(
                getattr(self.cfg.contact, "enabled", False)
            ):
                raise ValueError(
                    "critic_observes_fingertip_forces requires contact.enabled: "
                    "the net contact force tensor is only wrapped when contact "
                    "reporting is on"
                )
            self.num_privileged_obs = (
                self.num_obs
                + self.critic_force_observation_dim
                + self.critic_parameter_dim
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
        self._load_transform_bank()
        (
            self.rsi_distribution,
            self.rsi_max_start_index,
            self.rsi_pregrasp_start_index,
            self.rsi_early_probability,
        ) = resolve_rsi_settings(self.cfg.env, self.reference.last_index)
        # A config.json written before the assist existed simply keeps the
        # disabled class default, so replaying an older run is unaffected.
        self.object_assist_settings = resolve_object_assist_settings(
            self.cfg.object_assist, self.reference.last_index
        )
        self.object_assist_enabled = self.object_assist_settings.enabled
        self.object_assist_gates_object_reward = bool(
            getattr(self.cfg.object_assist, "gate_object_reward", True)
        )
        # Iteration 0 of the schedule until PPO says otherwise. Evaluation
        # processes never advance it and pin the scale to zero instead.
        self.object_assist_scale = assist_scale_at(
            self.object_assist_settings, 0
        )

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

    def _load_transform_bank(self) -> None:
        """Load the retargeted reference clips this run draws episodes from.

        The bank is an offline artefact rather than a startup step: solving the
        inverse kinematics for hundreds of transforms takes minutes, and a
        transform has to be proven feasible over the clip's *whole* length
        before any episode is allowed to start inside it. Building it separately
        also keeps ``pytorch_kinematics`` out of the training process.
        """
        randomization = self.cfg.object_randomization
        bank_path = ROOT_DIR / str(randomization.bank_path)
        if not bank_path.is_file():
            raise FileNotFoundError(
                "No transform bank at {}. Build one first:\n"
                "    PYTHONPATH=. python scripts/build_transform_bank.py "
                "--output {}".format(bank_path, bank_path)
            )
        bank = TransformBank.load(bank_path)
        if bank.sample_count != self.reference.sample_count:
            raise ValueError(
                "The transform bank has {} frames but the demonstration has "
                "{}. The bank was built from a different clip.".format(
                    bank.sample_count, self.reference.sample_count
                )
            )
        lever_arm = float(self.cfg.rewards.palm_lever_arm_m)
        if bank.reference_keypoints.shape[-2:] != (9, 3):
            raise ValueError("The transform bank carries malformed keypoints")
        self.transform_bank = bank.to(device=self.device, dtype=torch.float32)
        self.palm_lever_arm_m = lever_arm
        print(
            "Transform bank: {} transforms, {:.1f}% of sampled transforms were "
            "feasible".format(
                self.transform_bank.transform_count,
                100.0 * self.transform_bank.acceptance,
            )
        )

        # The cuboid's own symmetries, derived from its extents rather than
        # typed: eight for a bar with two equal sides. See envs/cuboid_symmetry.
        self.cuboid_symmetries = cuboid_rotation_symmetries(
            [0.5 * float(value) for value in self.cfg.object.size_m]
        ).to(device=self.device, dtype=torch.float32)

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
            self.gym,
            asset,
            self.demo_to_asset,
            arm_stiffness_scale=self.cfg.control.arm_stiffness_scale,
            arm_damping_scale=self.cfg.control.arm_damping_scale,
            hand_stiffness_scale=self.cfg.control.hand_stiffness_scale,
            hand_damping_scale=self.cfg.control.hand_damping_scale,
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
        self.default_hand_positions = self.default_positions[len(ARM_JOINT_NAMES):]
        if torch.any(self.default_positions < self.joint_lower_limits) or torch.any(
            self.default_positions > self.joint_upper_limits
        ):
            raise ValueError("Default pose exceeds the URDF position limits")
        self.demo_to_asset_tensor = torch.as_tensor(
            self.demo_to_asset, dtype=torch.long, device=self.device
        )
        # The Jacobian's columns are in ASSET order, which demo_to_asset is a
        # genuine permutation of -- so the arm's six columns have to be selected
        # by index. Slicing the first six, as one would with a joint vector in
        # demonstration order, silently picks the wrong joints.
        self.arm_asset_columns = self.demo_to_asset_tensor[: len(ARM_JOINT_NAMES)]

        q = self.reference.q
        if torch.any(q < self.joint_lower_limits - 1e-6) or torch.any(
            q > self.joint_upper_limits + 1e-6
        ):
            raise ValueError("The demonstration exceeds the robot position limits")
        return asset

    def _configure_robot_contact_properties(self, asset) -> None:
        """Configure materials and external robot collision-filter bits.

        The UR5e arm up to ``wrist_2_link`` is filtered against the cube, while
        the whole hand assembly can contact it. With fixed joints collapsed the
        static DG5F mount/base/palm shapes belong to ``wrist_3_link``, so that
        body is grouped with the articulated ``rl_dg_*`` fingers: filtering it
        out would leave the palm unable to support a grasp. Its own wrist mesh
        rides along, which is physically right and practically irrelevant --
        that mesh sits ~0.099 m behind the flange while the palm is ~0.074 m in
        front of it, so the cube reaches one and not the other.
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
            is_hand = body_name.startswith("rl_dg_") or body_name == "wrist_3_link"
            (hand_body_names if is_hand else arm_body_names).append(body_name)
            for shape_index in range(
                shape_range.start, shape_range.start + shape_range.count
            ):
                properties = shape_properties[shape_index]
                properties.friction = float(self.cfg.asset.friction)
                properties.restitution = float(self.cfg.asset.restitution)
                # Every robot shape filters the table. Arm shapes share the
                # cube bit; hand-assembly shapes (wrist 3 and the articulated
                # DG5F fingers) do not.
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

    def _set_cube_body_properties(self, env, actor: int, env_index: int = 0) -> None:
        properties = self.gym.get_actor_rigid_body_properties(env, actor)
        # Mass and inertia scale together: a heavier cube of the same size and
        # material has proportionally larger inertia, and scaling mass alone
        # would produce an object with no physical counterpart.
        mass_scale = self.domain_randomization.multiplier("object_mass", env_index)
        properties[0].mass = float(self.cfg.object.mass_kg) * mass_scale
        inertia = [float(v) * mass_scale for v in self.cfg.object.inertia_kg_m2]
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
        if self.domain_randomization.enabled:
            print("Domain randomisation active:")
            for name, (low, high) in self.domain_randomization.summary().items():
                print("  {:<20s} x[{:.3f}, {:.3f}]".format(name, low, high))
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
            self.gym.set_actor_dof_properties(
                env, actor, self._randomized_pd_properties(env_index)
            )
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
            self._set_cube_body_properties(env, cube, env_index)
            self._randomize_shape_friction(env, cube, "object_friction", env_index)
            self._randomize_shape_friction(env, actor, "fingertip_friction", env_index)
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
        """Resolve the surviving rigid bodies used by the 108D observation."""
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
        # The cuboid is a single-body actor, so its only body is where the
        # assist wrench is applied.
        self.cube_body_index = int(
            self.gym.get_actor_rigid_body_index(
                env, self.cube_handles[0], 0, gymapi.DOMAIN_ENV
            )
        )
        if self.cube_body_index < 0:
            raise ValueError("The cuboid rigid body was not found")
        self.cube_body_index_tensor = torch.tensor(
            [self.cube_body_index], dtype=torch.long, device=self.device
        )
        # Impulses land on the robot's own links, never the cube's body or the
        # table's, so a "robot" disturbance cannot secretly push the object.
        robot_bodies = self.gym.get_actor_rigid_body_count(env, actor)
        self.robot_body_indices = torch.arange(
            robot_bodies, dtype=torch.long, device=self.device
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
        # getattr keeps configurations written before the field existed usable;
        # 90 degrees is Isaac Gym's own default and what the recorded training
        # videos have always framed with.
        camera_properties.horizontal_fov = float(
            getattr(self.cfg.viewer, "training_camera_fov_deg", 90.0)
        )
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
        self.world_up = torch.tensor(
            [0.0, 0.0, 1.0], dtype=torch.float32, device=self.device
        ).expand(self.num_envs, -1)
        self.robot_base_position = torch.tensor(
            self.cfg.init_state.pos, dtype=torch.float32, device=self.device
        )
        self.object_half_extents = torch.tensor(
            self.cfg.object.size_m, dtype=torch.float32, device=self.device
        ) / 2.0
        # Isaac Gym takes one force and one torque per rigid body in the whole
        # simulation, so the buffers cover every body and stay zero outside the
        # cube column. They are only allocated when the assist can ever be on.
        self.rigid_body_forces = None
        self.rigid_body_torques = None
        self.gravity_vector = torch.tensor(
            self.cfg.sim.gravity, dtype=torch.float32, device=self.device
        )
        if self.object_assist_enabled or self.domain_randomization.impulses_enabled:
            self.rigid_body_forces = torch.zeros(
                (self.num_envs * rigid_bodies_per_env, 3),
                dtype=torch.float32,
                device=self.device,
            )
            self.rigid_body_torques = torch.zeros_like(self.rigid_body_forces)
            self.cube_body_forces = self.rigid_body_forces.view(
                self.num_envs, rigid_bodies_per_env, 3
            )[:, self.cube_body_index]
            self.cube_body_torques = self.rigid_body_torques.view(
                self.num_envs, rigid_bodies_per_env, 3
            )[:, self.cube_body_index]
        self.robot_jacobian = gymtorch.wrap_tensor(
            self.gym.acquire_jacobian_tensor(self.sim, "robot")
        )
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        if self.contact_enabled:
            self.gym.refresh_net_contact_force_tensor(self.sim)
        self._resolve_jacobian_index()

    def _resolve_jacobian_index(self) -> None:
        """Find the Jacobian's row for the wrist, and check what it is built on.

        Every assumption the task-space controller rests on is cheap to test once
        and expensive to debug later, so they are all tested here: that the base
        is fixed, that the base is unrotated, and that the wrist index resolved
        against the rigid-body tensor also addresses the Jacobian.
        """
        shape = tuple(self.robot_jacobian.shape)
        if len(shape) != 4 or shape[0] != self.num_envs or shape[2] != 6:
            raise ValueError(
                "Unexpected robot Jacobian shape {}".format(shape)
            )
        # A floating base would add six columns here, and would also break the
        # claim below that a world-frame Jacobian is a base-frame one.
        if shape[3] != len(JOINT_NAMES):
            raise ValueError(
                "The Jacobian has {} columns for {} DOFs, so the robot asset is "
                "not fixed-base; operational-space control needs "
                "asset.fix_base_link".format(shape[3], len(JOINT_NAMES))
            )
        # self.wrist_body_index was resolved with DOMAIN_ENV, but the Jacobian is
        # indexed per actor. They agree only while the robot leads its envs.
        actor_wrist = self.gym.find_actor_rigid_body_index(
            self.envs[0],
            self.robot_handles[0],
            PALM_PARENT_BODY_NAME,
            gymapi.DOMAIN_ACTOR,
        )
        if int(actor_wrist) != int(self.wrist_body_index):
            raise ValueError(
                "The robot is not the leading actor in its environment, so the "
                "rigid-body and Jacobian body indices disagree"
            )
        # Isaac Gym drops the immovable base link from a fixed-base actor's
        # Jacobian, so its link axis is one shorter than the rigid-body tensor's.
        # Derive which convention is in force rather than assuming either.
        robot_bodies = self.gym.get_actor_rigid_body_count(
            self.envs[0], self.robot_handles[0]
        )
        if shape[1] == robot_bodies:
            self.arm_ee_jacobian_index = int(self.wrist_body_index)
        elif shape[1] == robot_bodies - 1:
            self.arm_ee_jacobian_index = int(self.wrist_body_index) - 1
        else:
            raise ValueError(
                "The Jacobian has {} links against the actor's {} rigid "
                "bodies; neither convention applies".format(shape[1], robot_bodies)
            )
        if not 0 <= self.arm_ee_jacobian_index < shape[1]:
            raise ValueError(
                "Resolved Jacobian link index {} is out of range".format(
                    self.arm_ee_jacobian_index
                )
            )
        # The point the reported Jacobian is taken at. Read from the asset rather
        # than assumed zero: nothing randomizes the robot's inertial properties,
        # so one read covers every environment.
        wrist_com = self.gym.get_actor_rigid_body_properties(
            self.envs[0], self.robot_handles[0]
        )[int(self.wrist_body_index)].com
        self.wrist_com_in_link = torch.tensor(
            [wrist_com.x, wrist_com.y, wrist_com.z],
            dtype=torch.float32,
            device=self.device,
        ).expand(self.num_envs, -1)
        # The controller commands a world-frame twist and inverts a world-frame
        # Jacobian, and calls the result a base-frame command. That identity only
        # holds while the base is unrotated.
        root_orientation = self.robot_root_state[:, 3:7]
        identity = torch.tensor(
            [0.0, 0.0, 0.0, 1.0], dtype=torch.float32, device=self.device
        ).expand_as(root_orientation)
        if not torch.allclose(root_orientation.abs(), identity.abs(), atol=1e-5):
            raise ValueError(
                "Operational-space control assumes the robot base is unrotated "
                "in world, so that the world-frame Jacobian is also the "
                "base-frame Jacobian"
            )

    def _allocate_buffers(self) -> None:
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_obs), dtype=torch.float32, device=self.device
        )
        self.critic_obs_buf = (
            torch.zeros(
                self.num_envs,
                self.num_privileged_obs,
                dtype=torch.float32,
                device=self.device,
            )
            if (self.critic_force_observation_dim or self.critic_parameter_dim)
            else None
        )
        self._critic_force_features = None
        self.action_delay = ActionDelay(
            self.num_envs,
            self.num_actions,
            self.domain_randomization.action_delay_steps,
            self.device,
        )
        self.observation_position_bias = sample_position_bias(
            self.num_envs,
            self.num_actions,
            self.domain_randomization.obs_q_bias_rad
            if self.domain_randomization.enabled
            else 0.0,
            self.device,
        )
        if self.critic_parameter_dim:
            self.critic_parameter_table = (
                self.domain_randomization.privileged_table(device=self.device)
            )
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
        # Which transform of the scene this episode is playing, and which of the
        # cuboid's symmetry relabellings its frame was resolved to. Both are
        # drawn once at reset and held: the symmetry choice in particular must
        # not be recomputed per step, because the bar turns while it is lifted
        # and a re-choice can flip the reference frame mid-grasp.
        self.transform_index = torch.zeros_like(self.episode_length_buf)
        self.episode_translation = torch.zeros(
            self.num_envs, 3, dtype=torch.float32, device=self.device
        )
        self.episode_yaw_rad = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.symmetry_index = torch.zeros_like(self.episode_length_buf)
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
        # Magnitudes of the wrench actually applied in the last step, kept for
        # logging: they say how much of the task the crutch is still doing.
        self.object_assist_force_n = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.object_assist_torque_nm = torch.zeros_like(
            self.object_assist_force_n
        )
        self.actions = torch.zeros(
            (self.num_envs, self.num_actions), dtype=torch.float32, device=self.device
        )
        # a_{t-1} for the action-rate regularization term.
        self.previous_actions = torch.zeros_like(self.actions)
        # What the in-loop IK was asked for, what it delivered, and what it gave
        # up in between. The residual is the feasibility trace; the clipped flag
        # is how often the per-joint clamp bound.
        twist_shape = (self.num_envs, 6)
        self.requested_twist = torch.zeros(
            twist_shape, dtype=torch.float32, device=self.device
        )
        self.achieved_twist = torch.zeros_like(self.requested_twist)
        self.applied_arm_q_delta = torch.zeros_like(self.requested_twist)
        self.ik_residual_norm = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.arm_joint_delta_norm = torch.zeros_like(self.ik_residual_norm)
        self.arm_joint_delta_clipped = torch.zeros_like(self.ik_residual_norm)
        # The arm cannot seed a_{t-1} to "the action that holds this pose" the way
        # the hand can, so the first step after a reset is exempted from the EE
        # action-rate penalty. See reset_idx.
        self.suppress_ee_action_rate = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        # The targets command_targets last applied. The arm's mapping is an
        # integrator, so a diagnostic consumer has to read this rather than call
        # the mapping again.
        self._last_command_targets = torch.zeros(
            (self.num_envs, self.num_actions),
            dtype=torch.float32,
            device=self.device,
        )
        self.episode_sums = {
            name: torch.zeros(
                self.num_envs, dtype=torch.float32, device=self.device
            )
            for name in (
                "reward",
                "palm_keypoint_reward",
                "fingertip_keypoint_reward",
                "palm_keypoint_error_m",
                "fingertip_keypoint_error_m",
                "palm_tilt_reward",
                "palm_tilt_error_rad",
                "ee_action_rate_reward",
                "arm_joint_rate_reward",
                "ik_residual_reward",
                "ik_residual_norm",
                "arm_joint_delta_norm",
                "arm_joint_delta_clipped",
                "hand_position_reward",
                "hand_velocity_reward",
                "hand_action_rate_reward",
                "object_position_reward",
                "object_orientation_reward",
                "fingertip_object_distance_reward",
                "fingertip_object_distance_m",
                "rms_position_error",
                "rms_velocity_error",
                "rms_ee_action_rate",
                "rms_arm_joint_rate",
                "rms_hand_position_error",
                "rms_hand_velocity_error",
                "rms_hand_action_rate",
                "object_position_error_m",
                "object_orientation_error_rad",
                "fingertip_contact_reward",
                "fingertip_contact_fraction",
                "fingertip_contact_force_n",
                "object_assist_force_n",
                "object_assist_torque_nm",
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

    def _palm_pose_world(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """The palm frame in world coordinates, from the wrist plus a fixed offset.

        Isaac Gym collapses the fixed joints between ``wrist_3_link`` and the
        palm, so the palm is not a body it reports; the offset is re-applied
        here exactly as the MuJoCo runner does.
        """
        wrist = self.rigid_body_state[:, self.wrist_body_index]
        wrist_orientation = _normalize_canonical_quaternion(wrist[:, 3:7])
        palm_position_world = wrist[:, 0:3] + _quat_rotate(
            wrist_orientation, self.palm_position_in_wrist
        )
        palm_orientation_world = _normalize_canonical_quaternion(
            _quat_multiply(wrist_orientation, self.palm_orientation_in_wrist)
        )
        return palm_position_world, palm_orientation_world

    def _palm_jacobian_arm(self) -> torch.Tensor:
        """``(num_envs, 6, 6)`` world-frame palm Jacobian over the six arm DOFs.

        Isaac Gym reports ``wrist_3_link``, because the fixed wrist -> mount ->
        palm chain is collapsed when the asset loads. The palm is moved onto it
        by the same reconstruction :meth:`_palm_pose_world` uses for the pose,
        which is what makes the policy's ``dx`` move the frame the keypoint
        reward is measured at rather than one 73.8 mm behind it.
        """
        jacobian = self.robot_jacobian[:, self.arm_ee_jacobian_index]
        wrist = self.rigid_body_state[:, self.wrist_body_index]
        wrist_orientation = _normalize_canonical_quaternion(wrist[:, 3:7])
        # From the CENTRE OF MASS, not the link origin. Isaac Gym reports a
        # body's Jacobian at its centre of mass, and collapsing the hand
        # assembly into wrist_3_link puts that 4.8 cm up the link's own z. Using
        # the link origin leaves the angular rows exactly right and the linear
        # rows wrong by that lever arm, so the arm still moves smoothly while
        # every rotation command drags the palm about 5 cm per radian.
        offset_world = _quat_rotate(
            wrist_orientation, self.palm_position_in_wrist - self.wrist_com_in_link
        )
        palm_jacobian = transfer_jacobian(jacobian, offset_world)
        return palm_jacobian[:, :, self.arm_asset_columns]

    def _operational_space_arm_targets(
        self, arm_actions: torch.Tensor
    ) -> torch.Tensor:
        """Six arm joint targets from a base-frame end-effector twist command.

        The delta accumulates onto the previous *target* and is never re-anchored
        to the measured pose. That is deliberate: the command is free to run ahead
        of the robot, so drive error builds and the arm can press against the
        cuboid's weight instead of going slack the moment it is loaded. The drift
        this allows is the feasibility signal -- a twist the arm cannot follow
        opens a gap between the commanded palm and the real one, and the keypoint
        tracking reward charges for the gap without anyone having to price it.
        """
        # Saturated by magnitude, not clipped per component, and the two halves
        # separately because metres and radians do not share a norm. Clipping
        # each component at +/-1 instead made the policy's whole range past the
        # rail a dead zone: the action stopped reaching the environment, so the
        # advantage went flat there and nothing pulled the mean back -- measured
        # drifting to |a| = 14 with 46% of arm components pinned. Prior runs here
        # show why that was self-inflicted: the joint-space scheme set its own
        # clip so wide it never bound, and the policy routinely used |a| ~ 20.
        max_translation = self.arm_translation_speed * self.dt
        max_rotation = self.arm_rotation_speed * self.dt
        desired_twist = torch.cat(
            (
                saturate_direction_preserving(
                    arm_actions[:, 0:3] * max_translation, max_translation
                ),
                # A first-order axis-angle increment, packed straight into the
                # twist rather than composed as a quaternion. At 1 rad/s that is
                # 16.7 mrad per step, where the two agree to under 1e-6 rad.
                saturate_direction_preserving(
                    arm_actions[:, 3:6] * max_rotation, max_rotation
                ),
            ),
            dim=1,
        )

        jacobian = self._palm_jacobian_arm()
        q_delta = damped_least_squares_step(
            jacobian, desired_twist, self.ik_damping
        )

        unclipped_q_delta = q_delta
        # Per joint, not a norm rescale. Clamping this way distorts the commanded
        # direction as well as its magnitude, which is accepted: the distortion
        # lands in the residual below and in a real palm tracking gap, and that
        # is a truer picture of what the arm did than a direction-preserving
        # rescale would give.
        q_delta = q_delta.clamp(-self.ik_max_joint_delta, self.ik_max_joint_delta)
        previous = self.previous_arm_targets
        targets = (previous + q_delta).clamp(
            self.arm_lower_limits, self.arm_upper_limits
        )
        applied_q_delta = targets - previous

        achieved_twist = torch.bmm(
            jacobian, applied_q_delta.unsqueeze(-1)
        ).squeeze(-1)
        self.requested_twist.copy_(desired_twist)
        self.achieved_twist.copy_(achieved_twist)
        self.applied_arm_q_delta.copy_(applied_q_delta)
        self.ik_residual_norm.copy_((desired_twist - achieved_twist).norm(dim=-1))
        self.arm_joint_delta_norm.copy_(applied_q_delta.norm(dim=-1))
        self.arm_joint_delta_clipped.copy_(
            (unclipped_q_delta.abs() > self.ik_max_joint_delta)
            .any(dim=-1)
            .to(dtype=torch.float32)
        )
        return targets

    def canonical_cube_orientation(self) -> torch.Tensor:
        """The cuboid's orientation in the labelling the demonstration used.

        Applies the symmetry element chosen at reset. Choosing here instead,
        every step, is a bug: the bar turns as it is lifted, and re-choosing was
        measured to flip the representative partway through the motion, moving
        the reference frame 0.35 m in the middle of the grasp.
        """
        return apply_cuboid_symmetry(
            _normalize_canonical_quaternion(self.cube_orientation),
            self.cuboid_symmetries,
            self.symmetry_index,
        )

    def _hand_keypoints_world(self) -> torch.Tensor:
        """``(num_envs, 9, 3)`` palm and fingertip keypoints in world space.

        The anchor is chosen by the caller, because the two halves of the
        keypoint reward want different ones. See ``_hand_keypoints_anchored``.
        """
        palm_position_world, palm_orientation_world = self._palm_pose_world()
        return hand_keypoints(
            palm_position_world,
            palm_orientation_world,
            self._fingertip_positions_world(),
            self.palm_lever_arm_m,
        )

    def _hand_keypoints_anchored(
        self,
        keypoints_world: torch.Tensor,
        anchor_position: torch.Tensor,
        anchor_orientation: torch.Tensor,
    ) -> torch.Tensor:
        """``(num_envs, 9, 3)`` hand keypoints expressed in an anchor's frame.

        Choosing the anchor is choosing what the resulting reward is a
        statement *about*, which is why the palm and the fingertips use
        different ones. See docs/adr/0001.

        Anchored on the cuboid's *measured* pose the error is hand-bar relative
        geometry, so the term still points the right way when the bar has been
        nudged -- but it is also blind to the pair moving together, and the
        grasp makes it uncorrectable. Anchored on the *reference* pose the
        error is where the hand is in the world, which is what the lift needs.
        """
        return keypoints_in_object_frame(
            keypoints_world, anchor_position, anchor_orientation
        )

    def _palm_tilt(self) -> torch.Tensor:
        """``(num_envs, 3)`` gravity direction in the palm frame.

        The palm's pitch and roll with its yaw discarded, which is what makes it
        comparable against one per-frame reference table however the episode's
        bar was turned. World and base axes coincide here -- the base is asserted
        unrotated in _resolve_jacobian_index -- so the world vertical is usable
        directly.
        """
        _, palm_orientation_world = self._palm_pose_world()
        return _quat_rotate_inverse(palm_orientation_world, self.world_up)

    def _task_space_observation_components(self) -> Tuple[torch.Tensor, ...]:
        """Return the five task-space blocks appended to the old 79D.

        Palm pose is expressed in the robot actor-base frame. Fingertip
        positions and cube pose are expressed relative to the palm frame. Cube
        and palm velocities are deliberately absent from this 108D experiment.
        """
        palm_position_world, palm_orientation_world = self._palm_pose_world()
        if self.critic_force_observation_dim:
            # Cached here, where the palm frame is already computed, and
            # consumed by compute_observations below.
            self._critic_force_features = self._fingertip_force_features(
                palm_orientation_world
            )
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
                self.canonical_cube_orientation(),
            )
        )

        fingertip_positions_world = self._fingertip_positions_world()
        fingertip_positions_palm = _quat_rotate_inverse(
            palm_orientation_world.unsqueeze(1).expand(-1, 5, -1),
            fingertip_positions_world - palm_position_world.unsqueeze(1),
        ).reshape(self.num_envs, 15)
        # Both rotations go out as the continuous 6D representation rather than
        # as quaternions. A quaternion is a double cover, so q and -q are the
        # same rotation and the w >= 0 canonicalisation only moves the
        # discontinuity to w == 0 rather than removing it. That seam is a real
        # cost here: yaw is randomised over 112 degrees and the bar turns while
        # it is lifted, so the network would meet it often.
        components = (
            palm_position_robot,
            quat_to_rotation_6d(palm_orientation_robot),
            fingertip_positions_palm,
            quat_to_rotation_6d(cube_orientation_palm),
            cube_center_palm,
        )
        if not self.contact_observation_enabled:
            return components
        return components + (
            self._fingertip_force_features(palm_orientation_world),
        )

    def _fingertip_force_features(
        self, palm_orientation_world: torch.Tensor
    ) -> torch.Tensor:
        """Scaled, clipped fingertip contact forces in the palm frame.

        Contact forces go into the palm frame like the fingertip positions, so
        "pressed from this direction" reads the same wherever the arm happens
        to be. Shared by the actor observation and the privileged critic
        observation so the two can never drift apart.
        """
        fingertip_forces_world = select_fingertip_forces(
            self.net_contact_forces, self.contact_fingertip_body_indices
        )
        num_contact_tips = fingertip_forces_world.shape[1]
        fingertip_forces_palm = _quat_rotate_inverse(
            palm_orientation_world.unsqueeze(1).expand(-1, num_contact_tips, -1),
            fingertip_forces_world,
        )
        return fingertip_force_observation(
            fingertip_forces_palm,
            self.contact_observation_force_scale_n,
            self.contact_observation_clip,
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

    def _cube_reference_root_states(
        self, sample, env_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Convert a bank track to the episode's exact continuous transform."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
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
        bank_yaw = self.transform_bank.yaw_rad[self.transform_index[env_ids]]
        bank_translation = self.transform_bank.translation[
            self.transform_index[env_ids]
        ]
        delta_yaw = self.episode_yaw_rad[env_ids] - bank_yaw
        half = 0.5 * delta_yaw
        delta_quaternion = torch.stack(
            (
                torch.zeros_like(half),
                torch.zeros_like(half),
                torch.sin(half),
                torch.cos(half),
            ),
            dim=1,
        )
        # At frame zero the cuboid centre is the transform pivot, so the bank
        # centre differs from the demo centre by exactly bank_translation.
        bank_start_sample = self.transform_bank.sample(
            self.transform_index[env_ids], torch.zeros_like(env_ids)
        )
        bank_start_position = (
            self.robot_base_position
            + bank_start_sample.cube_pose[:, :3] * self.world_axis_sign
        )
        actual_start_position = (
            bank_start_position - bank_translation
            + self.episode_translation[env_ids]
        )
        position = actual_start_position + _quat_rotate(
            delta_quaternion, position - bank_start_position
        )
        quaternion_world = _quat_multiply(delta_quaternion, quaternion_world)
        linear_velocity = _quat_rotate(
            delta_quaternion, sample.cube_linear_velocity * self.world_axis_sign
        )
        angular_velocity = _quat_rotate(
            delta_quaternion, sample.cube_angular_velocity * self.world_axis_sign
        )
        return torch.cat(
            (
                position,
                quaternion_world,
                linear_velocity,
                angular_velocity,
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
            self._cube_reference_root_states(sample, env_ids)
        )
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state_all),
            gymtorch.unwrap_tensor(cube_actor_ids),
            cube_actor_ids.numel(),
        )

    def set_training_iteration(self, iteration: int) -> float:
        """Advance the assist curriculum and return the resulting scale.

        PPO calls this once per update. Nothing else in the environment depends
        on the iteration counter, so a process that never calls it (evaluation,
        the headless tests) simply keeps the schedule's iteration-0 value.
        """
        self.object_assist_scale = assist_scale_at(
            self.object_assist_settings, int(iteration)
        )
        return self.object_assist_scale

    def object_reward_gate(self) -> float:
        return object_reward_gate(
            self.object_assist_enabled,
            self.object_assist_gates_object_reward,
            self.object_assist_scale,
        )

    def set_object_assist_scale(self, scale: float) -> float:
        """Pin the assist scale, overriding the schedule until it is advanced."""
        scale = float(scale)
        if not math.isfinite(scale) or scale < 0.0:
            raise ValueError("The object-assist scale must be finite and non-negative")
        self.object_assist_scale = scale
        return self.object_assist_scale

    def _compute_object_assist(self, reference) -> None:
        """Fill the cube's annealed PD + gravity-compensation wrench buffers.

        The target is the reference sample the robot is being driven toward in
        this step, so the assist and the position targets pull the same way.
        """
        active = self.reference_index >= (
            self.object_assist_settings.active_from_reference_index
        )
        force, torque = object_assist_wrench(
            self.cube_position,
            self.cube_orientation,
            self.cube_linear_velocity,
            self.cube_angular_velocity,
            self._cube_reference_root_states(reference),
            self.object_assist_settings,
            self.object_assist_scale,
            float(self.cfg.object.mass_kg),
            self.gravity_vector,
            active,
        )
        self.cube_body_forces.copy_(force)
        self.cube_body_torques.copy_(torque)
        self.object_assist_force_n.copy_(
            torch.linalg.vector_norm(force, dim=1)
        )
        self.object_assist_torque_nm.copy_(
            torch.linalg.vector_norm(torque, dim=1)
        )

    def _apply_disturbances(self) -> None:
        """Add this step's random impulses into the force buffer.

        Written into the same buffer the object assist uses, so both reach
        PhysX through one apply call. Additive rather than overwriting: with
        the assist on, a disturbance should perturb the assisted cube, not
        replace the assist.
        """
        randomization = self.domain_randomization
        if not randomization.impulses_enabled or self.rigid_body_forces is None:
            return
        bodies = self.rigid_body_forces.shape[0] // self.num_envs
        view = self.rigid_body_forces.view(self.num_envs, bodies, 3)
        if (
            randomization.robot_impulse_probability > 0.0
            and randomization.robot_impulse_n > 0.0
        ):
            view += sample_impulses(
                self.num_envs,
                bodies,
                randomization.robot_impulse_probability,
                randomization.robot_impulse_n,
                self.device,
                body_indices=self.robot_body_indices,
            )
        if (
            randomization.object_impulse_probability > 0.0
            and randomization.object_impulse_n > 0.0
        ):
            view += sample_impulses(
                self.num_envs,
                bodies,
                randomization.object_impulse_probability,
                randomization.object_impulse_n,
                self.device,
                body_indices=self.cube_body_index_tensor,
            )

    def _push_object_assist(self) -> None:
        """Queue the buffered wrench for the next ``simulate`` call.

        PhysX consumes applied forces once per simulation step, so with a
        decimation above one the same zero-order-hold wrench is queued again
        before every substep.
        """
        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.rigid_body_forces),
            gymtorch.unwrap_tensor(self.rigid_body_torques),
            gymapi.ENV_SPACE,
        )

    def scale_hand_actions(self, hand_actions: torch.Tensor) -> torch.Tensor:
        """Apply the unbounded residual-action mapping to the twenty hand joints."""
        residual = (hand_actions * self.hand_action_scale).clamp(
            -self.action_target_clip, self.action_target_clip
        )
        return self.default_hand_positions + residual

    def saturated_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Which action components ask for more than the contract will pass on.

        The two halves saturate on different things -- the arm's twist is capped
        by magnitude per half, so all three components of a saturated half are
        reported together, while a hand residual binds only once scaled past
        ``clip_joint_target`` -- so the definition lives here instead of being
        spelled out again in each runner that counts it.
        """
        arm = actions[:, : len(ARM_JOINT_NAMES)]
        translation_saturated = arm[:, 0:3].norm(dim=1, keepdim=True) > 1.0
        rotation_saturated = arm[:, 3:6].norm(dim=1, keepdim=True) > 1.0
        hand = (
            actions[:, len(ARM_JOINT_NAMES):].abs() * self.hand_action_scale
            > self.action_target_clip
        )
        return torch.cat(
            (
                translation_saturated.expand(-1, 3),
                rotation_saturated.expand(-1, 3),
                hand,
            ),
            dim=1,
        )

    def command_targets(self, actions: torch.Tensor) -> torch.Tensor:
        """The full 26-joint position target for one control step.

        Unlike the joint-space mapping this replaces, it is **stateful**: the arm
        half integrates onto the previous target and reads the current Jacobian.
        Call it exactly once per step, before the targets are written -- calling
        it twice would integrate the same command twice. Nothing may reconstruct
        a target by calling it again after the fact, which is why the result is
        cached here for the evaluation plotter.
        """
        targets = torch.cat(
            (
                self._operational_space_arm_targets(
                    actions[:, : len(ARM_JOINT_NAMES)]
                ),
                self.scale_hand_actions(actions[:, len(ARM_JOINT_NAMES):]),
            ),
            dim=1,
        )
        # Copied into the owned buffer rather than rebound: PPO collects inside
        # torch.inference_mode(), and rebinding would leave an inference tensor
        # behind for later readers.
        self._last_command_targets.copy_(targets)
        return targets

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

    def positions_to_hand_actions(self, hand_positions: torch.Tensor) -> torch.Tensor:
        """Invert the hand's residual mapping for ideal reference playback.

        Hand-only: the arm's mapping runs through a damped, clamped, saturating
        IK onto an accumulator and has no closed-form inverse. See
        :meth:`next_reference_action` for what replaces it.
        """
        return (hand_positions - self.default_hand_positions) / self.hand_action_scale

    def demonstration_hand_action_delta(
        self, reference_velocity: torch.Tensor
    ) -> torch.Tensor:
        """Convert the hand's ``dq_ref * dt`` into residual-action space.

        Hand actions are dimensionless residuals with
        ``q_target = q_default + hand_action_scale * action``, so a demonstrated
        joint displacement must be divided by that scale before it can be
        compared with ``a_t - a_{t-1}``.
        """
        return (
            reference_velocity[:, len(ARM_JOINT_NAMES):]
            * self.dt
            / self.hand_action_scale
        )

    @property
    def _palm_kinematics(self):
        """URDF kinematics for the reference-playback harnesses only.

        float64 on the CPU, and it pulls in ``pytorch_kinematics``, so it is built
        on first use and must never be touched from :meth:`step`. Training never
        reaches it.
        """
        if getattr(self, "_palm_kinematics_cache", None) is None:
            from simtoolreal_animrl.envs.retarget import PalmKinematics

            self._palm_kinematics_cache = PalmKinematics(
                ROOT_DIR / self.cfg.asset.file, device="cpu"
            )
        return self._palm_kinematics_cache

    def next_reference_action(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the ideal 26-action command and the complete next pose.

        The hand half still inverts its residual mapping. The arm half cannot:
        its ideal action is the twist carrying the palm pose of the *current
        accumulated target* onto the palm pose of the next reference, scaled by
        the same speeds :meth:`_operational_space_arm_targets` divides them by.
        Forward kinematics runs on the target rather than on the measured joints
        because the target is what the integrator advances.
        """
        from simtoolreal_animrl.envs.retarget import pose_error

        next_indices = (self.reference_index + 1).clamp(
            max=self.reference.last_index
        )
        target = self.transform_bank.sample(self.transform_index, next_indices).q
        kinematics = self._palm_kinematics
        current_pose = kinematics.palm_matrices(
            self.previous_arm_targets.double().cpu()
        )
        target_pose = kinematics.palm_matrices(
            target[:, : len(ARM_JOINT_NAMES)].double().cpu()
        )
        twist = pose_error(current_pose, target_pose).to(
            dtype=torch.float32, device=self.device
        )
        arm_action = torch.cat(
            (
                twist[:, :3] / (self.arm_translation_speed * self.dt),
                twist[:, 3:] / (self.arm_rotation_speed * self.dt),
            ),
            dim=1,
        )
        # Saturated the same way the controller will saturate it, so the action
        # this hands back is one the controller can actually carry out.
        arm_action = torch.cat(
            (
                saturate_direction_preserving(arm_action[:, 0:3], 1.0),
                saturate_direction_preserving(arm_action[:, 3:6], 1.0),
            ),
            dim=1,
        )
        hand_action = self.positions_to_hand_actions(
            target[:, len(ARM_JOINT_NAMES):]
        )
        return torch.cat((arm_action, hand_action), dim=1), target

    def reset_idx(
        self,
        env_ids: torch.Tensor,
        reference_indices: Optional[torch.Tensor] = None,
        transform_indices: Optional[torch.Tensor] = None,
        episode_translation: Optional[torch.Tensor] = None,
        episode_yaw_rad: Optional[torch.Tensor] = None,
    ) -> None:
        if env_ids.numel() == 0:
            return
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        if transform_indices is None:
            randomization = self.cfg.object_randomization
            count = env_ids.numel()
            if (episode_translation is None) != (episode_yaw_rad is None):
                raise ValueError(
                    "episode_translation and episode_yaw_rad must be given together"
                )
            if episode_translation is None:
                x = torch.empty(count, device=self.device).uniform_(
                    float(randomization.translation_x_min_m),
                    float(randomization.translation_x_max_m),
                )
                y = torch.empty(count, device=self.device).uniform_(
                    float(randomization.translation_y_min_m),
                    float(randomization.translation_y_max_m),
                )
                yaw = torch.empty(count, device=self.device).uniform_(
                    math.radians(float(randomization.yaw_min_deg)),
                    math.radians(float(randomization.yaw_max_deg)),
                )
                episode_translation = torch.stack(
                    (x, y, torch.zeros_like(x)), dim=1
                )
            else:
                # A caller-chosen continuous placement, resolved to a bank
                # reference exactly as the uniform sampler is: the cuboid uses
                # the requested transform while the arm reference comes from
                # its nearest bank entry. An interactive evaluator therefore
                # sees the same approximation training does, not a different
                # one.
                episode_translation = episode_translation.to(
                    device=self.device, dtype=self.episode_translation.dtype
                )
                yaw = episode_yaw_rad.to(
                    device=self.device, dtype=self.episode_yaw_rad.dtype
                )
                if episode_translation.ndim == 1:
                    episode_translation = episode_translation.unsqueeze(0).repeat(
                        count, 1
                    )
                if yaw.ndim == 0:
                    yaw = yaw.repeat(count)
                if episode_translation.shape != (count, 3):
                    raise ValueError("episode_translation has the wrong shape")
                if yaw.shape != (count,):
                    raise ValueError("episode_yaw_rad has the wrong shape")

            transform_indices = nearest_transform_indices(
                episode_translation,
                yaw,
                self.transform_bank.translation,
                self.transform_bank.yaw_rad,
                float(randomization.nearest_yaw_lever_arm_m),
            )
            self.episode_translation[env_ids] = episode_translation
            self.episode_yaw_rad[env_ids] = yaw
        else:
            if episode_translation is not None or episode_yaw_rad is not None:
                raise ValueError(
                    "transform_indices and episode_translation are alternatives"
                )
            # Explicit transforms exist so an evaluator can sweep the envelope
            # deterministically -- a success rate that averages over a randomly
            # drawn set of poses hides where the envelope gives out.
            transform_indices = transform_indices.to(
                device=self.device, dtype=torch.long
            )
            if transform_indices.ndim == 0:
                transform_indices = transform_indices.repeat(env_ids.numel())
            if transform_indices.shape != (env_ids.numel(),):
                raise ValueError("transform_indices has the wrong shape")
            if torch.any(transform_indices < 0) or torch.any(
                transform_indices >= self.transform_bank.transform_count
            ):
                raise ValueError("Transform index outside the bank")
            # Explicit evaluator/debug selections replay the exact bank pose.
            self.episode_translation[env_ids] = (
                self.transform_bank.translation[transform_indices]
            )
            self.episode_yaw_rad[env_ids] = (
                self.transform_bank.yaw_rad[transform_indices]
            )
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

        sample = self.transform_bank.sample(transform_indices, reference_indices)
        self.reference_index[env_ids] = reference_indices
        self.transform_index[env_ids] = transform_indices
        self.episode_length_buf[env_ids] = 0
        self.arm_violation_steps[env_ids] = 0
        self.hand_violation_steps[env_ids] = 0
        self.object_violation_steps[env_ids] = 0
        self.reset_buf[env_ids] = False
        self.time_out_buf[env_ids] = False
        if hasattr(self, "episode_sums"):
            for values in self.episode_sums.values():
                values[env_ids] = 0.0
        reset_q, reset_dq = perturb_reference_pose(
            sample.q,
            sample.dq,
            len(ARM_JOINT_NAMES),
            self.cfg.env.rsi_position_noise_arm_rad,
            self.cfg.env.rsi_position_noise_hand_rad,
            self.cfg.env.rsi_velocity_noise_scale,
            lower_limits=self.joint_lower_limits,
            upper_limits=self.joint_upper_limits,
        )
        if hasattr(self, "previous_actions"):
            # Seed a_{t-1} with the action that reproduces the pose the robot
            # is ACTUALLY reset to, noise included. Seeding it from the clean
            # reference would charge the first step an action-rate penalty for
            # the perturbation itself, taxing the randomisation.
            #
            # The arm seeds to zero instead: its command is an integrator now, and
            # a zero twist is the only action that holds the pose the robot starts
            # from whatever that pose is. But zero is "do not move", not "what the
            # policy would have asked for", so unlike the hand it cannot absorb the
            # perturbation -- the first step would otherwise be charged the full
            # magnitude of the policy's first command. suppress_ee_action_rate
            # exempts that one step.
            reset_action = torch.zeros(
                (env_ids.numel(), self.num_actions),
                dtype=self.actions.dtype,
                device=self.device,
            )
            reset_action[:, len(ARM_JOINT_NAMES):] = self.positions_to_hand_actions(
                reset_q[:, len(ARM_JOINT_NAMES):]
            )
            self.actions[env_ids] = reset_action
            self.previous_actions[env_ids] = reset_action
            self.suppress_ee_action_rate[env_ids] = True
            # Seed the delay line with the reset pose, so a delayed environment
            # is not commanded to the previous episode's last target.
            self.action_delay.reset(env_ids, reset_action)
            # Stale IK telemetry would otherwise describe the previous episode.
            self.requested_twist[env_ids] = 0.0
            self.achieved_twist[env_ids] = 0.0
            self.applied_arm_q_delta[env_ids] = 0.0
            self.ik_residual_norm[env_ids] = 0.0
            self.arm_joint_delta_norm[env_ids] = 0.0
            self.arm_joint_delta_clipped[env_ids] = 0.0

        state_subset = self.dof_state[env_ids]
        state_subset[:, self.demo_to_asset_tensor, 0] = reset_q
        state_subset[:, self.demo_to_asset_tensor, 1] = reset_dq
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
        # Resolve the cuboid's frame ambiguity once, towards the pose this
        # episode was reset to, and hold the choice for the episode. In
        # simulation the cuboid is placed at exactly the reference pose, so this
        # selects the identity and changes nothing; it is insurance for a real
        # pose estimate, which is free to return any of the eight relabellings
        # of a bar that never moved.
        reference_root = self._cube_reference_root_states(sample, env_ids)
        _, chosen_symmetry = canonicalize_cuboid_orientation(
            _normalize_canonical_quaternion(self.cube_orientation[env_ids]),
            self.cuboid_symmetries,
            _normalize_canonical_quaternion(reference_root[:, 3:7]),
            return_index=True,
        )
        self.symmetry_index[env_ids] = chosen_symmetry.to(
            dtype=self.symmetry_index.dtype
        )
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
        # Beside the rigid-body refresh, always: the palm Jacobian is built from
        # the wrist quaternion in that tensor, so the two must describe the same
        # configuration or the point transfer is applied with a stale rotation.
        self.gym.refresh_jacobian_tensors(self.sim)

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

    def reset(
        self,
        reference_index: Optional[int] = None,
        translation_xy: Optional[Tuple[float, float]] = None,
        yaw_rad: Optional[float] = None,
    ) -> torch.Tensor:
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        indices = None
        if reference_index is not None:
            indices = torch.full(
                (self.num_envs,),
                int(reference_index),
                dtype=torch.long,
                device=self.device,
            )
        translation = None
        yaw = None
        if translation_xy is not None or yaw_rad is not None:
            if translation_xy is None or yaw_rad is None:
                raise ValueError("translation_xy and yaw_rad must be given together")
            translation = torch.tensor(
                (float(translation_xy[0]), float(translation_xy[1]), 0.0),
                device=self.device,
            ).unsqueeze(0).repeat(self.num_envs, 1)
            yaw = torch.full(
                (self.num_envs,), float(yaw_rad), device=self.device
            )
        self.reset_idx(
            env_ids,
            indices,
            episode_translation=translation,
            episode_yaw_rad=yaw,
        )
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        if self.contact_enabled:
            # Otherwise the first observation of the episode would carry the
            # forces from just before the reset.
            self.gym.refresh_net_contact_force_tensor(self.sim)
        self.compute_observations()
        return self.obs_buf

    def compute_observations(self) -> None:
        phase = (
            self.reference_index.float() / float(self.reference.last_index)
        ).unsqueeze(1)
        measured_q, measured_dq = self.q, self.dq
        if self.domain_randomization.observation_noise_enabled:
            measured_q, measured_dq = add_observation_noise(
                self.q,
                self.dq,
                self.domain_randomization.obs_q_noise_rad,
                self.domain_randomization.obs_dq_noise_rad_s,
                self.domain_randomization.obs_q_bias_rad,
                bias=self.observation_position_bias,
            )
        self.obs_buf.copy_(
            torch.cat(
                (
                    self.normalize_positions(measured_q),
                    self.previous_targets,
                    measured_dq,
                    phase,
                    *self._task_space_observation_components(),
                ),
                dim=1,
            )
        )
        if self.obs_buf.shape != (self.num_envs, self.num_obs):
            raise RuntimeError("Observation shape does not match configuration")
        if self.critic_obs_buf is not None:
            parts = [self.obs_buf]
            if self.critic_force_observation_dim:
                if self._critic_force_features is None:
                    raise RuntimeError(
                        "Privileged critic observation requested but the "
                        "fingertip forces were never computed"
                    )
                parts.append(self._critic_force_features)
            if self.critic_parameter_table is not None:
                parts.append(self.critic_parameter_table)
            self.critic_obs_buf.copy_(torch.cat(parts, dim=1))

    def _compute_reward_and_errors(self) -> Dict[str, torch.Tensor]:
        reference = self.transform_bank.sample(
            self.transform_index, self.reference_index
        )
        reference_arm_q = reference.q[:, : len(ARM_JOINT_NAMES)]
        reference_arm_dq = reference.dq[:, : len(ARM_JOINT_NAMES)]
        reference_hand_q = reference.q[:, len(ARM_JOINT_NAMES):]
        reference_hand_dq = reference.dq[:, len(ARM_JOINT_NAMES):]
        arm_q_error = self.arm_q - reference_arm_q
        arm_dq_error = self.arm_dq - reference_arm_dq
        hand_q_error = self.hand_q - reference_hand_q
        hand_dq_error = self.hand_dq - reference_hand_dq
        action_delta = self.actions - self.previous_actions
        hand_action_delta_error = (
            action_delta[:, len(ARM_JOINT_NAMES):]
            - self.demonstration_hand_action_delta(reference.dq)
        )
        # The arm term is pure command smoothness: the EE twist the policy asked
        # for, changing step to step. Unlike the hand it is not compared against
        # the demonstration, whose retargeted motion can legitimately want a
        # different command. Exempt the first step of an episode, where a_{t-1}
        # is the seeded zero rather than a command the policy chose.
        # step() clears the flag once the reward has been computed, so that this
        # stays free of side effects and can be called twice for the same state.
        ee_action_delta = torch.where(
            self.suppress_ee_action_rate.unsqueeze(1),
            torch.zeros_like(action_delta[:, : len(ARM_JOINT_NAMES)]),
            action_delta[:, : len(ARM_JOINT_NAMES)],
        )

        # Arm and hand keep separate Gaussians: averaging one MSE over all 26
        # joints would let the 20 hand joints outvote the 6 arm joints in a
        # single term and dilute the gradient each block needs.
        # Pitch and roll of the palm against the demonstration, yaw excluded.
        # Squared chord distance between the two unit vectors rather than the
        # angle between them: it equals the angle squared to second order and
        # has a finite derivative at zero, where arccos does not.
        palm_tilt = self._palm_tilt()
        reference_palm_tilt = self.transform_bank.palm_tilt_at(self.reference_index)
        palm_tilt_mse = (palm_tilt - reference_palm_tilt).square().sum(dim=1)
        palm_tilt_error_rad = torch.arccos(
            (palm_tilt * reference_palm_tilt).sum(dim=1).clamp(-1.0, 1.0)
        )

        rewards_cfg = self.cfg.rewards
        # Kept as diagnostics only. Nothing rewards arm joint tracking now, so
        # this reads as null-space drift: how far the arm has wandered from the
        # reference configuration while still reaching the commanded palm pose.
        position_mse = arm_q_error.square().mean(dim=1)
        velocity_mse = arm_dq_error.square().mean(dim=1)
        ee_action_rate_mse = ee_action_delta.square().mean(dim=1)
        arm_joint_rate_mse = self.applied_arm_q_delta.square().mean(dim=1)
        ik_residual_mse = self.ik_residual_norm.square()
        hand_position_mse = hand_q_error.square().mean(dim=1)
        hand_velocity_mse = hand_dq_error.square().mean(dim=1)
        hand_action_rate_mse = hand_action_delta_error.square().mean(dim=1)

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
        actual_cube_orientation = _normalize_canonical_quaternion(
            self.cube_orientation
        )
        reference_cube_orientation = _normalize_canonical_quaternion(
            reference_cube_root_state[:, 3:7]
        )
        # Modulo the bar's own symmetry. Its cross-section is square, so half a
        # turn about the long axis is the same physical pose; the plain geodesic
        # angle charged up to pi for a bar that was exactly where it should be.
        # Measured over 128 environments during the lift: the plain angle
        # averaged 1.372 rad with 21% of samples above 2.5 rad, while the
        # symmetry-invariant distance averaged 0.460. Two thirds of what this
        # term was pricing was relabelling, not error -- which is why raising
        # its weight and widening its sigma both moved the result by under 10%.
        #
        # This picks a fresh element every step, which canonical_cube_orientation
        # deliberately does not. The difference is what the result is used for:
        # a frame that jumps mid-motion drags the keypoints with it, a scalar
        # distance has no frame to jump.
        object_orientation_error_rad = symmetry_invariant_orientation_error(
            actual_cube_orientation,
            reference_cube_orientation,
            self.cuboid_symmetries,
        )

        gaussian = lambda mse, std: torch.exp(-mse / (2.0 * float(std) ** 2))

        def width(name, configured, mse):
            """The width for one term: adaptive when enabled, else the config."""
            tracker = self.adaptive_sigmas.get(name)
            if tracker is None:
                return float(configured)
            return tracker.update(float(mse.mean().item()))

        position_hand_std = width(
            "position_hand", rewards_cfg.position_hand_std_rad, hand_position_mse
        )
        ee_action_rate_std = width(
            "ee_action_rate", rewards_cfg.ee_action_rate_std, ee_action_rate_mse
        )
        hand_action_rate_std = width(
            "hand_action_rate",
            rewards_cfg.hand_action_rate_std,
            hand_action_rate_mse,
        )
        palm_tilt_reward = gaussian(palm_tilt_mse, rewards_cfg.palm_tilt_std_rad)
        ee_action_rate_reward = gaussian(ee_action_rate_mse, ee_action_rate_std)
        arm_joint_rate_reward = gaussian(
            arm_joint_rate_mse, rewards_cfg.arm_joint_rate_std_rad
        )
        ik_residual_reward = gaussian(
            ik_residual_mse, rewards_cfg.ik_residual_std
        )
        hand_position_reward = gaussian(
            hand_position_mse, position_hand_std
        )
        hand_velocity_reward = gaussian(
            hand_velocity_mse, rewards_cfg.velocity_hand_std_rad_per_s
        )
        hand_action_rate_reward = gaussian(
            hand_action_rate_mse, hand_action_rate_std
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

        # The object-centric terms. The hand is nine points -- the palm origin,
        # three more at a lever arm along its axes, and the five fingertips --
        # and the reward is how far they are from where the demonstration put
        # them *in the cuboid's frame*. Move the bar and the target moves with
        # it, which is the whole generalisation.
        #
        # The reference is looked up by frame alone: expressed in the cuboid's
        # frame it does not depend on which transform this episode is playing,
        # because a rigid motion of the whole scene cancels between hand and
        # bar. tests/test_retarget.py asserts that.
        #
        # Palm and fingertips keep separate Gaussians for the same reason the
        # arm and hand joints do: averaged into one term, five fingertips
        # outvote four palm points and the approach stops being paid for.
        #
        # The two halves take different anchors on purpose (docs/adr/0001).
        # The fingertips price the grasp, whose subject matter is relative
        # geometry, so they stay on the measured bar. The palm prices the
        # transport: on the measured bar it cannot see the lift at all, since
        # a rigid grasp carries hand and bar together and leaves the relative
        # pose untouched, so it anchors on the reference bar instead and its
        # target rises with the reference.
        #
        # No symmetry element on the reference side. canonical_cube_orientation
        # exists to bring the *measured* quaternion into the labelling the
        # demonstration used; the reference quaternion is the demonstration's,
        # so re-applying it would move the frame rather than align it.
        keypoints_world = self._hand_keypoints_world()
        palm_frame = self._hand_keypoints_anchored(
            keypoints_world,
            reference_cube_root_state[:, 0:3],
            reference_cube_orientation,
        )
        fingertip_frame = self._hand_keypoints_anchored(
            keypoints_world,
            self.cube_position,
            self.canonical_cube_orientation(),
        )
        reference_keypoints = self.transform_bank.keypoints_at(self.reference_index)
        actual_palm, _ = split_palm_and_fingertips(palm_frame)
        _, actual_fingertips = split_palm_and_fingertips(fingertip_frame)
        reference_palm, reference_fingertips = split_palm_and_fingertips(
            reference_keypoints
        )
        palm_keypoint_mse = keypoint_tracking_error(actual_palm, reference_palm)
        fingertip_keypoint_mse = keypoint_tracking_error(
            actual_fingertips, reference_fingertips
        )
        palm_keypoint_error_m = palm_keypoint_mse.clamp_min(0.0).sqrt()
        fingertip_keypoint_error_m = fingertip_keypoint_mse.clamp_min(0.0).sqrt()
        palm_keypoint_reward = keypoint_gaussian(
            palm_keypoint_mse, rewards_cfg.palm_keypoint_std_m
        )
        fingertip_keypoint_reward = keypoint_gaussian(
            fingertip_keypoint_mse, rewards_cfg.fingertip_keypoint_std_m
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
            fingertip_contact_reward = torch.zeros_like(palm_keypoint_reward)
            fingertip_contact_fraction = torch.zeros_like(palm_keypoint_reward)
            mean_fingertip_contact_force_n = torch.zeros_like(palm_keypoint_reward)
            fingertip_force_n = palm_keypoint_reward.new_zeros(
                (self.num_envs, len(FINGERTIP_BODY_NAMES))
            )
        self.rew_buf.copy_(
            float(rewards_cfg.palm_keypoint_weight) * palm_keypoint_reward
            + float(rewards_cfg.fingertip_keypoint_weight)
            * fingertip_keypoint_reward
            # The only term that sees the palm's absolute pose. Everything else
            # is measured either in the cube's frame or on the cube itself, and
            # is therefore satisfied by a hand that turns the cube up on its
            # wrist instead of carrying it.
            + float(rewards_cfg.palm_tilt_weight) * palm_tilt_reward
            # One regularizer on each side of the IK: the twist the policy asked
            # for, and the joint delta the solver emitted for it. The arm's
            # joint-space *tracking* terms are gone -- they only survived as
            # null-space selection, and the accumulating solver settles that
            # structurally by deforming continuously from the reset pose rather
            # than jumping IK branches.
            + float(rewards_cfg.ee_action_rate_weight) * ee_action_rate_reward
            + float(rewards_cfg.arm_joint_rate_weight) * arm_joint_rate_reward
            + float(rewards_cfg.ik_residual_weight) * ik_residual_reward
            # The hand's joint terms stay: each finger has four joints serving a
            # three-dimensional fingertip target, so five finger degrees of
            # freedom are otherwise unconstrained.
            + float(rewards_cfg.position_hand_weight) * hand_position_reward
            + float(rewards_cfg.velocity_hand_weight) * hand_velocity_reward
            + float(rewards_cfg.hand_action_rate_weight)
            * hand_action_rate_reward
            + self.object_reward_gate()
            * (
                float(rewards_cfg.object_position_weight)
                * object_position_reward
                + float(rewards_cfg.object_orientation_weight)
                * object_orientation_reward
            )
            + self.proximity_weight * fingertip_object_distance_reward
            + self.contact_shaping_weight * fingertip_contact_reward
        )
        return {
            "palm_keypoint_error_m": palm_keypoint_error_m,
            "fingertip_keypoint_error_m": fingertip_keypoint_error_m,
            "palm_keypoint_reward": palm_keypoint_reward,
            "fingertip_keypoint_reward": fingertip_keypoint_reward,
            "q_error": arm_q_error,
            "dq_error": arm_dq_error,
            "hand_q_error": hand_q_error,
            "hand_dq_error": hand_dq_error,
            "position_mse": position_mse,
            "velocity_mse": velocity_mse,
            "ee_action_rate_mse": ee_action_rate_mse,
            "arm_joint_rate_mse": arm_joint_rate_mse,
            "hand_position_mse": hand_position_mse,
            "hand_velocity_mse": hand_velocity_mse,
            "hand_action_rate_mse": hand_action_rate_mse,
            "palm_tilt_reward": palm_tilt_reward,
            "palm_tilt_error_rad": palm_tilt_error_rad,
            "ee_action_rate_reward": ee_action_rate_reward,
            "arm_joint_rate_reward": arm_joint_rate_reward,
            "ik_residual_reward": ik_residual_reward,
            "ik_residual_norm": self.ik_residual_norm,
            "arm_joint_delta_norm": self.arm_joint_delta_norm,
            "arm_joint_delta_clipped": self.arm_joint_delta_clipped,
            "requested_twist": self.requested_twist,
            "achieved_twist": self.achieved_twist,
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

    def threshold_violation(self, palm_keypoint_error_m: torch.Tensor) -> torch.Tensor:
        """Has the palm left the neighbourhood of where the reference puts it?

        In task space, not joint space. The reward deliberately lets the arm
        leave the retargeted joint angles -- that null-space freedom is the
        point of tracking a palm pose instead of six joints -- so a joint-error
        threshold would end episodes the reward is perfectly happy with.

        The error is the RMS over all four palm keypoints, so a palm in the
        right place but turned the wrong way still trips it; thresholding the
        origin alone would miss a pure orientation failure entirely.
        """
        return palm_keypoint_error_m > float(
            self.cfg.termination.palm_keypoint_threshold_m
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
        palm_keypoint_error_m: torch.Tensor,
        hand_q_error: torch.Tensor,
        object_position_error_m: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Arm, hand and object each keep an independent grace counter. Returning
        # inside one threshold clears only that source's count, regardless of
        # what the other tracking errors are doing.
        if bool(self.cfg.termination.enabled):
            self.arm_violation = self.threshold_violation(palm_keypoint_error_m)
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
        for name in (
            "palm_keypoint_reward",
            "fingertip_keypoint_reward",
            "palm_keypoint_error_m",
            "fingertip_keypoint_error_m",
        ):
            self.episode_sums[name] += metrics[name]
        self.episode_sums["palm_tilt_reward"] += metrics["palm_tilt_reward"]
        self.episode_sums["palm_tilt_error_rad"] += metrics["palm_tilt_error_rad"]
        self.episode_sums["ee_action_rate_reward"] += metrics[
            "ee_action_rate_reward"
        ]
        self.episode_sums["arm_joint_rate_reward"] += metrics[
            "arm_joint_rate_reward"
        ]
        self.episode_sums["ik_residual_reward"] += metrics["ik_residual_reward"]
        self.episode_sums["ik_residual_norm"] += metrics["ik_residual_norm"]
        self.episode_sums["arm_joint_delta_norm"] += metrics["arm_joint_delta_norm"]
        self.episode_sums["arm_joint_delta_clipped"] += metrics[
            "arm_joint_delta_clipped"
        ]
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
        self.episode_sums["rms_ee_action_rate"] += metrics[
            "ee_action_rate_mse"
        ].sqrt()
        self.episode_sums["rms_arm_joint_rate"] += metrics[
            "arm_joint_rate_mse"
        ].sqrt()
        self.episode_sums["object_assist_force_n"] += (
            self.object_assist_force_n
        )
        self.episode_sums["object_assist_torque_nm"] += (
            self.object_assist_torque_nm
        )

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
            # The object-centric tracking terms, per step of the episodes that
            # ended. These are the two numbers that say whether the policy is
            # reproducing the demonstrated motion relative to the bar.
            "palm_keypoint_reward": (
                self.episode_sums["palm_keypoint_reward"][done] / lengths
            ).mean(),
            "fingertip_keypoint_reward": (
                self.episode_sums["fingertip_keypoint_reward"][done] / lengths
            ).mean(),
            "palm_keypoint_error_m": (
                self.episode_sums["palm_keypoint_error_m"][done] / lengths
            ).mean(),
            "fingertip_keypoint_error_m": (
                self.episode_sums["fingertip_keypoint_error_m"][done] / lengths
            ).mean(),
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
        # The robot receives a possibly older command; self.actions keeps this
        # step's, so the action-rate reward still judges what the policy asked
        # for rather than what the delayed path delivered.
        complete_target_q = self.command_targets(self.action_delay(self.actions))
        next_indices = (self.reference_index + 1).clamp(
            max=self.reference.last_index
        )
        next_reference = self.transform_bank.sample(
            self.transform_index, next_indices
        )
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
        # The helper wrench is computed from the pre-physics cube state against
        # the same reference sample the position targets chase.
        if self.object_assist_enabled:
            self._compute_object_assist(next_reference)

        # Physics Step
        if self.domain_randomization.impulses_enabled:
            if not self.object_assist_enabled:
                # Nothing else writes this buffer, so clear last step's impulse
                # before adding this one; otherwise pushes would accumulate into
                # a sustained load.
                self.rigid_body_forces.zero_()
            self._apply_disturbances()
        for _ in range(int(self.cfg.control.decimation)):
            if self.object_assist_enabled or self.domain_randomization.impulses_enabled:
                self._push_object_assist()
            self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        # Refreshed at the end of the step, so the Jacobian the next step's IK
        # reads describes the configuration this step left the arm in.
        self.gym.refresh_jacobian_tensors(self.sim)
        if self.contact_enabled:
            self.gym.refresh_net_contact_force_tensor(self.sim)
        self.render(sync_frame_time=True)

        # Post-Physics Step: update the reference index, compute rewards and termination, and reset completed environments.
        self.reference_index.add_(1).clamp_(max=self.reference.last_index)
        self.episode_length_buf += 1
        metrics = self._compute_reward_and_errors()
        # The post-reset exemption has now been spent on the step it was set for.
        self.suppress_ee_action_rate.fill_(False)
        self._accumulate_episode_metrics(metrics)
        done, early, timeout = self._compute_termination(
            metrics["palm_keypoint_error_m"],
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
            "rms_ee_action_rate": metrics["ee_action_rate_mse"].sqrt(),
            "rms_arm_joint_rate": metrics["arm_joint_rate_mse"].sqrt(),
            "rms_hand_position_error": metrics["hand_position_mse"].sqrt(),
            "rms_hand_velocity_error": metrics["hand_velocity_mse"].sqrt(),
            "rms_hand_action_rate": metrics["hand_action_rate_mse"].sqrt(),
            # The object-centric terms the evaluator scores checkpoints on.
            "palm_keypoint_reward": metrics["palm_keypoint_reward"],
            "fingertip_keypoint_reward": metrics["fingertip_keypoint_reward"],
            "palm_keypoint_error_m": metrics["palm_keypoint_error_m"],
            "fingertip_keypoint_error_m": metrics["fingertip_keypoint_error_m"],
            "palm_tilt_reward": metrics["palm_tilt_reward"],
            "palm_tilt_error_rad": metrics["palm_tilt_error_rad"],
            "ee_action_rate_reward": metrics["ee_action_rate_reward"],
            "arm_joint_rate_reward": metrics["arm_joint_rate_reward"],
            "ik_residual_reward": metrics["ik_residual_reward"],
            "ik_residual_norm": metrics["ik_residual_norm"].clone(),
            "arm_joint_delta_norm": metrics["arm_joint_delta_norm"].clone(),
            "arm_joint_delta_clipped": metrics["arm_joint_delta_clipped"].clone(),
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
            # The wrench that was applied before this step's physics, so a
            # rollout can report how much help the object still receives.
            "object_assist_force_n": self.object_assist_force_n.clone(),
            "object_assist_torque_nm": self.object_assist_torque_nm.clone(),
            "object_assist_scale": self.object_assist_scale,
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

    def _randomize_shape_friction(self, env, actor, key, env_index) -> None:
        """Scale one actor's contact friction for this environment."""
        if not self.domain_randomization.enabled:
            return
        scale = self.domain_randomization.multiplier(key, env_index)
        if scale == 1.0:
            return
        shapes = self.gym.get_actor_rigid_shape_properties(env, actor)
        for shape in shapes:
            shape.friction = float(shape.friction) * scale
        self.gym.set_actor_rigid_shape_properties(env, actor, shapes)

    def _randomized_pd_properties(self, env_index):
        """This environment's drive gains, or the shared ones when off.

        Isaac Gym applies DOF properties when the actor is built, so the gains
        are fixed per environment for the run. With hundreds of environments the
        population covers the range on every iteration, which is what the policy
        actually experiences.
        """
        randomization = self.domain_randomization
        if not randomization.enabled:
            return self.pd_properties
        properties = np.copy(self.pd_properties)
        arm_indices = self.demo_to_asset[: len(ARM_JOINT_NAMES)]
        hand_indices = self.demo_to_asset[len(ARM_JOINT_NAMES):]
        for indices, stiffness_key, damping_key in (
            (arm_indices, "arm_stiffness", "arm_damping"),
            (hand_indices, "hand_stiffness", "hand_damping"),
        ):
            properties["stiffness"][indices] *= randomization.multiplier(
                stiffness_key, env_index
            )
            properties["damping"][indices] *= randomization.multiplier(
                damping_key, env_index
            )
        return properties

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
