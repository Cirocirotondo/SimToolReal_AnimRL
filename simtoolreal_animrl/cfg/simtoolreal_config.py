"""UR5e + DG5F motion-imitation configurations."""

from .base_config import BaseEnvCfg, BaseTrainCfg


class SimToolRealCfg(BaseEnvCfg):
    class env(BaseEnvCfg.env):
        num_envs = 4096
        # Match the 2026-08-28 no_object_reward reference run.
        episode_length = 360
        num_actions = 26
        # Existing 79D proprioception, followed by palm pose in the robot-base
        # frame (3+4), five fingertip positions relative to the palm (15), and
        # cube orientation/center relative to the palm (4+3).
        num_observations = 108
        num_privileged_obs = None
        reference_state_initialization = True
        # The reference run sampled uniformly from every valid start frame.
        # The demonstration has indices 0..1107 and 1107 is reserved for the
        # reference-end timeout, hence the inclusive maximum start is 1106.
        reference_init_distribution = "uniform"
        rsi_early_probability = 0.20 # used only when reference_init_distribution = "pregrasp_mixture"
        rsi_pregrasp_start_index = 740 # proximity reward starts from this demonstration index
        rsi_max_start_index = 1106

    class asset:
        file = "assets/urdf/ur5e_delto_description/ur5e_right_dg5f_mount_60deg.urdf"
        fix_base_link = True
        disable_gravity = True
        collapse_fixed_joints = True
        flip_visual_attachments = True
        thickness = 0.001
        angular_damping = 0.01
        linear_damping = 0.01
        use_physx_armature = True
        # PhysX aggregate self-collision flag. With it on, the twenty finger
        # bodies collide with each other and with the palm, which measures
        # 32.4 ms per step against 21.4 with reference actions and 50.8 against
        # 23.1 once the policy makes the fingers jitter. Off, the hand can pass
        # through itself, while external robot-cube contacts remain enabled.
        self_collision = False
        friction = 0.5
        fingertip_friction = 1.5
        restitution = 0.0

    class object:
        # Matches demo_viewer_isaacgym_cube_RSI_test.py exactly.
        size_m = [0.15, 0.05, 0.05]
        mass_kg = 0.2
        inertia_kg_m2 = [0.00008333, 0.00041667, 0.00041667]
        friction = 0.5
        restitution = 0.0
        color = [0.78, 0.78, 0.82]

    class object_assist:
        # External helper wrench applied at the cube's centre of mass: a PD
        # controller toward the demonstrated cube pose plus gravity
        # compensation. Its scale decays linearly to zero over the configured
        # iteration window, so the policy progressively takes over the force
        # the hand has to supply. Off by default; train.py --object-assist and
        # --set object_assist.* enable and tune it without changing any of the
        # existing experiment launchers.
        enabled = False
        # "linear" anneals initial_scale -> final_scale between the two
        # iterations below; "constant" holds initial_scale for the whole run
        # and exists for ablations.
        schedule = "linear"
        start_iteration = 0
        end_iteration = 6000
        initial_scale = 1.0
        final_scale = 0.0
        # Critically-to-slightly-overdamped gains for the 0.2 kg cuboid:
        # omega = sqrt(kp/m) ~ 24 rad/s, well inside the 60 Hz control rate at
        # which the wrench is held constant.
        position_stiffness_n_per_m = 120.0
        position_damping_ns_per_m = 12.0
        # Sized on the cuboid's smallest principal inertia (8.33e-5 kg m^2),
        # which is the axis that would go unstable first.
        orientation_stiffness_nm_per_rad = 0.06
        orientation_damping_nms_per_rad = 0.006
        torque_enabled = True
        # Cancels m*g, so at scale 1 a cube already at its target floats there
        # instead of needing a steady-state PD offset to stay put.
        gravity_compensation = True
        # Direction-preserving saturation, so a large transient pose error
        # cannot launch the cube. m*g is only 1.96 N.
        max_force_n = 30.0
        max_torque_nm = 0.2
        # The wrench is gated off before this demonstration index. Zero assists
        # over the whole motion, including while the cube rests on the table.
        active_from_reference_index = 0
        # Scale the object position/orientation rewards by (1 - assist scale),
        # so the policy is only paid for the cube tracking it produces itself.
        # Without this the assist hands it those terms for free and there is no
        # pressure to take the load over before the assist anneals away.
        # Inert when the assist is off, where the scale is always zero.
        gate_object_reward = True

    class table:
        size_m = [0.475, 0.4, 0.3]
        surface_below_robot_base_m = 0.035
        friction = 0.5
        restitution = 0.0
        color = [0.82, 0.56, 0.35]

    class init_state:
        # Matches the independently verified Isaac Gym demonstration viewer.
        # At z=0 the demonstrated wrist intersects the ground around RSI 732.
        pos = [0.0, 0.6, 0.55]
        rot = [0.0, 0.0, 0.0, 1.0]
        # AnimRL-style fixed default pose used by the residual action mapping.
        # This is sample zero of the processed demonstration.
        default_arm_joint_angles = [
            -1.5707905480,
            -1.0499914063,
            1.9499972045,
            -0.9000079470,
            1.5709867791,
            -2.6179869035,
        ]
        # Sample zero of the demonstration for the 20 DG5F joints, in the
        # rj_dg_<finger>_<joint> order of controller.HAND_JOINT_NAMES.
        default_hand_joint_angles = [
            0.3823220950, -0.1954499712, 0.0374738305, 0.0331612558,
            -0.1989675347, 0.0575958653, 0.0000000000, 0.2456188158,
            0.1364082419, 0.3909537524, 0.0191986218, 0.0575942737,
            0.2129301687, 0.3926990817, 0.0418879020, 0.0314159265,
            0.2112131611, 0.3211405824, 0.3385938749, 0.0261799388,
        ]

    class motion:
        file = "demonstrations/demo_20260727_152551_335339_60hz_cube_collision_resolved_stable_grasp.npz"
        frequency_hz = 60.0

    class control:
        control_type = "P"
        decimation = 1
        action_parameterization = "animrl_residual"
        scale_joint_target = 0.25
        # Finer residual for the hand. Its joints travel a median of 0.414 rad
        # from the default pose against the arm's 0.458, and the smaller scale
        # buys resolution for the contact work that follows.
        scale_hand_joint_target = 0.15
        # At 100.0 the residual clamp never binds. The finger movements are 
        # effectively capped by the early terminations.
        clip_joint_target = 100.0
        # Multipliers on the low-level position-drive gains in envs/controller.py.
        # 1.0 reproduces the gains every run so far has used. Softening a limb
        # lowers its bandwidth sqrt(k/J) so the drive filters the policy's
        # step-to-step chatter rather than tracking it into the joint, and
        # raises the damping ratio d/(2 sqrt(kJ)) at the same time.
        arm_stiffness_scale = 1.0
        arm_damping_scale = 1.0
        hand_stiffness_scale = 1.0
        hand_damping_scale = 1.0

    class contact:
        # Optional GPU contact shaping for the three fingers used by the
        # grasp.  When disabled, MotionImitationEnv keeps PhysX contact
        # reporting at CC_NEVER and does not acquire/refresh its tensor.
        enabled = False
        # Isaac Gym ContactCollection value: 1 = CC_LAST_SUBSTEP and
        # 2 = CC_ALL_SUBSTEPS. LAST_SUBSTEP is the cheaper production default
        # for the stable contacts expected during a grasp.
        collection = 1
        force_threshold_n = 0.5
        # DG5F semantic mapping: finger 1=thumb, 2=index, 3=middle.
        fingertip_names = ["thumb", "index", "middle"]
        # Add this amount once per selected fingertip over the force threshold:
        # 0, x, 2x or 3x at each control step with the default selection. Gated
        # by reward_enabled below rather than by `enabled`, so switching the
        # force tensor on for the observation does not quietly change the
        # reward function too.
        reward_per_finger = 0.05
        reward_enabled = False
        # Append one 3D contact-force vector per selected fingertip to the
        # observation vector, rotated into the palm frame like the fingertip
        # positions and the cube pose already are. Needs `enabled`, which is
        # what acquires the PhysX force tensor. Off by default, so every
        # existing experiment keeps its 108D observation and its checkpoints
        # stay loadable. train.py --contact-observations turns both on.
        observe_fingertip_forces = False
        # Asymmetric actor-critic: the actor stays blind while the critic also
        # reads the fingertip contact forces. Privileged information is legal
        # in the critic because the critic is discarded at deployment, and a
        # value function that can see contact explains the returns a blind
        # actor cannot, which lowers the advantage noise the actor learns from.
        critic_observes_fingertip_forces = False
        # The palm-frame force is divided by this before it reaches the policy,
        # so a firm grasp lands near unit scale instead of tens of newtons.
        observation_force_scale_n = 10.0
        # Symmetric per-component clip applied after that scaling. A collision
        # spike is worth several hundred newtons and would otherwise swamp the
        # 108 inputs it sits beside.
        observation_clip = 5.0

    class rewards:
        # Same weights and Gaussian widths as no_object_reward. Arm and hand
        # terms sum to a maximum per-step reward of 1.92.
        position_arm_weight = 0.8
        velocity_arm_weight = 0.2
        action_rate_arm_weight = 0.2
        position_arm_std_rad = 0.223607
        velocity_arm_std_rad_per_s = 1.0
        # Demonstration-aware action-delta tracking, per block. The error is
        # (a_t - a_{t-1}) - dq_demo * dt / action_scale because policy actions
        # are dimensionless residuals rather than joint angles.
        action_rate_arm_std = 5

        position_hand_weight = 0.48
        velocity_hand_weight = 0.12
        action_rate_hand_weight = 0.12
        position_hand_std_rad = 0.223607
        velocity_hand_std_rad_per_s = 1.0
        action_rate_hand_std = 5
        # Adaptive widths. Off by default, so every existing run reproduces.
        # When on, position_* and action_rate_* sigmas follow a slow average of
        # their own MSE and hold the term near adaptive_sigma_target_reward, so
        # it keeps a live gradient however good the policy gets. This is the
        # sharpening ladder done continuously: the fixed widths above stop
        # paying once the policy passes them, which is why mean_reward kept
        # rising while the deployment score fell after iteration 1500.
        adaptive_sigma_enabled = False
        adaptive_sigma_target_reward = 0.6
        adaptive_sigma_decay = 0.999
        # Floors, below which a term stops demanding improvement. Set at the
        # smoothest policy trained here (2026-08-26_sharpen_sigma), because
        # asking for better than that has never been necessary and a width that
        # chases the policy indefinitely is how blind_sharp traded its grasp
        # away for tracking it did not need.
        adaptive_sigma_position_arm_floor = 0.0061
        adaptive_sigma_position_hand_floor = 0.0103
        adaptive_sigma_action_rate_arm_floor = 0.0076
        adaptive_sigma_action_rate_hand_floor = 0.0076

        # Cube position tracking rewards
        object_scale = 1
        object_position_weight = 0.8 * object_scale
        object_orientation_weight = 0.2 * object_scale
        object_position_std_m = 0.05
        object_orientation_std_rad = 0.5

        # Fingertip-object distance rewards
        fingertip_object_distance_weight = 0 * 0.2 * object_scale
        fingertip_object_distance_std_m = 0.04
        fingertip_object_distance_names = ["thumb", "index", "middle"]

    class termination:
        enabled = True
        arm_position_threshold_rad = 0.35
        hand_position_threshold_rad = 1.35
        # End an episode when the physical cube remains farther than this
        # Euclidean center distance from the demonstrated cube target.
        object_position_enabled = True
        object_position_threshold_m = 0.07
        grace_steps = 5


class SimToolRealTrainCfg(BaseTrainCfg):
    """AnimRL Walk/Cartwheel PPO values, reserved for the next milestone."""

    class runner(BaseTrainCfg.runner):
        experiment_name = "simtoolreal"
        run_name = "new_demo_lr5em5_ec1em3_lenenv360_numenv4096"
        max_iterations = 9000
