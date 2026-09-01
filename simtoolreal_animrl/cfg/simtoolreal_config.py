"""UR5e + DG5F motion-imitation configurations."""

from .base_config import BaseEnvCfg, BaseTrainCfg


class SimToolRealCfg(BaseEnvCfg):
    class env(BaseEnvCfg.env):
        num_envs = 4096
        episode_length = 300
        num_actions = 26
        # Existing 79D proprioception, followed by: palm pose in the robot-base
        # frame (3+4), cube pose relative to palm (3+4), cube center relative
        # to five fingertips in the palm frame (15), and cube linear/angular
        # velocity relative to the palm (3+3).
        num_observations = 114
        num_privileged_obs = None
        reference_state_initialization = True
        # Uniform over every start the cube can survive. The 830 ceiling still
        # keeps resets out of the unstable post-grasp part of the 1108-frame
        # demonstration, but within it each frame is now equally likely. The
        # pregrasp_mixture alternative put 80% of resets in [740, 830], making
        # a pre-grasp frame 32x likelier than an approach frame and leaving the
        # approach at 3-8% episode coverage: enough to learn the grasp from an
        # RSI reset, not enough to reach it from frame 0 at deployment. The two
        # fields below are read only by "pregrasp_mixture" and are kept so the
        # skewed distribution can be restored without re-deriving its bounds.
        reference_init_distribution = "uniform"
        rsi_early_probability = 0.20
        rsi_pregrasp_start_index = 740
        rsi_max_start_index = 830

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
        file = "demonstrations/demo_20260727_152551_335339_60hz_cube.npz"
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
        # 0, x, 2x or 3x at each control step with the default selection.
        reward_per_finger = 0.05

    class rewards:
        # Arm and hand keep separate Gaussian terms so neither dilutes the
        # other's gradient. The hand weights are 0.6x the arm's, which is what
        # the handmult_0.6 run is named after. Robot terms sum to 1.92 and the
        # object pose terms add 1.0. Proximity adds at most 0.2 from frame 740,
        # for a maximum per-step reward of 3.12 while contact shaping is off.
        position_arm_weight = 0.8
        velocity_arm_weight = 0.2
        action_rate_arm_weight = 0.2
        position_arm_std_rad = 0.223607
        velocity_arm_std_rad_per_s = 1.0
        # Action-rate regularization on a_t - a_{t-1}, per block.
        action_rate_arm_std = 5

        position_hand_weight = 0.48
        velocity_hand_weight = 0.12
        action_rate_hand_weight = 0.12
        position_hand_std_rad = 0.223607
        velocity_hand_std_rad_per_s = 1.0
        action_rate_hand_std = 5

        # Full object-pose imitation. Position is the Euclidean center error in
        # metres; orientation is the sign-invariant quaternion geodesic angle
        # in radians. Cube velocity is deliberately not rewarded.
        object_position_weight = 0.8
        object_orientation_weight = 0.2
        object_position_std_m = 0.05
        object_orientation_std_rad = 0.5

        # Dense grasp shaping for thumb, index and middle fingertips. It uses
        # exact distance to the oriented cuboid surface and is gated off until
        # the reference reaches env.rsi_pregrasp_start_index.
        fingertip_object_distance_weight = 0.2
        fingertip_object_distance_std_m = 0.04
        fingertip_object_distance_names = ["thumb", "index", "middle"]

    class termination:
        enabled = True
        arm_position_threshold_rad = 0.35
        hand_position_threshold_rad = 1.35
        object_position_enabled = True
        # End an episode when the physical cube remains farther than this
        # Euclidean center distance from the demonstrated cube target.
        object_position_threshold_m = 0.05
        grace_steps = 5


class SimToolRealTrainCfg(BaseTrainCfg):
    """AnimRL Walk/Cartwheel PPO values, reserved for the next milestone."""

    class runner(BaseTrainCfg.runner):
        experiment_name = "simtoolreal"
        run_name = "handmult_0.6"
        max_iterations = 6000
