"""UR5e + DG5F motion-imitation configurations."""

from .base_config import BaseEnvCfg, BaseTrainCfg


class SimToolRealCfg(BaseEnvCfg):
    class env(BaseEnvCfg.env):
        num_envs = 256
        # Match the 2026-08-28 no_object_reward reference run.
        episode_length = 360
        num_actions = 26
        # 79D proprioception, followed by palm pose in the robot-base frame
        # (3 + 6), five fingertip positions relative to the palm (15), and the
        # cuboid's orientation/centre relative to the palm (6 + 3).
        #
        # Both rotations are the continuous 6D representation rather than
        # quaternions. This is 112 where every run before the object-centric
        # reference was 108, so no earlier checkpoint loads into it -- which is
        # fine, because every run here is trained from scratch.
        num_observations = 112
        num_privileged_obs = None
        reference_init_distribution = "pregrasp_mixture"
        rsi_early_probability = 0.20 # used only when reference_init_distribution = "pregrasp_mixture"
        # Perturbation applied to the reference pose at reset, in radians. The
        # real arm can never be placed exactly on a demonstration frame, so a
        # policy trained only on exact frames has never seen the states it will
        # actually start from. It also blurs the RSI distribution's edges: the
        # measured 54x jitter jump at reference frame 690 is a seam where a
        # barely-trained approach meets a heavily-trained grasp, and noisy
        # starts spread mass across that boundary instead of stacking it on one
        # side. Zero reproduces every run so far.
        rsi_position_noise_arm_rad = 0.0
        rsi_position_noise_hand_rad = 0.0
        # Velocities are differentiated encoder counts on hardware, never the
        # exact values the demonstration carries.
        rsi_velocity_noise_scale = 0.0
        rsi_pregrasp_start_index = 740 # proximity reward starts from this demonstration index
        rsi_max_start_index = 830

    class domain_randomization:
        # Per-environment physical variation, sampled once at creation.
        #
        # Added after a measured transfer failure: pg830_blind512_n256, the
        # ROUGHEST policy trained here, moved to MuJoCo well, while the 25x and
        # 47x smoother blind_quiet2 and adapt_sigma failed there with matching
        # PD gains in both simulators. The smooth policies grip at 1.6-2.9 N
        # where the rough one uses 6.3 N; that margin is a property of one
        # contact model, not of the task. A policy cannot fit a friction
        # coefficient that differs in every environment.
        #
        # Each value is a fractional spread about the nominal: 0.4 means
        # uniform in [0.6, 1.4]. Zero disables that parameter, and enabled =
        # False reproduces every run so far.
        enabled = False
        arm_stiffness_range = 0.0
        arm_damping_range = 0.0
        hand_stiffness_range = 0.0
        hand_damping_range = 0.0
        fingertip_friction_range = 0.0
        object_friction_range = 0.0
        object_mass_range = 0.0
        # Extra physical parameters, all multiplicative like the rest.
        table_friction_range = 0.0
        robot_link_mass_range = 0.0
        # External impulses. Domain randomisation varies the world's parameters
        # but leaves it deterministic -- nothing ever pushes the robot. A policy
        # that has only been disturbed by its own actions has no recovery
        # behaviour, and a real arm is knocked by cable drag while a real cube
        # is nudged by an imperfect placement.
        #
        # Probability is per environment per control step, so 0.02 at 60 Hz is
        # roughly one push per environment per second. Sparse on purpose: a
        # continuous push is a force field the policy learns to lean against.
        robot_impulse_probability = 0.0
        robot_impulse_n = 0.0
        # Deliberately light. The cube is 0.2 kg, so 1 N for one 60 Hz step is
        # about 0.08 m/s -- enough to require a correction, not enough to throw
        # the object out of the hand.
        object_impulse_probability = 0.0
        object_impulse_n = 0.0
        # Feed the sampled multipliers to the critic. Only possible on a run
        # trained from scratch: it widens the critic input, so no existing value
        # network can be warm-started into it. Without this the value function
        # sees identical observations from environments with different friction
        # and must average over outcomes it cannot explain, and that unexplained
        # variance lands in the advantages the actor learns from.
        critic_observes_parameters = False
        # Sensor realism. The policy reads exact joint angles and velocities
        # here; hardware reads encoder counts and a differentiated velocity.
        # Beyond robustness this suppresses chatter for a reason rather than by
        # decree: a high-gain reactive policy amplifies measurement noise into
        # action noise and is punished for it. Velocity takes the larger share
        # because differentiating a quantised position is where real noise lives.
        obs_q_noise_rad = 0.0
        obs_dq_noise_rad_s = 0.0
        # A constant per-joint offset standing in for an encoder zero error,
        # drawn once per environment: a bias the policy could average away over
        # a few steps would not be a bias.
        obs_q_bias_rad = 0.0
        # Control latency in whole control steps, drawn once per environment.
        # A command issued now reaches a real joint later, so an aggressive
        # corrector overshoots -- training with delay forces the low-gain
        # behaviour that survives the transfer. Fixed per environment rather
        # than per step, because per-step variation is jitter and averages out.
        action_delay_max_steps = 0

    class object_randomization:
        # Where the cuboid may be, per episode. The reference is retargeted for
        # each sampled pose offline; see scripts/build_transform_bank.py.
        #
        # The yaw range is a (low, high) pair rather than a +/- scalar because
        # the arm's feasible envelope is genuinely not symmetric. Measured at
        # this translation, accepting a transform only if its whole clip solves,
        # stays off the joint limits, and stays under half the arm's joint
        # velocity limit:
        #
        #   yaw    -30   -15    0   +15   +30   +45   +60   +75   +90
        #   accept 69%   93%  100%  96%   84%   63%   49%   63%   75%
        #
        # A symmetric range would clamp to about +/-15 degrees and discard the
        # whole reachable positive side. Kept wide deliberately: every clip the
        # bank admits is feasible and trackable, the sampling density is simply
        # thinner above +30. If a run generalises poorly at high yaw, that
        # thinness is the first thing to suspect -- a single scalar score
        # averages it away.
        #
        # Velocity is what binds, not reach. An earlier range was chosen from a
        # sweep that checked only reachability and joint limits; it admitted
        # clips demanding 12.2 rad/s of a joint capped at pi.
        # Continuous per-episode sampling range. The offline bank is only the
        # nearest-neighbour source for arm IK; the physical cuboid uses the
        # exact continuously sampled transform.
        translation_x_min_m = -0.09
        translation_x_max_m = 0.09
        translation_y_min_m = 0.00
        translation_y_max_m = 0.15
        yaw_min_deg = -22.5
        yaw_max_deg = 45.0
        # Converts yaw separation to an equivalent Cartesian distance for the
        # nearest-bank lookup: 10 degrees is about 1.75 cm at 0.1 m.
        nearest_yaw_lever_arm_m = 0.10
        # Relative to the repository root. Built offline rather than at startup:
        # solving hundreds of clips takes minutes, and a transform must be
        # proven feasible over the clip's whole length before an episode is
        # allowed to start inside it.
        bank_path = "banks/stage1.pt"

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
        # Grown from 0.475 x 0.4 for the randomised cuboid pose. The bar's
        # in-plane half-diagonal is 0.079 m, so at +/-0.20 m of translation it
        # reaches 0.279 m from the nominal centre, which is itself offset from
        # the table centre by (-0.018, -0.020). The old top would have let the
        # bar overhang its edge and tip before the episode began.
        size_m = [0.75, 0.75, 0.3]
        surface_below_robot_base_m = 0.035
        friction = 0.5
        restitution = 0.0
        color = [0.82, 0.56, 0.35]

    class init_state:
        # Matches the independently verified Isaac Gym demonstration viewer.
        # At z=0 the demonstrated wrist intersects the ground around RSI 732.
        pos = [0.0, 0.6, 0.55]
        rot = [0.0, 0.0, 0.0, 1.0]
        # Fixed default pose used by the residual action mapping:
        # q_target = default + scale * action.
        #
        # The MEAN pose over the transform bank, not demonstration frame zero
        # and not the bank's mid-range.
        #
        # Retargeting for a cuboid anywhere in the sampled envelope drives the
        # arm far wider than the single recorded clip did -- wrist_1 alone spans
        # 5.3 rad across the bank against the demonstration's 0.59 -- so the
        # default matters more than it used to. A policy starts at action mean
        # zero, which puts the arm exactly here, so what matters is the distance
        # to a TYPICAL reference pose, not to the furthest one:
        #
        #   default          mean |a|   p50    p95    max
        #   demo frame 0        5.79    6.16   8.99  14.86
        #   bank mid-range      6.12    6.13   7.78  10.55
        #   bank mean           4.10    3.62   7.36  15.87
        #
        # Mid-range was tried first and is a trap: it minimises the worst case
        # and leaves the typical one slightly worse, pushing wrist_1's median
        # residual from 1.86 to 6.01. Every episode then terminated early
        # (measured: early fraction 1.00 against 0.59 with demo frame 0),
        # because an untrained policy sits 1.5 rad from every reference.
        #
        # Recompute this whenever the bank's sampling range changes.
        default_arm_joint_angles = [
            -1.7809368836,
            -1.1789974103,
            1.8940494728,
            -0.6477838573,
            1.1408713253,
            -3.6591747714,
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
        # --- object-centric tracking, the dominant terms -------------------
        # The hand as nine keypoints measured in the cuboid's frame: the palm
        # origin plus three points at a lever arm along its axes, and the five
        # fingertips. Palm inherits the arm's old 0.8 and fingertips the hand's
        # old 0.48, so the relative calibration this project already trusts
        # carries over instead of being invented fresh.
        palm_keypoint_weight = 0.80
        fingertip_keypoint_weight = 0.48
        # Sigmas are RMS keypoint distances in metres. 0.05 matches
        # object_position_std_m; 0.025 is the bar's short half-extent, because
        # the grasp needs more precision than the approach.
        palm_keypoint_std_m = 0.05
        fingertip_keypoint_std_m = 0.025
        # The lever arm is the exchange rate between a metre of palm position
        # error and a radian of palm rotation error: a point this far out moves
        # by L * theta. 0.1 m reproduces the 0.05 m / 0.5 rad ratio the object
        # terms already use. Must match the value the bank was built with.
        palm_lever_arm_m = 0.1

        # --- joint space, demoted to null-space selection -------------------
        # These no longer do the tracking. A 6-DOF arm has a null space for a
        # given palm pose and several IK branches; each finger has four joints
        # serving a 3D fingertip target, so five finger degrees of freedom are
        # otherwise unconstrained. Small but non-zero is what picks one
        # solution, which is all that is wanted from them now.
        position_arm_weight = 0.06
        velocity_arm_weight = 0.0
        action_rate_arm_weight = 0.2
        position_arm_std_rad = 0.223607
        velocity_arm_std_rad_per_s = 1.0
        # Pure command smoothness for the arm: unlike joint-space imitation,
        # this remains meaningful when the palm trajectory is retargeted.
        action_rate_arm_std = 5

        position_hand_weight = 0.05
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
        # How far a width may relax above its tightest value when the policy
        # regresses. 1.0 pins it to the all-time best, which sounds strict but
        # left adapt_sigma's arm term at ~0.06 reward for 1500 iterations with
        # no gradient pointing back; 1.5 gives 0.29 there instead.
        adaptive_sigma_slack = 1.5
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
        object_orientation_weight = 0.4 * object_scale
        object_position_std_m = 0.05
        object_orientation_std_rad = 0.5

        # Fingertip-object distance rewards
        fingertip_object_distance_weight = 0 * 0.2 * object_scale
        fingertip_object_distance_std_m = 0.04
        fingertip_object_distance_names = ["thumb", "index", "middle"]

    class termination:
        enabled = True
        # Task space, not joint space. The reward deliberately allows the arm to
        # leave the retargeted joint angles -- that null-space freedom is the
        # point of tracking a palm pose rather than six joints -- so the old
        # 0.35 rad joint threshold would have ended episodes the reward was
        # perfectly happy with.
        #
        # The error is the RMS over all four palm keypoints, so it catches a
        # palm that is in the right place but turned the wrong way.
        #
        # Calibrated rather than guessed. Perturbing the retargeted arm pose so
        # that exactly one joint sits at the old 0.35 rad limit moves the palm
        # keypoints by a median of 0.213 m (p25 0.156, p75 0.277). So 0.20 m
        # reproduces roughly the old criterion's strictness, holding the
        # early-termination curriculum constant while changing its definition.
        # The 0.12 m first written here sat at the 12th percentile and would
        # have terminated episodes far sooner than any run before it.
        palm_keypoint_threshold_m = 0.20
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
