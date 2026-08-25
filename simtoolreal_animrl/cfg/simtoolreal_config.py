"""UR5e + DG5F motion-imitation configurations."""

from .base_config import BaseEnvCfg, BaseTrainCfg


class SimToolRealCfg(BaseEnvCfg):
    class env(BaseEnvCfg.env):
        num_envs = 4096
        episode_length = 100
        num_actions = 6
        # 6 normalized arm q + 6 previous physical arm targets + 6 arm dq
        # + 1 normalized reference phase. The hand follows the demonstration
        # directly and is neither observed nor controlled by the policy.
        num_observations = 19
        num_privileged_obs = None
        reference_state_initialization = True
        reference_init_distribution = "uniform"

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

    class motion:
        file = "demonstrations/demo_20260727_152551_335339_60hz.npz"
        frequency_hz = 60.0

    class control:
        control_type = "P"
        decimation = 1
        action_parameterization = "animrl_residual"
        scale_joint_target = 0.25
        clip_joint_target = 100.0

    class rewards:
        position_weight = 0.8
        velocity_weight = 0.2
        position_std_rad = 0.223607
        velocity_std_rad_per_s = 0.3
        # Action-rate regularization on a_t - a_{t-1}. The processed
        # demonstration never exceeds an RMS action delta of 0.012, so a
        # 0.05 std leaves perfect tracking essentially unpenalized while
        # suppressing the high-frequency chatter of an unregularized policy.
        action_rate_weight = 0.2
        action_rate_std = 5

    class termination:
        enabled = True
        arm_position_threshold_rad = 0.35
        grace_steps = 5


class SimToolRealTrainCfg(BaseTrainCfg):
    """AnimRL Walk/Cartwheel PPO values, reserved for the next milestone."""

    class runner(BaseTrainCfg.runner):
        experiment_name = "simtoolreal"
        run_name = "llcfix_animrl"
        max_iterations = 3000
