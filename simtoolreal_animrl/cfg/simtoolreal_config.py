"""UR5e + DG5F motion-imitation configurations."""

from .base_config import BaseEnvCfg, BaseTrainCfg


class SimToolRealCfg(BaseEnvCfg):
    class env(BaseEnvCfg.env):
        num_envs = 4096
        episode_length = 500
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

    class motion:
        file = "demonstrations/demo_20260727_152551_335339_60hz.npz"
        frequency_hz = 60.0

    class control:
        control_type = "P"
        decimation = 1

    class rewards:
        position_weight = 0.8
        velocity_weight = 0.2
        position_std_rad = 0.223607
        velocity_std_rad_per_s = 7.071068

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
