"""Python configuration classes following cmm-25-a3-animrl."""

import inspect


class ABCConfig:
    """Recursively instantiate nested configuration classes."""

    def __init__(self) -> None:
        self._init_member_classes(self)

    @classmethod
    def _init_member_classes(cls, obj) -> None:
        for key in dir(obj):
            if key == "__class__":
                continue
            value = getattr(obj, key)
            if inspect.isclass(value):
                instance = value()
                setattr(obj, key, instance)
                cls._init_member_classes(instance)


class BaseEnvCfg(ABCConfig):
    """AnimRL-compatible environment configuration surface."""

    seed = 42

    class sim:
        # Robot-specific exception to AnimRL's generic 5 ms / decimation 4:
        # one control step must match one sample of the 60 Hz demonstration.
        dt = 1.0 / 60.0
        substeps = 2
        gravity = [0.0, 0.0, -9.81]
        use_gpu_pipeline = True

        class physx:
            use_gpu = True
            num_threads = 6
            solver_type = 1
            num_position_iterations = 8
            num_velocity_iterations = 0
            contact_offset = 0.002
            rest_offset = 0.0
            bounce_threshold_velocity = 0.2
            max_depenetration_velocity = 2.0
            max_gpu_contact_pairs = 8 * 1024 * 1024
            default_buffer_size_multiplier = 25.0
            contact_collection = 0

    class env:
        # Values from AnimRL WalkCfg/CartwheelCfg.
        num_envs = 4096
        episode_length = 100
        env_spacing = 2.0

        num_observations = None
        num_privileged_obs = None
        num_actions = None
        reference_state_initialization = True
        play = False
        debug = False

    class terrain:
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0

    class viewer:
        enable_viewer = False
        camera_position = [-1.8,-2.0, 1.5] # [1.8, 2.0, 1.5]
        camera_lookat = [0.0, 0.6, 0.75]


class BaseTrainCfg(ABCConfig):
    """AnimRL runner configuration retained for the future PPO milestone."""

    algorithm_name = "PPO"

    class policy:
        log_std_init = 0.0
        actor_hidden_dims = [512, 256]
        critic_hidden_dims = [512, 256]
        activation = "elu"

    class algorithm:
        value_loss_coef = 0.5
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        surrogate_coef = 1.0
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 1.0e-4
        schedule = "fixed"
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0
        bootstrap = True

    class runner:
        num_steps_per_env = 24
        max_iterations = 3000
        normalize_observation = True
        save_interval = 100
        record_gif = True
        record_gif_interval = 100
        record_iters = 10
        experiment_name = "simtoolreal"
        run_name = "llcfix_animrl"
        tensorboard = True
        tensorboard_flush_secs = 10
        evaluation_enabled = True
        evaluation_interval = 500
        evaluation_num_envs = 64
        evaluation_seed = 123
        evaluation_fixed_phases = [0.0, 0.25, 0.5, 0.75]
        # W&B remains a future optional backend; this milestone logs locally.
        wandb = False
        wandb_group = "default"
