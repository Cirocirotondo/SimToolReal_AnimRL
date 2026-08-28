import unittest

import numpy as np
import torch

from simtoolreal_animrl.cfg import (
    SimToolRealCfg,
    SimToolRealTrainCfg,
    config_to_dict,
    update_config_from_dict,
)
from simtoolreal_animrl.runners.modules import (
    EmpiricalNormalization,
    Policy,
    Value,
)
from simtoolreal_animrl.runners.evaluation import (
    DeterministicEvaluator,
    preserve_random_state,
)
from simtoolreal_animrl.runners.algorithms.ppo import PPO
from simtoolreal_animrl.runners.storage import RolloutStorage


class _NullRunner:
    """Stub runner: the evaluator only flips its train/eval mode."""

    def eval_mode(self):
        pass

    def train_mode(self):
        pass


class DivergenceGuardTest(unittest.TestCase):
    """The guard that aborts a run whose action std has run away.

    ``_divergence_reason`` only reads ``self.cfg`` and ``self.divergence_streak``,
    so it is exercised on a stub rather than a full PPO runner.
    """

    def setUp(self):
        self.guard = PPO._divergence_reason.__get__(self._stub())

    def _stub(self, **overrides):
        cfg = SimToolRealTrainCfg().runner
        for name, value in overrides.items():
            setattr(cfg, name, value)

        class Stub:
            pass

        stub = Stub()
        stub.cfg = cfg
        stub.divergence_streak = 0
        self.stub = stub
        return stub

    @staticmethod
    def _stats(action_std=1.0, clipped=0.0, value_loss=1.0):
        return {
            "mean_action_std": action_std,
            "action_target_clipped_fraction": clipped,
            "value_loss": value_loss,
            "surrogate_loss": -0.003,
            "mean_reward": 1.9,
        }

    def test_healthy_run_never_aborts(self):
        # The 2026-08-26 runs peak at std 2.45 and never clip a target.
        for _ in range(50):
            self.assertIsNone(self.guard(self._stats(action_std=2.45)))
        self.assertEqual(self.stub.divergence_streak, 0)

    def test_action_std_aborts_only_after_patience(self):
        patience = int(self.stub.cfg.abort_patience)
        for _ in range(patience - 1):
            self.assertIsNone(self.guard(self._stats(action_std=25.0)))
        reason = self.guard(self._stats(action_std=25.0))
        self.assertIsNotNone(reason)
        self.assertIn("mean_action_std", reason)

    def test_a_healthy_iteration_resets_the_streak(self):
        self.guard(self._stats(action_std=25.0))
        self.guard(self._stats(action_std=25.0))
        self.assertIsNone(self.guard(self._stats(action_std=2.0)))
        self.assertEqual(self.stub.divergence_streak, 0)
        # The streak restarts from scratch rather than resuming.
        self.assertIsNone(self.guard(self._stats(action_std=25.0)))

    def test_clipped_fraction_threshold(self):
        for _ in range(int(self.stub.cfg.abort_patience)):
            reason = self.guard(self._stats(clipped=0.85))
        self.assertIn("action_target_clipped_fraction", reason)
        # Heavy but sub-threshold clipping is tolerated.
        guard = PPO._divergence_reason.__get__(self._stub())
        for _ in range(50):
            self.assertIsNone(guard(self._stats(clipped=0.75)))

    def test_non_finite_metrics_abort(self):
        """A NaN std would pass every ``>`` test, so it is checked explicitly."""
        for _ in range(int(self.stub.cfg.abort_patience)):
            reason = self.guard(self._stats(action_std=float("nan")))
        self.assertIn("not finite", reason)

    def test_guard_can_be_disabled(self):
        guard = PPO._divergence_reason.__get__(
            self._stub(abort_on_divergence=False)
        )
        for _ in range(50):
            self.assertIsNone(guard(self._stats(action_std=1.0e6)))


class RunnerModulesTest(unittest.TestCase):
    def setUp(self):
        self.env_cfg = SimToolRealCfg()
        self.train_cfg = SimToolRealTrainCfg()

    def test_animrl_network_state_dict_contract(self):
        policy = Policy(
            self.env_cfg.env.num_observations,
            self.env_cfg.env.num_actions,
            self.train_cfg.policy.actor_hidden_dims,
            self.train_cfg.policy.activation,
            self.train_cfg.policy.log_std_init,
        )
        value = Value(
            self.env_cfg.env.num_observations,
            self.train_cfg.policy.critic_hidden_dims,
            self.train_cfg.policy.activation,
        )
        self.assertEqual(
            set(policy.state_dict()),
            {
                "log_std",
                "policy_latent_net.0.weight",
                "policy_latent_net.0.bias",
                "policy_latent_net.2.weight",
                "policy_latent_net.2.bias",
                "action_mean_net.weight",
                "action_mean_net.bias",
            },
        )
        self.assertEqual(
            set(value.state_dict()),
            {
                "value.0.weight",
                "value.0.bias",
                "value.2.weight",
                "value.2.bias",
                "value.4.weight",
                "value.4.bias",
            },
        )

        observations = torch.zeros(8, self.env_cfg.env.num_observations)
        actions, log_prob = policy.act_and_log_prob(observations)
        self.assertEqual(actions.shape, (8, self.env_cfg.env.num_actions))
        self.assertEqual(log_prob.shape, (8,))
        self.assertEqual(value(observations).shape, (8, 1))

    def test_normalizer_and_rollout_shapes(self):
        normalizer = EmpiricalNormalization(19)
        normalized = normalizer(torch.randn(16, 19))
        self.assertEqual(normalized.shape, (16, 19))
        self.assertEqual(
            set(normalizer.state_dict()), {"_mean", "_var", "_std"}
        )

        storage = RolloutStorage(8, 24, 19, 19, 6)
        self.assertEqual(storage.observations.shape, (24, 8, 19))
        self.assertEqual(storage.actions.shape, (24, 8, 6))
        self.assertEqual(storage.values.shape, (24, 8, 1))

    def test_training_config_matches_animrl_cartwheel(self):
        train = self.train_cfg
        self.assertEqual(train.runner.num_steps_per_env, 24)
        self.assertEqual(train.algorithm.num_learning_epochs, 5)
        self.assertEqual(train.algorithm.num_mini_batches, 4)
        self.assertEqual(train.algorithm.learning_rate, 1.0e-4)
        self.assertEqual(train.algorithm.schedule, "fixed")
        self.assertTrue(train.runner.tensorboard)
        self.assertFalse(train.runner.wandb)
        self.assertTrue(train.runner.evaluation_enabled)
        self.assertEqual(train.runner.evaluation_interval, 500)
        self.assertEqual(train.runner.evaluation_num_envs, 64)
        self.assertEqual(
            train.runner.evaluation_fixed_phases, [0.0, 0.25, 0.5, 0.75]
        )
        self.assertEqual(train.policy.actor_hidden_dims, [512, 256])
        self.assertEqual(train.policy.critic_hidden_dims, [512, 256])

        snapshot = config_to_dict(train)
        self.assertEqual(snapshot["algorithm"]["learning_rate"], 1.0e-4)
        self.assertEqual(snapshot["runner"]["num_steps_per_env"], 24)

        restored = SimToolRealTrainCfg()
        restored.algorithm.learning_rate = 9.9
        update_config_from_dict(restored, snapshot)
        self.assertEqual(restored.algorithm.learning_rate, 1.0e-4)
        with self.assertRaises(KeyError):
            update_config_from_dict(restored, {"unknown": 1})

    def test_evaluation_cohorts_are_repeatable(self):
        class Reference:
            last_index = 1107

        class Env:
            num_envs = 7
            device = torch.device("cpu")
            reference = Reference()

        first = DeterministicEvaluator(Env(), 100, 123, [0.0, 0.5, 1.0])
        second = DeterministicEvaluator(Env(), 100, 123, [0.0, 0.5, 1.0])
        self.assertEqual(
            first.fixed_indices.tolist(), [0, 553, 1106, 0, 553, 1106, 0]
        )
        self.assertTrue(
            torch.equal(first.uniform_indices, second.uniform_indices)
        )

    def test_final_iteration_is_always_evaluated(self):
        """The last update is ranked even when it is off the interval."""

        class Reference:
            last_index = 1107

        class Env:
            num_envs = 4
            device = torch.device("cpu")
            reference = Reference()

        evaluator = DeterministicEvaluator(Env(), 500, 123, [0.0, 0.5])
        calls = []
        evaluator._evaluate_suite = lambda runner, indices: calls.append(
            indices
        ) or {"position_score": 1.0}

        # Off-interval iterations stay skipped.
        self.assertIsNone(evaluator(2999, runner=None))
        self.assertEqual(calls, [])

        # The same iteration marked as final is evaluated.
        stats = evaluator(2999, runner=_NullRunner(), is_final=True)
        self.assertEqual(stats["evaluation_score"], 1.0)
        self.assertEqual(len(calls), 2)  # fixed and uniform suites

    def _score_with_object_weights(self, position_weight, orientation_weight):
        """Run one evaluation step over a stub env and return its statistics."""
        cfg = SimToolRealCfg()
        cfg.rewards.object_position_weight = position_weight
        cfg.rewards.object_orientation_weight = orientation_weight

        num_envs = 4
        ones = torch.ones(num_envs)
        infos = {
            "position_reward": 0.6 * ones,
            "velocity_reward": ones,
            "action_rate_reward": ones,
            "rms_action_rate": torch.zeros(num_envs),
            "hand_position_reward": 0.4 * ones,
            "hand_velocity_reward": ones,
            "hand_action_rate_reward": ones,
            # Perfect position, worst orientation: the two object terms differ
            # so a mis-weighted mixture would not go unnoticed.
            "object_position_reward": ones.clone(),
            "object_orientation_reward": torch.zeros(num_envs),
            "object_position_error_m": torch.zeros(num_envs),
            "object_orientation_error_rad": torch.zeros(num_envs),
            "rms_hand_position_error": torch.zeros(num_envs),
            "max_abs_hand_position_error": torch.zeros(num_envs),
            "rms_position_error": torch.zeros(num_envs),
            "rms_velocity_error": torch.zeros(num_envs),
            "max_abs_arm_position_error": torch.zeros(num_envs),
            "early_termination": torch.zeros(num_envs, dtype=torch.bool),
            "time_outs": torch.ones(num_envs, dtype=torch.bool),
        }

        class Gym:
            def refresh_dof_state_tensor(self, sim):
                pass

        class Reference:
            last_index = 1107

        class Env:
            num_envs = 4
            num_actions = 2
            max_episode_length = 1
            device = torch.device("cpu")
            reference = Reference()
            gym = Gym()
            sim = None
            action_scales = torch.ones(2)
            action_target_clip = 100.0

            def __init__(self, cfg):
                self.cfg = cfg

            def reset_idx(self, env_ids, indices):
                pass

            def compute_observations(self):
                pass

            def get_observations(self):
                return torch.zeros(self.num_envs, 3)

            def step(self, actions):
                return (
                    torch.zeros(self.num_envs, 3),
                    None,
                    ones.clone(),
                    torch.ones(self.num_envs, dtype=torch.bool),
                    infos,
                )

        class Policy:
            def act_inference(self, observations):
                return torch.zeros(observations.shape[0], 2)

        class Runner(_NullRunner):
            policy = Policy()

            @staticmethod
            def actor_obs_normalizer(observations):
                return observations

        evaluator = DeterministicEvaluator(Env(cfg), 100, 123, [0.0])
        return evaluator._evaluate_suite(Runner(), evaluator.fixed_indices)

    def test_object_pose_score_uses_configured_weights(self):
        stats = self._score_with_object_weights(0.8, 0.2)
        self.assertAlmostEqual(stats["mean_object_pose_score"], 0.8, places=6)
        # 0.5 * (0.5 * (0.6 + 0.4) + 0.8), with no early terminations.
        self.assertAlmostEqual(stats["position_score"], 0.65, places=6)

    def test_zero_object_weights_score_on_robot_pose_alone(self):
        """Disabling the object reward must not put a NaN in the score.

        A NaN would compare False against every best score in PPO, so no best
        checkpoint would ever be written for the run.
        """
        stats = self._score_with_object_weights(0.0, 0.0)
        self.assertEqual(stats["mean_object_pose_score"], 0.0)
        # Robot pose alone, on the same 0-1 scale: 0.5 * (0.6 + 0.4).
        self.assertAlmostEqual(stats["position_score"], 0.5, places=6)
        for name, value in stats.items():
            if isinstance(value, float):
                self.assertFalse(
                    np.isnan(value), "{} is NaN".format(name)
                )

    def test_evaluation_preserves_torch_and_numpy_rng(self):
        torch.manual_seed(17)
        np.random.seed(17)
        expected_torch = torch.rand(4)
        expected_numpy = np.random.rand(4)

        torch.manual_seed(17)
        np.random.seed(17)
        with preserve_random_state():
            torch.rand(20)
            np.random.rand(20)
        self.assertTrue(torch.equal(torch.rand(4), expected_torch))
        self.assertTrue(np.array_equal(np.random.rand(4), expected_numpy))


if __name__ == "__main__":
    unittest.main()
