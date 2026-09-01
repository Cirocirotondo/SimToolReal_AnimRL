import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from simtoolreal_animrl.cfg import (
    SimToolRealCfg,
    SimToolRealTrainCfg,
    config_to_dict,
    update_config_from_dict,
)
from simtoolreal_animrl.envs.contact import (
    fingertip_contact_diagnostics,
    fingertip_force_norms,
)
from simtoolreal_animrl.envs.proximity import fingertip_cuboid_proximity
from simtoolreal_animrl.envs.rsi import resolve_rsi_settings, sample_rsi_indices
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

    @staticmethod
    def _checkpoint_runner():
        """Build the checkpoint-relevant part of PPO without an Isaac env."""
        runner = PPO.__new__(PPO)
        runner.device = torch.device("cpu")
        runner.policy = Policy(3, 2, [4], "elu", -0.4)
        runner.value = Value(3, [4], "elu")
        runner.optimizer = torch.optim.Adam(
            list(runner.policy.parameters()) + list(runner.value.parameters()),
            lr=1.0e-4,
        )
        runner.normalize_observation = True
        runner.actor_obs_normalizer = EmpiricalNormalization(3)
        runner.critic_obs_normalizer = EmpiricalNormalization(3)
        runner.total_timesteps = 0
        runner.total_time_s = 0.0
        runner.best_evaluation_score = -float("inf")
        runner.best_evaluation_iteration = -1
        return runner

    def test_policy_initialization_leaves_critic_and_optimizer_fresh(self):
        source = self._checkpoint_runner()
        with torch.no_grad():
            for parameter in source.policy.parameters():
                parameter.fill_(0.25)
            for parameter in source.value.parameters():
                parameter.fill_(0.75)
        source.actor_obs_normalizer(torch.randn(7, 3))
        source.critic_obs_normalizer(torch.randn(7, 3))
        # Populate Adam's state so accidentally loading it is observable.
        source.optimizer.zero_grad()
        sum(parameter.sum() for parameter in source.policy.parameters()).backward()
        source.optimizer.step()
        infos = {
            "total_timesteps": 12345,
            "total_time_s": 67.0,
            "best_evaluation_score": 0.9,
            "best_evaluation_iteration": 42,
            "actor_normalizer_count": source.actor_obs_normalizer.count,
            "critic_normalizer_count": source.critic_obs_normalizer.count,
        }

        target = self._checkpoint_runner()
        untouched_value = {
            name: tensor.clone() for name, tensor in target.value.state_dict().items()
        }
        with TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "source.pt"
            source.save(checkpoint, infos=infos)
            returned_infos = target.initialize_policy(checkpoint)

        self.assertEqual(returned_infos, infos)
        for name, tensor in source.policy.state_dict().items():
            self.assertTrue(torch.equal(target.policy.state_dict()[name], tensor))
        for name, tensor in untouched_value.items():
            self.assertTrue(torch.equal(target.value.state_dict()[name], tensor))
        self.assertEqual(target.optimizer.state, {})
        self.assertEqual(target.total_timesteps, 0)
        self.assertEqual(target.best_evaluation_iteration, -1)
        self.assertEqual(
            target.actor_obs_normalizer.count,
            source.actor_obs_normalizer.count,
        )

    def test_training_config_matches_animrl_cartwheel(self):
        train = self.train_cfg
        self.assertEqual(train.runner.num_steps_per_env, 24)
        self.assertEqual(train.algorithm.num_learning_epochs, 5)
        self.assertEqual(train.algorithm.num_mini_batches, 4)
        self.assertEqual(train.algorithm.learning_rate, 1.0e-4)
        self.assertEqual(train.algorithm.schedule, "fixed")
        self.assertTrue(train.runner.tensorboard)
        self.assertFalse(train.runner.record_video)
        self.assertEqual(train.runner.record_video_interval, 500)
        self.assertEqual(train.runner.record_video_duration_s, 10.0)
        self.assertEqual(train.runner.record_video_fps, 60)
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

    def test_pregrasp_rsi_mixture_respects_ranges_and_probability(self):
        settings = resolve_rsi_settings(self.env_cfg.env, 1107)
        self.assertEqual(settings, ("pregrasp_mixture", 830, 740, 0.20))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(123)
        indices = sample_rsi_indices(
            100000, torch.device("cpu"), *settings, generator=generator
        )
        self.assertGreaterEqual(int(indices.min()), 0)
        self.assertLessEqual(int(indices.max()), 830)
        early_fraction = float((indices < 740).float().mean())
        self.assertAlmostEqual(early_fraction, 0.20, delta=0.01)
        self.assertTrue(bool((indices[indices >= 740] <= 830).all()))

    def test_evaluation_cohorts_are_repeatable(self):
        class Reference:
            last_index = 1107

        class Env:
            num_envs = 7
            device = torch.device("cpu")
            reference = Reference()
            rsi_distribution = "pregrasp_mixture"
            rsi_max_start_index = 830
            rsi_pregrasp_start_index = 740
            rsi_early_probability = 0.20

        first = DeterministicEvaluator(Env(), 100, 123, [0.0, 0.5, 1.0])
        second = DeterministicEvaluator(Env(), 100, 123, [0.0, 0.5, 1.0])
        self.assertEqual(
            first.fixed_indices.tolist(), [0, 553, 830, 0, 553, 830, 0]
        )
        self.assertTrue(
            torch.equal(first.uniform_indices, second.uniform_indices)
        )
        self.assertLessEqual(int(first.uniform_indices.max()), 830)

    def test_final_iteration_is_always_evaluated(self):
        """The last update is ranked even when it is off the interval."""

        class Reference:
            last_index = 1107

        class Env:
            num_envs = 4
            device = torch.device("cpu")
            reference = Reference()
            rsi_distribution = "pregrasp_mixture"
            rsi_max_start_index = 830
            rsi_pregrasp_start_index = 740
            rsi_early_probability = 0.20

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
            "fingertip_object_distance_reward": torch.zeros(num_envs),
            "fingertip_object_distance_m": torch.zeros(num_envs),
            "object_position_error_m": torch.zeros(num_envs),
            "object_orientation_error_rad": torch.zeros(num_envs),
            "object_com_height_m": 0.55 * ones,
            "object_com_lift_m": torch.zeros(num_envs),
            "fingertip_contact_reward": torch.zeros(num_envs),
            "fingertip_contact_fraction": torch.zeros(num_envs),
            "mean_fingertip_contact_force_n": torch.zeros(num_envs),
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
            rsi_distribution = "pregrasp_mixture"
            rsi_max_start_index = 830
            rsi_pregrasp_start_index = 740
            rsi_early_probability = 0.20

            def __init__(self, cfg):
                self.cfg = cfg
                self.cube_position = torch.tensor(
                    [[0.0, 0.0, 0.55]] * self.num_envs
                )

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
        self.assertAlmostEqual(
            stats["mean_peak_object_com_height_m"], 0.55, places=6
        )
        self.assertAlmostEqual(stats["max_peak_object_com_lift_m"], 0.0, places=6)
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

    def test_contact_reward_configuration_round_trip(self):
        cfg = SimToolRealCfg()
        self.assertFalse(cfg.contact.enabled)
        self.assertEqual(cfg.contact.collection, 1)
        self.assertEqual(
            cfg.contact.fingertip_names, ["thumb", "index", "middle"]
        )
        self.assertGreater(cfg.contact.force_threshold_n, 0.0)
        self.assertGreater(cfg.contact.reward_per_finger, 0.0)

        snapshot = config_to_dict(cfg)
        restored = SimToolRealCfg()
        restored.contact.enabled = True
        update_config_from_dict(restored, snapshot)
        self.assertFalse(restored.contact.enabled)
        self.assertEqual(
            restored.contact.fingertip_names,
            snapshot["contact"]["fingertip_names"],
        )

    def test_fingertip_cuboid_proximity_is_surface_based_and_phase_gated(self):
        half_extents = torch.tensor([0.075, 0.025, 0.025])
        # Along +z these points are respectively on the surface, 4 cm away and
        # 8 cm away. Duplicate them across two environments to isolate gating.
        points = torch.tensor(
            [
                [[0.0, 0.0, 0.025], [0.0, 0.0, 0.065], [0.0, 0.0, 0.105]],
                [[0.0, 0.0, 0.025], [0.0, 0.0, 0.065], [0.0, 0.0, 0.105]],
            ]
        )
        reward, mean_distance, per_finger = fingertip_cuboid_proximity(
            points,
            half_extents,
            std_m=0.04,
            active=torch.tensor([False, True]),
        )
        self.assertEqual(float(reward[0]), 0.0)
        expected = torch.exp(
            -torch.tensor([0.0, 0.04, 0.08]).square() / (2.0 * 0.04**2)
        ).mean()
        self.assertTrue(torch.allclose(reward[1], expected))
        self.assertTrue(
            torch.allclose(mean_distance, torch.full((2,), 0.04))
        )
        # The per-finger distances are what the mean above averages away.
        self.assertEqual(per_finger.shape, (2, 3))
        self.assertTrue(
            torch.allclose(
                per_finger, torch.tensor([[0.0, 0.04, 0.08]]).expand(2, 3)
            )
        )
        self.assertTrue(
            torch.allclose(per_finger.mean(dim=1), mean_distance)
        )

    def test_proximity_mean_hides_an_uneven_hand(self):
        """Two very different hands share one mean distance."""
        half_extents = torch.tensor([0.075, 0.025, 0.025])
        points = torch.tensor(
            [
                # Even: every finger 4 cm off the surface.
                [[0.0, 0.0, 0.065], [0.0, 0.0, 0.065], [0.0, 0.0, 0.065]],
                # Uneven: two fingers touching, one 12 cm away. Same mean.
                [[0.0, 0.0, 0.025], [0.0, 0.0, 0.025], [0.0, 0.0, 0.145]],
            ]
        )
        _, mean_distance, per_finger = fingertip_cuboid_proximity(
            points,
            half_extents,
            std_m=0.04,
            active=torch.tensor([True, True]),
        )
        self.assertTrue(
            torch.allclose(mean_distance, torch.full((2,), 0.04), atol=1e-6)
        )
        self.assertFalse(torch.allclose(per_finger[0], per_finger[1]))

    def test_fingertip_force_norms_keep_each_finger_separate(self):
        """The averaged metric hides which finger carries the load."""
        forces = torch.zeros(2, 5, 3)
        forces[0, 0, 0] = 3.0   # thumb pushes alone
        forces[1, 0, 0] = 1.0   # all three push equally
        forces[1, 2, 0] = 1.0
        forces[1, 4, 0] = 1.0
        selected = torch.tensor([0, 2, 4])

        per_finger = fingertip_force_norms(forces, selected)
        self.assertEqual(per_finger.shape, (2, 3))
        self.assertTrue(
            torch.allclose(
                per_finger,
                torch.tensor([[3.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
            )
        )
        # Both environments average to 1.0 N, which is exactly why the
        # per-finger series is worth plotting.
        _, _, mean_force = fingertip_contact_diagnostics(forces, selected, 0.5)
        self.assertTrue(torch.allclose(mean_force, torch.tensor([1.0, 1.0])))

    def test_fingertip_force_norms_use_the_vector_magnitude(self):
        forces = torch.zeros(1, 5, 3)
        forces[0, 1] = torch.tensor([3.0, 4.0, 0.0])
        per_finger = fingertip_force_norms(forces, torch.tensor([1]))
        self.assertTrue(torch.allclose(per_finger, torch.tensor([[5.0]])))

    def test_fingertip_contact_diagnostics_count_each_selected_finger(self):
        forces = torch.zeros(2, 5, 3)
        # Environment zero: thumb and middle exceed 0.5 N, index does not.
        forces[0, 0, 0] = 0.6
        forces[0, 2, 1] = 0.5
        forces[0, 4, 2] = 2.0
        # Environment one: all three selected fingers exceed the threshold.
        forces[1, 0, 0] = 1.0
        forces[1, 2, 0] = 1.0
        forces[1, 4, 0] = 1.0
        selected = torch.tensor([0, 2, 4])

        count, fraction, mean_force = fingertip_contact_diagnostics(
            forces, selected, 0.5
        )
        self.assertTrue(torch.equal(count, torch.tensor([2.0, 3.0])))
        self.assertTrue(
            torch.allclose(fraction, torch.tensor([2.0 / 3.0, 1.0]))
        )
        self.assertTrue(
            torch.allclose(mean_force, torch.tensor([3.1 / 3.0, 1.0]))
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
