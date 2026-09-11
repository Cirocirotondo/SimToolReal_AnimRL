"""Noisy sensors and a delayed command path, as real hardware has."""

import unittest

import torch

from simtoolreal_animrl.envs.sensing import (
    ActionDelay,
    add_observation_noise,
    sample_position_bias,
)


class ObservationNoiseTest(unittest.TestCase):
    def test_zero_noise_reproduces_every_run_so_far(self):
        q, dq = torch.zeros(8, 26), torch.ones(8, 26)
        out_q, out_dq = add_observation_noise(q, dq, 0.0, 0.0)
        torch.testing.assert_close(out_q, q)
        torch.testing.assert_close(out_dq, dq)

    def test_noise_has_the_requested_scale(self):
        q, dq = torch.zeros(4096, 26), torch.zeros(4096, 26)
        out_q, out_dq = add_observation_noise(q, dq, 0.005, 0.4243)
        self.assertAlmostEqual(float(out_q.std()), 0.005, places=3)
        self.assertAlmostEqual(float(out_dq.std()), 0.4243, places=2)

    def test_velocity_noise_is_far_larger_than_position_noise(self):
        """Differentiating a quantised encoder is where real noise lives."""
        q, dq = torch.zeros(2048, 26), torch.zeros(2048, 26)
        out_q, out_dq = add_observation_noise(q, dq, 0.005, 0.4243)
        self.assertGreater(float(out_dq.std()), 10.0 * float(out_q.std()))

    def test_the_bias_is_constant_across_a_call_not_resampled(self):
        bias = sample_position_bias(64, 26, 0.005, "cpu")
        q, dq = torch.zeros(64, 26), torch.zeros(64, 26)
        a, _ = add_observation_noise(q, dq, 0.0, 0.0, 0.005, bias=bias)
        b, _ = add_observation_noise(q, dq, 0.0, 0.0, 0.005, bias=bias)
        torch.testing.assert_close(a, b)

    def test_the_bias_differs_between_environments(self):
        bias = sample_position_bias(256, 26, 0.005, "cpu")
        self.assertGreater(len(set(bias[:, 0].tolist())), 200)
        self.assertLessEqual(float(bias.abs().max()), 0.005 + 1e-9)

    def test_no_bias_requested_means_no_bias_tensor(self):
        self.assertIsNone(sample_position_bias(8, 26, 0.0, "cpu"))

    def test_the_caller_s_tensors_are_untouched(self):
        q, dq = torch.zeros(8, 26), torch.ones(8, 26)
        original = q.clone()
        add_observation_noise(q, dq, 0.01, 0.5)
        torch.testing.assert_close(q, original)


class ActionDelayTest(unittest.TestCase):
    def test_zero_delay_passes_actions_straight_through(self):
        delay = ActionDelay(8, 26, 0, "cpu")
        a = torch.randn(8, 26)
        torch.testing.assert_close(delay(a), a)

    def test_a_delayed_environment_receives_an_older_action(self):
        delay = ActionDelay(2, 3, 2, "cpu")
        delay.steps = torch.tensor([0, 2])
        first = torch.tensor([[1.0, 1, 1], [1.0, 1, 1]])
        second = torch.tensor([[2.0, 2, 2], [2.0, 2, 2]])
        third = torch.tensor([[3.0, 3, 3], [3.0, 3, 3]])
        delay(first)
        delay(second)
        out = delay(third)
        self.assertAlmostEqual(float(out[0, 0]), 3.0)  # no delay
        self.assertAlmostEqual(float(out[1, 0]), 1.0)  # two steps behind

    def test_the_delay_is_fixed_per_environment_not_resampled(self):
        """A delay that changes every step is jitter, which averages out."""
        delay = ActionDelay(512, 26, 2, "cpu")
        before = delay.steps.clone()
        for _ in range(5):
            delay(torch.randn(512, 26))
        torch.testing.assert_close(delay.steps, before)

    def test_delays_span_the_configured_range(self):
        delay = ActionDelay(4096, 26, 2, "cpu")
        self.assertEqual(set(delay.steps.unique().tolist()), {0, 1, 2})

    def test_a_reset_does_not_leak_the_previous_episode_s_commands(self):
        delay = ActionDelay(4, 3, 2, "cpu")
        delay(torch.full((4, 3), 9.0))
        seed = torch.zeros(2, 3)
        delay.reset(torch.tensor([0, 1]), seed)
        out = delay(torch.full((4, 3), 1.0))
        self.assertEqual(float(delay.buffer[:, 0].abs().max()), 1.0)

    def test_a_negative_delay_is_rejected(self):
        with self.assertRaises(ValueError):
            ActionDelay(8, 26, -1, "cpu")


if __name__ == "__main__":
    unittest.main()
