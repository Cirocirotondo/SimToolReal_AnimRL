"""A reward width that follows the policy, so the term never stops paying."""

import math
import unittest

from simtoolreal_animrl.envs.adaptive_sigma import AdaptiveSigma, sigma_for_target


def reward(mse, sigma):
    return math.exp(-mse / (2.0 * sigma * sigma))


class SigmaForTargetTest(unittest.TestCase):
    def test_the_width_it_returns_scores_the_target(self):
        for mse in (0.001, 0.05, 6.39):
            sigma = sigma_for_target(mse, 0.6)
            self.assertAlmostEqual(reward(mse, sigma), 0.6, places=6)

    def test_a_zero_or_broken_error_yields_no_width(self):
        self.assertIsNone(sigma_for_target(0.0, 0.6))
        self.assertIsNone(sigma_for_target(float("nan"), 0.6))

    def test_a_target_outside_the_open_unit_interval_is_rejected(self):
        for bad in (0.0, 1.0, -0.5, 2.0):
            with self.assertRaises(ValueError):
                sigma_for_target(0.05, bad)


class AdaptiveSigmaTest(unittest.TestCase):
    def test_it_tightens_as_the_policy_improves(self):
        tracker = AdaptiveSigma(initial=0.2236, floor=0.001, decay=0.0)
        wide = tracker.update(0.0048)
        tight = tracker.update(0.0013)
        self.assertLess(tight, wide)

    def test_the_term_keeps_a_live_gradient_at_every_scale(self):
        """The whole point: reward never saturates and never dies."""
        tracker = AdaptiveSigma(initial=5.0, floor=1e-4, decay=0.0)
        for mse in (6.39, 0.5, 0.074, 0.0043, 0.0013):
            sigma = tracker.update(mse)
            self.assertTrue(0.2 < reward(mse, sigma) < 0.9,
                            "reward %.3f at mse %.4g" % (reward(mse, sigma), mse))

    def test_a_fixed_width_saturates_where_the_adaptive_one_does_not(self):
        """blind_sharp's arm term measured 0.936 -- no gradient left."""
        fixed = reward(0.0335 ** 2, 0.10)
        tracker = AdaptiveSigma(initial=0.10, floor=1e-4, decay=0.0)
        adaptive = reward(0.0335 ** 2, tracker.update(0.0335 ** 2))
        self.assertGreater(fixed, 0.9)
        self.assertAlmostEqual(adaptive, 0.6, places=6)

    def test_it_never_tightens_below_the_floor(self):
        tracker = AdaptiveSigma(initial=0.10, floor=0.02, decay=0.0)
        for _ in range(50):
            tracker.update(1e-9)
        self.assertEqual(tracker.sigma, 0.02)

    def test_it_never_loosens_again(self):
        """A width that grew back would forgive a regression already paid for."""
        tracker = AdaptiveSigma(initial=0.10, floor=1e-4, decay=0.0)
        tight = tracker.update(0.0001)
        after_regression = tracker.update(10.0)
        self.assertEqual(after_regression, tight)

    def test_the_decay_smooths_a_single_bad_batch(self):
        slow = AdaptiveSigma(initial=0.10, floor=1e-4, decay=0.999)
        for _ in range(10):
            slow.update(0.0100)
        before = slow.sigma
        slow.update(1e-8)
        self.assertAlmostEqual(slow.sigma, before, places=4)

    def test_a_nonsense_batch_is_ignored_rather_than_fatal(self):
        tracker = AdaptiveSigma(initial=0.10, floor=1e-4, decay=0.9)
        tracker.update(0.01)
        before = tracker.sigma
        self.assertEqual(tracker.update(float("nan")), before)
        self.assertEqual(tracker.update(-1.0), before)

    def test_an_invalid_floor_or_decay_is_rejected(self):
        with self.assertRaises(ValueError):
            AdaptiveSigma(initial=0.1, floor=0.0)
        with self.assertRaises(ValueError):
            AdaptiveSigma(initial=0.1, floor=0.01, decay=1.0)


if __name__ == "__main__":
    unittest.main()
