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

    def test_a_regression_relaxes_the_width_only_up_to_the_slack(self):
        tracker = AdaptiveSigma(initial=0.10, floor=1e-4, decay=0.0, slack=1.5)
        tight = tracker.update(0.0001)
        after_regression = tracker.update(10.0)
        self.assertAlmostEqual(after_regression, tight * 1.5)

    def test_slack_one_reproduces_the_strict_ratchet(self):
        tracker = AdaptiveSigma(initial=0.10, floor=1e-4, decay=0.0, slack=1.0)
        tight = tracker.update(0.0001)
        self.assertAlmostEqual(tracker.update(10.0), tight)

    def test_a_regressing_term_keeps_a_usable_gradient(self):
        """adapt_sigma's failure: the arm width locked at 0.0268 while error
        drifted to 0.0747, leaving the term at almost no reward for 1500
        iterations, so nothing pointed the policy back."""
        strict = AdaptiveSigma(initial=0.10, floor=1e-4, decay=0.0, slack=1.0)
        slack = AdaptiveSigma(initial=0.10, floor=1e-4, decay=0.0, slack=1.5)
        best, regressed = 0.0318 ** 2, 0.0747 ** 2
        for tracker in (strict, slack):
            tracker.update(best)
        # 0.060 with the strict ratchet against 0.286 with slack: nearly 5x
        # the gradient to climb back on.
        self.assertLess(reward(regressed, strict.update(regressed)), 0.07)
        self.assertGreater(reward(regressed, slack.update(regressed)), 0.25)

    def test_a_slack_below_one_is_rejected(self):
        with self.assertRaises(ValueError):
            AdaptiveSigma(initial=0.1, floor=0.01, slack=0.9)

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

    def test_the_ladder_survives_a_save_and_resume(self):
        """A resumed run must not be handed back the width it started from.

        Rebuilding the widths from the configuration is not a neutral restart:
        the term jumps back to a sigma the policy has long passed, is pinned at
        1 again, and stops paying for the tracking it was holding. Measured on
        rot6d_adaptive, arm error went 0.044 -> 0.136 over the 1200 iterations
        after a resume, with nothing else changed.
        """
        trained = AdaptiveSigma(initial=5.0, floor=0.0076)
        for _ in range(50):
            trained.update(0.04)
        self.assertLess(trained.sigma, 0.25)

        resumed = AdaptiveSigma(initial=5.0, floor=0.0076)
        resumed.load_state(trained.state())
        self.assertAlmostEqual(resumed.sigma, trained.sigma, places=12)
        self.assertAlmostEqual(
            resumed.mean_squared_error, trained.mean_squared_error, places=12
        )
        # The two now agree on the next width as well, which is the point.
        self.assertAlmostEqual(
            resumed.update(0.04), trained.update(0.04), places=12
        )

    def test_a_checkpoint_without_a_ladder_leaves_the_width_alone(self):
        """Checkpoints predating the saved ladder must still resume."""
        tracker = AdaptiveSigma(initial=0.20, floor=0.0076)
        tracker.load_state(None)
        self.assertEqual(tracker.sigma, 0.20)
        tracker.load_state({"sigma": None, "mean_squared_error": None})
        self.assertEqual(tracker.sigma, 0.20)
        self.assertIsNone(tracker.mean_squared_error)

    def test_a_restored_width_still_respects_the_floor(self):
        tracker = AdaptiveSigma(initial=0.20, floor=0.05)
        tracker.load_state({"sigma": 0.001, "mean_squared_error": 1e-6})
        self.assertEqual(tracker.sigma, 0.05)


if __name__ == "__main__":
    unittest.main()
