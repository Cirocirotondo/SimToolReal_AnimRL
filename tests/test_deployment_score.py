"""The deployment score has to rank policies the way the robot would."""

import unittest

from simtoolreal_animrl.runners.deployment_score import (
    ANCHORS,
    deployment_score,
    deployment_terms,
    grasp_gate,
)


def metrics(et=0.0, lift=0.24, rate=0.0076, arm=0.0061, hand=0.0103):
    return {
        "evaluation_uniform_early_termination_fraction": et,
        "evaluation_uniform_mean_peak_object_com_lift_m": lift,
        "evaluation_fixed_mean_rms_action_rate": rate,
        "evaluation_fixed_mean_rms_position_error": arm,
        "evaluation_fixed_mean_rms_hand_position_error": hand,
    }


class GraspGateTest(unittest.TestCase):
    def test_a_policy_that_drops_the_cube_scores_zero_however_smooth(self):
        """The whole reason the grasp gates rather than adds."""
        perfect_but_dropping = metrics(et=0.0, lift=0.0)
        self.assertEqual(deployment_score(perfect_but_dropping), 0.0)

    def test_every_episode_terminating_early_scores_zero(self):
        self.assertEqual(deployment_score(metrics(et=1.0)), 0.0)

    def test_matching_the_demonstration_lift_opens_the_gate_fully(self):
        self.assertAlmostEqual(grasp_gate(0.0, 0.24), 1.0)

    def test_lifting_beyond_the_demonstration_does_not_score_above_one(self):
        self.assertAlmostEqual(grasp_gate(0.0, 0.50), 1.0)

    def test_the_gate_is_the_product_of_surviving_and_lifting(self):
        self.assertAlmostEqual(grasp_gate(0.25, 0.12), 0.75 * 0.5)


class ScoreTest(unittest.TestCase):
    def test_the_reference_best_policy_scores_one(self):
        self.assertAlmostEqual(deployment_score(metrics()), 1.0, places=6)

    def test_the_base_blind_run_scores_about_zero(self):
        """pg830_blind512 at 6500: lifts well, but 300x too rough to deploy."""
        base = metrics(et=0.0, lift=0.2137, rate=2.5275, arm=0.0692, hand=0.2124)
        self.assertLess(deployment_score(base), 0.01)

    def test_it_ranks_the_night_s_runs_in_the_order_we_believe(self):
        base = metrics(et=0.0, lift=0.2137, rate=2.5275, arm=0.0692, hand=0.2124)
        sharp = metrics(et=0.0, lift=0.1970, rate=0.2623, arm=0.0341, hand=0.0809)
        quiet = metrics(et=0.0, lift=0.1954, rate=0.1010, arm=0.0365, hand=0.0975)
        self.assertLess(deployment_score(base), deployment_score(sharp))
        self.assertLess(deployment_score(sharp), deployment_score(quiet))

    def test_smoothness_carries_the_largest_weight(self):
        """Vibration is what blocks the real robot, so a perfect smoothness
        term is worth more than a perfect arm or hand term alone."""
        worst = {name: worst for name, (worst, _) in ANCHORS.items()}
        only_smooth = metrics(
            rate=ANCHORS["action_rate"][1],
            arm=worst["arm_position_error"],
            hand=worst["hand_position_error"],
        )
        only_arm = metrics(
            rate=worst["action_rate"],
            arm=ANCHORS["arm_position_error"][1],
            hand=worst["hand_position_error"],
        )
        only_hand = metrics(
            rate=worst["action_rate"],
            arm=worst["arm_position_error"],
            hand=ANCHORS["hand_position_error"][1],
        )
        self.assertAlmostEqual(deployment_score(only_smooth), 0.5, places=6)
        self.assertAlmostEqual(deployment_score(only_arm), 0.3, places=6)
        self.assertAlmostEqual(deployment_score(only_hand), 0.2, places=6)

    def test_a_broken_zero_reading_does_not_win_every_checkpoint(self):
        """A zero rms is a failed measurement, not a perfect policy."""
        self.assertLess(deployment_score(metrics(rate=0.0)), 1.0)

    def test_missing_inputs_yield_no_score_rather_than_a_wrong_one(self):
        incomplete = metrics()
        del incomplete["evaluation_fixed_mean_rms_action_rate"]
        self.assertIsNone(deployment_score(incomplete))
        self.assertIsNone(deployment_score({}))

    def test_the_score_is_free_of_every_reward_sigma(self):
        """Changing a reward sigma must not move an unchanged policy's score."""
        m = metrics(rate=0.10, arm=0.03, hand=0.09)
        before = deployment_score(m)
        m["evaluation_score"] = 0.87
        m["evaluation_fixed_mean_position_reward"] = 0.42
        self.assertEqual(deployment_score(m), before)


class TermsTest(unittest.TestCase):
    def test_the_terms_explain_the_score(self):
        terms = deployment_terms(metrics(rate=2.5275, arm=0.0692, hand=0.2124))
        self.assertAlmostEqual(terms["deployment_smooth"], 0.0, places=3)
        self.assertAlmostEqual(terms["deployment_grasp_gate"], 1.0)

    def test_anchors_run_from_worst_to_best(self):
        for worst, best in ANCHORS.values():
            self.assertGreater(worst, best)


if __name__ == "__main__":
    unittest.main()
