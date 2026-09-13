"""The deployment score has to rank policies the way the robot would."""

import unittest

from simtoolreal_animrl.runners.deployment_score import (
    ANCHORS,
    deployment_score,
    deployment_terms,
    WEIGHTS,
    grasp_gate,
)


def metrics(et=0.0, lift=0.24, hand=0.0103):
    return {
        "evaluation_uniform_early_termination_fraction": et,
        "evaluation_uniform_mean_peak_object_com_lift_m": lift,
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
    """The score is the grasp gate alone while the quality terms are suspended.

    Every quality anchor was measured under the joint-tracking reward this
    project has replaced, and the task-space policy sits outside all of them.
    An out-of-range reading clamps to 0, and one zero factor zeroed the whole
    product -- so the score read 0.0 at every evaluation and
    best_deployment_model.pt never updated. The gate is what survives: lift
    times survival, in metres and fractions that no reward sigma can move.
    """

    def test_matching_the_demonstration_lift_scores_one(self):
        self.assertAlmostEqual(deployment_score(metrics()), 1.0, places=6)

    def test_dropping_the_cube_scores_zero(self):
        self.assertAlmostEqual(deployment_score(metrics(lift=0.0)), 0.0, places=6)

    def test_terminating_early_scores_zero(self):
        self.assertAlmostEqual(deployment_score(metrics(et=1.0)), 0.0, places=6)

    def test_a_partial_lift_scores_in_proportion(self):
        self.assertAlmostEqual(
            deployment_score(metrics(lift=0.12)), 0.5, places=6
        )

    def test_it_tracks_the_lift_the_first_task_space_run_produced(self):
        """The signal evaluation_score is blind to, and the reason for this fix.

        Measured lift went 0.0037 -> 0.0758 m over iterations 6500-8000 while
        evaluation_score FELL from 0.094 to -0.093, because that score rewards
        palm tracking and cannot see a lift at all.
        """
        early = deployment_score(metrics(et=0.594, lift=0.0037))
        late = deployment_score(metrics(et=0.844, lift=0.0758))
        self.assertGreater(late, early)

    def test_no_quality_metric_can_move_it_while_they_are_suspended(self):
        base = metrics()
        for key, value in (
            ("evaluation_fixed_mean_rms_position_error", 0.21),
            ("evaluation_fixed_mean_rms_hand_position_error", 0.25),
            ("evaluation_fixed_mean_rms_ee_action_rate", 0.5),
        ):
            polluted = dict(base)
            polluted[key] = value
            self.assertEqual(deployment_score(polluted), deployment_score(base))

    def test_missing_inputs_yield_no_score_rather_than_a_wrong_one(self):
        incomplete = metrics()
        del incomplete["evaluation_uniform_mean_peak_object_com_lift_m"]
        self.assertIsNone(deployment_score(incomplete))
        self.assertIsNone(deployment_score({}))

    def test_the_score_is_free_of_every_reward_sigma(self):
        """Changing a reward sigma must not move an unchanged policy's score."""
        m = metrics(hand=0.09)
        before = deployment_score(m)
        m["evaluation_score"] = 0.87
        m["evaluation_fixed_mean_palm_keypoint_reward"] = 0.42
        self.assertEqual(deployment_score(m), before)


class TermsTest(unittest.TestCase):
    def test_the_terms_explain_the_score(self):
        terms = deployment_terms(metrics(hand=0.2124))
        self.assertAlmostEqual(terms["deployment_hand"], 0.0, places=3)
        self.assertAlmostEqual(terms["deployment_grasp_gate"], 1.0)

    def test_anchors_run_from_worst_to_best(self):
        for worst, best in ANCHORS.values():
            self.assertGreater(worst, best)


if __name__ == "__main__":
    unittest.main()
