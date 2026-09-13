"""A second policy score, built to answer "can this go on the real robot?".

``evaluation_score`` cannot answer that, for two reasons this project has paid
for repeatedly. It is computed on the *fixed* cohort, three of whose four start
phases lie outside the window ``pregrasp_mixture`` actually trains, and it
subtracts that cohort's early-termination fraction outright -- so a healthy
policy reads as failing whenever those under-trained starts wobble. And it is
built from Gaussian reward terms, so tightening a reward sigma lowers the score
of an unchanged policy, which makes it meaningless across runs.

This score uses only quantities that are free of both problems: the cohort that
follows the training distribution, and physical measurements in metres, radians
and action units that no sigma can move.

    deployment_score = grasp_gate            # quality terms await re-anchoring

The grasp is a *gate*, not a summand. A policy that drops the cube is worthless
however smooth it is, and an additive term would let one be traded for the
other -- the mirror of the failure that makes evaluation_score misleading.
"""

import math


# The demonstration's own peak lift. A policy matching it scores the full gate.
DEMONSTRATION_LIFT_M = 0.24

# Anchors for the three log-scaled terms, measured on this box: `worst` is the
# base blind run pg830_blind512 at iteration 6500, `best` is 2026-08-26_sharpen
# _sigma, the smoothest policy trained here. A term is 0 at `worst` and 1 at
# `best`, and the scale is logarithmic because these span two decades.
ANCHORS = {
    "hand_position_error": (0.2124, 0.0103),
}

# The smoothness term is suspended, not renamed. Its anchors (2.5275, 0.0076)
# were measured on joint-space action deltas, and the arm's action is now an
# end-effector twist: feeding the new metric into the old scale would silently
# mis-score half of this number. Re-anchor it from the first task-space runs and
# restore the 0.5 weight then. Until that happens the remaining two terms are
# renormalized so the score still spans [0, 1] and stays comparable within the
# task-space era -- though not against any pre-change run.
#
# arm_position_error is gone for the same reason, and dropping it was a
# correction rather than a plan: it was first kept as a "diagnostic input",
# which was wrong. Its anchors were measured on runs that TRAINED arm joint
# tracking. With that reward removed the arm wanders its null space freely, the
# error settles around 0.21 rad -- three times worse than the anchor's worst
# case -- and _log_term clamps to 0. A zero term zeroed the whole product, so
# the score read 0.0 for every checkpoint and best_deployment_model.pt was
# never once updated. A metric outside its own calibration range does not
# degrade gracefully; it has to come out.
#
# Hand tracking is still trained, so that term is still meaningful and now
# carries the quality half alone.
WEIGHTS = {"hand_position_error": 1.0}


def _log_term(value, worst, best):
    """Map a lower-is-better measurement onto [0, 1] on a log scale."""
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        # A zero or negative reading is not a perfect policy, it is a broken
        # measurement; scoring it 1.0 would let it win every checkpoint.
        return 0.0
    span = math.log(worst / best)
    return min(1.0, max(0.0, math.log(worst / value) / span))


def grasp_gate(early_termination_fraction, peak_lift_m):
    """How much of the task the policy actually completes, in [0, 1]."""
    if early_termination_fraction is None or peak_lift_m is None:
        return None
    survived = 1.0 - min(1.0, max(0.0, float(early_termination_fraction)))
    lifted = min(1.0, max(0.0, float(peak_lift_m) / DEMONSTRATION_LIFT_M))
    return survived * lifted


def deployment_score(metrics):
    """Score a metrics dict from an evaluation. Returns None if inputs are absent.

    ``metrics`` is the flat dict a periodic evaluation logs, so this reads the
    same keys the dashboard plots.
    """
    gate = grasp_gate(
        metrics.get("evaluation_uniform_early_termination_fraction"),
        metrics.get("evaluation_uniform_mean_peak_object_com_lift_m"),
    )
    if gate is None:
        return None
    # The quality block is suspended in full, not weighted down. Every anchor
    # in it was measured under the joint-tracking reward this project has since
    # replaced, and THIS policy sits outside all of them: arm joint error lands
    # near 0.21 rad against a 0.069 worst case, hand error near 0.25 against
    # 0.212. _log_term clamps an out-of-range reading to 0, a zero factor zeroes
    # the product, and the score read exactly 0.0 at every evaluation of the
    # first task-space run -- best_deployment_model.pt was never updated once.
    #
    # So the gate stands alone for now. It is the honest half: lift x survival,
    # both physical measurements no reward sigma can move, and it is the only
    # signal here that can see the policy starting to lift the cube. Restore the
    # quality terms once they have been re-anchored on task-space runs, and note
    # that scores from before that point are gate-only and not comparable.
    return gate


def deployment_terms(metrics):
    """The individual terms, for logging and for explaining a score."""
    terms = {
        "deployment_grasp_gate": grasp_gate(
            metrics.get("evaluation_uniform_early_termination_fraction"),
            metrics.get("evaluation_uniform_mean_peak_object_com_lift_m"),
        ),
        "deployment_hand": _log_term(
            metrics.get("evaluation_fixed_mean_rms_hand_position_error"),
            *ANCHORS["hand_position_error"]
        ),
    }
    return {k: v for k, v in terms.items() if v is not None}
