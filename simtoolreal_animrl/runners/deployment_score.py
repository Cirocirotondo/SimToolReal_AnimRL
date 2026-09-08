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

    deployment_score = grasp_gate * (0.5*smooth + 0.3*arm + 0.2*hand)

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
    "action_rate": (2.5275, 0.0076),
    "arm_position_error": (0.0692, 0.0061),
    "hand_position_error": (0.2124, 0.0103),
}

WEIGHTS = {"action_rate": 0.5, "arm_position_error": 0.3, "hand_position_error": 0.2}


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
    sources = {
        "action_rate": metrics.get("evaluation_fixed_mean_rms_action_rate"),
        "arm_position_error": metrics.get(
            "evaluation_fixed_mean_rms_position_error"
        ),
        "hand_position_error": metrics.get(
            "evaluation_fixed_mean_rms_hand_position_error"
        ),
    }
    quality = 0.0
    for name, value in sources.items():
        term = _log_term(value, *ANCHORS[name])
        if term is None:
            return None
        quality += WEIGHTS[name] * term
    return gate * quality


def deployment_terms(metrics):
    """The individual terms, for logging and for explaining a score."""
    terms = {
        "deployment_grasp_gate": grasp_gate(
            metrics.get("evaluation_uniform_early_termination_fraction"),
            metrics.get("evaluation_uniform_mean_peak_object_com_lift_m"),
        ),
        "deployment_smooth": _log_term(
            metrics.get("evaluation_fixed_mean_rms_action_rate"),
            *ANCHORS["action_rate"]
        ),
        "deployment_arm": _log_term(
            metrics.get("evaluation_fixed_mean_rms_position_error"),
            *ANCHORS["arm_position_error"]
        ),
        "deployment_hand": _log_term(
            metrics.get("evaluation_fixed_mean_rms_hand_position_error"),
            *ANCHORS["hand_position_error"]
        ),
    }
    return {k: v for k, v in terms.items() if v is not None}
