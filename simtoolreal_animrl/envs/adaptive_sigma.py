"""Reward widths that follow the policy instead of being set once.

Every Gaussian reward term here is ``exp(-mse / (2 sigma^2))``, which has almost
no gradient in two places: far outside sigma, where it is numerically zero, and
well inside it, where it is pinned at 1. A fixed sigma therefore stops paying
the moment the policy passes it, and the optimiser -- which is still doing its
job -- reallocates capacity to whatever else still earns reward. That is the
mechanism behind "more training made it worse": the deployment score peaked at
iteration 1500 of blind_quiet2 while mean_reward kept climbing to the end.

An adaptive sigma tracks a slow average of the term's own MSE and holds the
reward near a target value, so the term keeps a live gradient however good the
policy gets. It is the sharpening ladder -- 0.2236 -> 0.10 -> 0.05, each rung
recalibrated by hand -- done continuously and without the recalibration.

    reward = target  =>  sigma^2 = mse / (2 ln(1 / target))

so with target 0.6 the width sits a little under the running RMS.
"""

import math


def sigma_for_target(mean_squared_error, target_reward):
    """The width that scores ``target_reward`` at this error."""
    target_reward = float(target_reward)
    if not 0.0 < target_reward < 1.0:
        raise ValueError("Adaptive sigma target reward must lie in (0, 1)")
    mse = float(mean_squared_error)
    if not math.isfinite(mse) or mse <= 0.0:
        return None
    return math.sqrt(mse / (2.0 * math.log(1.0 / target_reward)))


class AdaptiveSigma:
    """One reward term's width, tracked across iterations.

    ``floor`` is the width below which the term stops tightening. Without it
    the ladder never ends: sigma chases the policy down indefinitely and the
    term keeps demanding improvement in a dimension that no longer matters,
    which is how blind_sharp traded its grasp away for tracking it did not need.
    """

    def __init__(self, initial, floor, target_reward=0.6, decay=0.999, slack=1.5):
        if float(floor) <= 0.0:
            raise ValueError("Adaptive sigma floor must be positive")
        if not 0.0 <= float(decay) < 1.0:
            raise ValueError("Adaptive sigma decay must lie in [0, 1)")
        self.sigma = float(initial)
        self.floor = float(floor)
        self.target_reward = float(target_reward)
        self.decay = float(decay)
        if float(slack) < 1.0:
            raise ValueError("Adaptive sigma slack must be at least 1.0")
        self.slack = float(slack)
        self.tightest = float(initial)
        self.mean_squared_error = None

    def update(self, batch_mean_squared_error):
        """Fold in one iteration's MSE and return the width to use next."""
        mse = float(batch_mean_squared_error)
        if not math.isfinite(mse) or mse < 0.0:
            return self.sigma
        if self.mean_squared_error is None:
            self.mean_squared_error = mse
        else:
            self.mean_squared_error = (
                self.decay * self.mean_squared_error + (1.0 - self.decay) * mse
            )
        target = sigma_for_target(self.mean_squared_error, self.target_reward)
        if target is not None:
            target = max(self.floor, target)
            self.tightest = min(self.tightest, target)
            self.sigma = max(self.floor, min(target, self.tightest * self.slack))
        return self.sigma

    def state(self):
        return {
            "sigma": self.sigma,
            "tightest": self.tightest,
            "mean_squared_error": self.mean_squared_error,
        }

    def load_state(self, state):
        """Restore a ladder saved by ``state()``.

        Without this a resumed run silently rebuilds every width from the
        configuration, which hands the policy back the wide sigma it started
        from. The term is then pinned at 1 again, stops paying, and the
        tracking it was holding drifts -- measured on rot6d_adaptive, arm error
        went 0.044 -> 0.136 over the 1200 iterations after a resume, with no
        other change.
        """
        if not isinstance(state, dict):
            return
        sigma = state.get("sigma")
        if sigma is not None and math.isfinite(float(sigma)):
            self.sigma = max(self.floor, float(sigma))
        tightest = state.get("tightest")
        if tightest is not None and math.isfinite(float(tightest)):
            self.tightest = max(self.floor, float(tightest))
        else:
            self.tightest = min(self.tightest, self.sigma)
        mse = state.get("mean_squared_error")
        self.mean_squared_error = (
            None if mse is None or not math.isfinite(float(mse)) else float(mse)
        )
