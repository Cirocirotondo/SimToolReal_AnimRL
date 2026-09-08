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

    def __init__(self, initial, floor, target_reward=0.6, decay=0.999):
        if float(floor) <= 0.0:
            raise ValueError("Adaptive sigma floor must be positive")
        if not 0.0 <= float(decay) < 1.0:
            raise ValueError("Adaptive sigma decay must lie in [0, 1)")
        self.sigma = float(initial)
        self.floor = float(floor)
        self.target_reward = float(target_reward)
        self.decay = float(decay)
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
            # Never loosen: a width that grows again would forgive regressions
            # the policy has already been paid to fix.
            self.sigma = max(self.floor, min(self.sigma, target))
        return self.sigma

    def state(self):
        return {"sigma": self.sigma, "mean_squared_error": self.mean_squared_error}
