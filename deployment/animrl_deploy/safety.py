"""Rate limits, discontinuity detection and the arming prompt.

The action-spike monitor is not generic caution. The 108-D observation encodes
the palm and cube rotations as quaternions canonicalized to ``w >= 0``, which
cuts the double cover at ``w = 0``: a palm rotating smoothly through that plane
negates all four components at once and steps the observation by 2.0 while
nothing physical moves. A continuous network answers a discontinuous input with
a discontinuous output. In the run that motivated the 6-D rotation encoding, the
policy replied to exactly that crossing with a 4.03 action-unit step on
``rj_dg_3_4`` -- 0.63 rad -- out of a stream whose neighbouring steps moved by
0.009, and then rang for thirty frames.

On hardware that is a finger snapping through roughly 36 degrees in one control
tick. ``SpikeMonitor`` watches the raw action stream for that signature and, in
``stop`` mode, aborts before the target reaches the robot. ``TargetLimiter``
is the second line: even an undetected spike cannot leave the joint limits or
move further than one step allowance per tick.
"""

from __future__ import annotations

import sys
from typing import Optional

import numpy as np

ARM_DOF = 6
HAND_DOF = 20
ACTION_DIM = ARM_DOF + HAND_DOF


class SafetyAbort(RuntimeError):
    """A safety monitor stopped the rollout."""


class TargetLimiter:
    """Smooth, rate-limit and clip position targets before they are sent.

    Applied in that order: smoothing shapes the trajectory, the step clamp
    bounds per-tick motion, and the limit clip is the final hard bound. The
    clamp is measured against the previously *emitted* target, not the measured
    position, so a policy that ramps away is followed at a bounded rate rather
    than being repeatedly pulled back.
    """

    def __init__(
        self,
        lower_limits: np.ndarray,
        upper_limits: np.ndarray,
        *,
        max_arm_step_rad: float = 0.02,
        max_hand_step_rad: float = 0.05,
        smoothing: float = 0.0,
    ) -> None:
        lower = np.asarray(lower_limits, dtype=np.float64)
        upper = np.asarray(upper_limits, dtype=np.float64)
        if lower.shape != (ACTION_DIM,) or upper.shape != (ACTION_DIM,):
            raise ValueError("Joint limits must have shape (26,)")
        if np.any(lower >= upper):
            raise ValueError("Joint limits must be non-empty intervals")
        if max_arm_step_rad <= 0.0 or max_hand_step_rad <= 0.0:
            raise ValueError("Step limits must be positive")
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must lie in [0, 1)")
        self.lower = lower
        self.upper = upper
        self.step_limit = np.concatenate(
            (
                np.full(ARM_DOF, float(max_arm_step_rad)),
                np.full(HAND_DOF, float(max_hand_step_rad)),
            )
        )
        self.smoothing = float(smoothing)
        self.previous: Optional[np.ndarray] = None
        self.filtered: Optional[np.ndarray] = None

    def reset(self, initial_target: np.ndarray) -> None:
        target = np.asarray(initial_target, dtype=np.float64)
        if target.shape != (ACTION_DIM,):
            raise ValueError("Initial target must have shape (26,)")
        self.previous = target.copy()
        self.filtered = target.copy()

    def apply(self, requested: np.ndarray) -> tuple:
        requested = np.asarray(requested, dtype=np.float64)
        if requested.shape != (ACTION_DIM,):
            raise ValueError("Requested target must have shape (26,)")
        if not np.all(np.isfinite(requested)):
            raise SafetyAbort("Policy produced a non-finite position target.")
        if self.previous is None or self.filtered is None:
            raise RuntimeError("reset() must be called before apply()")

        if self.smoothing > 0.0:
            self.filtered = (
                self.smoothing * self.filtered + (1.0 - self.smoothing) * requested
            )
            shaped = self.filtered.copy()
        else:
            self.filtered = requested.copy()
            shaped = requested.copy()

        raw_step = shaped - self.previous
        limited_step = np.clip(raw_step, -self.step_limit, self.step_limit)
        stepped = self.previous + limited_step
        final = np.clip(stepped, self.lower, self.upper)

        info = {
            "max_requested_step_rad": float(np.max(np.abs(raw_step))),
            "step_limited_joints": np.flatnonzero(
                np.abs(raw_step) > self.step_limit + 1e-12
            ),
            "limit_clipped_joints": np.flatnonzero(
                np.abs(final - stepped) > 1e-12
            ),
        }
        self.previous = final.copy()
        return final, info


class SpikeMonitor:
    """Detect a discontinuity in a per-step signal (raw actions, or targets)."""

    MODES = ("off", "warn", "stop")

    def __init__(
        self,
        threshold: float,
        *,
        mode: str = "warn",
        name: str = "action",
        labels: Optional[list] = None,
        grace_steps: int = 0,
    ) -> None:
        if mode not in self.MODES:
            raise ValueError("mode must be one of {}".format(self.MODES))
        if threshold <= 0.0:
            raise ValueError("threshold must be positive")
        self.threshold = float(threshold)
        self.mode = mode
        self.name = name
        self.labels = labels
        # The first policy steps after a reset carry an inherent transient: the
        # policy settles from the reference pose onto its own trajectory, and
        # in sim2sim that first step alone exceeds 1.0 action units. Aborting
        # on it would stop every run at step 1, so the grace window suppresses
        # the abort (never the report) while the transient passes.
        self.grace_steps = max(0, int(grace_steps))
        self.updates = 0
        self.previous: Optional[np.ndarray] = None
        self.worst = 0.0
        self.worst_index = -1
        self.detections = 0

    def reset(self, initial: Optional[np.ndarray] = None) -> None:
        self.updates = 0
        self.previous = (
            None if initial is None else np.asarray(initial, dtype=np.float64).copy()
        )

    def _label(self, index: int) -> str:
        if self.labels is not None and 0 <= index < len(self.labels):
            return str(self.labels[index])
        return "index {}".format(index)

    def update(self, values: np.ndarray) -> Optional[dict]:
        current = np.asarray(values, dtype=np.float64)
        if self.mode == "off":
            self.previous = current.copy()
            return None
        if self.previous is None:
            self.previous = current.copy()
            return None
        self.updates += 1
        delta = np.abs(current - self.previous)
        index = int(np.argmax(delta))
        magnitude = float(delta[index])
        if magnitude > self.worst:
            self.worst = magnitude
            self.worst_index = index
        self.previous = current.copy()
        if magnitude <= self.threshold:
            return None
        self.detections += 1
        report = {
            "name": self.name,
            "magnitude": magnitude,
            "index": index,
            "label": self._label(index),
            "threshold": self.threshold,
        }
        message = (
            "{} discontinuity: {} moved {:.4f} in one step "
            "(threshold {:.4f}). This is the signature of the quaternion "
            "double-cover crossing; see deployment/README.md.".format(
                self.name.capitalize(), report["label"], magnitude, self.threshold
            )
        )
        if self.mode == "stop" and self.updates > self.grace_steps:
            raise SafetyAbort(message)
        if self.updates <= self.grace_steps:
            message += " (within the {}-step startup grace window)".format(
                self.grace_steps
            )
        print("WARNING: " + message)
        return report


def wait_for_key(prompt: str, accept: tuple = (" ", "\r", "\n")) -> None:
    """Block until an accepted key, raising KeyboardInterrupt on 'q'."""
    print(prompt, end="", flush=True)
    if not sys.stdin.isatty():
        answer = input().strip().lower()
        if answer == "q":
            raise KeyboardInterrupt
        return
    import termios
    import tty

    descriptor = sys.stdin.fileno()
    settings = termios.tcgetattr(descriptor)
    try:
        tty.setcbreak(descriptor)
        while True:
            key = sys.stdin.read(1)
            if key.lower() == "q":
                print()
                raise KeyboardInterrupt
            if key in accept:
                print()
                return
    finally:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, settings)


def confirm_send(outputs: list) -> None:
    """Require a typed SEND before any physical output is armed."""
    print()
    print("=" * 70)
    print("ARMING PHYSICAL OUTPUT:")
    for output in outputs:
        print("  - {}".format(output))
    print("Clear the workspace. Keep the e-stop within reach.")
    print("=" * 70)
    answer = input("Type SEND to arm, anything else to abort: ")
    if answer.strip() != "SEND":
        raise KeyboardInterrupt("Aborted at the arming prompt.")
