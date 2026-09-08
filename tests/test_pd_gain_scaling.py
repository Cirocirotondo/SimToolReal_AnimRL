"""Softening the low-level drive is what makes a policy safe on the real robot."""

import unittest

import numpy as np

from simtoolreal_animrl.envs.pd_gains import (
    ARM_PD_DAMPING,
    ARM_PD_STIFFNESS,
    HAND_PD_DAMPING,
    HAND_PD_STIFFNESS,
    scale_gains,
)


class ScaleGainsTest(unittest.TestCase):
    def test_unit_scale_reproduces_the_gains_every_run_so_far_used(self):
        self.assertEqual(scale_gains(HAND_PD_STIFFNESS, 1.0), tuple(
            float(g) for g in HAND_PD_STIFFNESS
        ))

    def test_halving_the_stiffness_halves_every_joint(self):
        scaled = scale_gains(HAND_PD_STIFFNESS, 0.5)
        for original, value in zip(HAND_PD_STIFFNESS, scaled):
            self.assertAlmostEqual(value, original * 0.5)

    def test_the_outlier_joint_is_scaled_like_the_rest(self):
        """Hand joint 1 sits at 400 against the others' 42.97; it must not be
        special-cased, or softening the hand would leave one stiff joint."""
        scaled = scale_gains(HAND_PD_STIFFNESS, 0.25)
        self.assertAlmostEqual(scaled[1], 100.0)

    def test_a_softer_drive_raises_the_damping_ratio(self):
        """zeta = d / (2 sqrt(k J)): lowering k with d fixed increases zeta,
        which is why softening is not merely slower but better damped."""
        inertia = 1e-4
        stiff = HAND_PD_STIFFNESS[0]
        damping = HAND_PD_DAMPING[0]
        zeta = lambda k: damping / (2.0 * np.sqrt(k * inertia))
        self.assertGreater(zeta(stiff * 0.5), zeta(stiff))
        self.assertAlmostEqual(zeta(stiff * 0.25) / zeta(stiff), 2.0, places=6)

    def test_a_nonpositive_or_nonfinite_scale_is_rejected(self):
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                scale_gains(ARM_PD_STIFFNESS, bad)

    def test_arm_and_hand_scale_independently(self):
        arm = scale_gains(ARM_PD_DAMPING, 2.0)
        hand = scale_gains(HAND_PD_DAMPING, 0.5)
        self.assertAlmostEqual(arm[0], ARM_PD_DAMPING[0] * 2.0)
        self.assertAlmostEqual(hand[0], HAND_PD_DAMPING[0] * 0.5)


if __name__ == "__main__":
    unittest.main()
