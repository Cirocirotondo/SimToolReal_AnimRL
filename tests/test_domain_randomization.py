"""Physical variation the policy cannot observe, so it cannot fit it."""

import unittest

import numpy as np

from simtoolreal_animrl.envs.domain_randomization import DomainRandomization


class _Cfg:
    def __init__(self, enabled=True, **kw):
        self.enabled = enabled
        self.arm_stiffness_range = kw.get("arm_stiffness_range", 0.0)
        self.arm_damping_range = kw.get("arm_damping_range", 0.0)
        self.hand_stiffness_range = kw.get("hand_stiffness_range", 0.0)
        self.hand_damping_range = kw.get("hand_damping_range", 0.0)
        self.fingertip_friction_range = kw.get("fingertip_friction_range", 0.0)
        self.object_friction_range = kw.get("object_friction_range", 0.0)
        self.object_mass_range = kw.get("object_mass_range", 0.0)
        self.table_friction_range = kw.get("table_friction_range", 0.0)
        self.robot_link_mass_range = kw.get("robot_link_mass_range", 0.0)
        self.robot_impulse_probability = kw.get("robot_impulse_probability", 0.0)
        self.robot_impulse_n = kw.get("robot_impulse_n", 0.0)
        self.object_impulse_probability = kw.get("object_impulse_probability", 0.0)
        self.object_impulse_n = kw.get("object_impulse_n", 0.0)
        self.critic_observes_parameters = kw.get("critic_observes_parameters", False)


class DomainRandomizationTest(unittest.TestCase):
    def test_disabled_reproduces_the_fixed_values_exactly(self):
        dr = DomainRandomization(_Cfg(enabled=False, hand_stiffness_range=0.4), 64)
        self.assertTrue(all(dr.multiplier("hand_stiffness", i) == 1.0
                            for i in range(64)))

    def test_a_zero_range_leaves_that_parameter_alone(self):
        dr = DomainRandomization(_Cfg(hand_stiffness_range=0.4), 64)
        self.assertTrue(all(dr.multiplier("object_mass", i) == 1.0
                            for i in range(64)))

    def test_multipliers_stay_inside_the_requested_band(self):
        dr = DomainRandomization(_Cfg(hand_stiffness_range=0.4), 4096)
        values = dr.samples["hand_stiffness"]
        self.assertGreaterEqual(values.min(), 0.6)
        self.assertLessEqual(values.max(), 1.4)
        self.assertAlmostEqual(values.mean(), 1.0, places=1)

    def test_environments_differ_from_one_another(self):
        """The whole point: a policy cannot tune itself to one friction."""
        dr = DomainRandomization(_Cfg(object_friction_range=0.4), 256)
        self.assertGreater(len(set(dr.samples["object_friction"].tolist())), 200)

    def test_parameters_are_drawn_independently(self):
        """Correlated draws would let one compensating ratio be learned."""
        dr = DomainRandomization(
            _Cfg(hand_stiffness_range=0.4, object_friction_range=0.4), 4096
        )
        r = np.corrcoef(dr.samples["hand_stiffness"], dr.samples["object_friction"])
        self.assertLess(abs(r[0, 1]), 0.1)

    def test_it_is_reproducible_from_the_seed(self):
        a = DomainRandomization(_Cfg(object_mass_range=0.25), 128, seed=7)
        b = DomainRandomization(_Cfg(object_mass_range=0.25), 128, seed=7)
        np.testing.assert_allclose(a.samples["object_mass"], b.samples["object_mass"])

    def test_an_out_of_bounds_range_is_rejected(self):
        for bad in (-0.1, 1.0, 2.0):
            with self.assertRaises(ValueError):
                DomainRandomization(_Cfg(hand_stiffness_range=bad), 8)

    def test_the_critic_row_is_centred_on_zero_and_one_per_parameter(self):
        dr = DomainRandomization(_Cfg(hand_stiffness_range=0.4), 256)
        row = dr.privileged_row(0)
        self.assertEqual(row.shape, (9,))
        self.assertEqual(dr.privileged_dim, 0)  # off unless the critic asks
        rows = np.stack([dr.privileged_row(i) for i in range(256)])
        self.assertAlmostEqual(float(rows.mean()), 0.0, places=1)

    def test_a_disabled_randomisation_adds_no_critic_input(self):
        dr = DomainRandomization(_Cfg(enabled=False), 8)
        self.assertEqual(dr.privileged_dim, 0)

    def test_the_critic_sees_the_parameters_only_when_asked(self):
        off = DomainRandomization(_Cfg(hand_stiffness_range=0.4), 16)
        on = DomainRandomization(
            _Cfg(hand_stiffness_range=0.4, critic_observes_parameters=True), 16
        )
        self.assertEqual(off.privileged_dim, 0)
        self.assertEqual(on.privileged_dim, 9)
        self.assertEqual(tuple(on.privileged_table().shape), (16, 9))

    def test_impulses_are_off_until_both_probability_and_size_are_set(self):
        self.assertFalse(
            DomainRandomization(_Cfg(robot_impulse_probability=0.02), 8).impulses_enabled
        )
        self.assertTrue(
            DomainRandomization(
                _Cfg(robot_impulse_probability=0.02, robot_impulse_n=8.0), 8
            ).impulses_enabled
        )


if __name__ == "__main__":
    unittest.main()
