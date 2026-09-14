import unittest

import torch

from simtoolreal_animrl.cfg import SimToolRealCfg
from simtoolreal_animrl.envs.object_assist import object_reward_gate
from simtoolreal_animrl.envs.contact import (
    fingertip_force_observation,
    fingertip_force_observation_dim,
    select_fingertip_forces,
)


class ContactObservationConfigTest(unittest.TestCase):
    def test_the_block_is_off_by_default(self):
        cfg = SimToolRealCfg()
        self.assertFalse(cfg.contact.observe_fingertip_forces)
        self.assertEqual(fingertip_force_observation_dim(cfg.contact), 0)
        # 112 since the object-centric reference: both rotations in the
        # observation became the continuous 6D representation.
        self.assertEqual(cfg.env.num_observations, 112)

    def test_three_fingertips_add_nine_numbers(self):
        cfg = SimToolRealCfg()
        cfg.contact.observe_fingertip_forces = True
        self.assertEqual(cfg.contact.fingertip_names, ["thumb", "index", "middle"])
        self.assertEqual(fingertip_force_observation_dim(cfg.contact), 9)

    def test_the_width_follows_the_selected_fingertips(self):
        cfg = SimToolRealCfg()
        cfg.contact.observe_fingertip_forces = True
        cfg.contact.fingertip_names = ["thumb", "index"]
        self.assertEqual(fingertip_force_observation_dim(cfg.contact), 6)

    def test_a_configuration_without_the_field_reads_as_off(self):
        """The dimension helper also sees configs saved before the feature."""

        class LegacyContactCfg:
            fingertip_names = ["thumb", "index", "middle"]

        self.assertEqual(fingertip_force_observation_dim(LegacyContactCfg()), 0)


class SelectFingertipForcesTest(unittest.TestCase):
    def test_the_configured_order_is_preserved(self):
        net_contact_forces = torch.zeros(2, 6, 3)
        net_contact_forces[:, 4] = torch.tensor([1.0, 0.0, 0.0])
        net_contact_forces[:, 1] = torch.tensor([0.0, 2.0, 0.0])
        net_contact_forces[:, 3] = torch.tensor([0.0, 0.0, 3.0])
        indices = torch.tensor([4, 1, 3], dtype=torch.long)
        selected = select_fingertip_forces(net_contact_forces, indices)
        self.assertEqual(tuple(selected.shape), (2, 3, 3))
        torch.testing.assert_close(selected[0, 0], torch.tensor([1.0, 0.0, 0.0]))
        torch.testing.assert_close(selected[0, 1], torch.tensor([0.0, 2.0, 0.0]))
        torch.testing.assert_close(selected[0, 2], torch.tensor([0.0, 0.0, 3.0]))

    def test_bodies_outside_the_selection_are_ignored(self):
        net_contact_forces = torch.zeros(1, 5, 3)
        net_contact_forces[:, 2] = torch.tensor([9.0, 9.0, 9.0])
        selected = select_fingertip_forces(
            net_contact_forces, torch.tensor([0, 1], dtype=torch.long)
        )
        torch.testing.assert_close(selected, torch.zeros(1, 2, 3))


class FingertipForceObservationTest(unittest.TestCase):
    def test_the_block_is_flattened_finger_major(self):
        forces = torch.tensor(
            [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]
        )
        block = fingertip_force_observation(forces, force_scale_n=1.0, clip=100.0)
        self.assertEqual(tuple(block.shape), (1, 6))
        torch.testing.assert_close(
            block, torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
        )

    def test_the_scale_divides_every_component(self):
        forces = torch.full((3, 3, 3), 20.0)
        block = fingertip_force_observation(forces, force_scale_n=10.0, clip=100.0)
        torch.testing.assert_close(block, torch.full((3, 9), 2.0))

    def test_the_clip_is_symmetric_and_applied_after_the_scale(self):
        forces = torch.tensor([[[300.0, -300.0, 5.0]]])
        block = fingertip_force_observation(forces, force_scale_n=10.0, clip=5.0)
        torch.testing.assert_close(block, torch.tensor([[5.0, -5.0, 0.5]]))

    def test_no_contact_reads_as_exactly_zero(self):
        block = fingertip_force_observation(
            torch.zeros(4, 3, 3), force_scale_n=10.0, clip=5.0
        )
        torch.testing.assert_close(block, torch.zeros(4, 9))

    def test_the_direction_survives_a_component_clip(self):
        """A clipped component keeps its sign, so a squeeze never reads as a pull."""
        forces = torch.tensor([[[-80.0, 0.0, 0.0]]])
        block = fingertip_force_observation(forces, force_scale_n=10.0, clip=5.0)
        self.assertLess(float(block[0, 0]), 0.0)

    def test_the_configured_defaults_keep_a_firm_grasp_inside_the_clip(self):
        """5 N per fingertip -- a solid grasp -- must not saturate."""
        cfg = SimToolRealCfg()
        forces = torch.full((1, 3, 3), 0.0)
        forces[:, :, 2] = 5.0
        block = fingertip_force_observation(
            forces,
            cfg.contact.observation_force_scale_n,
            cfg.contact.observation_clip,
        )
        self.assertLess(float(block.abs().max()), cfg.contact.observation_clip)


class ContactRewardGateTest(unittest.TestCase):
    def test_the_reward_gate_is_independent_of_the_force_tensor(self):
        """Acquiring forces must stay separable from paying a reward for them.

        Both ship on now -- the grasp released the bar at the lift under ideal
        actions, so the shaping reward earns its place -- but the two flags
        must remain independent, or turning the tensor on for the critic would
        silently change the reward function too.
        """
        cfg = SimToolRealCfg()
        self.assertEqual(cfg.contact.reward_per_finger, 0.05)
        cfg.contact.reward_enabled = False
        self.assertTrue(cfg.contact.enabled)
        self.assertFalse(cfg.contact.reward_enabled)

    def test_the_actor_stays_blind_to_contact_by_default(self):
        """The deployed policy must not depend on fingertip force sensing.

        The critic may read the forces -- it is discarded at deployment -- but
        the actor's observation must not, or the policy could not run on a
        robot without force sensors.
        """
        cfg = SimToolRealCfg()
        self.assertFalse(cfg.contact.observe_fingertip_forces)
        self.assertTrue(cfg.contact.critic_observes_fingertip_forces)

    def test_the_weight_survives_for_callers_that_ask_for_it(self):
        cfg = SimToolRealCfg()
        cfg.contact.enabled = True
        cfg.contact.reward_enabled = True
        self.assertEqual(cfg.contact.reward_per_finger, 0.05)


class ObjectRewardGateTest(unittest.TestCase):
    gate = staticmethod(object_reward_gate)

    def test_no_assist_pays_the_object_reward_in_full(self):
        self.assertEqual(self.gate(False, True, 0.0), 1.0)
        self.assertEqual(self.gate(False, True, 1.0), 1.0)

    def test_full_assist_pays_nothing(self):
        self.assertEqual(self.gate(True, True, 1.0), 0.0)

    def test_the_share_grows_as_the_assist_anneals_out(self):
        self.assertAlmostEqual(self.gate(True, True, 0.75), 0.25)
        self.assertAlmostEqual(self.gate(True, True, 0.5), 0.5)
        self.assertAlmostEqual(self.gate(True, True, 0.0), 1.0)

    def test_gating_can_be_switched_off_for_an_ablation(self):
        self.assertEqual(self.gate(True, False, 1.0), 1.0)

    def test_the_gate_never_goes_negative(self):
        self.assertEqual(self.gate(True, True, 1.5), 0.0)

    def test_the_configuration_default_gates(self):
        self.assertTrue(SimToolRealCfg().object_assist.gate_object_reward)


if __name__ == "__main__":
    unittest.main()
