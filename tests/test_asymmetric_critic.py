"""The critic may see contact the blind actor cannot."""

import unittest

import torch

from simtoolreal_animrl.envs.contact import fingertip_force_observation


class _Cfg:
    """Minimal stand-in for the contact config block."""

    def __init__(self, critic=False, actor=False, enabled=True):
        self.enabled = enabled
        self.observe_fingertip_forces = actor
        self.critic_observes_fingertip_forces = critic
        self.fingertip_names = ("thumb", "index", "middle")


def privileged_width(cfg, num_obs):
    """Mirror of the width rule in MotionImitationEnv.__init__."""
    if not bool(getattr(cfg, "critic_observes_fingertip_forces", False)):
        return None
    return num_obs + 3 * len(cfg.fingertip_names)


class AsymmetricCriticTest(unittest.TestCase):
    def test_the_feature_off_leaves_ppo_on_its_symmetric_path(self):
        self.assertIsNone(privileged_width(_Cfg(critic=False), 108))

    def test_the_critic_is_wider_than_the_blind_actor_by_the_force_vectors(self):
        self.assertEqual(privileged_width(_Cfg(critic=True), 108), 117)

    def test_the_actor_width_is_untouched_by_the_critic_setting(self):
        """The whole point: the deployed policy stays blind at 108."""
        from simtoolreal_animrl.envs.contact import (
            fingertip_force_observation_dim,
        )
        self.assertEqual(fingertip_force_observation_dim(_Cfg(critic=True)), 0)

    def test_actor_and_critic_can_both_see_forces(self):
        cfg = _Cfg(critic=True, actor=True)
        from simtoolreal_animrl.envs.contact import (
            fingertip_force_observation_dim,
        )
        actor_extra = fingertip_force_observation_dim(cfg)
        self.assertEqual(actor_extra, 9)
        self.assertEqual(privileged_width(cfg, 108 + actor_extra), 126)

    def test_the_critic_vector_is_the_actor_vector_then_the_forces(self):
        obs = torch.arange(2 * 108, dtype=torch.float32).reshape(2, 108)
        forces = torch.ones(2, 3, 3)
        features = fingertip_force_observation(forces, 10.0, 5.0)
        critic = torch.cat((obs, features), dim=1)
        self.assertEqual(critic.shape, (2, 117))
        torch.testing.assert_close(critic[:, :108], obs)
        torch.testing.assert_close(critic[:, 108:], features)

    def test_forces_are_scaled_and_clipped_before_the_critic_sees_them(self):
        """An unclipped impulse would swamp the value normaliser."""
        forces = torch.full((1, 3, 3), 1000.0)
        features = fingertip_force_observation(forces, 10.0, 5.0)
        self.assertTrue(torch.all(features <= 5.0))
        self.assertEqual(features.shape, (1, 9))


if __name__ == "__main__":
    unittest.main()
