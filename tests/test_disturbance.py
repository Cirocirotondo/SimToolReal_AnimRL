"""External pushes, so the policy has a recovery behaviour at all."""

import unittest

import torch

from simtoolreal_animrl.envs.disturbance import sample_impulses


class SampleImpulsesTest(unittest.TestCase):
    def test_zero_probability_or_magnitude_disturbs_nothing(self):
        for p, m in ((0.0, 5.0), (1.0, 0.0)):
            f = sample_impulses(64, 30, p, m, "cpu")
            self.assertEqual(float(f.abs().sum()), 0.0)

    def test_the_shape_matches_the_force_buffer(self):
        self.assertEqual(sample_impulses(64, 30, 0.5, 5.0, "cpu").shape, (64, 30, 3))

    def test_roughly_the_requested_fraction_of_environments_is_hit(self):
        f = sample_impulses(8192, 30, 0.10, 5.0, "cpu")
        hit = (f.abs().sum(dim=(1, 2)) > 0).float().mean().item()
        self.assertGreater(hit, 0.08)
        self.assertLess(hit, 0.12)

    def test_at_most_one_body_per_environment_is_pushed(self):
        """A push on several links at once is a force field, not an impulse."""
        f = sample_impulses(512, 30, 1.0, 5.0, "cpu")
        pushed = (f.abs().sum(dim=2) > 0).sum(dim=1)
        self.assertTrue(bool((pushed <= 1).all()))

    def test_magnitude_never_exceeds_the_requested_bound(self):
        f = sample_impulses(4096, 30, 1.0, 5.0, "cpu")
        self.assertLessEqual(float(f.norm(dim=2).max()), 5.0 + 1e-4)

    def test_magnitudes_vary_rather_than_being_one_fixed_push(self):
        f = sample_impulses(4096, 30, 1.0, 5.0, "cpu")
        norms = f.norm(dim=2).sum(dim=1)
        self.assertGreater(float(norms.std()), 0.5)

    def test_directions_are_isotropic(self):
        """A biased direction is a gravity change, not a disturbance."""
        f = sample_impulses(16384, 4, 1.0, 5.0, "cpu")
        mean = f.sum(dim=(0, 1)) / f.norm(dim=2).sum()
        self.assertLess(float(mean.abs().max()), 0.05)

    def test_only_the_listed_bodies_are_pushed(self):
        allowed = torch.tensor([3, 7])
        f = sample_impulses(1024, 30, 1.0, 5.0, "cpu", body_indices=allowed)
        pushed = torch.nonzero(f.abs().sum(dim=2) > 0)[:, 1].unique()
        self.assertTrue(set(pushed.tolist()).issubset({3, 7}))


if __name__ == "__main__":
    unittest.main()
