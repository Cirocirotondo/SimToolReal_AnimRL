"""A reset the real robot could actually reproduce."""

import unittest

import torch

from simtoolreal_animrl.envs.rsi_noise import perturb_reference_pose


ARM = 6


def pose(n=8, joints=26):
    return torch.zeros(n, joints), torch.ones(n, joints)


class PerturbTest(unittest.TestCase):
    def test_zero_noise_reproduces_every_run_so_far(self):
        q, dq = pose()
        out_q, out_dq = perturb_reference_pose(q, dq, ARM, 0.0, 0.0, 0.0)
        torch.testing.assert_close(out_q, q)
        torch.testing.assert_close(out_dq, dq)

    def test_arm_and_hand_take_separate_magnitudes(self):
        """The arm lands within a milliradian; the tendon hand does not."""
        q, dq = pose(n=512)
        out_q, _ = perturb_reference_pose(q, dq, ARM, 0.01, 0.05, 0.0)
        self.assertLessEqual(out_q[:, :ARM].abs().max().item(), 0.01 + 1e-6)
        self.assertLessEqual(out_q[:, ARM:].abs().max().item(), 0.05 + 1e-6)
        self.assertGreater(out_q[:, ARM:].abs().max().item(), 0.01)

    def test_the_perturbation_is_hard_bounded(self):
        """Uniform, not Gaussian: a tail sample putting a finger through the
        cube would inject a contact impulse at reset."""
        q, dq = pose(n=4096)
        out_q, _ = perturb_reference_pose(q, dq, ARM, 0.02, 0.02, 0.0)
        self.assertLessEqual(out_q.abs().max().item(), 0.02 + 1e-6)

    def test_it_is_centred_on_the_reference(self):
        q, dq = pose(n=8192)
        out_q, _ = perturb_reference_pose(q, dq, ARM, 0.02, 0.02, 0.0)
        self.assertAlmostEqual(out_q.mean().item(), 0.0, places=2)

    def test_velocity_noise_is_proportional_so_still_joints_stay_still(self):
        q = torch.zeros(256, 26)
        dq = torch.zeros(256, 26)
        dq[:, 0] = 2.0
        _, out_dq = perturb_reference_pose(q, dq, ARM, 0.0, 0.0, 0.10)
        self.assertTrue(torch.all(out_dq[:, 1:] == 0.0))
        self.assertLessEqual((out_dq[:, 0] - 2.0).abs().max().item(), 0.2 + 1e-6)

    def test_positions_are_clamped_inside_the_joint_limits(self):
        """A reset outside the limits is a state the robot cannot occupy."""
        q = torch.zeros(256, 26)
        dq = torch.zeros(256, 26)
        lower = torch.full((26,), -0.005)
        upper = torch.full((26,), 0.005)
        out_q, _ = perturb_reference_pose(
            q, dq, ARM, 0.05, 0.05, 0.0, lower_limits=lower, upper_limits=upper
        )
        self.assertLessEqual(out_q.max().item(), 0.005 + 1e-9)
        self.assertGreaterEqual(out_q.min().item(), -0.005 - 1e-9)

    def test_the_caller_s_tensors_are_not_modified(self):
        q, dq = pose()
        original = q.clone()
        perturb_reference_pose(q, dq, ARM, 0.02, 0.02, 0.1)
        torch.testing.assert_close(q, original)


if __name__ == "__main__":
    unittest.main()
