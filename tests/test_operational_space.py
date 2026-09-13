"""The frame convention behind task-space arm control, checked without a simulator.

The headline test is :meth:`TransferTest.test_transferred_wrist_jacobian_is_the_
palm_jacobian`. Isaac Gym can only report a Jacobian for ``wrist_3_link`` -- the
palm is collapsed out of the asset -- so the controller moves that Jacobian onto
the palm itself. If that transfer is wrong the arm still moves smoothly and
plausibly, just not where the reward is measured, which is the kind of bug a
training curve hides for days. ``pytorch_kinematics`` can build a chain to either
link, so the transfer has an exact ground truth and needs no Isaac Gym.
"""

import math
import unittest

import numpy as np
import torch

from simtoolreal_animrl import ROOT_DIR
from simtoolreal_animrl.envs.operational_space import (
    damped_least_squares_step,
    saturate_direction_preserving,
    skew,
    transfer_jacobian,
)
from simtoolreal_animrl.envs.retarget import (
    ARM_JOINT_COUNT,
    PALM_LINK_NAME,
    PalmKinematics,
    pose_error,
)


URDF = ROOT_DIR / "assets/urdf/ur5e_delto_description/ur5e_right_dg5f_mount_60deg.urdf"
DEMO = ROOT_DIR / (
    "demonstrations/"
    "demo_20260727_152551_335339_60hz_cube_collision_resolved_stable_grasp.npz"
)
WRIST_LINK_NAME = "wrist_3_link"
# Sampled from the demonstration rather than drawn uniformly: a transfer error
# scales with the wrist's angular rate, so it must be exercised at the wrist
# angles the robot actually visits, not at arbitrary ones.
FRAME_STRIDE = 37
CONTROL_DT = 1.0 / 60.0


class OperationalSpaceFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import pytorch_kinematics as pk

        cls.kinematics = PalmKinematics(URDF, device="cpu")
        data = URDF.read_bytes()
        cls.wrist_chain = pk.build_serial_chain_from_urdf(
            data, WRIST_LINK_NAME
        ).to(dtype=torch.float64)
        with np.load(str(DEMO)) as archive:
            arm = np.asarray(archive["arm_q"])[::FRAME_STRIDE]
        cls.arm_q = torch.as_tensor(arm, dtype=torch.float64)

    def wrist_offset_world(self, arm_q):
        """The palm origin minus the wrist origin, in the base frame."""
        wrist = self.wrist_chain.forward_kinematics(arm_q).get_matrix()
        palm = self.kinematics.palm_matrices(arm_q)
        return palm[:, :3, 3] - wrist[:, :3, 3]


class SkewTest(unittest.TestCase):
    def test_skew_matrix_reproduces_the_cross_product(self):
        generator = torch.Generator().manual_seed(0)
        a = torch.randn(32, 3, dtype=torch.float64, generator=generator)
        b = torch.randn(32, 3, dtype=torch.float64, generator=generator)
        expected = torch.linalg.cross(a, b)
        actual = (skew(a) @ b.unsqueeze(-1)).squeeze(-1)
        torch.testing.assert_close(actual, expected)


class SaturationTest(unittest.TestCase):
    """Bounding the twist must not redirect it.

    A per-component clip bounds the magnitude and silently turns the vector,
    which costs the policy control of direction exactly where it is asking
    hardest. It also makes the action unreachable past the rail, which is what
    let |action| drift to 14 with 46% of arm components pinned on the first
    training run.
    """

    def test_a_vector_inside_the_limit_is_untouched(self):
        vectors = torch.tensor([[0.3, -0.4, 0.0], [0.0, 0.0, 0.5]])
        torch.testing.assert_close(
            saturate_direction_preserving(vectors, 1.0), vectors
        )

    def test_a_vector_at_the_limit_is_untouched(self):
        vectors = torch.tensor([[1.0, 0.0, 0.0], [0.6, 0.8, 0.0]])
        torch.testing.assert_close(
            saturate_direction_preserving(vectors, 1.0), vectors
        )

    def test_an_over_range_vector_keeps_its_direction(self):
        vectors = torch.tensor([[10.0, 20.0, 0.0], [-3.0, 0.0, 4.0]])
        saturated = saturate_direction_preserving(vectors, 1.0)
        torch.testing.assert_close(
            saturated.norm(dim=1), torch.ones(2), atol=1e-6, rtol=0.0
        )
        expected = vectors / vectors.norm(dim=1, keepdim=True)
        torch.testing.assert_close(saturated, expected)

    def test_it_differs_from_clipping_each_component(self):
        """The whole point, stated as a test rather than left to a comment."""
        vectors = torch.tensor([[10.0, 20.0, 0.0]])
        saturated = saturate_direction_preserving(vectors, 1.0)
        clipped = vectors.clamp(-1.0, 1.0)
        # Clipping returns the 45 degree diagonal for a 2:1 request.
        torch.testing.assert_close(clipped, torch.tensor([[1.0, 1.0, 0.0]]))
        cosine = float(
            (saturated[0] / saturated[0].norm()) @ (clipped[0] / clipped[0].norm())
        )
        self.assertLess(cosine, 0.95)

    def test_a_zero_vector_survives(self):
        zeros = torch.zeros(4, 3)
        result = saturate_direction_preserving(zeros, 1.0)
        self.assertTrue(bool(torch.isfinite(result).all()))
        torch.testing.assert_close(result, zeros)


class PalmTiltTest(OperationalSpaceFixture):
    """Pitch and roll of the palm, with yaw discarded.

    The reward this feeds exists because every other term is blind to the palm's
    absolute pose: hand keypoints are measured in the cube's frame, so a hand
    that rotates together with the cube scores full marks. The first task-space
    run exploited exactly that, tipping the cube up on the wrist instead of
    carrying it.

    Yaw has to stay out of it because the bar's yaw is randomised per episode and
    the hand must follow it. What makes that cheap is the invariance tested here:
    one table indexed by frame serves every transform in the bank.
    """

    def test_tilt_is_invariant_to_rotation_about_the_world_vertical(self):
        """The property the whole design rests on, as algebra rather than data.

        ``(R_z R)^T z = R^T R_z^T z = R^T z``, because the world vertical is the
        axis being rotated about.
        """
        from simtoolreal_animrl.envs.transform_bank import palm_tilt_in_palm_frame

        reference = palm_tilt_in_palm_frame(self.kinematics, self.arm_q)
        for yaw_deg in (-22.5, 15.0, 45.0, 89.9):
            yaw = math.radians(yaw_deg)
            spin = torch.tensor(
                [
                    [math.cos(yaw), -math.sin(yaw), 0.0],
                    [math.sin(yaw), math.cos(yaw), 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=torch.float64,
            )
            matrices = self.kinematics.palm_matrices(self.arm_q).clone()
            matrices[:, :3, :3] = spin @ matrices[:, :3, :3]
            up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
            spun = torch.nn.functional.normalize(
                matrices[:, :3, :3].transpose(1, 2) @ up, dim=-1
            )
            torch.testing.assert_close(spun, reference, atol=1e-12, rtol=0.0)

    def test_tilt_is_not_invariant_to_pitch_or_roll(self):
        """Guards the test above from passing on a quantity that ignores everything."""
        from simtoolreal_animrl.envs.transform_bank import palm_tilt_in_palm_frame

        reference = palm_tilt_in_palm_frame(self.kinematics, self.arm_q)
        angle = math.radians(20.0)
        pitch = torch.tensor(
            [
                [math.cos(angle), 0.0, math.sin(angle)],
                [0.0, 1.0, 0.0],
                [-math.sin(angle), 0.0, math.cos(angle)],
            ],
            dtype=torch.float64,
        )
        matrices = self.kinematics.palm_matrices(self.arm_q).clone()
        matrices[:, :3, :3] = pitch @ matrices[:, :3, :3]
        up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
        tilted = torch.nn.functional.normalize(
            matrices[:, :3, :3].transpose(1, 2) @ up, dim=-1
        )
        angles = torch.arccos(
            (tilted * reference).sum(dim=1).clamp(-1.0, 1.0)
        )
        torch.testing.assert_close(
            angles, torch.full_like(angles, angle), atol=1e-9, rtol=0.0
        )

    def test_the_vectors_are_unit_length(self):
        from simtoolreal_animrl.envs.transform_bank import palm_tilt_in_palm_frame

        tilt = palm_tilt_in_palm_frame(self.kinematics, self.arm_q)
        torch.testing.assert_close(
            tilt.norm(dim=1), torch.ones(tilt.shape[0], dtype=tilt.dtype)
        )

    def test_the_squared_chord_matches_the_angle_for_small_errors(self):
        """Why the reward uses the chord instead of arccos.

        The chord equals the angle to second order and its derivative is finite
        at zero, where arccos's is not -- and zero error is exactly where a
        converged policy sits.
        """
        # chord/angle is sinc(angle/2) = 1 - angle^2/24 + ..., so the agreement
        # is second order rather than exact; 0.1% up to 5 degrees is the claim.
        for degrees in (0.5, 2.0, 5.0):
            angle = math.radians(degrees)
            chord = math.sqrt(2.0 * (1.0 - math.cos(angle)))
            self.assertLess(abs(chord / angle - 1.0), 1e-3)
        # And it is still monotonic far out, so a large error is not rewarded.
        chords = [
            2.0 * (1.0 - math.cos(math.radians(d))) for d in (10, 45, 90, 135)
        ]
        self.assertEqual(chords, sorted(chords))


class TransferTest(OperationalSpaceFixture):
    def test_transferred_wrist_jacobian_is_the_palm_jacobian(self):
        """The whole frame argument for this controller, in one assertion.

        The tolerance is set by ``pytorch_kinematics``, not by the transfer. The
        angular rows are copied across verbatim -- no arithmetic of ours touches
        them -- and they still disagree with the palm chain by 2.7e-8, slightly
        more than the linear rows this function actually computes. That residual
        is two independently built chains composing the same fixed joints in a
        different order, so 1e-7 is the floor any comparison against this ground
        truth can have.
        """
        wrist_jacobian = self.wrist_chain.jacobian(self.arm_q)
        transferred = transfer_jacobian(
            wrist_jacobian, self.wrist_offset_world(self.arm_q)
        )
        expected = self.kinematics.jacobian(self.arm_q)
        self.assertEqual(tuple(transferred.shape), tuple(expected.shape))
        torch.testing.assert_close(transferred, expected, atol=1e-7, rtol=0.0)

    def test_a_zero_offset_leaves_the_jacobian_alone(self):
        wrist_jacobian = self.wrist_chain.jacobian(self.arm_q)
        offset = torch.zeros(
            (wrist_jacobian.shape[0], 3), dtype=wrist_jacobian.dtype
        )
        torch.testing.assert_close(
            transfer_jacobian(wrist_jacobian, offset), wrist_jacobian
        )

    def test_the_angular_rows_survive_the_transfer(self):
        """A rigid attachment cannot change the body's angular velocity."""
        wrist_jacobian = self.wrist_chain.jacobian(self.arm_q)
        transferred = transfer_jacobian(
            wrist_jacobian, self.wrist_offset_world(self.arm_q)
        )
        torch.testing.assert_close(transferred[:, 3:], wrist_jacobian[:, 3:])

    def test_the_transfer_is_worth_doing(self):
        """Guards the tests above from passing on an accidental no-op.

        Skipping the transfer and using the wrist Jacobian directly misplaces the
        linear rows by 0.0738 -- the palm offset itself, which is what a lever arm
        error of exactly ``|r|`` looks like. The transfer removes an error six
        orders of magnitude larger than the residual it leaves behind.
        """
        wrist_jacobian = self.wrist_chain.jacobian(self.arm_q)
        palm_jacobian = self.kinematics.jacobian(self.arm_q)
        untransferred = (wrist_jacobian[:, :3] - palm_jacobian[:, :3]).abs().max()
        self.assertAlmostEqual(float(untransferred), 0.0738, places=4)


class DampedLeastSquaresTest(OperationalSpaceFixture):
    def test_one_step_matches_the_offline_solver(self):
        """The duplication of solve_palm_ik's formula is deliberate; it must agree.

        ``retarget.solve_palm_ik`` runs this same algebra in a tolerance loop to
        build the transform bank. This reproduces its first iteration exactly.
        """
        damping = 0.05
        jacobian = self.kinematics.jacobian(self.arm_q)
        twist = pose_error(
            self.kinematics.palm_matrices(self.arm_q),
            self.kinematics.palm_matrices(self.arm_q + 0.01),
        )
        actual = damped_least_squares_step(jacobian, twist, damping)

        gram = jacobian @ jacobian.transpose(-1, -2)
        gram = gram + (damping ** 2) * torch.eye(6, dtype=gram.dtype)
        expected = (
            jacobian.transpose(-1, -2) @ torch.linalg.solve(gram, twist.unsqueeze(-1))
        ).squeeze(-1)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0.0)

    def test_a_zero_twist_asks_for_no_motion(self):
        jacobian = self.kinematics.jacobian(self.arm_q)
        twist = torch.zeros((jacobian.shape[0], 6), dtype=jacobian.dtype)
        step = damped_least_squares_step(jacobian, twist, 0.05)
        torch.testing.assert_close(step, torch.zeros_like(step))

    def test_a_commanded_twist_reduces_the_pose_error(self):
        """The round trip a sign error survives the Jacobian tests but not this.

        Walks one control step of the real pipeline: the twist carrying the palm
        onto the next demonstration frame, scaled into an action, scaled back out,
        inverted, and applied. The palm must end up closer than it started.
        """
        translation_speed, rotation_speed = 0.40, 1.0
        current_q = self.arm_q[:-1]
        target = self.kinematics.palm_matrices(self.arm_q[1:])
        twist = pose_error(self.kinematics.palm_matrices(current_q), target)

        action = torch.cat(
            (
                twist[:, :3] / (translation_speed * CONTROL_DT),
                twist[:, 3:] / (rotation_speed * CONTROL_DT),
            ),
            dim=1,
        ).clamp(-1.0, 1.0)
        commanded = torch.cat(
            (
                action[:, :3] * translation_speed * CONTROL_DT,
                action[:, 3:] * rotation_speed * CONTROL_DT,
            ),
            dim=1,
        )
        step = damped_least_squares_step(
            self.kinematics.jacobian(current_q), commanded, 0.05
        )
        moved = self.kinematics.palm_matrices(current_q + step)

        before = pose_error(self.kinematics.palm_matrices(current_q), target)
        after = pose_error(moved, target)
        self.assertLess(
            float(after.norm(dim=1).max()), float(before.norm(dim=1).max())
        )


class DemonstrationScaleTest(OperationalSpaceFixture):
    """The speed scales in cfg.control are derived from these numbers."""

    def test_the_demonstration_fits_inside_the_speed_limit(self):
        """Saturation bounds the twist by MAGNITUDE, so that is what must fit.

        A per-axis check would pass while a diagonal command still saturated: at
        |a| = 0.39 per axis, a three-axis request has norm up to 0.68.
        """
        with np.load(str(DEMO)) as archive:
            arm_q = torch.as_tensor(
                np.asarray(archive["arm_q"]), dtype=torch.float64
            )
        poses = self.kinematics.palm_matrices(arm_q)
        twist = pose_error(poses[:-1], poses[1:])
        action = torch.cat(
            (
                twist[:, :3] / (0.40 * CONTROL_DT),
                twist[:, 3:] / (1.0 * CONTROL_DT),
            ),
            dim=1,
        )
        # 0.373 per axis and 0.394 by norm when this was written. The margin
        # below 1.0 is what the policy has left for correcting RSI perturbations
        # and rejecting disturbances, so a regression here means the speed
        # scales need revisiting.
        self.assertLess(float(action.abs().max()), 0.5)
        self.assertLess(float(action[:, :3].norm(dim=1).max()), 0.6)
        self.assertLess(float(action[:, 3:].norm(dim=1).max()), 0.6)


if __name__ == "__main__":
    unittest.main()
