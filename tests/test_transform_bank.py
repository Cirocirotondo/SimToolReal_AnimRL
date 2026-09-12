"""The per-episode bank of feasible transforms and their retargeted clips."""

import math
import unittest

import torch

from simtoolreal_animrl import ROOT_DIR
from simtoolreal_animrl.envs.demonstration import JointDemonstration60Hz
from simtoolreal_animrl.envs.retarget import (
    ARM_JOINT_COUNT,
    PalmKinematics,
    cube_pose_from_base_frame,
    cube_pose_to_base_frame,
)
from simtoolreal_animrl.envs.transform_bank import (
    TransformBank,
    build_transform_bank,
    nearest_transform_indices,
)


URDF = ROOT_DIR / "assets/urdf/ur5e_delto_description/ur5e_right_dg5f_mount_60deg.urdf"
DEMO = ROOT_DIR / (
    "demonstrations/"
    "demo_20260727_152551_335339_60hz_cube_collision_resolved_stable_grasp.npz"
)
FRAME_STRIDE = 37
LEVER_ARM = 0.1
SOLVER_TOLERANCE = 2e-4


class NearestTransformTest(unittest.TestCase):
    def test_translation_and_wrapped_yaw_select_the_nearest_entry(self):
        bank_translation = torch.tensor(
            [[0.0, 0.0, 0.0], [0.08, 0.10, 0.0]], dtype=torch.float32
        )
        bank_yaw = torch.deg2rad(torch.tensor([179.0, 20.0]))
        episode_translation = torch.tensor(
            [[0.079, 0.099, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32
        )
        episode_yaw = torch.deg2rad(torch.tensor([21.0, -179.0]))
        selected = nearest_transform_indices(
            episode_translation, episode_yaw,
            bank_translation, bank_yaw, yaw_lever_arm_m=0.1,
        )
        torch.testing.assert_close(selected, torch.tensor([1, 0]))


def subsampled_demonstration():
    """A short clip with the same interface, so the tests are seconds not minutes."""
    full = JointDemonstration60Hz.load(DEMO, device="cpu")
    take = lambda tensor: tensor[::FRAME_STRIDE]
    return JointDemonstration60Hz(
        path=full.path,
        timestamp=take(full.timestamp),
        monotonic_timestamp=take(full.monotonic_timestamp),
        q=take(full.q),
        dq=take(full.dq),
        cube_pose=take(full.cube_pose),
        cube_linear_velocity=take(full.cube_linear_velocity),
        cube_angular_velocity=take(full.cube_angular_velocity),
        frequency_hz=full.frequency_hz,
    )


class FrameRoundTripTest(unittest.TestCase):
    def test_the_ur_base_conversion_inverts_exactly(self):
        """Applying the forward map twice returns -q, not q: two pi rotations
        make 2 pi, which is -1 in quaternion space. The inverse uses the
        conjugate, and the demonstration carries w < 0 poses that would
        otherwise come back flipped."""
        demonstration = subsampled_demonstration()
        original = demonstration.cube_pose.double()
        torch.testing.assert_close(
            cube_pose_from_base_frame(cube_pose_to_base_frame(original)),
            original,
            atol=1e-7,
            rtol=0.0,
        )

    def test_the_forward_map_applied_twice_negates_the_quaternion(self):
        original = subsampled_demonstration().cube_pose.double()
        twice = cube_pose_to_base_frame(cube_pose_to_base_frame(original))
        torch.testing.assert_close(twice[:, :3], original[:, :3], atol=1e-7, rtol=0.0)
        torch.testing.assert_close(twice[:, 3:], -original[:, 3:], atol=1e-7, rtol=0.0)


class BankFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.demonstration = subsampled_demonstration()
        cls.kinematics = PalmKinematics(URDF, device="cpu")


class IdentityBankTest(BankFixture):
    """A bank of nothing-but-identity transforms must be the demonstration."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.bank = build_transform_bank(
            cls.kinematics,
            cls.demonstration,
            transform_count=2,
            translation_m=0.0,
            yaw_low_rad=0.0,
            yaw_high_rad=0.0,
            lever_arm_m=LEVER_ARM,
            seed=0,
            batch=4,
            verbose=False,
        )

    def test_the_joint_trajectory_comes_back(self):
        torch.testing.assert_close(
            self.bank.q[0],
            self.demonstration.q.double(),
            atol=1e-3,
            rtol=0.0,
        )

    def test_the_recorded_velocities_come_back(self):
        """The Jacobian mapping exists so this holds; finite differencing the
        retargeted path would not reproduce measured velocities."""
        torch.testing.assert_close(
            self.bank.dq[0],
            self.demonstration.dq.double(),
            atol=1e-3,
            rtol=0.0,
        )

    def test_the_cube_track_comes_back_in_the_frame_the_environment_expects(self):
        torch.testing.assert_close(
            self.bank.cube_pose[0],
            self.demonstration.cube_pose.double(),
            atol=1e-6,
            rtol=0.0,
        )
        torch.testing.assert_close(
            self.bank.cube_linear_velocity[0],
            self.demonstration.cube_linear_velocity.double(),
            atol=1e-9,
            rtol=0.0,
        )

    def test_the_reference_curve_is_stored_once(self):
        """Not once per transform: in the bar's frame it does not depend on the
        transform, and storing it per transform would invite them to diverge."""
        self.assertEqual(
            tuple(self.bank.reference_keypoints.shape),
            (self.demonstration.sample_count, 9, 3),
        )


class SampledBankTest(BankFixture):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.bank = build_transform_bank(
            cls.kinematics,
            cls.demonstration,
            transform_count=16,
            translation_m=0.20,
            yaw_low_rad=math.radians(-22.5),
            yaw_high_rad=math.radians(90.0),
            lever_arm_m=LEVER_ARM,
            seed=1,
            batch=32,
            verbose=False,
        )

    def test_every_transform_lands_inside_the_requested_range(self):
        self.assertGreaterEqual(float(self.bank.yaw_rad.min()), math.radians(-22.5))
        self.assertLessEqual(float(self.bank.yaw_rad.max()), math.radians(90.0))
        self.assertLessEqual(float(self.bank.translation[:, :2].abs().max()), 0.20)
        torch.testing.assert_close(
            self.bank.translation[:, 2],
            torch.zeros(self.bank.transform_count, dtype=torch.float64),
        )

    def test_the_acceptance_rate_is_reported(self):
        """A silently biased distribution is the failure this guards against."""
        self.assertGreater(self.bank.acceptance, 0.0)
        self.assertLessEqual(self.bank.acceptance, 1.0)

    def test_sampling_gathers_one_pair_per_environment(self):
        transforms = torch.tensor([0, 7, 15])
        frames = torch.tensor([0, 5, self.bank.last_index])
        sample = self.bank.sample(transforms, frames)
        self.assertEqual(tuple(sample.q.shape), (3, 26))
        self.assertEqual(tuple(sample.cube_pose.shape), (3, 7))
        torch.testing.assert_close(sample.q[1], self.bank.q[7, 5])

    def test_keypoints_need_no_transform_index(self):
        frames = torch.tensor([0, 3, 9])
        torch.testing.assert_close(
            self.bank.keypoints_at(frames), self.bank.reference_keypoints[frames]
        )

    def test_an_index_outside_the_bank_is_refused(self):
        with self.assertRaises(IndexError):
            self.bank.sample(
                torch.tensor([self.bank.transform_count]), torch.tensor([0])
            )
        with self.assertRaises(IndexError):
            self.bank.sample(
                torch.tensor([0]), torch.tensor([self.bank.sample_count])
            )

    def test_it_survives_a_save_and_load(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = "{}/bank.pt".format(directory)
            self.bank.save(path)
            restored = TransformBank.load(path)
        torch.testing.assert_close(restored.q, self.bank.q)
        torch.testing.assert_close(restored.cube_pose, self.bank.cube_pose)
        self.assertEqual(restored.acceptance, self.bank.acceptance)

    def test_the_fingers_are_copied_not_solved(self):
        """Stage 1's whole shortcut: a rigid transform preserves the hand, so
        only the six arm joints are solved for."""
        for index in range(self.bank.transform_count):
            torch.testing.assert_close(
                self.bank.q[index][:, ARM_JOINT_COUNT:],
                self.demonstration.q.double()[:, ARM_JOINT_COUNT:],
            )

    def test_a_clip_demanding_impossible_joint_speeds_is_rejected(self):
        """Every accepted clip must be trackable by the real arm.

        A palm path passing near a wrist singularity solves at every individual
        frame and is still useless: following it needs joint speeds the arm does
        not have. Measured on a bank built without this check, 9.6% of accepted
        transforms exceeded the limit, the worst asking 12.2 rad/s of a joint
        capped at pi.
        """
        from simtoolreal_animrl.envs.transform_bank import (
            ARM_JOINT_VELOCITY_LIMIT_RAD_S,
        )

        intervals = torch.diff(self.demonstration.monotonic_timestamp.double())
        dt = float(intervals.median())
        speed = (
            (self.bank.q[:, 1:, :ARM_JOINT_COUNT]
             - self.bank.q[:, :-1, :ARM_JOINT_COUNT]).abs().amax(dim=-1) / dt
        )
        self.assertLessEqual(
            float(speed.max()), 0.5 * ARM_JOINT_VELOCITY_LIMIT_RAD_S + 1e-9
        )

    def test_reference_velocities_stay_physical(self):
        """The stored dq is what the velocity reward asks the policy to match."""
        self.assertLess(float(self.bank.dq.abs().max()), 50.0)

    def test_an_impossible_range_fails_loudly(self):
        with self.assertRaises(RuntimeError):
            build_transform_bank(
                self.kinematics,
                self.demonstration,
                transform_count=8,
                translation_m=1.5,
                yaw_low_rad=0.0,
                yaw_high_rad=0.0,
                lever_arm_m=LEVER_ARM,
                seed=2,
                batch=8,
                max_attempts_multiplier=1,
                verbose=False,
            )


if __name__ == "__main__":
    unittest.main()
