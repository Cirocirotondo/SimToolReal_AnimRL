"""Rewriting the demonstration for a bar that is somewhere else.

The headline test here is :meth:`InvariantTest.test_the_reference_does_not_
depend_on_the_transform`. Everything else is scaffolding for it.
"""

import math
import unittest

import numpy as np
import torch

from simtoolreal_animrl import ROOT_DIR
from simtoolreal_animrl.envs.cuboid_symmetry import (
    apply_cuboid_symmetry,
    canonicalize_cuboid_orientation,
    cuboid_rotation_symmetries,
)
from simtoolreal_animrl.envs.keypoints import keypoints_in_object_frame
from simtoolreal_animrl.envs.retarget import (
    ARM_JOINT_COUNT,
    PalmKinematics,
    cube_pose_to_base_frame,
    reference_keypoints_in_object_frame,
    retarget_clip,
    retarget_clip_best_branch,
    solve_palm_ik,
    transform_points,
    yaw_quaternion,
)
from simtoolreal_animrl.envs.rotations import quat_multiply, quat_to_matrix


URDF = ROOT_DIR / "assets/urdf/ur5e_delto_description/ur5e_right_dg5f_mount_60deg.urdf"
DEMO = ROOT_DIR / (
    "demonstrations/"
    "demo_20260727_152551_335339_60hz_cube_collision_resolved_stable_grasp.npz"
)
BAR = [0.075, 0.025, 0.025]
LEVER_ARM = 0.1
# Every 37th frame: 30 samples spanning approach, grasp and lift, which is
# enough to catch a frame-convention error without a minute of solver time.
FRAME_STRIDE = 37
# The IK stops at 1e-4; nothing downstream of it can be tighter than that.
SOLVER_TOLERANCE_M = 2e-4


class RetargetFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kinematics = PalmKinematics(URDF, device="cpu")
        with np.load(str(DEMO)) as archive:
            arm = np.asarray(archive["arm_q"])[::FRAME_STRIDE]
            hand = np.asarray(archive["hand_q_measured"])[::FRAME_STRIDE]
            cube = np.asarray(archive["cube_pose"])[::FRAME_STRIDE]
        cls.demo_arm_q = torch.as_tensor(arm, dtype=torch.float64)
        cls.demo_q = torch.as_tensor(
            np.concatenate((arm, hand), axis=1), dtype=torch.float64
        )
        cls.demo_cube = cube_pose_to_base_frame(
            torch.as_tensor(cube, dtype=torch.float64)
        )
        cls.pivot = cls.demo_cube[0, :3]
        cls.symmetries = cuboid_rotation_symmetries(BAR)


class ConventionTest(RetargetFixture):
    def test_the_ur_base_frame_is_the_robot_base_turned_by_pi(self):
        """The demonstration records the cube in the UR controller's frame,
        which is base_link rotated pi about z."""
        pose = torch.tensor(
            [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=torch.float64
        )
        converted = cube_pose_to_base_frame(pose)
        torch.testing.assert_close(
            converted[:3], torch.tensor([-0.1, -0.2, 0.3], dtype=torch.float64)
        )
        torch.testing.assert_close(
            quat_to_matrix(converted[3:7]),
            quat_to_matrix(
                torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float64)
            ),
        )

    def test_yaw_is_a_rotation_about_the_vertical_axis(self):
        vertical = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
        matrix = quat_to_matrix(
            yaw_quaternion(torch.tensor(math.radians(37.0), dtype=torch.float64))
        )
        torch.testing.assert_close(matrix @ vertical, vertical)

    def test_the_pivot_stays_put(self):
        """Yaw turns the bar in place; it does not swing it around the robot."""
        pivot = torch.tensor([0.3, -0.4, 0.5], dtype=torch.float64)
        moved = transform_points(
            pivot,
            torch.tensor(math.radians(80.0), dtype=torch.float64),
            torch.zeros(3, dtype=torch.float64),
            pivot,
        )
        torch.testing.assert_close(moved, pivot)

    def test_zero_yaw_is_a_pure_translation(self):
        points = torch.randn(
            9, 3, generator=torch.Generator().manual_seed(2), dtype=torch.float64
        )
        offset = torch.tensor([0.1, -0.05, 0.0], dtype=torch.float64)
        torch.testing.assert_close(
            transform_points(
                points,
                torch.zeros((), dtype=torch.float64),
                offset,
                self.pivot,
            ),
            points + offset,
        )


class SolverTest(RetargetFixture):
    def test_the_chain_to_the_palm_is_the_arm_and_nothing_else(self):
        """If the fingers ever appeared here, copying them verbatim would be
        wrong and the whole Stage 1 shortcut would collapse."""
        self.assertEqual(len(self.kinematics.joint_names), ARM_JOINT_COUNT)
        self.assertEqual(self.kinematics.joint_names[0], "shoulder_pan_joint")
        self.assertEqual(self.kinematics.joint_names[-1], "wrist_3_joint")

    def test_a_target_already_reached_leaves_the_seed_alone(self):
        seed = self.demo_arm_q[:4]
        solved = solve_palm_ik(
            self.kinematics, self.kinematics.palm_matrices(seed), seed
        )
        torch.testing.assert_close(solved, seed, atol=1e-9, rtol=0.0)

    def test_fingertips_come_from_the_urdf_tip_links(self):
        keypoints = self.kinematics.hand_keypoints(self.demo_q[:3], LEVER_ARM)
        self.assertEqual(tuple(keypoints.shape), (3, 9, 3))

    def test_it_refuses_the_wrong_number_of_joints(self):
        with self.assertRaises(ValueError):
            self.kinematics.hand_keypoints(self.demo_arm_q[:2], LEVER_ARM)


class IdentityTest(RetargetFixture):
    def test_the_identity_transform_reproduces_the_demonstration(self):
        result = retarget_clip(
            self.kinematics,
            self.demo_arm_q,
            torch.zeros(1, dtype=torch.float64),
            torch.zeros(1, 3, dtype=torch.float64),
            self.pivot,
        )
        self.assertLess(float(result.position_residual_m.max()), SOLVER_TOLERANCE_M)
        self.assertLess(
            float((result.arm_q[:, 0, :] - self.demo_arm_q).abs().max()), 1e-3
        )

    def test_an_unreachable_transform_is_reported_not_hidden(self):
        """Rejection depends on the residual being honest about failure."""
        result = retarget_clip(
            self.kinematics,
            self.demo_arm_q,
            torch.zeros(1, dtype=torch.float64),
            torch.tensor([[0.0, -0.85, 0.0]], dtype=torch.float64),
            self.pivot,
        )
        self.assertGreater(float(result.position_residual_m.max()), 0.01)

    def test_branch_search_avoids_the_yaw_52_wrist_singularity(self):
        """Regression for a reachable pose the demo-seeded branch rejected."""
        with np.load(str(DEMO)) as archive:
            arm = torch.as_tensor(
                np.asarray(archive["arm_q"])[::2], dtype=torch.float64
            )
        result = retarget_clip_best_branch(
            self.kinematics,
            arm,
            torch.tensor([math.radians(52.0)], dtype=torch.float64),
            torch.zeros(1, 3, dtype=torch.float64),
            self.pivot,
            control_dt=2.0 / 60.0,
        )
        peak_speed = float(
            ((result.arm_q[1:] - result.arm_q[:-1]).abs() / (2.0 / 60.0)).max()
        )
        self.assertLess(float(result.position_residual_m.max()), 1e-3)
        self.assertGreater(float(result.limit_margin_rad.min()), 0.05)
        self.assertLess(peak_speed, 0.5 * math.pi)
        frame_zero_q = torch.cat(
            (result.arm_q[0, 0], self.demo_q[0, ARM_JOINT_COUNT:]), dim=0
        ).unsqueeze(0)
        poses = self.kinematics.full_chain.forward_kinematics(frame_zero_q)
        elbow_z = poses["forearm_link"].get_matrix()[0, 2, 3]
        wrist_z = poses["wrist_1_link"].get_matrix()[0, 2, 3]
        self.assertGreater(
            float(elbow_z), float(wrist_z),
            "the fallback must keep the physical elbow above the wrist",
        )

    def test_branch_search_recovers_the_offset_minus_22_point(self):
        """Regression for the second visually reachable false negative."""
        with np.load(str(DEMO)) as archive:
            arm = torch.as_tensor(
                np.asarray(archive["arm_q"])[::2], dtype=torch.float64
            )
        result = retarget_clip_best_branch(
            self.kinematics,
            arm,
            torch.tensor([math.radians(-22.5)], dtype=torch.float64),
            torch.tensor([[0.06, -0.15, 0.0]], dtype=torch.float64),
            self.pivot,
            control_dt=2.0 / 60.0,
        )
        peak_speed = float(
            ((result.arm_q[1:] - result.arm_q[:-1]).abs() / (2.0 / 60.0)).max()
        )
        self.assertLess(float(result.position_residual_m.max()), 1e-3)
        self.assertGreater(float(result.limit_margin_rad.min()), 0.05)
        self.assertLess(peak_speed, 0.5 * math.pi)
        self.assertGreater(
            float(self.kinematics.elbow_height_margin(result.arm_q[:, 0]).min()),
            0.0,
        )
        self.assertGreater(
            float(result.arm_q[0, 0, 0]), 0.0,
            "use the requested opposite-side shoulder/wrist branch",
        )
        self.assertLess(float(result.arm_q[0, 0, 3]), 1.5)


class InvariantTest(RetargetFixture):
    def test_the_reference_does_not_depend_on_the_transform(self):
        """Expressed in the bar's frame, the retargeted hand lands exactly where
        the demonstration's hand was. That is why one reference curve serves
        every episode, and why the reward needs no per-transform lookup.

        This exercises the transform convention, the IK, the keypoint
        construction and the symmetry canonicalisation at once. If it fails,
        one of those four has drifted.
        """
        reference = reference_keypoints_in_object_frame(
            self.kinematics, self.demo_q, self.demo_cube, LEVER_ARM
        )
        frames = self.demo_q.shape[0]
        yaws = torch.tensor(
            [0.0, math.radians(30.0), math.radians(-60.0), math.radians(88.0)],
            dtype=torch.float64,
        )
        translations = torch.tensor(
            [[0.0, 0.0, 0.0], [0.08, 0.05, 0.0], [-0.05, 0.10, 0.0], [0.12, -0.04, 0.0]],
            dtype=torch.float64,
        )
        result = retarget_clip(
            self.kinematics, self.demo_arm_q, yaws, translations, self.pivot
        )

        for index in range(len(yaws)):
            with self.subTest(yaw_degrees=round(math.degrees(float(yaws[index])))):
                self.assertLess(
                    float(result.position_residual_m[:, index].max()),
                    SOLVER_TOLERANCE_M,
                    "this transform should be reachable",
                )
                joint_positions = torch.cat(
                    (result.arm_q[:, index, :], self.demo_q[:, ARM_JOINT_COUNT:]),
                    dim=-1,
                )
                cube_position = transform_points(
                    self.demo_cube[:, :3], yaws[index], translations[index], self.pivot
                )
                cube_orientation = quat_multiply(
                    yaw_quaternion(yaws[index]).expand(frames, 4),
                    self.demo_cube[:, 3:7],
                )
                # Choose the symmetry representative once, from the pose the
                # episode resets to, then hold it -- see cuboid_symmetry.
                _, chosen = canonicalize_cuboid_orientation(
                    cube_orientation[0],
                    self.symmetries,
                    cube_orientation[0],
                    return_index=True,
                )
                actual = keypoints_in_object_frame(
                    self.kinematics.hand_keypoints(joint_positions, LEVER_ARM),
                    cube_position,
                    apply_cuboid_symmetry(
                        cube_orientation, self.symmetries, chosen.expand(frames)
                    ),
                )
                self.assertLess(
                    float((actual - reference).abs().max()), SOLVER_TOLERANCE_M
                )


if __name__ == "__main__":
    unittest.main()
