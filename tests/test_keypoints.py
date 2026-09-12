"""The hand as nine points, measured from the bar."""

import math
import unittest

import torch

from simtoolreal_animrl.envs.keypoints import (
    KEYPOINT_COUNT,
    hand_keypoints,
    keypoint_gaussian,
    keypoint_tracking_error,
    keypoints_in_object_frame,
    split_palm_and_fingertips,
)
from simtoolreal_animrl.envs.rotations import (
    normalize_canonical_quaternion,
    quat_multiply,
    quat_rotate,
)


LEVER_ARM = 0.1


def scene(count=16, seed=0):
    generator = torch.Generator().manual_seed(seed)
    palm_position = torch.randn(count, 3, generator=generator, dtype=torch.float64)
    palm_orientation = normalize_canonical_quaternion(
        torch.randn(count, 4, generator=generator, dtype=torch.float64)
    )
    fingertips = torch.randn(count, 5, 3, generator=generator, dtype=torch.float64)
    return palm_position, palm_orientation, fingertips


class HandKeypointTest(unittest.TestCase):
    def test_it_produces_four_palm_points_and_five_fingertips(self):
        keypoints = hand_keypoints(*scene(), LEVER_ARM)
        self.assertEqual(tuple(keypoints.shape), (16, KEYPOINT_COUNT, 3))

    def test_the_first_point_is_the_palm_origin(self):
        palm_position, palm_orientation, fingertips = scene()
        keypoints = hand_keypoints(palm_position, palm_orientation, fingertips, LEVER_ARM)
        torch.testing.assert_close(keypoints[:, 0, :], palm_position)

    def test_the_axis_points_sit_at_the_lever_arm(self):
        """The lever arm is the exchange rate between a metre of position error
        and a radian of orientation error, so it has to be exactly what it says."""
        palm_position, palm_orientation, fingertips = scene()
        keypoints = hand_keypoints(palm_position, palm_orientation, fingertips, LEVER_ARM)
        offsets = keypoints[:, 1:4, :] - palm_position.unsqueeze(1)
        torch.testing.assert_close(
            torch.linalg.vector_norm(offsets, dim=-1),
            torch.full((16, 3), LEVER_ARM, dtype=torch.float64),
        )

    def test_the_axis_points_are_rigidly_attached(self):
        """They must turn with the palm, or they carry no orientation at all."""
        palm_position, palm_orientation, fingertips = scene()
        keypoints = hand_keypoints(palm_position, palm_orientation, fingertips, LEVER_ARM)
        axes = torch.eye(3, dtype=torch.float64) * LEVER_ARM
        expected = palm_position.unsqueeze(1) + quat_rotate(
            palm_orientation.unsqueeze(1).expand(16, 3, 4), axes.expand(16, 3, 3)
        )
        torch.testing.assert_close(keypoints[:, 1:4, :], expected)

    def test_the_last_five_are_the_fingertips(self):
        palm_position, palm_orientation, fingertips = scene()
        keypoints = hand_keypoints(palm_position, palm_orientation, fingertips, LEVER_ARM)
        torch.testing.assert_close(keypoints[:, 4:, :], fingertips)

    def test_it_rejects_a_non_positive_lever_arm(self):
        with self.assertRaises(ValueError):
            hand_keypoints(*scene(), 0.0)


class ObjectFrameTest(unittest.TestCase):
    def test_the_reference_does_not_depend_on_where_the_bar_is(self):
        """The invariant the whole approach rests on: move the bar and the hand
        together, and the keypoints in the bar's frame do not change. One
        reference curve serves every episode."""
        palm_position, palm_orientation, fingertips = scene(count=8, seed=7)
        keypoints = hand_keypoints(palm_position, palm_orientation, fingertips, LEVER_ARM)
        object_position = torch.randn(
            8, 3, generator=torch.Generator().manual_seed(11), dtype=torch.float64
        )
        object_orientation = normalize_canonical_quaternion(
            torch.randn(
                8, 4, generator=torch.Generator().manual_seed(12), dtype=torch.float64
            )
        )
        baseline = keypoints_in_object_frame(
            keypoints, object_position, object_orientation
        )

        transform = normalize_canonical_quaternion(
            torch.randn(
                8, 4, generator=torch.Generator().manual_seed(13), dtype=torch.float64
            )
        )
        offset = torch.randn(
            8, 3, generator=torch.Generator().manual_seed(14), dtype=torch.float64
        )
        moved_keypoints = quat_rotate(
            transform.unsqueeze(1).expand(8, KEYPOINT_COUNT, 4), keypoints
        ) + offset.unsqueeze(1)
        moved = keypoints_in_object_frame(
            moved_keypoints,
            quat_rotate(transform, object_position) + offset,
            quat_multiply(transform, object_orientation),
        )
        torch.testing.assert_close(moved, baseline)

    def test_a_bar_at_the_origin_with_no_rotation_changes_nothing(self):
        keypoints = hand_keypoints(*scene(), LEVER_ARM)
        identity = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64).expand(16, 4)
        torch.testing.assert_close(
            keypoints_in_object_frame(
                keypoints, torch.zeros(16, 3, dtype=torch.float64), identity
            ),
            keypoints,
        )


class TrackingErrorTest(unittest.TestCase):
    def test_it_is_the_mean_squared_keypoint_distance(self):
        keypoints = torch.zeros(2, KEYPOINT_COUNT, 3, dtype=torch.float64)
        reference = torch.zeros_like(keypoints)
        reference[0, :, 0] = 0.03          # every keypoint off by 3 cm in x
        error = keypoint_tracking_error(keypoints, reference)
        self.assertAlmostEqual(float(error[0]), 0.03 ** 2, places=12)
        self.assertAlmostEqual(float(error[1]), 0.0, places=12)

    def test_a_sigma_in_metres_means_what_it_reads_as(self):
        """An RMS error of exactly sigma should score exp(-1/2)."""
        keypoints = torch.zeros(1, KEYPOINT_COUNT, 3, dtype=torch.float64)
        reference = torch.zeros_like(keypoints)
        reference[0, :, 0] = 0.05
        reward = keypoint_gaussian(
            keypoint_tracking_error(keypoints, reference), 0.05
        )
        self.assertAlmostEqual(float(reward), math.exp(-0.5), places=12)

    def test_mismatched_shapes_are_refused(self):
        with self.assertRaises(ValueError):
            keypoint_tracking_error(
                torch.zeros(2, KEYPOINT_COUNT, 3), torch.zeros(2, 5, 3)
            )

    def test_the_split_keeps_palm_and_fingertips_apart(self):
        """Averaged into one term, five fingertips outvote four palm points."""
        keypoints = hand_keypoints(*scene(), LEVER_ARM)
        palm, fingertips = split_palm_and_fingertips(keypoints)
        self.assertEqual(tuple(palm.shape), (16, 4, 3))
        self.assertEqual(tuple(fingertips.shape), (16, 5, 3))
        torch.testing.assert_close(
            torch.cat((palm, fingertips), dim=-2), keypoints
        )


if __name__ == "__main__":
    unittest.main()
