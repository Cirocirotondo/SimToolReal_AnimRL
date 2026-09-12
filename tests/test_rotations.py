"""Rotation arithmetic the object-frame reference is built on."""

import math
import unittest

import torch

from simtoolreal_animrl.envs.rotations import (
    matrix_to_quat,
    normalize_canonical_quaternion,
    quat_rotate,
    quat_to_matrix,
    quat_to_rotation_6d,
    rotation_6d_to_matrix,
)


def random_quaternions(count=512, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return normalize_canonical_quaternion(
        torch.randn(count, 4, generator=generator, dtype=torch.float64)
    )


def rotation_about(axis, angle_rad):
    axis = torch.nn.functional.normalize(
        torch.as_tensor(axis, dtype=torch.float64), dim=-1
    )
    half = angle_rad / 2.0
    return normalize_canonical_quaternion(
        torch.cat(
            (axis * math.sin(half), torch.tensor([math.cos(half)], dtype=torch.float64))
        )
    )


class QuaternionMatrixTest(unittest.TestCase):
    def test_the_matrix_agrees_with_the_quaternion_rotation(self):
        """Two ways to rotate a vector must never disagree."""
        quaternions = random_quaternions()
        vectors = torch.randn(
            512, 3, generator=torch.Generator().manual_seed(1), dtype=torch.float64
        )
        matrices = quat_to_matrix(quaternions)
        torch.testing.assert_close(
            (matrices @ vectors.unsqueeze(-1)).squeeze(-1),
            quat_rotate(quaternions, vectors),
        )

    def test_the_matrix_is_a_rotation(self):
        matrices = quat_to_matrix(random_quaternions())
        identity = torch.eye(3, dtype=torch.float64).expand_as(matrices)
        torch.testing.assert_close(matrices @ matrices.transpose(-1, -2), identity)
        torch.testing.assert_close(
            torch.det(matrices), torch.ones(matrices.shape[0], dtype=torch.float64)
        )

    def test_matrix_to_quat_round_trips(self):
        quaternions = random_quaternions()
        torch.testing.assert_close(
            matrix_to_quat(quat_to_matrix(quaternions)), quaternions
        )

    def test_matrix_to_quat_survives_a_half_turn(self):
        """Half the cuboid symmetry group is exactly 180 degrees, where the
        trace-only formula divides by zero."""
        for axis in ((1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 1, 1), (0, 1, -1)):
            matrix = quat_to_matrix(rotation_about(axis, math.pi))
            self.assertAlmostEqual(float(matrix.diagonal().sum()), -1.0, places=12)
            torch.testing.assert_close(
                quat_to_matrix(matrix_to_quat(matrix)), matrix
            )

    def test_identity_maps_to_the_identity_quaternion(self):
        torch.testing.assert_close(
            matrix_to_quat(torch.eye(3, dtype=torch.float64)),
            torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64),
        )


class Rotation6DTest(unittest.TestCase):
    def test_it_round_trips_through_gram_schmidt(self):
        matrices = quat_to_matrix(random_quaternions())
        torch.testing.assert_close(
            rotation_6d_to_matrix(quat_to_rotation_6d(matrix_to_quat(matrices))),
            matrices,
        )

    def test_the_double_cover_disappears(self):
        """The reason the observation uses 6D at all: q and -q are the same
        rotation, and a network reading the raw four numbers cannot know that."""
        quaternions = random_quaternions()
        torch.testing.assert_close(
            quat_to_rotation_6d(quaternions), quat_to_rotation_6d(-quaternions)
        )

    def test_batch_shapes_survive(self):
        quaternions = random_quaternions(count=35).reshape(5, 7, 4)
        self.assertEqual(tuple(quat_to_rotation_6d(quaternions).shape), (5, 7, 6))


if __name__ == "__main__":
    unittest.main()
