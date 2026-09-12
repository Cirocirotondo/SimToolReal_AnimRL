"""A bar that never moved must not appear to jump."""

import math
import unittest

import torch

from simtoolreal_animrl.envs.cuboid_symmetry import (
    apply_cuboid_symmetry,
    canonicalize_cuboid_orientation,
    cuboid_rotation_symmetries,
)
from simtoolreal_animrl.envs.rotations import (
    normalize_canonical_quaternion,
    quat_multiply,
    quat_to_matrix,
)


BAR = [0.075, 0.025, 0.025]  # the 0.15 x 0.05 x 0.05 cuboid this project grasps


def yaw(degrees):
    half = math.radians(degrees) / 2.0
    return normalize_canonical_quaternion(
        torch.tensor([0.0, 0.0, math.sin(half), math.cos(half)], dtype=torch.float64)
    )


def yaw_degrees(quaternion):
    return math.degrees(2.0 * math.atan2(float(quaternion[2]), float(quaternion[3])))


class SymmetryGroupTest(unittest.TestCase):
    def test_the_group_order_follows_from_the_extents(self):
        """Derived, not typed: two equal extents give eight, a true cube
        twenty-four, three distinct extents four."""
        self.assertEqual(len(cuboid_rotation_symmetries(BAR)), 8)
        self.assertEqual(len(cuboid_rotation_symmetries([0.05, 0.05, 0.05])), 24)
        self.assertEqual(len(cuboid_rotation_symmetries([0.075, 0.05, 0.025])), 4)

    def test_every_element_maps_the_bar_onto_itself(self):
        extents = torch.tensor(BAR, dtype=torch.float64)
        matrices = quat_to_matrix(cuboid_rotation_symmetries(BAR))
        torch.testing.assert_close(
            matrices.abs() @ extents, extents.expand(len(matrices), 3)
        )

    def test_the_elements_are_distinct(self):
        symmetries = cuboid_rotation_symmetries(BAR)
        unique = {tuple(q.tolist()) for q in symmetries.round(decimals=9)}
        self.assertEqual(len(unique), len(symmetries))

    def test_it_rejects_a_degenerate_box(self):
        with self.assertRaises(ValueError):
            cuboid_rotation_symmetries([0.075, 0.0, 0.025])


class CanonicalisationTest(unittest.TestCase):
    def setUp(self):
        self.symmetries = cuboid_rotation_symmetries(BAR)
        self.reference = yaw(0.0)

    def canonical(self, degrees):
        return canonicalize_cuboid_orientation(
            yaw(degrees), self.symmetries, self.reference
        )

    def test_yaw_folds_into_a_half_turn(self):
        """A bar at +135 degrees is a bar at -45 degrees, which is why a
        sampling range of 180 degrees covers every planar orientation."""
        self.assertAlmostEqual(yaw_degrees(self.canonical(135.0)), -45.0, places=6)
        self.assertAlmostEqual(yaw_degrees(self.canonical(-135.0)), 45.0, places=6)
        self.assertAlmostEqual(abs(yaw_degrees(self.canonical(180.0))), 0.0, places=6)

    def test_orientations_inside_the_range_are_left_alone(self):
        for degrees in (-80.0, -45.0, 0.0, 45.0, 80.0):
            self.assertAlmostEqual(
                yaw_degrees(self.canonical(degrees)), degrees, places=6
            )

    def test_it_is_idempotent(self):
        once = self.canonical(123.0)
        twice = canonicalize_cuboid_orientation(once, self.symmetries, self.reference)
        torch.testing.assert_close(twice, once)

    def test_all_eight_relabellings_collapse_to_one(self):
        """The property the whole module exists for."""
        generator = torch.Generator().manual_seed(3)
        orientation = normalize_canonical_quaternion(
            torch.randn(64, 4, generator=generator, dtype=torch.float64)
        )
        expected = canonicalize_cuboid_orientation(
            orientation, self.symmetries, self.reference
        )
        count = len(self.symmetries)
        relabelled = quat_multiply(
            orientation.unsqueeze(1).expand(64, count, 4),
            self.symmetries.expand(64, count, 4),
        )
        torch.testing.assert_close(
            canonicalize_cuboid_orientation(
                relabelled, self.symmetries, self.reference
            ),
            expected.unsqueeze(1).expand(64, count, 4),
        )


class HeldChoiceTest(unittest.TestCase):
    """Choosing once and replaying must equal choosing, or the episode-long
    hold would silently differ from the reset-time decision."""

    def test_applying_the_returned_index_reproduces_the_choice(self):
        symmetries = cuboid_rotation_symmetries(BAR)
        reference = yaw(0.0)
        generator = torch.Generator().manual_seed(5)
        orientation = normalize_canonical_quaternion(
            torch.randn(32, 4, generator=generator, dtype=torch.float64)
        )
        canonical, index = canonicalize_cuboid_orientation(
            orientation, symmetries, reference, return_index=True
        )
        torch.testing.assert_close(
            apply_cuboid_symmetry(orientation, symmetries, index), canonical
        )

    def test_a_held_choice_does_not_flip_while_the_bar_turns(self):
        """Measured failure this guards: canonicalising every step against a
        fixed reference flipped the representative partway through the lift,
        moving the reference frame by 0.35 m mid-grasp."""
        symmetries = cuboid_rotation_symmetries(BAR)
        reference = yaw(-88.0)
        turning = torch.stack([yaw(-88.0 + 0.6 * step) for step in range(60)])
        _, index = canonicalize_cuboid_orientation(
            turning[0], symmetries, reference, return_index=True
        )
        held = apply_cuboid_symmetry(
            turning, symmetries, index.expand(len(turning))
        )
        steps = (held[1:] - held[:-1]).abs().amax(dim=-1)
        self.assertLess(float(steps.max()), 0.05)


if __name__ == "__main__":
    unittest.main()
