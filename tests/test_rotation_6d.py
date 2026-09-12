"""The observation's rotation encoding must be continuous and sign-free.

blind_quiet2 answered a w = 0 crossing mid-approach with a 4.03 action-unit
step on rj_dg_3_4 -- 0.63 rad -- out of a stream whose neighbouring steps moved
by 0.009, then rang for thirty frames. Nothing physical happened there: the
canonicalized quaternion negated all four components, and a continuous network
cannot answer a discontinuous input with a continuous output. These tests pin
the property that fixed it.
"""

import math
import unittest

import numpy as np
import torch

from simtoolreal_animrl.envs.rotations import (
    normalize_canonical_quaternion,
    quaternion_to_rotation_6d,
)
from simtoolreal_animrl.sim2sim.observation import (
    quaternion_to_rotation_6d as quaternion_to_rotation_6d_numpy,
)


def _quaternion_about_x(angle):
    """Unit xyzw quaternion rotating by `angle` about x; w = cos(angle/2)."""
    return [math.sin(angle / 2.0), 0.0, 0.0, math.cos(angle / 2.0)]


class Rotation6DTest(unittest.TestCase):
    def test_sign_invariance(self):
        """q and -q are the same rotation, so they must encode identically."""
        generator = torch.Generator().manual_seed(0)
        quaternions = torch.nn.functional.normalize(
            torch.randn(64, 4, generator=generator), dim=-1
        )
        torch.testing.assert_close(
            quaternion_to_rotation_6d(quaternions),
            quaternion_to_rotation_6d(-quaternions),
        )

    def test_encoding_is_continuous_where_the_quaternion_jumps(self):
        """Sweep through w = 0 and compare the two encodings step for step."""
        angles = torch.linspace(math.pi - 0.05, math.pi + 0.05, 101)
        quaternions = torch.tensor(
            [_quaternion_about_x(float(a)) for a in angles], dtype=torch.float32
        )
        # The sweep really does cross the cut this is all about.
        canonical = normalize_canonical_quaternion(quaternions)
        quaternion_steps = (canonical[1:] - canonical[:-1]).norm(dim=-1)
        self.assertGreater(float(quaternion_steps.max()), 1.9)

        encoded = quaternion_to_rotation_6d(quaternions)
        encoded_steps = (encoded[1:] - encoded[:-1]).norm(dim=-1)
        # Same physical sweep, no jump: every step stays near the 0.001 rad
        # the sweep actually advances, against the quaternion's 2.0.
        self.assertLess(float(encoded_steps.max()), 0.01)

    def test_matches_the_numpy_contract_used_on_hardware(self):
        """The deployed observation is built by the sim2sim implementation."""
        generator = torch.Generator().manual_seed(1)
        quaternions = torch.nn.functional.normalize(
            torch.randn(32, 4, generator=generator), dim=-1
        ).double()
        expected = np.stack(
            [quaternion_to_rotation_6d_numpy(q.numpy()) for q in quaternions]
        )
        np.testing.assert_allclose(
            quaternion_to_rotation_6d(quaternions).numpy(), expected, atol=1e-12
        )

    def test_encodes_the_rotation_matrix_columns(self):
        """A quarter turn about z maps x -> y and y -> -x."""
        quaternion = torch.tensor(
            [[0.0, 0.0, math.sin(math.pi / 4.0), math.cos(math.pi / 4.0)]]
        )
        torch.testing.assert_close(
            quaternion_to_rotation_6d(quaternion),
            torch.tensor([[0.0, 1.0, 0.0, -1.0, 0.0, 0.0]]),
            atol=1e-6,
            rtol=0.0,
        )


if __name__ == "__main__":
    unittest.main()
