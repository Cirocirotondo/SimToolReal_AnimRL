"""Quaternion and rotation-representation helpers, free of isaacgym.

These lived as private functions inside ``motion_imitation.py``, which cannot be
imported without a CUDA-capable Isaac Gym installation. The object-centric
reference needs the same arithmetic in the retargeter, the cuboid-symmetry
canonicaliser and the keypoint reward, none of which run a simulation. Putting
them here keeps one implementation instead of four, and keeps all three of those
modules unit-testable on a machine with no GPU.

Every quaternion in this project is **xyzw** and canonicalised to ``w >= 0``.

The 6D rotation representation is Zhou et al. 2019: the first two columns of the
rotation matrix, recovered by Gram-Schmidt. Quaternions are a double cover, so
``q`` and ``-q`` are the same rotation and any network reading one has a
discontinuity to learn around; the ``w >= 0`` canonicalisation does not remove
that, it just moves the seam to ``w == 0``. The 6D form is continuous
everywhere, which is why the observation uses it.
"""

import torch


def quat_conjugate(quaternion: torch.Tensor) -> torch.Tensor:
    return torch.cat((-quaternion[..., :3], quaternion[..., 3:4]), dim=-1)


def quat_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Hamilton product for xyzw quaternions."""
    left_xyz, left_w = left[..., :3], left[..., 3:4]
    right_xyz, right_w = right[..., :3], right[..., 3:4]
    xyz = (
        left_w * right_xyz
        + right_w * left_xyz
        + torch.cross(left_xyz, right_xyz, dim=-1)
    )
    w = left_w * right_w - (left_xyz * right_xyz).sum(dim=-1, keepdim=True)
    return torch.cat((xyz, w), dim=-1)


def quat_rotate(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate a vector by a unit xyzw quaternion without constructing matrices."""
    quaternion_xyz = quaternion[..., :3]
    uv = torch.cross(quaternion_xyz, vector, dim=-1)
    uuv = torch.cross(quaternion_xyz, uv, dim=-1)
    return vector + 2.0 * (quaternion[..., 3:4] * uv + uuv)


def quat_rotate_inverse(
    quaternion: torch.Tensor, vector: torch.Tensor
) -> torch.Tensor:
    return quat_rotate(quat_conjugate(quaternion), vector)


def normalize_canonical_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    quaternion = torch.nn.functional.normalize(quaternion, dim=-1)
    return torch.where(quaternion[..., 3:4] < 0.0, -quaternion, quaternion)


def quat_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """Return the ``(..., 3, 3)`` rotation matrix for an xyzw quaternion.

    Built by rotating the three basis vectors rather than by writing out the
    nine entries, so it can only ever agree with ``quat_rotate`` above.
    """
    shape = quaternion.shape[:-1]
    basis = torch.eye(3, dtype=quaternion.dtype, device=quaternion.device)
    basis = basis.expand(*shape, 3, 3)
    expanded = quaternion.unsqueeze(-2).expand(*shape, 3, 4)
    # Columns of R are the images of the basis vectors, so rotate rows of the
    # identity and transpose.
    rotated = quat_rotate(expanded, basis)
    return rotated.transpose(-1, -2)


def matrix_to_quat(matrix: torch.Tensor) -> torch.Tensor:
    """Return the canonical xyzw quaternion for a ``(..., 3, 3)`` rotation.

    Shepperd's method: form the quaternion from whichever of the four
    components is largest, so the division is never by a near-zero number. The
    naive trace-only formula loses all precision at a 180 degree rotation, and
    half the cuboid symmetry group is exactly 180 degree rotations.
    """
    m = matrix
    trace = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    candidates = torch.stack(
        (trace, m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]), dim=-1
    )
    choice = candidates.argmax(dim=-1)

    quarter = 0.25
    w0 = torch.sqrt((1.0 + trace).clamp_min(0.0)) * 2.0
    s0 = w0.clamp_min(torch.finfo(m.dtype).tiny)
    by_trace = torch.stack(
        (
            (m[..., 2, 1] - m[..., 1, 2]) / s0,
            (m[..., 0, 2] - m[..., 2, 0]) / s0,
            (m[..., 1, 0] - m[..., 0, 1]) / s0,
            quarter * w0,
        ),
        dim=-1,
    )

    def diagonal_branch(index):
        i, j, k = index, (index + 1) % 3, (index + 2) % 3
        raw = torch.sqrt(
            (1.0 + m[..., i, i] - m[..., j, j] - m[..., k, k]).clamp_min(0.0)
        ) * 2.0
        s = raw.clamp_min(torch.finfo(m.dtype).tiny)
        parts = [None, None, None]
        parts[i] = quarter * raw
        parts[j] = (m[..., j, i] + m[..., i, j]) / s
        parts[k] = (m[..., k, i] + m[..., i, k]) / s
        w = (m[..., k, j] - m[..., j, k]) / s
        return torch.stack((parts[0], parts[1], parts[2], w), dim=-1)

    stacked = torch.stack(
        (by_trace, diagonal_branch(0), diagonal_branch(1), diagonal_branch(2)),
        dim=-2,
    )
    picked = torch.gather(
        stacked, -2, choice[..., None, None].expand(*choice.shape, 1, 4)
    ).squeeze(-2)
    return normalize_canonical_quaternion(picked)


def quat_to_rotation_6d(quaternion: torch.Tensor) -> torch.Tensor:
    """Return the ``(..., 6)`` continuous representation of a rotation.

    The first two columns of the rotation matrix. The third is their cross
    product and carries no extra information, which is the whole point: six
    numbers, no discontinuity, no unit-norm constraint for a network to satisfy.
    """
    matrix = quat_to_matrix(quaternion)
    return torch.cat((matrix[..., :, 0], matrix[..., :, 1]), dim=-1)


def rotation_6d_to_matrix(rotation_6d: torch.Tensor) -> torch.Tensor:
    """Invert :func:`quat_to_rotation_6d` by Gram-Schmidt.

    Present for tests and diagnostics rather than for the hot path: nothing in
    the environment needs to go back from 6D, but a representation nobody can
    invert is a representation nobody can check.
    """
    first = torch.nn.functional.normalize(rotation_6d[..., 0:3], dim=-1)
    second = rotation_6d[..., 3:6]
    second = second - (first * second).sum(dim=-1, keepdim=True) * first
    second = torch.nn.functional.normalize(second, dim=-1)
    third = torch.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-1)
