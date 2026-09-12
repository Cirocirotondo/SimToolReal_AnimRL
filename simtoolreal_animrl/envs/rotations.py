"""Rotation encodings for the observation vector.

Kept free of isaacgym so the contract can be tested without a simulator, like
``rsi.py`` and ``adaptive_sigma.py`` beside it.
"""

import torch


def normalize_canonical_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    """Unit xyzw quaternion with a non-negative w.

    Safe wherever only the *rotation* is used -- rotating a vector by q and by
    -q gives the same answer -- and unsafe anywhere the four numbers reach the
    policy directly. See ``quaternion_to_rotation_6d``.
    """
    quaternion = torch.nn.functional.normalize(quaternion, dim=-1)
    return torch.where(quaternion[..., 3:4] < 0.0, -quaternion, quaternion)


def quaternion_to_rotation_6d(quaternion: torch.Tensor) -> torch.Tensor:
    """The first two columns of the rotation matrix, flattened to six values.

    An observation must be a continuous function of the physical state, and a
    canonicalized quaternion is not. ``normalize_canonical_quaternion`` cuts the
    double cover at ``w = 0``, so a palm rotating smoothly through that plane
    negates all four components at once: the observation jumps by 2.0 while
    nothing physical happens, and the network -- being continuous in its input
    -- has no choice but to jump with it. That is measured, not feared:
    blind_quiet2 crosses w = 0 mid-approach and answers with a 4.03 action-unit
    step on rj_dg_3_4 (0.63 rad) out of a stream whose neighbouring steps move
    by 0.009, then rings for thirty frames.

    q and -q are the same rotation and produce the same matrix, so this encoding
    has no sign to choose, needs no history to stay continuous, and costs two
    extra numbers per rotation. See Zhou et al. 2019, "On the Continuity of
    Rotation Representations in Neural Networks".
    """
    quaternion = torch.nn.functional.normalize(quaternion, dim=-1)
    quaternion_xyz = quaternion[..., :3]
    quaternion_w = quaternion[..., 3:4]

    def rotate(vector):
        uv = torch.cross(quaternion_xyz, vector, dim=-1)
        uuv = torch.cross(quaternion_xyz, uv, dim=-1)
        return vector + 2.0 * (quaternion_w * uv + uuv)

    x_axis = torch.zeros_like(quaternion_xyz)
    x_axis[..., 0] = 1.0
    y_axis = torch.zeros_like(quaternion_xyz)
    y_axis[..., 1] = 1.0
    return torch.cat((rotate(x_axis), rotate(y_axis)), dim=-1)
