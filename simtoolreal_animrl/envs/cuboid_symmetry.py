"""Fold a cuboid's pose ambiguity away before anything else reads it.

A 0.15 x 0.05 x 0.05 bar maps onto itself under eight different rotations: four
about its long axis, because the two short extents are equal, and four more that
flip it end for end. Nothing physical distinguishes them -- same shape, same
mass distribution, same grasp -- but they are eight different quaternions, and a
pose estimate is free to return any of them for a bar that never moved. A policy
reading the raw orientation would see a discontinuous jump, and an object-frame
reference would be told to approach from the wrong side.

So the frame is canonicalised: the label the measurement returns is swapped for
the one the demonstration would have used, and everything downstream -- the
observation, the object-frame keypoint reward -- reads that.

Two things this module is often expected to do, and only one of them it does:

* It is **not** what makes 180 degrees of yaw enough. A bar at +135 degrees *is*
  a bar at -45 degrees, so a sampling range of 180 degrees already covers every
  planar orientation the bar can take. That is a statement about how wide to
  sample, and it needs no code.
* It **is** insurance against a pose estimate that relabels an object that never
  moved. In simulation the cuboid is spawned at exactly the reference pose, so
  the canonical choice is the identity and this is a no-op. On hardware it is
  not.

Get the reference right or it does damage. The choice must be made against the
pose the episode was **reset to**, and then held for the episode -- not against
some fixed frame-0 orientation. The bar's orientation is not constant during the
motion; it is picked up and turned. Canonicalising every step against a fixed
frame-0 reference was measured to flip the representative partway through the
lift at yaw -88 degrees, moving the reference frame by 0.35 m in the middle of
the grasp. Choose once with :func:`canonicalize_cuboid_orientation`, keep the
returned index, replay it with :func:`apply_cuboid_symmetry`.

Canonicalise in a *fixed* frame (the robot base), never in the palm frame: the
palm moves, so a palm-frame reference orientation would make the chosen
representative depend on where the arm happens to be.
"""

import itertools

import torch

from simtoolreal_animrl.envs.rotations import (
    matrix_to_quat,
    normalize_canonical_quaternion,
    quat_multiply,
)


def cuboid_rotation_symmetries(half_extents, atol: float = 1e-9) -> torch.Tensor:
    """Return the ``(K, 4)`` xyzw rotations that map the cuboid onto itself.

    Derived from the extents rather than typed out, so it stays correct if the
    object's proportions ever change: a bar with two equal extents gives eight,
    a true cube gives twenty-four, a box with three distinct extents gives four.

    Every such rotation is a signed permutation of the axes with determinant
    +1 -- there are twenty-four of those in total, and one is a symmetry exactly
    when permuting the extents leaves them unchanged.
    """
    extents = torch.as_tensor(half_extents, dtype=torch.float64).reshape(-1)
    if extents.numel() != 3:
        raise ValueError("Expected three half extents")
    if torch.any(extents <= 0.0):
        raise ValueError("Half extents must be positive")

    matrices = []
    for permutation in itertools.permutations(range(3)):
        if any(
            abs(float(extents[permutation[j]] - extents[j])) > atol
            for j in range(3)
        ):
            continue
        for signs in itertools.product((1.0, -1.0), repeat=3):
            matrix = torch.zeros(3, 3, dtype=torch.float64)
            for column, (row, sign) in enumerate(zip(permutation, signs)):
                matrix[row, column] = sign
            if torch.det(matrix) < 0.0:
                # An improper signed permutation is a reflection: it maps the
                # box onto itself as a set of points but is not a rotation, so
                # no rigid motion realises it.
                continue
            matrices.append(matrix)

    # itertools iterates in a fixed order and the matrices are exact, so the
    # result is byte-identical run to run: a bank built today and a bank built
    # tomorrow pick the same representative for the same pose.
    return matrix_to_quat(torch.stack(matrices))


def canonicalize_cuboid_orientation(
    orientation: torch.Tensor,
    symmetries: torch.Tensor,
    reference: torch.Tensor,
    return_index: bool = False,
):
    """Pick the symmetry-equivalent orientation closest to ``reference``.

    ``orientation`` is ``(..., 4)`` xyzw in a fixed frame and ``symmetries`` is
    ``(K, 4)`` from :func:`cuboid_rotation_symmetries`. ``reference`` is the
    orientation to resolve towards, in the same frame -- either one ``(4,)``
    shared by everything, or ``(..., 4)`` broadcasting against ``orientation``
    so each environment can resolve towards the pose it was reset to.

    The symmetry multiplies on the right: it relabels the *body* frame, which is
    what the ambiguity actually is. ``abs`` on the dot product is there because
    ``q`` and ``-q`` are the same rotation, so the nearest representative must
    not be chosen by a sign the double cover made up.

    With ``return_index`` the chosen element's index comes back too. Keep it for
    the rest of the episode and replay it through :func:`apply_cuboid_symmetry`;
    see this module's docstring for why re-choosing every step is a bug.
    """
    if symmetries.ndim != 2 or symmetries.shape[-1] != 4:
        raise ValueError("Expected symmetries with shape (K, 4)")
    reference = torch.as_tensor(
        reference, dtype=orientation.dtype, device=orientation.device
    )
    if reference.shape[-1] != 4:
        raise ValueError("Expected a reference orientation with shape (..., 4)")
    reference = normalize_canonical_quaternion(reference)
    symmetries = symmetries.to(dtype=orientation.dtype, device=orientation.device)

    orientation = normalize_canonical_quaternion(orientation)
    batch = orientation.shape[:-1]
    count = symmetries.shape[0]
    candidates = quat_multiply(
        orientation.unsqueeze(-2).expand(*batch, count, 4),
        symmetries.expand(*batch, count, 4),
    )
    alignment = (candidates * reference.unsqueeze(-2)).sum(dim=-1).abs()
    chosen = alignment.argmax(dim=-1)
    picked = torch.gather(
        candidates, -2, chosen[..., None, None].expand(*batch, 1, 4)
    ).squeeze(-2)
    canonical = normalize_canonical_quaternion(picked)
    if return_index:
        return canonical, chosen
    return canonical


def apply_cuboid_symmetry(
    orientation: torch.Tensor,
    symmetries: torch.Tensor,
    index: torch.Tensor,
) -> torch.Tensor:
    """Relabel ``orientation`` by an already-chosen symmetry element.

    ``index`` is ``(...,)`` long, broadcasting against ``orientation``'s batch
    shape -- one element per environment, fixed at reset. This is the half of
    the canonicalisation that runs every step; the choosing half runs once.
    """
    if symmetries.ndim != 2 or symmetries.shape[-1] != 4:
        raise ValueError("Expected symmetries with shape (K, 4)")
    symmetries = symmetries.to(dtype=orientation.dtype, device=orientation.device)
    selected = symmetries[index.reshape(-1).long()].reshape(*index.shape, 4)
    return normalize_canonical_quaternion(
        quat_multiply(normalize_canonical_quaternion(orientation), selected)
    )


def symmetry_invariant_orientation_error(
    orientation: torch.Tensor,
    reference: torch.Tensor,
    symmetries: torch.Tensor,
) -> torch.Tensor:
    """Geodesic angle between two orientations, modulo the cuboid's symmetry.

    ``(...,)`` radians: the smallest rotation carrying ``orientation`` onto any
    relabelling of ``reference``. For a bar with a square cross-section, half a
    turn about its long axis is the same physical pose, so the raw angle can
    charge up to pi for a state that is exactly right.

    Unlike :func:`canonicalize_cuboid_orientation` this picks a fresh element
    every call, and that is correct *here* precisely because the result is a
    scalar distance rather than a frame. The rule in this module's docstring --
    choose once, replay it -- protects quantities measured *in* the cuboid's
    frame, where re-choosing makes the frame jump mid-motion. A distance has no
    frame to jump: it is the geodesic on the quotient by the symmetry group,
    which is what "how far is this bar from where it should be" means.
    """
    if symmetries.ndim != 2 or symmetries.shape[-1] != 4:
        raise ValueError("Expected symmetries with shape (K, 4)")
    symmetries = symmetries.to(dtype=orientation.dtype, device=orientation.device)
    orientation = normalize_canonical_quaternion(orientation)
    reference = normalize_canonical_quaternion(
        torch.as_tensor(
            reference, dtype=orientation.dtype, device=orientation.device
        )
    )
    batch = orientation.shape[:-1]
    count = symmetries.shape[0]
    candidates = quat_multiply(
        orientation.unsqueeze(-2).expand(*batch, count, 4),
        symmetries.expand(*batch, count, 4),
    )
    # abs(): q and -q are the same rotation, so the double cover must not pick
    # the representative. max over the symmetry axis is the closest relabelling.
    alignment = (
        (candidates * reference.unsqueeze(-2)).sum(dim=-1).abs().amax(dim=-1)
    )
    return 2.0 * torch.acos(alignment.clamp(max=1.0))
