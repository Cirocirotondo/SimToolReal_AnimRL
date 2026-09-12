"""Rewrite the demonstration for a cuboid that is somewhere else.

The recorded clip is one grasp of one bar at one place. Every joint angle in it
is only correct for that place, so a policy trained to track those angles cannot
generalise: move the bar 10 cm and the right arm pose is further from the
recorded one than the termination threshold allows.

What *is* transferable is the motion seen from the bar. So for a rigid transform
``T`` -- a planar translation plus a yaw about the bar's own vertical axis --
this module solves for the joint trajectory that reproduces the demonstrated
palm pose *relative to the bar*, and hands back a clip in exactly the shape the
environment already consumes.

Two facts make this much cheaper than it sounds:

* ``T`` is rigid, so the twenty finger joints are **copied verbatim**. If the
  palm lands at ``T . palm_demo(t)`` then every fingertip lands at
  ``T . fingertip_demo(t)`` automatically, and the whole hand-object graph is
  preserved exactly. Only the six arm joints need solving.
* The demonstration itself is the seed. Solving frame ``t`` from the solution at
  ``t - 1`` keeps the solver on one IK branch for the whole clip, which is what
  a closed-form solver would make us choose by hand -- and it sidesteps joint
  wrapping entirely, which matters because ``wrist_3`` lives at -4.30 rad, well
  outside any principal-value range a closed-form solver would return.

Kinematics come from the URDF via ``pytorch_kinematics``, targeting the
``rl_dg_palm`` frame directly. That is deliberate: it makes the 60 degree mount,
the 0.0738 m palm offset and the pi rotation at ``base_link -> base_link_inertia``
the parser's problem rather than ours, and it avoids adding a third hand-typed
copy of constants that already exist twice.

All poses here are in the **robot base** (``base_link``) frame. The
demonstration records the cube in the UR controller's base frame, which is
``base_link`` turned by pi about z -- hence the ``(-1, -1, 1)`` position flip
that :func:`cube_pose_to_base_frame` applies.
"""

from pathlib import Path
from typing import NamedTuple, Optional, Union

import torch

from simtoolreal_animrl.envs.rotations import (
    matrix_to_quat,
    normalize_canonical_quaternion,
    quat_multiply,
    quat_to_matrix,
)


PALM_LINK_NAME = "rl_dg_palm"
# The URDF's fixed tip joints survive pytorch_kinematics parsing, unlike Isaac
# Gym's collapse_fixed_joints. Targeting them directly is what keeps this module
# from becoming a third copy of FINGERTIP_OFFSETS; verified equal to
# distal-link + offset to 5e-10 m.
FINGERTIP_LINK_NAMES = tuple("rl_dg_{}_tip".format(finger) for finger in range(1, 6))
ARM_JOINT_COUNT = 6
HAND_JOINT_COUNT = 20
# UR-base -> base_link is a pi rotation about z, so x and y flip sign.
UR_BASE_AXIS_SIGN = (-1.0, -1.0, 1.0)


class RetargetResult(NamedTuple):
    """One retargeted clip plus the evidence that it is usable."""

    arm_q: torch.Tensor              # (frames, 6)
    position_residual_m: torch.Tensor    # (frames,)
    rotation_residual_rad: torch.Tensor  # (frames,)
    limit_margin_rad: torch.Tensor       # (frames,) distance to the nearest limit


# Seed for the elbow-up solution on the opposite UR5e shoulder branch. It is
# deliberately only coarse configurations: solve_palm_ik moves each one onto
# the exact frame-0 target.  The ordinary demonstration seed remains the first
# choice; these are fallbacks for transforms where that branch approaches the
# wrist singularity (wrist_2 ~= 0) or otherwise fails a whole-clip criterion.
#
# "Elbow up" is a Cartesian property, not the sign of the elbow joint. With the
# opposite shoulder branch, the positive-elbow solutions put the actual elbow
# below the wrist. This seed converges to the branch whose forearm-link origin
# stays above the wrist, like the demonstration. Speed is a secondary criterion,
# not permission to change that qualitative posture.
_ALTERNATE_ARM_SEEDS = (
    # Preferred elbow-up / wrist-under family. The other exact elbow-up
    # solution routes wrist_1 on the opposite side; although slightly slower
    # (about 0.55 vs 0.36 rad/s in the -22.5 degree regression), this is the
    # mechanically desired posture and remains far below the speed limit.
    (2.1, -2.9, -1.0, 0.7, 0.6, 0.5),
)


def cube_pose_to_base_frame(cube_pose: torch.Tensor) -> torch.Tensor:
    """Convert recorded ``(..., 7)`` UR-base cube poses into the base_link frame.

    Position flips x and y; orientation is premultiplied by the same pi rotation
    about z, which in xyzw works out to ``(-y, x, w, -z)``. This is the identical
    convention the environment and the MuJoCo runner already use, restated here
    so the retargeter does not depend on an isaacgym import.

    Deliberately normalised but **not** canonicalised to ``w >= 0``: the recorded
    demonstration carries ``w < 0`` poses, and flipping them here would make a
    round trip fail to reproduce the clip it started from.

    This is **not** its own inverse. Applying a pi rotation twice is a 2 pi
    rotation, which is ``-1`` in quaternion space rather than ``+1`` -- the same
    rotation with the opposite sign. Use :func:`cube_pose_from_base_frame`,
    which premultiplies by the conjugate, to come back exactly.
    """
    if cube_pose.shape[-1] != 7:
        raise ValueError("Expected cube poses with shape (..., 7)")
    sign = torch.as_tensor(
        UR_BASE_AXIS_SIGN, dtype=cube_pose.dtype, device=cube_pose.device
    )
    position = cube_pose[..., :3] * sign
    x, y, z, w = cube_pose[..., 3].clone(), cube_pose[..., 4].clone(), \
        cube_pose[..., 5].clone(), cube_pose[..., 6].clone()
    orientation = torch.nn.functional.normalize(
        torch.stack((-y, x, w, -z), dim=-1), dim=-1
    )
    return torch.cat((position, orientation), dim=-1)


def cube_pose_from_base_frame(cube_pose: torch.Tensor) -> torch.Tensor:
    """Exact inverse of :func:`cube_pose_to_base_frame`.

    Premultiplies by the *conjugate* pi rotation, ``(0, 0, -1, 0)``, which in
    xyzw works out to ``(y, -x, -w, z)``. Composing the two recovers the input
    quaternion sign included, where applying the forward map twice would return
    its negation.
    """
    if cube_pose.shape[-1] != 7:
        raise ValueError("Expected cube poses with shape (..., 7)")
    sign = torch.as_tensor(
        UR_BASE_AXIS_SIGN, dtype=cube_pose.dtype, device=cube_pose.device
    )
    position = cube_pose[..., :3] * sign
    x, y, z, w = cube_pose[..., 3].clone(), cube_pose[..., 4].clone(), \
        cube_pose[..., 5].clone(), cube_pose[..., 6].clone()
    orientation = torch.nn.functional.normalize(
        torch.stack((y, -x, -w, z), dim=-1), dim=-1
    )
    return torch.cat((position, orientation), dim=-1)


def yaw_quaternion(yaw_rad: torch.Tensor) -> torch.Tensor:
    """``(..., 4)`` xyzw rotations of ``yaw_rad`` about the vertical axis."""
    half = yaw_rad * 0.5
    zeros = torch.zeros_like(half)
    return normalize_canonical_quaternion(
        torch.stack((zeros, zeros, torch.sin(half), torch.cos(half)), dim=-1)
    )


def transform_points(
    points: torch.Tensor,
    yaw_rad: torch.Tensor,
    translation: torch.Tensor,
    pivot: torch.Tensor,
) -> torch.Tensor:
    """Apply ``R_z(yaw)`` about ``pivot``, then translate.

    The pivot is the bar's own centre at frame 0, so yaw and translation are
    independent knobs: yaw turns the bar in place, translation slides it. A yaw
    about the world origin would couple the two and make the sampled ranges mean
    something different at every distance from the robot.
    """
    rotation = quat_to_matrix(yaw_quaternion(yaw_rad))
    centred = points - pivot
    rotated = torch.einsum("...ij,...j->...i", rotation, centred)
    return rotated + pivot + translation


class PalmKinematics:
    """Batched forward kinematics and Jacobians for the arm-to-palm chain."""

    def __init__(
        self,
        urdf_path: Union[str, Path],
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        import pytorch_kinematics as pk

        resolved = Path(urdf_path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError("URDF not found: {}".format(resolved))
        data = resolved.read_bytes()
        self.device = torch.device(device)
        self.dtype = dtype
        self.chain = pk.build_serial_chain_from_urdf(
            data, PALM_LINK_NAME
        ).to(dtype=dtype, device=self.device)
        self.forearm_chain = pk.build_serial_chain_from_urdf(
            data, "forearm_link"
        ).to(dtype=dtype, device=self.device)
        self.wrist_1_chain = pk.build_serial_chain_from_urdf(
            data, "wrist_1_link"
        ).to(dtype=dtype, device=self.device)
        # The full tree as well, for the fingertips: the serial chain to the
        # palm stops before the fingers branch off it.
        self.full_chain = pk.build_chain_from_urdf(data).to(
            dtype=dtype, device=self.device
        )
        joint_names = self.chain.get_joint_parameter_names()
        if len(joint_names) != ARM_JOINT_COUNT:
            raise ValueError(
                "Expected {} actuated joints from base to {}, found {}".format(
                    ARM_JOINT_COUNT, PALM_LINK_NAME, joint_names
                )
            )
        self.joint_names = tuple(joint_names)
        lower, upper = self.chain.get_joint_limits()
        self.lower_limits = torch.as_tensor(
            lower, dtype=dtype, device=self.device
        )
        self.upper_limits = torch.as_tensor(
            upper, dtype=dtype, device=self.device
        )

    def palm_matrices(self, arm_q: torch.Tensor) -> torch.Tensor:
        """``(B, 4, 4)`` palm poses in the base frame."""
        return self.chain.forward_kinematics(arm_q).get_matrix()

    def jacobian(self, arm_q: torch.Tensor) -> torch.Tensor:
        """``(B, 6, 6)`` geometric Jacobian, ``[linear; angular]`` in the base frame."""
        return self.chain.jacobian(arm_q)

    def elbow_height_margin(self, arm_q: torch.Tensor) -> torch.Tensor:
        """Physical elbow height minus wrist-1 height for each arm pose.

        The demonstration stays positive by 0.19--0.31 m. This Cartesian test
        distinguishes the desired elbow-up family across shoulder branches;
        the sign of the UR elbow joint does not.
        """
        shape = arm_q.shape[:-1]
        flat = arm_q.reshape(-1, ARM_JOINT_COUNT)
        elbow_z = self.forearm_chain.forward_kinematics(
            flat[:, :3]
        ).get_matrix()[:, 2, 3]
        wrist_z = self.wrist_1_chain.forward_kinematics(
            flat[:, :4]
        ).get_matrix()[:, 2, 3]
        return (elbow_z - wrist_z).reshape(shape)

    def hand_keypoints(
        self, joint_positions: torch.Tensor, lever_arm_m: float
    ) -> torch.Tensor:
        """``(B, 9, 3)`` hand keypoints in the base frame, from all 26 joints.

        The same nine points the reward uses, computed from joint angles rather
        than from a simulator's rigid-body tensor. That is what lets the
        reference curve be built offline, and what lets a test compare the two
        paths against each other.
        """
        from simtoolreal_animrl.envs.keypoints import hand_keypoints

        expected = ARM_JOINT_COUNT + HAND_JOINT_COUNT
        if joint_positions.shape[-1] != expected:
            raise ValueError(
                "Expected {} joint positions, got {}".format(
                    expected, joint_positions.shape[-1]
                )
            )
        poses = self.full_chain.forward_kinematics(
            joint_positions.to(dtype=self.dtype, device=self.device)
        )
        palm = poses[PALM_LINK_NAME].get_matrix()
        fingertips = torch.stack(
            [poses[name].get_matrix()[..., :3, 3] for name in FINGERTIP_LINK_NAMES],
            dim=-2,
        )
        return hand_keypoints(
            palm[..., :3, 3],
            matrix_to_quat(palm[..., :3, :3]),
            fingertips,
            lever_arm_m,
        )


def reference_keypoints_in_object_frame(
    kinematics: PalmKinematics,
    demo_joint_positions: torch.Tensor,
    demo_cube_pose_base: torch.Tensor,
    lever_arm_m: float,
) -> torch.Tensor:
    """The ``(frames, 9, 3)`` reference curve every episode tracks.

    Stored once rather than once per transform, because in the cuboid's frame it
    does not depend on the transform at all -- a rigid motion of the whole scene
    cancels between the hand and the bar. ``tests/test_retarget.py`` asserts
    that; if it ever fails, a frame convention has drifted.
    """
    from simtoolreal_animrl.envs.keypoints import keypoints_in_object_frame

    keypoints = kinematics.hand_keypoints(demo_joint_positions, lever_arm_m)
    return keypoints_in_object_frame(
        keypoints,
        demo_cube_pose_base[..., :3].to(dtype=kinematics.dtype, device=kinematics.device),
        demo_cube_pose_base[..., 3:7].to(dtype=kinematics.dtype, device=kinematics.device),
    )


def pose_error(current: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """``(B, 6)`` twist carrying ``current`` onto ``target``, in the base frame.

    Rotation error goes through the quaternion rather than the matrix logarithm
    because the canonical ``w >= 0`` form is already the shortest rotation, so
    the solver never takes the long way round a 350 degree error.
    """
    linear = target[..., :3, 3] - current[..., :3, 3]
    relative = target[..., :3, :3] @ current[..., :3, :3].transpose(-1, -2)
    quaternion = matrix_to_quat(relative)
    vector, scalar = quaternion[..., :3], quaternion[..., 3:4]
    norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    # angle = 2 atan2(|v|, w); the limit of angle/|v| as |v| -> 0 is 2/w, so the
    # small-angle branch is just 2v and stays differentiable at zero error.
    small = norm < 1e-12
    scale = torch.where(
        small,
        torch.full_like(norm, 2.0),
        2.0 * torch.atan2(norm, scalar) / norm.clamp_min(1e-30),
    )
    return torch.cat((linear, vector * scale), dim=-1)


def solve_palm_ik(
    kinematics: PalmKinematics,
    target: torch.Tensor,
    seed: torch.Tensor,
    iterations: int = 40,
    damping: float = 0.02,
    position_tolerance_m: float = 1e-4,
    rotation_tolerance_rad: float = 1e-4,
) -> torch.Tensor:
    """Damped least squares from ``seed`` to the ``(B, 4, 4)`` palm ``target``.

    Damping rather than a plain pseudo-inverse because the UR5e passes close to
    wrist and elbow singularities under some transforms, and an undamped step
    there asks for an unbounded joint velocity. The damping trades a little
    accuracy at the singularity for a step that stays finite; the residual is
    returned to the caller either way, so a transform that cannot be solved is
    rejected rather than quietly accepted.
    """
    q = seed.clone()
    for _ in range(int(iterations)):
        current = kinematics.palm_matrices(q)
        error = pose_error(current, target)
        position_error = torch.linalg.vector_norm(error[..., :3], dim=-1)
        rotation_error = torch.linalg.vector_norm(error[..., 3:], dim=-1)
        if bool(
            torch.all(position_error < position_tolerance_m)
            and torch.all(rotation_error < rotation_tolerance_rad)
        ):
            break
        jacobian = kinematics.jacobian(q)
        gram = jacobian @ jacobian.transpose(-1, -2)
        gram = gram + (damping ** 2) * torch.eye(
            6, dtype=gram.dtype, device=gram.device
        )
        step = jacobian.transpose(-1, -2) @ torch.linalg.solve(
            gram, error.unsqueeze(-1)
        )
        q = torch.clamp(
            q + step.squeeze(-1),
            kinematics.lower_limits,
            kinematics.upper_limits,
        )
    return q


def retarget_clip(
    kinematics: PalmKinematics,
    demo_arm_q: torch.Tensor,
    yaw_rad: torch.Tensor,
    translation: torch.Tensor,
    pivot: torch.Tensor,
    iterations: int = 40,
    first_frame_iterations: int = 200,
    initial_seed: Optional[torch.Tensor] = None,
) -> RetargetResult:
    """Solve a whole clip for a batch of transforms, chained frame to frame.

    ``demo_arm_q`` is ``(frames, 6)``; ``yaw_rad`` and ``translation`` are
    ``(B,)`` and ``(B, 3)``. Returns ``(frames, B, 6)`` joint angles alongside
    the residuals and the joint-limit margin, so the caller can reject a
    transform on evidence rather than on faith.

    Frame 0 gets far more iterations than the rest: it is seeded from the
    demonstration, which for a large transform is a long way from the answer,
    while every later frame is seeded from a solution one 60 Hz step away.
    """
    frames = demo_arm_q.shape[0]
    batch = yaw_rad.shape[0]
    dtype, device = kinematics.dtype, kinematics.device
    demo_arm_q = demo_arm_q.to(dtype=dtype, device=device)
    yaw_rad = yaw_rad.to(dtype=dtype, device=device)
    translation = translation.to(dtype=dtype, device=device)
    pivot = pivot.to(dtype=dtype, device=device).reshape(3)

    demo_palm = kinematics.palm_matrices(demo_arm_q)          # (frames, 4, 4)
    rotation = quat_to_matrix(yaw_quaternion(yaw_rad))        # (B, 3, 3)

    arm_q = torch.empty(frames, batch, ARM_JOINT_COUNT, dtype=dtype, device=device)
    position_residual = torch.empty(frames, batch, dtype=dtype, device=device)
    rotation_residual = torch.empty(frames, batch, dtype=dtype, device=device)

    if initial_seed is None:
        seed = demo_arm_q[0].unsqueeze(0).expand(batch, ARM_JOINT_COUNT).clone()
    else:
        seed = initial_seed.to(dtype=dtype, device=device)
        if seed.shape != (batch, ARM_JOINT_COUNT):
            raise ValueError(
                "initial_seed must have shape ({}, {})".format(
                    batch, ARM_JOINT_COUNT
                )
            )
        seed = seed.clone()
    for frame in range(frames):
        source = demo_palm[frame]
        target = torch.empty(batch, 4, 4, dtype=dtype, device=device)
        target[:, 3, :] = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=dtype, device=device)
        target[:, :3, :3] = rotation @ source[:3, :3].unsqueeze(0)
        target[:, :3, 3] = transform_points(
            source[:3, 3].unsqueeze(0).expand(batch, 3),
            yaw_rad,
            translation,
            pivot,
        )
        solved = solve_palm_ik(
            kinematics,
            target,
            seed,
            iterations=first_frame_iterations if frame == 0 else iterations,
        )
        residual = pose_error(kinematics.palm_matrices(solved), target)
        arm_q[frame] = solved
        position_residual[frame] = torch.linalg.vector_norm(residual[..., :3], dim=-1)
        rotation_residual[frame] = torch.linalg.vector_norm(residual[..., 3:], dim=-1)
        seed = solved

    margin = torch.minimum(
        arm_q - kinematics.lower_limits,
        kinematics.upper_limits - arm_q,
    ).amin(dim=-1)
    return RetargetResult(arm_q, position_residual, rotation_residual, margin)


def retarget_clip_preferred_branch(
    kinematics: PalmKinematics,
    demo_arm_q: torch.Tensor,
    yaw_rad: torch.Tensor,
    translation: torch.Tensor,
    pivot: torch.Tensor,
) -> RetargetResult:
    """Solve directly on the user-selected elbow-up/wrist-under branch.

    This is the interactive-viewer path. Unlike the exhaustive bank builder it
    does not first spend a complete clip proving that the demonstration branch
    fails, so a transform such as ``(+.06, -.15, -22.5 deg)`` takes one solve
    rather than two. The returned residuals still let the viewer report if this
    preferred branch itself is infeasible.
    """
    batch = int(yaw_rad.shape[0])
    seed = torch.as_tensor(
        _ALTERNATE_ARM_SEEDS[0],
        dtype=kinematics.dtype,
        device=kinematics.device,
    ).unsqueeze(0).expand(batch, ARM_JOINT_COUNT).clone()
    return retarget_clip(
        kinematics,
        demo_arm_q,
        yaw_rad,
        translation,
        pivot,
        initial_seed=seed,
    )


def retarget_clip_translation_continuation(
    kinematics: PalmKinematics,
    demo_arm_q: torch.Tensor,
    yaw_rad: torch.Tensor,
    translation: torch.Tensor,
    pivot: torch.Tensor,
    steps: int = 8,
) -> RetargetResult:
    """Carry the normal zero-translation branch continuously to ``translation``.

    Solving the destination from a generic frame-0 seed can jump to another
    shoulder/elbow solution. Here the accepted trajectory at ``(0, 0, yaw)`` is
    the anchor. The translation is introduced gradually and every target frame
    is seeded from that same frame of the preceding transform, preserving the
    branch in transform space as well as in time.

    This interactive path currently accepts one transform at a time.
    """
    if yaw_rad.shape != (1,) or translation.shape != (1, 3):
        raise ValueError("translation continuation expects one transform")
    if int(steps) < 1:
        raise ValueError("steps must be positive")

    dtype, device = kinematics.dtype, kinematics.device
    demo_arm_q = demo_arm_q.to(device=device, dtype=dtype)
    yaw_rad = yaw_rad.to(device=device, dtype=dtype)
    translation = translation.to(device=device, dtype=dtype)
    pivot = pivot.to(device=device, dtype=dtype).reshape(3)

    # This is precisely the unchecked/normal solution at (0, 0, yaw).
    anchor = retarget_clip(
        kinematics,
        demo_arm_q,
        yaw_rad,
        torch.zeros_like(translation),
        pivot,
    )
    seed_trajectory = anchor.arm_q[:, 0]
    demo_palm = kinematics.palm_matrices(demo_arm_q)
    rotation = quat_to_matrix(yaw_quaternion(yaw_rad))[0]
    frames = demo_arm_q.shape[0]

    result = anchor
    for fraction in torch.linspace(
        1.0 / int(steps), 1.0, int(steps), dtype=dtype, device=device
    ):
        target = torch.eye(4, dtype=dtype, device=device).expand(
            frames, 4, 4
        ).clone()
        target[:, :3, :3] = rotation @ demo_palm[:, :3, :3]
        frame_yaw = yaw_rad.expand(frames)
        frame_translation = (translation[0] * fraction).expand(frames, 3)
        target[:, :3, 3] = transform_points(
            demo_palm[:, :3, 3],
            frame_yaw,
            frame_translation,
            pivot,
        )
        solved = solve_palm_ik(
            kinematics,
            target,
            seed_trajectory,
            iterations=40,
        )
        residual = pose_error(kinematics.palm_matrices(solved), target)
        margin = torch.minimum(
            solved - kinematics.lower_limits,
            kinematics.upper_limits - solved,
        ).amin(dim=-1)
        result = RetargetResult(
            solved[:, None, :],
            torch.linalg.vector_norm(residual[:, :3], dim=-1)[:, None],
            torch.linalg.vector_norm(residual[:, 3:], dim=-1)[:, None],
            margin[:, None],
        )
        seed_trajectory = solved
    return result


def retarget_clip_best_branch(
    kinematics: PalmKinematics,
    demo_arm_q: torch.Tensor,
    yaw_rad: torch.Tensor,
    translation: torch.Tensor,
    pivot: torch.Tensor,
    control_dt: float,
    position_tolerance_m: float = 1e-3,
    rotation_tolerance_rad: float = 1e-2,
    limit_margin_rad: float = 0.05,
    velocity_limit_rad_s: float = 0.5 * 3.141592653589793,
) -> RetargetResult:
    """Retarget a clip while considering multiple UR5e IK branches.

    Chaining a numeric IK solve from the preceding frame gives excellent local
    continuity, but it cannot leave the branch selected at frame 0.  That made
    reachable transforms such as ``(x=0, y=0, yaw=52 deg)`` look infeasible:
    the demonstration branch drives ``wrist_2`` through a wrist singularity and
    asks ``wrist_1``/``wrist_3`` for more than 4 rad/s, while another exact IK
    branch needs less than 0.5 rad/s.

    To avoid changing established references unnecessarily, the demonstration-
    seeded trajectory wins whenever it passes every whole-clip criterion. Only
    failed transforms are replaced by an elbow-up alternate (the same Cartesian
    elbow-posture family as the demonstration) with the lowest peak finite-difference joint
    speed. If no such branch is feasible, the primary result is retained so its
    residuals still explain the rejection honestly.
    """
    if float(control_dt) <= 0.0:
        raise ValueError("control_dt must be positive")

    primary = retarget_clip(
        kinematics, demo_arm_q, yaw_rad, translation, pivot
    )

    def measurements(result):
        if result.arm_q.shape[0] < 2:
            peak_speed = torch.zeros(
                result.arm_q.shape[1],
                dtype=result.arm_q.dtype,
                device=result.arm_q.device,
            )
        else:
            peak_speed = (
                (result.arm_q[1:] - result.arm_q[:-1]).abs().amax(dim=-1)
                / float(control_dt)
            ).amax(dim=0)
        feasible = (
            (result.position_residual_m.amax(dim=0) <= position_tolerance_m)
            & (result.rotation_residual_rad.amax(dim=0) <= rotation_tolerance_rad)
            & (result.limit_margin_rad.amin(dim=0) >= limit_margin_rad)
            & (peak_speed <= velocity_limit_rad_s)
            & (kinematics.elbow_height_margin(result.arm_q).amin(dim=0) > 0.0)
        )
        return peak_speed, feasible

    primary_speed, primary_feasible = measurements(primary)
    failed = torch.nonzero(~primary_feasible, as_tuple=False).reshape(-1)
    if not failed.numel():
        return primary

    branch_count = len(_ALTERNATE_ARM_SEEDS)
    alternate_seed = torch.as_tensor(
        _ALTERNATE_ARM_SEEDS,
        dtype=kinematics.dtype,
        device=kinematics.device,
    )
    alternate_seed = alternate_seed.unsqueeze(0).expand(
        failed.numel(), branch_count, ARM_JOINT_COUNT
    ).reshape(-1, ARM_JOINT_COUNT)
    failed_yaw = yaw_rad.to(kinematics.device)[failed]
    failed_translation = translation.to(kinematics.device)[failed]
    alternates = retarget_clip(
        kinematics,
        demo_arm_q,
        failed_yaw.repeat_interleave(branch_count),
        failed_translation.repeat_interleave(branch_count, dim=0),
        pivot,
        initial_seed=alternate_seed,
    )
    alternate_speed, alternate_feasible = measurements(alternates)
    alternate_speed = alternate_speed.reshape(-1, branch_count)
    alternate_feasible = alternate_feasible.reshape(-1, branch_count)
    candidate_score = torch.where(
        alternate_feasible,
        alternate_speed,
        torch.full_like(alternate_speed, torch.inf),
    )
    selected_branch = candidate_score.argmin(dim=1)
    has_alternate = alternate_feasible.any(dim=1)
    if not has_alternate.any():
        return primary

    rows = torch.arange(failed.numel(), device=kinematics.device)
    flat_selected = rows * branch_count + selected_branch
    replace_env = failed[has_alternate]
    replace_alt = flat_selected[has_alternate]

    arm_q = primary.arm_q.clone()
    position = primary.position_residual_m.clone()
    rotation = primary.rotation_residual_rad.clone()
    margin = primary.limit_margin_rad.clone()
    arm_q[:, replace_env] = alternates.arm_q[:, replace_alt]
    position[:, replace_env] = alternates.position_residual_m[:, replace_alt]
    rotation[:, replace_env] = alternates.rotation_residual_rad[:, replace_alt]
    margin[:, replace_env] = alternates.limit_margin_rad[:, replace_alt]
    return RetargetResult(arm_q, position, rotation, margin)
