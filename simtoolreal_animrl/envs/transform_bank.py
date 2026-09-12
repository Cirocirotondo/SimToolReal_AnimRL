"""A bank of feasible transforms, each with its own retargeted clip.

An episode draws two indices at reset: which demonstration frame to start from,
as it always has, and now also which transform of the scene it is playing. The
second one indexes this bank.

Why a pre-built bank rather than solving at reset. A transform has to be
feasible over the clip's *whole* length. Solving only for the frame an episode
resets at cannot know that frame 950 is unreachable when the reset happens at
frame 700 -- the episode would start cleanly and then walk into a hole two
seconds later. Building offline lets every candidate be rejected on evidence
before any policy ever sees it, and lets the acceptance rate be *reported*: a
range that admits 40% of its samples is training on a silently biased
distribution, and that should be a number on the screen rather than a surprise
in a learning curve.

Two conventions worth knowing before changing anything here:

* Cube poses are stored in the **UR controller's base frame**, the same frame
  the recorded demonstration uses -- not the ``base_link`` frame the retargeter
  solves in. That is deliberate: it means the environment's existing
  ``_cube_reference_root_states`` conversion keeps working untouched.
* Arm velocities are mapped through the Jacobian rather than finite-differenced.
  A rigid transform rotates the palm twist and leaves its magnitude alone, so
  ``dq_new = J_new^+ . Ad_T . J_demo . dq_demo``. At the identity transform this
  returns the recorded velocities exactly, which finite differencing would not.

The reference keypoint curve is stored **once**, not once per transform. In the
cuboid's frame it does not depend on the transform at all.
"""

import hashlib
import json
from pathlib import Path
from typing import NamedTuple, Optional, Union

import torch

from simtoolreal_animrl.envs.retarget import (
    ARM_JOINT_COUNT,
    PalmKinematics,
    cube_pose_from_base_frame,
    cube_pose_to_base_frame,
    reference_keypoints_in_object_frame,
    retarget_clip,
    transform_points,
    yaw_quaternion,
)
from simtoolreal_animrl.envs.rotations import quat_multiply, quat_rotate, quat_to_matrix


# Near a wrist singularity the damped pseudo-inverse's gain peaks at 1/(2*lambda),
# so 1e-6 allowed a half-million-fold amplification and produced reference
# velocities of 145 rad/s on a joint limited to pi. At 1e-3 the worst gain is 500
# and a well-conditioned Jacobian is still inverted to within 1e-6.
VELOCITY_DAMPING = 1e-3

# Every UR5e arm joint declares velocity="3.141592653589793" in the URDF. A
# reference that asks for more than the robot can deliver is not a reference:
# the policy cannot track it, the drive saturates, and the tracking terms stay
# pinned low for that transform no matter how good the policy gets.
ARM_JOINT_VELOCITY_LIMIT_RAD_S = 3.141592653589793


def nearest_transform_indices(
    episode_translation: torch.Tensor,
    episode_yaw_rad: torch.Tensor,
    bank_translation: torch.Tensor,
    bank_yaw_rad: torch.Tensor,
    yaw_lever_arm_m: float,
) -> torch.Tensor:
    """Nearest bank transform under a Cartesian-equivalent SE(2) metric."""
    delta_xy = (
        episode_translation[:, None, :2] - bank_translation[None, :, :2]
    )
    delta_yaw = (
        episode_yaw_rad[:, None] - bank_yaw_rad[None, :] + torch.pi
    ).remainder(2.0 * torch.pi) - torch.pi
    return (
        delta_xy.square().sum(dim=2)
        + (float(yaw_lever_arm_m) * delta_yaw).square()
    ).argmin(dim=1)


class BankSample(NamedTuple):
    """Deliberately the same field names ``JointDemonstration60Hz`` returns."""

    q: torch.Tensor
    dq: torch.Tensor
    cube_pose: torch.Tensor
    cube_linear_velocity: torch.Tensor
    cube_angular_velocity: torch.Tensor


class TransformBank:
    """``(transforms, frames)`` of retargeted reference, indexed per episode."""

    def __init__(
        self,
        yaw_rad: torch.Tensor,
        translation: torch.Tensor,
        q: torch.Tensor,
        dq: torch.Tensor,
        cube_pose: torch.Tensor,
        cube_linear_velocity: torch.Tensor,
        cube_angular_velocity: torch.Tensor,
        reference_keypoints: torch.Tensor,
        acceptance: float,
    ) -> None:
        self.yaw_rad = yaw_rad
        self.translation = translation
        self.q = q
        self.dq = dq
        self.cube_pose = cube_pose
        self.cube_linear_velocity = cube_linear_velocity
        self.cube_angular_velocity = cube_angular_velocity
        self.reference_keypoints = reference_keypoints
        self.acceptance = float(acceptance)

    @property
    def transform_count(self) -> int:
        return int(self.q.shape[0])

    @property
    def sample_count(self) -> int:
        return int(self.q.shape[1])

    @property
    def last_index(self) -> int:
        return self.sample_count - 1

    def to(self, device, dtype=torch.float32) -> "TransformBank":
        move = lambda tensor: tensor.to(device=device, dtype=dtype)
        return TransformBank(
            move(self.yaw_rad),
            move(self.translation),
            move(self.q),
            move(self.dq),
            move(self.cube_pose),
            move(self.cube_linear_velocity),
            move(self.cube_angular_velocity),
            move(self.reference_keypoints),
            self.acceptance,
        )

    def sample(
        self, transform_indices: torch.Tensor, frame_indices: torch.Tensor
    ) -> BankSample:
        """Gather one ``(transform, frame)`` pair per environment."""
        transform_indices = transform_indices.long()
        frame_indices = frame_indices.long()
        if transform_indices.shape != frame_indices.shape:
            raise ValueError("Transform and frame indices must have one shape")
        if torch.any(transform_indices < 0) or torch.any(
            transform_indices >= self.transform_count
        ):
            raise IndexError("Transform index outside the bank")
        if torch.any(frame_indices < 0) or torch.any(frame_indices > self.last_index):
            raise IndexError("Reference index outside the demonstration")
        return BankSample(
            self.q[transform_indices, frame_indices],
            self.dq[transform_indices, frame_indices],
            self.cube_pose[transform_indices, frame_indices],
            self.cube_linear_velocity[transform_indices, frame_indices],
            self.cube_angular_velocity[transform_indices, frame_indices],
        )

    def keypoints_at(self, frame_indices: torch.Tensor) -> torch.Tensor:
        """``(N, 9, 3)`` reference keypoints -- no transform index needed."""
        return self.reference_keypoints[frame_indices.long()]

    def save(self, path: Union[str, Path]) -> None:
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "yaw_rad": self.yaw_rad,
                "translation": self.translation,
                "q": self.q,
                "dq": self.dq,
                "cube_pose": self.cube_pose,
                "cube_linear_velocity": self.cube_linear_velocity,
                "cube_angular_velocity": self.cube_angular_velocity,
                "reference_keypoints": self.reference_keypoints,
                "acceptance": self.acceptance,
            },
            str(destination),
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TransformBank":
        # Pure tensors and one float, so the restrictive loader is enough and
        # a cached bank can never execute anything.
        payload = torch.load(
            str(Path(path).expanduser().resolve()),
            map_location="cpu",
            weights_only=True,
        )
        return cls(
            payload["yaw_rad"],
            payload["translation"],
            payload["q"],
            payload["dq"],
            payload["cube_pose"],
            payload["cube_linear_velocity"],
            payload["cube_angular_velocity"],
            payload["reference_keypoints"],
            payload["acceptance"],
        )


def map_arm_velocities(
    kinematics: PalmKinematics,
    demo_arm_q: torch.Tensor,
    demo_arm_dq: torch.Tensor,
    solved_arm_q: torch.Tensor,
    yaw_rad: torch.Tensor,
) -> torch.Tensor:
    """Carry recorded joint velocities through the transform.

    A rigid transform rotates the palm's twist and changes nothing else about
    it, so the new joint velocities are whatever produces the rotated twist in
    the new configuration. At the identity transform the rotation is identity
    and the new configuration is the old one, so this returns ``demo_arm_dq``.
    """
    frames, batch = solved_arm_q.shape[0], solved_arm_q.shape[1]
    rotation = quat_to_matrix(yaw_quaternion(yaw_rad))          # (B, 3, 3)
    demo_jacobian = kinematics.jacobian(demo_arm_q)             # (frames, 6, 6)
    demo_twist = torch.einsum(
        "fij,fj->fi", demo_jacobian, demo_arm_dq
    )                                                            # (frames, 6)
    linear = torch.einsum("bij,fj->fbi", rotation, demo_twist[:, :3])
    angular = torch.einsum("bij,fj->fbi", rotation, demo_twist[:, 3:])
    target_twist = torch.cat((linear, angular), dim=-1)          # (frames, B, 6)

    flat_q = solved_arm_q.reshape(frames * batch, ARM_JOINT_COUNT)
    jacobian = kinematics.jacobian(flat_q)
    gram = jacobian @ jacobian.transpose(-1, -2)
    gram = gram + (VELOCITY_DAMPING ** 2) * torch.eye(
        6, dtype=gram.dtype, device=gram.device
    )
    solved = jacobian.transpose(-1, -2) @ torch.linalg.solve(
        gram, target_twist.reshape(frames * batch, 6, 1)
    )
    return solved.squeeze(-1).reshape(frames, batch, ARM_JOINT_COUNT)


def transform_cube_track(
    demo_cube_pose_base: torch.Tensor,
    demo_linear_velocity: torch.Tensor,
    demo_angular_velocity: torch.Tensor,
    yaw_rad: torch.Tensor,
    translation: torch.Tensor,
    pivot: torch.Tensor,
):
    """Apply the transform to the cuboid's whole recorded track.

    Velocities rotate but do not translate: a constant offset has no derivative.
    Returns poses in the base frame; the caller converts back to UR-base.
    """
    frames = demo_cube_pose_base.shape[0]
    batch = yaw_rad.shape[0]
    rotation_quaternion = yaw_quaternion(yaw_rad)
    rotation = quat_to_matrix(rotation_quaternion)

    positions = transform_points(
        demo_cube_pose_base[:, None, :3].expand(frames, batch, 3),
        yaw_rad[None, :].expand(frames, batch),
        translation[None, :, :].expand(frames, batch, 3),
        pivot,
    )
    orientations = quat_multiply(
        rotation_quaternion[None, :, :].expand(frames, batch, 4),
        demo_cube_pose_base[:, None, 3:7].expand(frames, batch, 4),
    )
    linear = torch.einsum("bij,fj->fbi", rotation, demo_linear_velocity)
    angular = torch.einsum("bij,fj->fbi", rotation, demo_angular_velocity)
    return torch.cat((positions, orientations), dim=-1), linear, angular


def build_transform_bank(
    kinematics: PalmKinematics,
    demonstration,
    transform_count: int,
    translation_m: float,
    yaw_low_rad: float,
    yaw_high_rad: float,
    lever_arm_m: float,
    seed: int = 0,
    batch: int = 256,
    max_attempts_multiplier: int = 4,
    position_tolerance_m: float = 1e-3,
    rotation_tolerance_rad: float = 1e-2,
    limit_margin_rad: float = 0.05,
    velocity_fraction: float = 0.5,
    control_dt: Optional[float] = None,
    verbose: bool = True,
) -> TransformBank:
    """Sample transforms until ``transform_count`` feasible clips are collected.

    ``demonstration`` is a ``JointDemonstration60Hz``. Rejection is on whole-clip
    evidence: the worst frame's IK residual, the tightest joint-limit margin, and
    the fastest joint motion the clip demands, all over the entire motion.

    The velocity test is not optional. A transform whose palm path passes near a
    wrist singularity is solvable at every individual frame and still useless:
    tracking it needs joint speeds the arm does not have. Measured on a bank
    built without this check, 9.6% of accepted transforms exceeded the limit,
    the worst asking 12.2 rad/s of a joint capped at 3.14.
    """
    if control_dt is None:
        # Read the step from the clip's own timestamps rather than assuming
        # 60 Hz. A subsampled clip is still a valid demonstration object and
        # would otherwise be judged at 1/60 s per frame, inflating every
        # measured joint speed by the subsampling factor.
        intervals = torch.diff(demonstration.monotonic_timestamp.double())
        control_dt = float(intervals.median())
    control_dt = float(control_dt)
    if not control_dt > 0.0:
        raise ValueError("control_dt must be positive")

    dtype, device = kinematics.dtype, kinematics.device
    demo_q = demonstration.q.to(dtype=dtype, device=device)
    demo_dq = demonstration.dq.to(dtype=dtype, device=device)
    demo_arm_q = demo_q[:, :ARM_JOINT_COUNT]
    demo_arm_dq = demo_dq[:, :ARM_JOINT_COUNT]
    demo_cube_ur = demonstration.cube_pose.to(dtype=dtype, device=device)
    demo_cube_base = cube_pose_to_base_frame(demo_cube_ur)
    pivot = demo_cube_base[0, :3]

    reference_keypoints = reference_keypoints_in_object_frame(
        kinematics, demo_q, demo_cube_base, lever_arm_m
    )

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    collected, attempted, accepted = [], 0, 0
    limit = transform_count * int(max_attempts_multiplier)
    while accepted < transform_count and attempted < limit:
        size = min(batch, limit - attempted)
        yaw = (
            torch.rand(size, generator=generator, dtype=torch.float64)
            * (yaw_high_rad - yaw_low_rad)
            + yaw_low_rad
        ).to(dtype=dtype, device=device)
        translation = torch.zeros(size, 3, dtype=dtype, device=device)
        translation[:, :2] = (
            (torch.rand(size, 2, generator=generator, dtype=torch.float64) * 2.0 - 1.0)
            * float(translation_m)
        ).to(dtype=dtype, device=device)

        # The training bank intentionally preserves the demonstrated joint
        # family. An alternative elbow/wrist branch may reach the same palm
        # pose, but it is not the manipulation posture selected for this task.
        result = retarget_clip(
            kinematics,
            demo_arm_q,
            yaw,
            translation,
            pivot,
        )
        peak_speed = (
            (result.arm_q[1:] - result.arm_q[:-1]).abs().amax(dim=-1) / control_dt
        ).amax(dim=0)
        feasible = (
            (result.position_residual_m.amax(dim=0) <= position_tolerance_m)
            & (result.rotation_residual_rad.amax(dim=0) <= rotation_tolerance_rad)
            & (result.limit_margin_rad.amin(dim=0) >= limit_margin_rad)
            & (
                peak_speed
                <= ARM_JOINT_VELOCITY_LIMIT_RAD_S * float(velocity_fraction)
            )
        )
        attempted += size
        keep = torch.nonzero(feasible, as_tuple=False).reshape(-1)
        if keep.numel():
            collected.append(
                (yaw[keep], translation[keep], result.arm_q[:, keep, :])
            )
            accepted += int(keep.numel())
        if verbose:
            print(
                "  attempted {:5d}  accepted {:5d}  ({:.1f}%)".format(
                    attempted, accepted, 100.0 * accepted / max(attempted, 1)
                ),
                flush=True,
            )

    if accepted < transform_count:
        raise RuntimeError(
            "Only {} of {} transforms were feasible after {} attempts. The "
            "requested range is too wide for this arm -- run "
            "scripts/sweep_transform_feasibility.py to find one that is not.".format(
                accepted, transform_count, attempted
            )
        )

    yaw = torch.cat([item[0] for item in collected])[:transform_count]
    translation = torch.cat([item[1] for item in collected])[:transform_count]
    arm_q = torch.cat([item[2] for item in collected], dim=1)[:, :transform_count, :]

    arm_dq = map_arm_velocities(kinematics, demo_arm_q, demo_arm_dq, arm_q, yaw)
    frames = demo_q.shape[0]
    hand_q = demo_q[:, None, ARM_JOINT_COUNT:].expand(
        frames, transform_count, demo_q.shape[1] - ARM_JOINT_COUNT
    )
    hand_dq = demo_dq[:, None, ARM_JOINT_COUNT:].expand_as(hand_q)
    q = torch.cat((arm_q, hand_q), dim=-1).permute(1, 0, 2).contiguous()
    dq = torch.cat((arm_dq, hand_dq), dim=-1).permute(1, 0, 2).contiguous()

    cube_base, linear, angular = transform_cube_track(
        demo_cube_base,
        demonstration.cube_linear_velocity.to(dtype=dtype, device=device),
        demonstration.cube_angular_velocity.to(dtype=dtype, device=device),
        yaw,
        translation,
        pivot,
    )
    # Back to the UR base frame, which is what the environment's existing
    # world conversion expects.
    cube_ur = cube_pose_from_base_frame(cube_base).permute(1, 0, 2).contiguous()

    return TransformBank(
        yaw,
        translation,
        q,
        dq,
        cube_ur,
        linear.permute(1, 0, 2).contiguous(),
        angular.permute(1, 0, 2).contiguous(),
        reference_keypoints,
        accepted / max(attempted, 1),
    )


def bank_cache_path(cache_dir: Union[str, Path], **parameters) -> Path:
    """A path that changes whenever any parameter that shaped the bank does."""
    digest = hashlib.sha256(
        json.dumps(parameters, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    return Path(cache_dir).expanduser().resolve() / "transform_bank_{}.pt".format(digest)
