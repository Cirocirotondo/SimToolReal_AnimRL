"""Task-space arm control: move a Jacobian onto the palm, then invert it once.

The arm's six actions are an end-effector twist rather than six joint offsets, so
something has to turn ``[dx, dy, dz, wx, wy, wz]`` into joint targets every
control step. That is the two functions here, plus the accumulator in
:mod:`simtoolreal_animrl.envs.motion_imitation` that owns the state.

Two things make this file worth having separately from the environment:

* It touches no Isaac Gym symbol, so the frame convention it encodes can be
  tested against ``pytorch_kinematics`` on a CPU with no simulator running --
  and the frame convention is the part most likely to be wrong.
* Isaac Gym reports a Jacobian for ``wrist_3_link``, not for the palm, because
  the fixed wrist -> mount -> palm chain is collapsed when the asset loads. The
  reward, the observation and the termination all live at the palm, so the
  Jacobian has to be moved there before it is inverted.

:func:`damped_least_squares_step` deliberately duplicates the formula in
``retarget.solve_palm_ik``. That one is float64, CPU, ``pytorch_kinematics``, and
iterates to a tolerance because it built the transform bank sitting on disk; this
one is float32, on the GPU, and runs once per control step for every environment.
Sharing six lines of algebra would put a lazy ``pytorch_kinematics`` import on
the hot path. ``tests/test_operational_space.py`` asserts the two agree instead.
"""

import torch


def skew(vectors: torch.Tensor) -> torch.Tensor:
    """``(..., 3) -> (..., 3, 3)``, the matrix with ``skew(a) @ b == a x b``."""
    if vectors.shape[-1] != 3:
        raise ValueError(
            "skew expects trailing dimension 3, got {}".format(tuple(vectors.shape))
        )
    x, y, z = vectors[..., 0], vectors[..., 1], vectors[..., 2]
    zero = torch.zeros_like(x)
    return torch.stack(
        (
            torch.stack((zero, -z, y), dim=-1),
            torch.stack((z, zero, -x), dim=-1),
            torch.stack((-y, x, zero), dim=-1),
        ),
        dim=-2,
    )


def saturate_direction_preserving(
    vectors: torch.Tensor, limit: float
) -> torch.Tensor:
    """Cap ``|vectors|`` at ``limit`` without turning them.

    Clipping each component against the limit independently would bound the
    magnitude and change the direction too -- a diagonal command comes back
    pointing along whichever axes were not clipped. Scaling the whole vector
    keeps the direction the policy asked for and leaves only the magnitude
    saturated, so the direction stays controllable across the policy's entire
    output range instead of going flat past the limit.

    The same reasoning the object-assist wrench already follows, for the same
    reason: a large request should be limited, not redirected.
    """
    norm = vectors.norm(dim=-1, keepdim=True)
    return vectors * (float(limit) / norm.clamp_min(float(limit)))


def transfer_jacobian(
    jacobian: torch.Tensor, offset_world: torch.Tensor
) -> torch.Tensor:
    """Move a geometric Jacobian from a body origin to a point rigidly fixed to it.

    ``jacobian`` is ``(B, 6, N)`` with rows ``[linear; angular]``; ``offset_world``
    is ``(B, 3)``, the point minus the body origin, expressed in the *same* frame
    the Jacobian is.

    A rigid attachment shares the body's angular velocity and picks up the lever
    arm on the linear part: ``v_P = v_O + omega x r``, and ``omega x r`` is
    ``-skew(r) omega``, so

        J_lin_P = J_lin_O - skew(r) @ J_ang_O
        J_ang_P = J_ang_O

    The palm's fixed 60 degree rotation relative to ``wrist_3_link`` never enters:
    the twist stays in the world frame, and only the origin of the controlled
    frame moves.
    """
    if jacobian.dim() != 3 or jacobian.shape[1] != 6:
        raise ValueError(
            "transfer_jacobian expects (B, 6, N), got {}".format(tuple(jacobian.shape))
        )
    if offset_world.shape != (jacobian.shape[0], 3):
        raise ValueError(
            "offset_world must be ({}, 3), got {}".format(
                jacobian.shape[0], tuple(offset_world.shape)
            )
        )
    linear, angular = jacobian[:, :3], jacobian[:, 3:]
    return torch.cat((linear - skew(offset_world) @ angular, angular), dim=1)


def damped_least_squares_step(
    jacobian: torch.Tensor, twist: torch.Tensor, damping: float
) -> torch.Tensor:
    """One Levenberg-Marquardt step: ``J^T (J J^T + lambda^2 I)^-1 twist``.

    Damped rather than a plain pseudo-inverse because the UR5e passes close to
    wrist and elbow singularities under some of the bank's transforms, where an
    undamped inverse asks for an unbounded joint velocity. The damping trades
    accuracy there for a step that stays finite, and the caller measures what it
    gave up by pushing the result back through ``jacobian``.
    """
    if jacobian.dim() != 3 or jacobian.shape[1] != 6:
        raise ValueError(
            "damped_least_squares_step expects (B, 6, N), got {}".format(
                tuple(jacobian.shape)
            )
        )
    if twist.shape != (jacobian.shape[0], 6):
        raise ValueError(
            "twist must be ({}, 6), got {}".format(
                jacobian.shape[0], tuple(twist.shape)
            )
        )
    transpose = jacobian.transpose(-1, -2)
    gram = jacobian @ transpose
    gram = gram + (float(damping) ** 2) * torch.eye(
        6, dtype=gram.dtype, device=gram.device
    )
    solved = transpose @ torch.linalg.solve(gram, twist.unsqueeze(-1))
    return solved.squeeze(-1)
