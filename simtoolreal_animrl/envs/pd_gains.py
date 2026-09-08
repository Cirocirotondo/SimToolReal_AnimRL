"""Low-level position-drive gains, and the scaling applied to them.

Kept apart from ``controller.py`` because that module imports ``isaacgym``,
which insists on being imported before ``torch``; these values are plain
numbers and are needed by tests and tools that never build a simulation.
"""

# Tuned against both the recorded motion and the synthetic all-arm-joints
# trajectory. The official UR Gazebo effort-controller profile was used only as
# the starting shape; these lower gains preserve tracking without the
# float32-max fallback previously inherited from the asset.
ARM_PD_STIFFNESS = (1000.0, 1000.0, 1000.0, 200.0, 200.0, 100.0)
ARM_PD_DAMPING = (100.0, 100.0, 100.0, 10.0, 10.0, 10.0)

# Source of truth copied from the corrected hand_dofs == 20 branch of
# simtoolreal/isaacgymenvs/tasks/simtoolreal/utils.py and from the verified
# Isaac Gym demonstration viewer.
HAND_PD_STIFFNESS = (
    42.9718, 400.0, 42.9718, 42.9718,
    42.9718, 42.9718, 42.9718, 42.9718,
    42.9718, 42.9718, 42.9718, 42.9718,
    42.9718, 42.9718, 42.9718, 42.9718,
    42.9718, 42.9718, 42.9718, 42.9718,
)
HAND_PD_DAMPING = (
    0.1, 0.9475, 0.3012, 0.1821,
    0.7523, 0.4126, 0.2856, 0.1365,
    0.7587, 0.4126, 0.2856, 0.1365,
    0.7274, 0.4126, 0.2856, 0.1365,
    0.2662, 0.4796, 0.3012, 0.1821,
)


def scale_gains(gains, scale):
    """Scale a per-joint gain tuple, rejecting values that would break the PD.

    Lowering a joint's stiffness lowers its closed-loop bandwidth
    ``omega = sqrt(k / J)``, so the drive filters the policy's step-to-step
    chatter instead of tracking it into the joint. It also *raises* the damping
    ratio ``zeta = d / (2 sqrt(k J))`` for a fixed damping, which is why a
    softer hand is both slower and better damped -- the combination that makes
    a policy safe to put on the real robot.
    """
    scale = float(scale)
    if scale != scale or scale in (float("inf"), float("-inf")) or scale <= 0.0:
        raise ValueError("PD gain scale must be finite and positive")
    return tuple(float(gain) * scale for gain in gains)
