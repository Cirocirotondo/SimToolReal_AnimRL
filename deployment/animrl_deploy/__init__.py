"""Hardware clients, kinematics and safety used to deploy AnimRL policies.

The policy contract itself is NOT redefined here. Observations and the
residual-action mapping are imported from ``simtoolreal_animrl.sim2sim`` so
that a policy behaves identically on the robot and in sim2sim. If that package
changes its observation dimension, deployment must be re-validated.
"""

from .arm_client import ArmClient, ArmClientError
from .cube_source import (
    CubeSource,
    CubeSourceError,
    DemonstrationCube,
    FrozenCube,
    PoseEstimationCube,
)
from .hand_client import HandClient, HandClientError
from .kinematics import HardwareKinematics
from .safety import (
    SafetyAbort,
    SpikeMonitor,
    TargetLimiter,
    confirm_send,
    wait_for_key,
)

__all__ = [
    "ArmClient",
    "ArmClientError",
    "CubeSource",
    "CubeSourceError",
    "DemonstrationCube",
    "FrozenCube",
    "PoseEstimationCube",
    "HandClient",
    "HandClientError",
    "HardwareKinematics",
    "SafetyAbort",
    "SpikeMonitor",
    "TargetLimiter",
    "confirm_send",
    "wait_for_key",
]
