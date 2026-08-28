#!/usr/bin/env python3
"""Visually verify the fingertip points reconstructed from the collapsed URDF.

Red spheres mark the origins of rl_dg_<finger>_4. Green spheres mark the
fingertip points used by the observation vector. A colored segment joins each
distal-link origin to its fingertip point.

This is a disposable debugging viewer; it does not modify the environment.
"""

import argparse
import os
import sys
from pathlib import Path


def _configure_graphics_environment() -> None:
    """Prevent ROS/Gazebo libraries from shadowing NVIDIA graphics libraries."""
    library_path = os.environ.get("LD_LIBRARY_PATH", "")
    kept_paths = [
        entry
        for entry in library_path.split(":")
        if "/opt/ros/" not in entry and "/gazebo" not in entry.lower()
    ]
    if kept_paths:
        os.environ["LD_LIBRARY_PATH"] = ":".join(kept_paths)
    else:
        os.environ.pop("LD_LIBRARY_PATH", None)
    nvidia_icd = "/usr/share/vulkan/icd.d/nvidia_icd.json"
    if os.path.isfile(nvidia_icd):
        os.environ.setdefault("VK_ICD_FILENAMES", nvidia_icd)
    os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")


_configure_graphics_environment()

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Isaac Gym has to be imported before torch.
from isaacgym import gymapi, gymutil  # noqa: E402
import torch  # noqa: E402

from simtoolreal_animrl.cfg import SimToolRealCfg  # noqa: E402
from simtoolreal_animrl.envs.motion_imitation import (  # noqa: E402
    FINGERTIP_BODY_NAMES,
    FINGERTIP_OFFSETS,
    MotionImitationEnv,
    _normalize_canonical_quaternion,
    _quat_rotate,
)


FINGER_COLORS = (
    (1.0, 0.2, 0.2),
    (1.0, 0.7, 0.1),
    (0.2, 0.7, 1.0),
    (0.8, 0.3, 1.0),
    (0.1, 0.9, 0.8),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--demo-index",
        type=int,
        default=800,
        help="Demonstration pose to display (default: 800).",
    )
    parser.add_argument(
        "--point-radius",
        type=float,
        default=0.015,
        help="Radius in metres of the debug spheres (default: 0.015).",
    )
    return parser.parse_args()


def draw_sphere(env, position, radius, color) -> None:
    geometry = gymutil.WireframeSphereGeometry(
        radius, 12, 12, None, color=color
    )
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(*[float(value) for value in position])
    gymutil.draw_lines(geometry, env.gym, env.viewer, env.envs[0], pose)


def draw_segment(env, start, end, color) -> None:
    vertices = torch.stack((start, end)).detach().cpu().numpy().astype("float32")
    colors = torch.tensor([color], dtype=torch.float32).numpy()
    env.gym.add_lines(
        env.viewer,
        env.envs[0],
        1,
        vertices,
        colors,
    )


def draw_cross(env, center, half_length, color) -> None:
    """Draw a world-aligned 3D cross whose exact intersection is the point."""
    for axis in range(3):
        displacement = torch.zeros(3, dtype=center.dtype, device=center.device)
        displacement[axis] = half_length
        draw_segment(env, center - displacement, center + displacement, color)


def fingertip_debug_points(env):
    states = env.rigid_body_state[0, env.fingertip_body_indices]
    origins = states[:, 0:3]
    orientations = _normalize_canonical_quaternion(states[:, 3:7])
    offsets = torch.tensor(
        FINGERTIP_OFFSETS, dtype=torch.float32, device=env.device
    )
    fingertips = origins + _quat_rotate(orientations, offsets)
    return origins, fingertips


def main() -> None:
    args = parse_args()
    cfg = SimToolRealCfg()
    cfg.viewer.reference_ghost = False
    env = MotionImitationEnv(
        cfg,
        sim_device="cuda:0",
        headless=False,
        num_envs_override=1,
    )
    try:
        if not 0 <= args.demo_index < env.reference.last_index:
            raise ValueError(
                "--demo-index must lie in [0, {}]".format(
                    env.reference.last_index - 1
                )
            )
        env.reset(reference_index=args.demo_index)
        # A first physics step is required to publish the tensor-written actor
        # state to Isaac Gym's graphics subsystem. Without it the debug lines
        # are visible, but the robot mesh may not have been uploaded yet.
        env.gym.simulate(env.sim)
        env.gym.fetch_results(env.sim, True)
        env.gym.refresh_dof_state_tensor(env.sim)
        env.gym.refresh_actor_root_state_tensor(env.sim)
        env.gym.refresh_rigid_body_state_tensor(env.sim)
        origins, fingertips = fingertip_debug_points(env)
        environment_origin = env.gym.get_env_origin(env.envs[0])
        environment_origin = torch.tensor(
            [environment_origin.x, environment_origin.y, environment_origin.z],
            dtype=torch.float32,
            device=env.device,
        )

        print("Demonstration frame: {}".format(args.demo_index))
        print("Red = distal-link origin; green = observation fingertip")
        for finger, (name, offset) in enumerate(
            zip(FINGERTIP_BODY_NAMES, FINGERTIP_OFFSETS), start=1
        ):
            print(
                "finger {}: {:>9s}, local offset={}, world fingertip={}".format(
                    finger,
                    name,
                    tuple(float(value) for value in offset),
                    tuple(
                        round(float(value), 6)
                        for value in fingertips[finger - 1]
                    ),
                )
            )

        while not env.viewer_closed():
            env.gym.clear_lines(env.viewer)
            for index, color in enumerate(FINGER_COLORS):
                # Rigid-body tensors use simulation/world coordinates, while
                # viewer debug lines attached to an env use env-local ones.
                origin = origins[index] - environment_origin
                fingertip = fingertips[index] - environment_origin
                draw_segment(env, origin, fingertip, color)
                draw_sphere(
                    env,
                    origin,
                    args.point_radius * 0.65,
                    (1.0, 0.0, 0.0),
                )
                draw_sphere(
                    env,
                    fingertip,
                    args.point_radius,
                    (0.0, 1.0, 0.0),
                )
                # The exact URDF tip-frame origin can lie inside its visual
                # mesh. This cross extends outside it while keeping its
                # intersection exactly on the measured point.
                draw_cross(
                    env,
                    fingertip,
                    args.point_radius * 1.75,
                    (0.0, 1.0, 0.0),
                )
            env.gym.step_graphics(env.sim)
            env.gym.draw_viewer(env.viewer, env.sim, False)
    finally:
        env.close()


if __name__ == "__main__":
    main()
