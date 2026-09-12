"""Build the bank of retargeted reference clips that training draws from.

Run this once per configuration; training loads the result and never solves any
inverse kinematics itself. Keeping it a separate artefact rather than a startup
step means the bank is reviewable, reproducible from its recorded arguments, and
that ``pytorch_kinematics`` stays out of the training process.

    PYTHONPATH=. /home/simone/.venv/bin/python scripts/build_transform_bank.py

Pick the range with ``scripts/sweep_transform_feasibility.py`` first. The
defaults here are what that sweep measured: translation +/- 0.20 m and yaw
[-22.5, +90] degrees, which is deliberately **asymmetric** because the arm's
reachable envelope is -- it turns the bar one way almost freely and the other way
barely at all.
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch

from simtoolreal_animrl import ROOT_DIR
from simtoolreal_animrl.envs.demonstration import JointDemonstration60Hz
from simtoolreal_animrl.envs.retarget import PalmKinematics
from simtoolreal_animrl.envs.transform_bank import build_transform_bank


DEFAULT_URDF = "assets/urdf/ur5e_delto_description/ur5e_right_dg5f_mount_60deg.urdf"
DEFAULT_DEMO = (
    "demonstrations/"
    "demo_20260727_152551_335339_60hz_cube_collision_resolved_stable_grasp.npz"
)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", default=str(ROOT_DIR / DEFAULT_URDF))
    parser.add_argument("--demo", default=str(ROOT_DIR / DEFAULT_DEMO))
    parser.add_argument("--output", default=str(ROOT_DIR / "banks/stage1.pt"))
    parser.add_argument("--transform-count", type=int, default=512)
    parser.add_argument("--translation-m", type=float, default=0.20)
    parser.add_argument("--yaw-min-deg", type=float, default=-22.5)
    parser.add_argument("--yaw-max-deg", type=float, default=90.0)
    parser.add_argument(
        "--lever-arm-m", type=float, default=0.1,
        help="Palm keypoint lever arm; must match cfg.rewards.palm_lever_arm_m.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--position-tolerance-m", type=float, default=1e-3)
    parser.add_argument("--rotation-tolerance-rad", type=float, default=1e-2)
    parser.add_argument("--limit-margin-rad", type=float, default=0.05)
    return parser.parse_args()


def resolve_device(requested):
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("  CUDA unavailable, falling back to CPU (slower, same result)")
        return "cpu"
    return requested


def main():
    arguments = parse_arguments()
    device = resolve_device(arguments.device)
    demonstration = JointDemonstration60Hz.load(arguments.demo, device=device)
    kinematics = PalmKinematics(arguments.urdf, device=device)

    print("Building the transform bank")
    print("  demonstration : {} frames at {:.2f} Hz".format(
        demonstration.sample_count, demonstration.frequency_hz))
    print("  transforms    : {}".format(arguments.transform_count))
    print("  translation   : +/- {:.3f} m".format(arguments.translation_m))
    print("  yaw           : [{:+.1f}, {:+.1f}] deg  (asymmetric on purpose)".format(
        arguments.yaw_min_deg, arguments.yaw_max_deg))
    print("  device        : {}".format(device))
    print()

    started = time.time()
    bank = build_transform_bank(
        kinematics,
        demonstration,
        transform_count=arguments.transform_count,
        translation_m=arguments.translation_m,
        yaw_low_rad=math.radians(arguments.yaw_min_deg),
        yaw_high_rad=math.radians(arguments.yaw_max_deg),
        lever_arm_m=arguments.lever_arm_m,
        seed=arguments.seed,
        batch=arguments.batch,
        position_tolerance_m=arguments.position_tolerance_m,
        rotation_tolerance_rad=arguments.rotation_tolerance_rad,
        limit_margin_rad=arguments.limit_margin_rad,
    )
    elapsed = time.time() - started

    # Stored as float32: the environment reads these every reset and the solver
    # tolerance is 1e-4 m, far above float32's resolution at these magnitudes.
    bank = bank.to(device="cpu", dtype=torch.float32)
    destination = Path(arguments.output).expanduser().resolve()
    bank.save(destination)

    sidecar = destination.with_suffix(".json")
    sidecar.write_text(json.dumps(vars(arguments), indent=2, sort_keys=True))

    print()
    print("built in {:.1f} s".format(elapsed))
    print("  acceptance      {:.1f}%".format(100.0 * bank.acceptance))
    print("  transforms      {}".format(bank.transform_count))
    print("  frames          {}".format(bank.sample_count))
    print("  yaw sampled     [{:+.1f}, {:+.1f}] deg".format(
        math.degrees(float(bank.yaw_rad.min())),
        math.degrees(float(bank.yaw_rad.max()))))
    print("  |translation|   up to {:.3f} m".format(
        float(bank.translation[:, :2].abs().max())))
    size_mb = sum(
        tensor.numel() * tensor.element_size()
        for tensor in (bank.q, bank.dq, bank.cube_pose,
                       bank.cube_linear_velocity, bank.cube_angular_velocity)
    ) / 1e6
    print("  tensors         {:.0f} MB".format(size_mb))
    print("\nwritten to {}".format(destination))
    print("arguments recorded in {}".format(sidecar))
    if bank.acceptance < 0.9:
        print("\n  WARNING: acceptance below 90%. The admitted transforms are "
              "denser near the\n  demonstration pose than at the edges, so the "
              "training distribution is biased.\n  Run "
              "scripts/sweep_transform_feasibility.py to find a cleaner range.")


if __name__ == "__main__":
    main()
