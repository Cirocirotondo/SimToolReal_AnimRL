"""Is this cuboid pose feasible, and if not, why?

Answers for an arbitrary ``(x, y, yaw)`` -- not only for points of the saved
grid -- by retargeting the whole demonstration for that transform and applying
the same four criteria ``build_transform_bank`` uses. A query takes a few
seconds; use ``--grid`` for an instant nearest-cell lookup instead.

    # exact, any pose
    PYTHONPATH=. python scripts/query_feasibility.py --pose 0.10 -0.05 45

    # several at once
    PYTHONPATH=. python scripts/query_feasibility.py \\
        --pose 0 0 0 --pose 0.18 0 30 --pose 0.10 0.10 60

    # instant, nearest cell of the saved grid
    PYTHONPATH=. python scripts/query_feasibility.py \\
        --grid banks/feasibility_grid.npz --pose 0.10 -0.05 45

``x`` and ``y`` are metres of planar translation of the cuboid from its
demonstrated position, ``yaw`` is degrees about the vertical axis through the
cuboid's own centre. A rejection names every criterion it failed, not just the
first: being out of reach and demanding impossible joint speeds usually happen
together, because once the inverse kinematics stops converging the trajectory it
returns is meaningless and its apparent speed is meaningless with it.
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from simtoolreal_animrl import ROOT_DIR
from simtoolreal_animrl.cfg import SimToolRealCfg
from simtoolreal_animrl.envs.demonstration import JointDemonstration60Hz
from simtoolreal_animrl.envs.retarget import (
    PalmKinematics,
    cube_pose_to_base_frame,
    retarget_clip,
)
from simtoolreal_animrl.envs.transform_bank import ARM_JOINT_VELOCITY_LIMIT_RAD_S


ARM_JOINT_NAMES = ("pan", "lift", "elbow", "wrist_1", "wrist_2", "wrist_3")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--pose", nargs=3, type=float, action="append", metavar=("X", "Y", "YAW_DEG"),
        required=True, help="Repeatable. Metres, metres, degrees.",
    )
    parser.add_argument(
        "--grid", default=None,
        help="Look the answer up in a saved feasibility grid instead of "
             "solving. Instant, but only as fine as the grid.",
    )
    parser.add_argument("--urdf", default=None)
    parser.add_argument("--demo", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--frame-stride", type=int, default=2,
        help="Subsampling smooths velocity, which is a rejection criterion. "
             "Stride 2 agrees with the full clip exactly; 16 does not.",
    )
    parser.add_argument("--position-tolerance-m", type=float, default=1e-3)
    parser.add_argument("--rotation-tolerance-rad", type=float, default=1e-2)
    parser.add_argument("--limit-margin-rad", type=float, default=0.05)
    parser.add_argument("--velocity-fraction", type=float, default=0.5)
    return parser.parse_args()


def describe(unreachable, too_fast, at_limit):
    reasons = []
    if unreachable:
        reasons.append("REACH")
    if too_fast:
        reasons.append("VELOCITY")
    if at_limit:
        reasons.append("JOINT LIMIT")
    return " + ".join(reasons)


def query_grid(path, poses):
    grid = np.load(str(Path(path).expanduser().resolve()), allow_pickle=True)
    x_axis, y_axis = grid["translation_x_m"], grid["translation_y_m"]
    yaw_axis = grid["yaw_deg"]
    print("Nearest-cell lookup in {}".format(path))
    print("  grid: {} x {} x {}, yaw {:+.1f}..{:+.1f} deg\n".format(
        len(x_axis), len(y_axis), len(yaw_axis), yaw_axis.min(), yaw_axis.max()))
    for x, y, yaw in poses:
        i = int(np.abs(x_axis - x).argmin())
        j = int(np.abs(y_axis - y).argmin())
        k = int(np.abs(yaw_axis - yaw).argmin())
        feasible = bool(grid["feasible"][i, j, k])
        verdict = "FEASIBLE" if feasible else "INFEASIBLE: " + describe(
            bool(grid["unreachable"][i, j, k]),
            bool(grid["too_fast"][i, j, k]),
            bool(grid["at_limit"][i, j, k]),
        )
        print("  ({:+.3f}, {:+.3f}, {:+6.1f} deg) -> cell ({:+.3f}, {:+.3f}, "
              "{:+.1f}) : {}".format(x, y, yaw, x_axis[i], y_axis[j],
                                     yaw_axis[k], verdict))
        print("      peak speed {:.2f} rad/s (limit {:.2f}), IK residual "
              "{:.3f} mm, limit margin {:.3f} rad, fastest at frame {}".format(
                  float(grid["peak_speed_rad_s"][i, j, k]),
                  float(grid["velocity_limit_rad_s"]),
                  1e3 * float(grid["max_position_residual_m"][i, j, k]),
                  float(grid["min_limit_margin_rad"][i, j, k]),
                  int(grid["fastest_frame"][i, j, k])))


def query_exact(arguments, poses):
    cfg = SimToolRealCfg()
    urdf = arguments.urdf or (ROOT_DIR / cfg.asset.file)
    demo_path = arguments.demo or (ROOT_DIR / cfg.motion.file)
    device = arguments.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    kinematics = PalmKinematics(urdf, device=device)
    demonstration = JointDemonstration60Hz.load(demo_path, device=device)
    stride = max(1, int(arguments.frame_stride))
    arm_q = demonstration.q.double()[::stride, :6]
    pivot = cube_pose_to_base_frame(demonstration.cube_pose.double())[0, :3]
    dt = stride / float(demonstration.frequency_hz)
    speed_limit = ARM_JOINT_VELOCITY_LIMIT_RAD_S * float(arguments.velocity_fraction)

    yaws = torch.tensor([math.radians(p[2]) for p in poses], dtype=torch.float64)
    translations = torch.tensor(
        [[p[0], p[1], 0.0] for p in poses], dtype=torch.float64
    )
    result = retarget_clip(kinematics, arm_q, yaws, translations, pivot)
    speed = (
        (result.arm_q[1:] - result.arm_q[:-1]).abs().amax(dim=-1) / dt
    ).cpu()

    print("Exact feasibility, solved over {} frames (every {}) of {}".format(
        arm_q.shape[0], stride, Path(demo_path).name))
    print("  criteria: IK residual <= {:.1f} mm and <= {:.1f} mrad, joint "
          "margin >= {:.2f} rad,\n            peak joint speed <= {:.2f} rad/s "
          "({:.0f}% of the arm's limit)\n".format(
              1e3 * arguments.position_tolerance_m,
              1e3 * arguments.rotation_tolerance_rad,
              arguments.limit_margin_rad, speed_limit,
              100 * arguments.velocity_fraction))

    for index, (x, y, yaw) in enumerate(poses):
        position = float(result.position_residual_m[:, index].max())
        rotation = float(result.rotation_residual_rad[:, index].max())
        margin = float(result.limit_margin_rad[:, index].min())
        peak = float(speed[:, index].max())
        frame = int(speed[:, index].argmax()) * stride
        joint = int(
            (result.arm_q[int(speed[:, index].argmax()) + 1, index]
             - result.arm_q[int(speed[:, index].argmax()), index]).abs().argmax()
        )
        unreachable = (position > arguments.position_tolerance_m) or (
            rotation > arguments.rotation_tolerance_rad
        )
        at_limit = margin < arguments.limit_margin_rad
        too_fast = peak > speed_limit
        feasible = not (unreachable or at_limit or too_fast)

        print("  ({:+.3f} m, {:+.3f} m, {:+6.1f} deg)  {}".format(
            x, y, yaw,
            "FEASIBLE" if feasible else
            "INFEASIBLE  -- " + describe(unreachable, too_fast, at_limit)))
        print("      reach    : {:8.3f} mm / {:7.3f} mrad {}".format(
            1e3 * position, 1e3 * rotation,
            "" if not unreachable else "  <-- over tolerance, worst at frame {}".format(
                int(result.position_residual_m[:, index].argmax()) * stride)))
        print("      velocity : {:8.2f} rad/s  ({} at frame {}) {}".format(
            peak, ARM_JOINT_NAMES[joint], frame,
            "" if not too_fast else "  <-- over limit"))
        print("      margin   : {:8.3f} rad {}".format(
            margin, "" if not at_limit else "  <-- against a joint stop"))
        if unreachable and too_fast:
            print("      note     : the speed reading is not meaningful here -- "
                  "once the IK stops\n                 converging, the joint "
                  "trajectory it returns is arbitrary.")
        print()


def main():
    arguments = parse_arguments()
    poses = [tuple(p) for p in arguments.pose]
    if arguments.grid:
        query_grid(arguments.grid, poses)
    else:
        query_exact(arguments, poses)


if __name__ == "__main__":
    main()
