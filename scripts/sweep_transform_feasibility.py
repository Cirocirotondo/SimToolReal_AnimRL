"""Measure where the arm actually gives out, instead of guessing a range.

The object-centric reference only works where the retargeted clip is
kinematically solvable for its *whole* length. A transform that is reachable at
the approach but not at the lift is worse than useless: an episode reset into it
starts fine and then walks into a hole.

So this sweeps a grid of planar translations and yaws, retargets the clip for
each, and reports three things:

* the overall acceptance rate, printed rather than inferred, because a range
  that admits 40% of its samples is training on a silently biased distribution;
* where the failures are, as a map, so the shape of the envelope is visible
  rather than summarised into one number;
* the largest symmetric box that clears an acceptance threshold, which is what
  the sampling range should actually be set to.

Run it before touching the sampling ranges in the config:

    PYTHONPATH=. /home/simone/.venv/bin/python scripts/sweep_transform_feasibility.py

Feasibility is judged on a subsampled clip. At 60 Hz the palm moves under a
millimetre between neighbouring frames, so every sixteenth frame traces the same
reachability envelope for a sixteenth of the solver time; the bank that training
actually uses is then built at full resolution.
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from simtoolreal_animrl import ROOT_DIR
from simtoolreal_animrl.envs.demonstration import JointDemonstration60Hz
from simtoolreal_animrl.envs.retarget import (
    ARM_JOINT_COUNT,
    PalmKinematics,
    cube_pose_from_base_frame,
    cube_pose_to_base_frame,
    reference_keypoints_in_object_frame,
    retarget_clip,
)
from simtoolreal_animrl.envs.transform_bank import (
    ARM_JOINT_VELOCITY_LIMIT_RAD_S,
    TransformBank,
    map_arm_velocities,
    transform_cube_track,
)


DEFAULT_URDF = (
    "assets/urdf/ur5e_delto_description/ur5e_right_dg5f_mount_60deg.urdf"
)
DEFAULT_DEMO = (
    "demonstrations/"
    "demo_20260727_152551_335339_60hz_cube_collision_resolved_stable_grasp.npz"
)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", default=str(ROOT_DIR / DEFAULT_URDF))
    parser.add_argument("--demo", default=str(ROOT_DIR / DEFAULT_DEMO))
    parser.add_argument("--translation-m", type=float, default=0.15)
    parser.add_argument(
        "--translation-y-center-m", type=float, default=0.07,
        help="Centre of the y-translation interval. The default shifts the "
             "whole grid 7 cm toward positive y, giving [-0.08, +0.22] m "
             "when --translation-m is 0.15.",
    )
    parser.add_argument(
        "--yaw-min-deg", type=float, default=-90.0,
        help="Low end of the yaw range. The reachable envelope is not "
             "symmetric, so this is a pair rather than a single +/- number.",
    )
    parser.add_argument("--yaw-max-deg", type=float, default=90.0)
    parser.add_argument("--translation-steps", type=int, default=11)
    parser.add_argument("--yaw-steps", type=int, default=9)
    parser.add_argument(
        "--frame-stride", type=int, default=2,
        help="Sample every Nth demonstration frame when judging feasibility. "
             "Subsampling smooths velocity, which is a rejection criterion: "
             "measured against the full clip, stride 2 agrees exactly, stride 4 "
             "on 99.2%% of cells and stride 16 on only 96.9%%, and stride 16 "
             "reports acceptance about 3 points optimistic.",
    )
    parser.add_argument(
        "--batch", type=int, default=64,
        help="Transforms solved together. Smaller batches print progress more "
             "frequently; 64 is a responsive CPU default.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--position-tolerance-m", type=float, default=1e-3,
        help="Largest palm position residual an accepted transform may have.",
    )
    parser.add_argument("--rotation-tolerance-rad", type=float, default=1e-2)
    parser.add_argument(
        "--lever-arm-m", type=float, default=0.1,
        help="Palm keypoint lever arm stored in the training bank.",
    )
    parser.add_argument(
        "--limit-margin-rad", type=float, default=0.05,
        help="How far every arm joint must stay from its limit.",
    )
    parser.add_argument(
        "--velocity-fraction", type=float, default=0.5,
        help="Fraction of the arm's joint velocity limit a clip may demand. "
             "Must match build_transform_bank, or this sweep recommends a box "
             "the bank will then reject a fifth of.",
    )
    parser.add_argument(
        "--save", default=None,
        help="Write the raw acceptance grid here as .npz for later analysis.",
    )
    parser.add_argument(
        "--save-bank", default=None,
        help="Also save every feasible full-resolution IK trajectory as a "
             "training TransformBank. Requires --frame-stride 1.",
    )
    parser.add_argument(
        "--acceptance", type=float, default=0.9,
        help="Acceptance fraction the reported box must clear.",
    )
    return parser.parse_args()


def load_demonstration(path, stride):
    with np.load(str(Path(path).expanduser().resolve())) as archive:
        arm_q = np.asarray(archive["arm_q"])
        cube_pose = np.asarray(archive["cube_pose"])
    total = len(arm_q)
    sampled = torch.as_tensor(arm_q[::stride], dtype=torch.float64)
    pivot = cube_pose_to_base_frame(
        torch.as_tensor(cube_pose[0], dtype=torch.float64)
    )[:3]
    return sampled, pivot, total


def build_grid(
    translation_m, translation_y_center_m, yaw_min_deg, yaw_max_deg,
    translation_steps, yaw_steps,
):
    x_axis = torch.linspace(
        -translation_m, translation_m, translation_steps, dtype=torch.float64
    )
    y_axis = torch.linspace(
        translation_y_center_m - translation_m,
        translation_y_center_m + translation_m,
        translation_steps,
        dtype=torch.float64,
    )
    yaws = torch.linspace(
        math.radians(yaw_min_deg), math.radians(yaw_max_deg), yaw_steps,
        dtype=torch.float64,
    )
    dx, dy, yaw = torch.meshgrid(x_axis, y_axis, yaws, indexing="ij")
    translations = torch.stack(
        (dx.reshape(-1), dy.reshape(-1), torch.zeros(dx.numel(), dtype=torch.float64)),
        dim=-1,
    )
    return x_axis, y_axis, yaws, translations, yaw.reshape(-1)


def evaluate(kinematics, demo_arm_q, yaws, translations, pivot, batch, arguments, dt):
    """Per-transform feasibility, the reason for each failure, and the measured
    quantities behind both -- so a saved grid can be re-thresholded without
    re-running the solver."""
    total = yaws.shape[0]
    feasible = torch.zeros(total, dtype=torch.bool)
    unreachable = torch.zeros(total, dtype=torch.bool)
    at_limit = torch.zeros(total, dtype=torch.bool)
    too_fast = torch.zeros(total, dtype=torch.bool)
    peak_speed = torch.zeros(total, dtype=torch.float32)
    max_position = torch.zeros(total, dtype=torch.float32)
    max_rotation = torch.zeros(total, dtype=torch.float32)
    min_margin = torch.zeros(total, dtype=torch.float32)
    worst_frame = torch.zeros(total, dtype=torch.int32)
    retained_arm_q = [] if arguments.save_bank else None
    started = time.monotonic()
    for start in range(0, total, batch):
        stop = min(start + batch, total)
        print(
            "  solving transforms {:5d}-{:5d} of {:5d} ({:5.1f}%) ...".format(
                start + 1,
                stop,
                total,
                100.0 * start / max(total, 1),
            ),
            flush=True,
        )
        # Only the demonstrated/approved joint family counts as feasible for
        # training. Reaching the palm through a rejected elbow/wrist flip does
        # not make this transform admissible.
        result = retarget_clip(
            kinematics,
            demo_arm_q,
            yaws[start:stop],
            translations[start:stop],
            pivot,
        )
        position = result.position_residual_m.amax(dim=0).cpu()
        rotation = result.rotation_residual_rad.amax(dim=0).cpu()
        margin = result.limit_margin_rad.amin(dim=0).cpu()
        # Same velocity test the bank applies. A palm path near a wrist
        # singularity solves at every frame and still demands joint speeds the
        # arm does not have, so leaving it out here made the sweep recommend
        # boxes the bank then rejected a fifth of.
        speed_series = (
            (result.arm_q[1:] - result.arm_q[:-1]).abs().amax(dim=-1) / dt
        ).cpu()
        peak_speed_batch = speed_series.amax(dim=0)
        solved = (position <= arguments.position_tolerance_m) & (
            rotation <= arguments.rotation_tolerance_rad
        )
        inside = (margin >= arguments.limit_margin_rad) & (
            peak_speed_batch
            <= ARM_JOINT_VELOCITY_LIMIT_RAD_S * float(arguments.velocity_fraction)
        )
        speed_ok = (
            peak_speed_batch
            <= ARM_JOINT_VELOCITY_LIMIT_RAD_S * float(arguments.velocity_fraction)
        )
        feasible[start:stop] = solved & inside
        unreachable[start:stop] = ~solved
        at_limit[start:stop] = margin < arguments.limit_margin_rad
        too_fast[start:stop] = ~speed_ok
        peak_speed[start:stop] = peak_speed_batch.float()
        max_position[start:stop] = position.float()
        max_rotation[start:stop] = rotation.float()
        min_margin[start:stop] = margin.float()
        # Where in the clip the fastest motion happens, in ORIGINAL frame
        # numbers, so it can be scrubbed to directly in a viewer.
        worst_frame[start:stop] = (
            speed_series.argmax(dim=0).cpu() * arguments.frame_stride
        ).int()
        if retained_arm_q is not None:
            # Retain the result already computed above. Keeping only the six
            # arm joints makes the temporary buffer modest; hand joints and
            # cube tracks are reconstructed once after feasibility filtering.
            retained_arm_q.append(result.arm_q.detach().cpu())
        elapsed = time.monotonic() - started
        completed = stop
        remaining = elapsed * (total - completed) / max(completed, 1)
        print(
            "    done: batch accepted {:5.1f}% | total {:5.1f}% | "
            "elapsed {:6.1f}s | ETA {:6.1f}s".format(
                100.0 * float((solved & inside).float().mean()),
                100.0 * float(feasible[:completed].float().mean()),
                elapsed,
                remaining,
            ),
            flush=True,
        )
    measured = {
        "feasible": feasible,
        "unreachable": unreachable,
        "at_limit": at_limit,
        "too_fast": too_fast,
        "peak_speed_rad_s": peak_speed,
        "max_position_residual_m": max_position,
        "max_rotation_residual_rad": max_rotation,
        "min_limit_margin_rad": min_margin,
        "fastest_frame": worst_frame,
    }
    arm_q = None
    if retained_arm_q is not None:
        arm_q = torch.cat(retained_arm_q, dim=1)
    return measured, arm_q


def save_training_bank(
    path,
    kinematics,
    demonstration,
    yaws,
    translations,
    feasible,
    solved_arm_q,
    lever_arm_m,
):
    """Create the training bank without repeating any palm IK solve."""
    keep = torch.nonzero(feasible, as_tuple=False).reshape(-1)
    if not keep.numel():
        raise RuntimeError("Cannot save a training bank: no grid point is feasible")

    device, dtype = kinematics.device, kinematics.dtype
    yaw = yaws[keep].to(device=device, dtype=dtype)
    translation = translations[keep].to(device=device, dtype=dtype)
    arm_q = solved_arm_q[:, keep, :].to(device=device, dtype=dtype)
    demo_q = demonstration.q.to(device=device, dtype=dtype)
    demo_dq = demonstration.dq.to(device=device, dtype=dtype)
    demo_cube_ur = demonstration.cube_pose.to(device=device, dtype=dtype)
    demo_cube_base = cube_pose_to_base_frame(demo_cube_ur)
    pivot = demo_cube_base[0, :3]

    arm_dq = map_arm_velocities(
        kinematics,
        demo_q[:, :ARM_JOINT_COUNT],
        demo_dq[:, :ARM_JOINT_COUNT],
        arm_q,
        yaw,
    )
    count = int(keep.numel())
    frames = demo_q.shape[0]
    hand_q = demo_q[:, None, ARM_JOINT_COUNT:].expand(frames, count, -1)
    hand_dq = demo_dq[:, None, ARM_JOINT_COUNT:].expand_as(hand_q)
    q = torch.cat((arm_q, hand_q), dim=-1).permute(1, 0, 2).contiguous()
    dq = torch.cat((arm_dq, hand_dq), dim=-1).permute(1, 0, 2).contiguous()

    cube_base, linear, angular = transform_cube_track(
        demo_cube_base,
        demonstration.cube_linear_velocity.to(device=device, dtype=dtype),
        demonstration.cube_angular_velocity.to(device=device, dtype=dtype),
        yaw,
        translation,
        pivot,
    )
    cube_ur = cube_pose_from_base_frame(cube_base).permute(1, 0, 2).contiguous()
    keypoints = reference_keypoints_in_object_frame(
        kinematics, demo_q, demo_cube_base, lever_arm_m
    )
    bank = TransformBank(
        yaw,
        translation,
        q,
        dq,
        cube_ur,
        linear.permute(1, 0, 2).contiguous(),
        angular.permute(1, 0, 2).contiguous(),
        keypoints,
        count / float(feasible.numel()),
    ).to(device="cpu", dtype=torch.float32)
    destination = Path(path).expanduser().resolve()
    bank.save(destination)
    size_mb = sum(
        tensor.numel() * tensor.element_size()
        for tensor in (
            bank.q, bank.dq, bank.cube_pose,
            bank.cube_linear_velocity, bank.cube_angular_velocity,
        )
    ) / 1e6
    print("\ntraining bank written to {}".format(destination))
    print("  feasible trajectories: {} / {}".format(count, feasible.numel()))
    print("  frames per trajectory: {}".format(bank.sample_count))
    print("  tensor storage: {:.0f} MB".format(size_mb))


def failure_reason(unreachable, at_limit, too_fast):
    """A transform can fail several criteria at once, so report them together.

    Collapsing this to one reason hides the common case. Roughly a third of
    rejected transforms are both out of reach *and* too fast, because once the
    IK stops converging the joint trajectory it returns is meaningless and its
    apparent speed is meaningless with it.
    """
    reasons = []
    if unreachable:
        reasons.append("reach")
    if too_fast:
        reasons.append("velocity")
    if at_limit:
        reasons.append("joint_limit")
    return "+".join(reasons) if reasons else "ok"


def write_flat_table(path, translations, yaws, measured):
    """One row per transform, independent flags per criterion."""
    flat = {name: value.reshape(-1) for name, value in measured.items()}
    with Path(path).open("w", encoding="utf-8") as handle:
        handle.write(
            "dx_m,dy_m,yaw_deg,feasible,reason,fails_reach,fails_velocity,"
            "fails_joint_limit,peak_speed_rad_s,max_position_residual_mm,"
            "min_limit_margin_rad,fastest_frame\n"
        )
        for index in range(len(flat["feasible"])):
            unreachable = bool(flat["unreachable"][index])
            at_limit = bool(flat["at_limit"][index])
            too_fast = bool(flat["too_fast"][index])
            handle.write(
                "{:.4f},{:.4f},{:.2f},{:d},{},{:d},{:d},{:d},"
                "{:.4f},{:.4f},{:.4f},{:d}\n".format(
                    float(translations[index, 0]),
                    float(translations[index, 1]),
                    math.degrees(float(yaws[index])),
                    int(bool(flat["feasible"][index])),
                    failure_reason(unreachable, at_limit, too_fast),
                    int(unreachable), int(too_fast), int(at_limit),
                    float(flat["peak_speed_rad_s"][index]),
                    1e3 * float(flat["max_position_residual_m"][index]),
                    float(flat["min_limit_margin_rad"][index]),
                    int(flat["fastest_frame"][index]),
                )
            )


def bar(fraction, width=16):
    filled = int(round(fraction * width))
    return "#" * filled + "." * (width - filled)


def cell(fraction):
    if fraction >= 0.999:
        return "#"
    if fraction >= 0.75:
        return "O"
    if fraction >= 0.5:
        return "o"
    if fraction >= 0.25:
        return "-"
    if fraction > 0.0:
        return "."
    return " "


def largest_box(grid, x_axis, y_axis, yaws, threshold, symmetric_yaw=True):
    """Largest (dx, dy, yaw) box whose acceptance clears ``threshold``.

    X stays symmetric about zero while Y stays symmetric about its configured
    shifted centre. Yaw gets both symmetric and asymmetric treatments because
    its measured envelope is not symmetric.
    """
    centre = len(x_axis) // 2
    yaw_steps = len(yaws)
    best = None
    for half in range(1, centre + 1):
        rows = slice(centre - half, centre + half + 1)
        if symmetric_yaw:
            zero = int(torch.argmin(yaws.abs()))
            reach = min(zero, yaw_steps - 1 - zero)
            spans = [(zero - k, zero + k) for k in range(1, reach + 1)]
        else:
            spans = [
                (low, high)
                for low in range(yaw_steps)
                for high in range(low + 1, yaw_steps)
            ]
        for low, high in spans:
            block = grid[rows, rows, low: high + 1]
            fraction = float(block.float().mean())
            if fraction < threshold:
                continue
            x_half_width = float(x_axis[centre + half])
            y_low = float(y_axis[centre - half])
            y_high = float(y_axis[centre + half])
            span = float(yaws[high] - yaws[low])
            volume = (2.0 * x_half_width) * (y_high - y_low) * span
            if best is None or volume > best[0]:
                best = (
                    volume,
                    x_half_width,
                    y_low,
                    y_high,
                    math.degrees(float(yaws[low])),
                    math.degrees(float(yaws[high])),
                    fraction,
                )
    return best


def report_box(label, best, threshold):
    print("\n{} clearing {:.0f}% acceptance:".format(label, 100 * threshold))
    if best is None:
        print("    none -- loosen the criteria or shrink the requested range")
        return
    _, x_half_width, y_low, y_high, yaw_low, yaw_high, fraction = best
    print("    x translation  +/- {:.3f} m".format(x_half_width))
    print("    y translation  [{:+.3f}, {:+.3f}] m".format(y_low, y_high))
    print("    yaw          [{:+.1f}, {:+.1f}] deg   (span {:.1f})".format(
        yaw_low, yaw_high, yaw_high - yaw_low))
    print("    acceptance    {:.1f}% inside it".format(100.0 * fraction))


def main():
    arguments = parse_arguments()
    if arguments.save_bank and arguments.frame_stride != 1:
        raise ValueError(
            "--save-bank requires --frame-stride 1: training needs all 1108 "
            "reference frames, not a subsampled feasibility trajectory"
        )
    torch.set_num_threads(max(1, torch.get_num_threads()))
    demo_arm_q, pivot, total_frames = load_demonstration(
        arguments.demo, arguments.frame_stride
    )
    kinematics = PalmKinematics(arguments.urdf, device=arguments.device)
    x_axis, y_axis, yaws_axis, translations, yaws = build_grid(
        arguments.translation_m,
        arguments.translation_y_center_m,
        arguments.yaw_min_deg,
        arguments.yaw_max_deg,
        arguments.translation_steps,
        arguments.yaw_steps,
    )

    print("Transform feasibility sweep")
    print("  demonstration : {} frames, every {} -> {}".format(
        total_frames, arguments.frame_stride, demo_arm_q.shape[0]))
    print("  grid          : dx {} x dy {} x yaw {} = {} transforms".format(
        arguments.translation_steps, arguments.translation_steps,
        arguments.yaw_steps, yaws.shape[0]))
    print("  translation   : x [{:+.2f}, {:+.2f}] m, y [{:+.2f}, {:+.2f}] m".format(
        float(x_axis[0]), float(x_axis[-1]), float(y_axis[0]), float(y_axis[-1])))
    print("  yaw           : [{:+.1f}, {:+.1f}] deg".format(
        arguments.yaw_min_deg, arguments.yaw_max_deg))
    print("  accept when   : position <= {:.1f} mm, rotation <= {:.1f} mrad, "
          "joint margin >= {:.2f} rad, speed <= {:.0f}% of limit".format(
              1e3 * arguments.position_tolerance_m,
              1e3 * arguments.rotation_tolerance_rad,
              arguments.limit_margin_rad, 100 * arguments.velocity_fraction))
    print()

    # The sampled clip's own step, so a subsampled sweep is not judged as if
    # its frames were 1/60 s apart.
    dt = arguments.frame_stride / 60.0
    measured, solved_arm_q = evaluate(
        kinematics, demo_arm_q, yaws, translations, pivot, arguments.batch,
        arguments, dt
    )
    feasible = measured["feasible"]
    unreachable = measured["unreachable"]
    at_limit = measured["at_limit"]

    shape = (arguments.translation_steps, arguments.translation_steps, arguments.yaw_steps)
    grid = feasible.reshape(shape)
    print()
    print("overall acceptance: {:.1f}%  ({} / {})".format(
        100.0 * float(feasible.float().mean()), int(feasible.sum()), feasible.numel()))
    print("  unsolvable      : {:.1f}%".format(100.0 * float(unreachable.float().mean())))
    print("  limit or speed  : {:.1f}%".format(100.0 * float(at_limit.float().mean())))

    print("\nacceptance by yaw")
    for index, value in enumerate(yaws_axis):
        fraction = float(grid[:, :, index].float().mean())
        print("  {:+6.1f} deg  {}  {:5.1f}%".format(
            math.degrees(float(value)), bar(fraction), 100.0 * fraction))

    print("\nacceptance over the table plane, marginal over yaw")
    print("  dy is towards the robot at +, away at -; dx is lateral")
    header = "        " + " ".join("{:+5.2f}".format(float(v)) for v in y_axis)
    print(header)
    for row, dx_value in enumerate(x_axis):
        cells = " ".join(
            "  {}  ".format(cell(float(grid[row, column, :].float().mean())))
            for column in range(len(y_axis))
        )
        print("  {:+5.2f} {}".format(float(dx_value), cells))
    print("  legend: # = 100%   O >= 75%   o >= 50%   - >= 25%   . > 0   blank = none")

    symmetric = largest_box(
        grid, x_axis, y_axis, yaws_axis, arguments.acceptance, True
    )
    asymmetric = largest_box(
        grid, x_axis, y_axis, yaws_axis, arguments.acceptance, False
    )
    report_box("largest box with a SYMMETRIC yaw range", symmetric, arguments.acceptance)
    report_box("largest box with an ASYMMETRIC yaw range", asymmetric, arguments.acceptance)
    if symmetric is not None and asymmetric is not None:
        gain = (asymmetric[5] - asymmetric[4]) - (symmetric[5] - symmetric[4])
        if gain > 1.0:
            print("\n  an asymmetric yaw range buys {:.0f} more degrees of span; "
                  "the envelope is not symmetric and forcing it to be\n"
                  "  discards the reachable side.".format(gain))

    if arguments.save:
        destination = Path(arguments.save).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            name: value.reshape(shape).numpy()
            for name, value in measured.items()
        }
        arrays.update(
            translation_x_m=x_axis.numpy(),
            translation_y_m=y_axis.numpy(),
            translation_y_center_m=np.float64(arguments.translation_y_center_m),
            yaw_rad=yaws_axis.numpy(),
            yaw_deg=np.degrees(yaws_axis.numpy()),
            # The thresholds this grid was judged against, so it can be
            # re-thresholded from the recorded measurements without re-solving.
            position_tolerance_m=np.float64(arguments.position_tolerance_m),
            rotation_tolerance_rad=np.float64(arguments.rotation_tolerance_rad),
            limit_margin_rad=np.float64(arguments.limit_margin_rad),
            velocity_limit_rad_s=np.float64(
                ARM_JOINT_VELOCITY_LIMIT_RAD_S * arguments.velocity_fraction
            ),
            frame_stride=np.int64(arguments.frame_stride),
            demonstration=np.str_(str(Path(arguments.demo).name)),
        )
        np.savez_compressed(str(destination), **arrays)
        print("\ngrid written to {}".format(destination))
        print("  indexed [x, y, yaw] -> "
              "{} x {} x {}".format(*shape))

        # A flat table alongside it: one row per transform, trivially loadable
        # from anything, no numpy needed.
        csv_path = destination.with_suffix(".csv")
        write_flat_table(csv_path, translations, yaws, measured)
        print("  flat table written to {}".format(csv_path))

    if arguments.save_bank:
        demonstration = JointDemonstration60Hz.load(
            arguments.demo, device=arguments.device
        )
        save_training_bank(
            arguments.save_bank,
            kinematics,
            demonstration,
            yaws,
            translations,
            feasible,
            solved_arm_q,
            arguments.lever_arm_m,
        )


if __name__ == "__main__":
    main()
