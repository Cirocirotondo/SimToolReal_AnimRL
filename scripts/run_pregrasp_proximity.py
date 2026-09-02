#!/usr/bin/env python3
"""Run one scratch training focused entirely on the pre-grasp RSI window."""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PREGRASP_START_INDEX = 740
PREGRASP_END_INDEX = 830
EPISODE_LENGTH = 200


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--iterations",
        type=int,
        default=12000,
        help="Number of PPO updates (default: 12000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=43,
        help="Training seed (default: 43, the stable robustness seed).",
    )
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--eval-num-envs", type=int, default=None)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Fresh output directory for this run.",
    )
    parser.add_argument(
        "--object-position-weight",
        type=float,
        default=0.8,
        help="Cube-position reward weight (default: 0.8).",
    )
    parser.add_argument(
        "--object-orientation-weight",
        type=float,
        default=0.2,
        help="Cube-orientation reward weight (default: 0.2).",
    )
    parser.add_argument(
        "--proximity-weight",
        type=float,
        default=0.2,
        help="Thumb/index/middle proximity reward weight (default: 0.2).",
    )
    parser.add_argument(
        "--no-periodic-eval",
        action="store_true",
        help="Disable periodic deterministic evaluation.",
    )
    parser.add_argument(
        "--no-final-eval",
        action="store_true",
        help="Disable final deterministic evaluation.",
    )
    video_group = parser.add_mutually_exclusive_group()
    video_group.add_argument(
        "--record-video",
        dest="record_video",
        action="store_true",
        default=None,
        help="Enable periodic training video.",
    )
    video_group.add_argument(
        "--no-record-video",
        dest="record_video",
        action="store_false",
        help="Force the no-graphics headless path.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="Additional train.py override; may be repeated.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generated command without starting Isaac Gym.",
    )
    return parser.parse_args()


def number_text(value):
    return format(float(value), ".12g")


def build_command(args, log_dir):
    # Preserve virtualenv symlinks: resolving .venv/bin/python can select the
    # system interpreter instead of the environment containing Isaac Gym.
    python = args.python.expanduser().absolute()
    command = [
        str(python),
        str(REPO_ROOT / "scripts/train.py"),
        "--log-dir",
        str(log_dir),
        "--iterations",
        str(args.iterations),
        "--seed",
        str(args.seed),
        "--sim-device",
        args.sim_device,
    ]
    if args.num_envs is not None:
        command.extend(("--num-envs", str(args.num_envs)))
    if args.save_interval is not None:
        command.extend(("--save-interval", str(args.save_interval)))
    if args.eval_interval is not None:
        command.extend(("--eval-interval", str(args.eval_interval)))
    if args.eval_num_envs is not None:
        command.extend(("--eval-num-envs", str(args.eval_num_envs)))
    if args.no_periodic_eval:
        command.append("--no-periodic-eval")
    if args.no_final_eval:
        command.append("--no-final-eval")
    if args.record_video is not None:
        command.append(
            "--record-video" if args.record_video else "--no-record-video"
        )
    for override in args.overrides:
        command.extend(("--set", override))

    # Keep the defining settings last so additional common overrides cannot
    # silently turn this into a different RSI or reward experiment.
    defining_overrides = (
        "env.episode_length={}".format(EPISODE_LENGTH),
        "env.reference_init_distribution=pregrasp_mixture",
        "env.rsi_early_probability=0",
        "env.rsi_pregrasp_start_index={}".format(PREGRASP_START_INDEX),
        "env.rsi_max_start_index={}".format(PREGRASP_END_INDEX),
        "rewards.object_position_weight={}".format(
            number_text(args.object_position_weight)
        ),
        "rewards.object_orientation_weight={}".format(
            number_text(args.object_orientation_weight)
        ),
        "rewards.fingertip_object_distance_weight={}".format(
            number_text(args.proximity_weight)
        ),
        "contact.enabled=false",
        "termination.object_position_enabled=true",
    )
    for override in defining_overrides:
        command.extend(("--set", override))
    return command


def validate_args(args):
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    if args.num_envs is not None and args.num_envs <= 0:
        raise ValueError("--num-envs must be positive")
    for name in (
        "object_position_weight",
        "object_orientation_weight",
        "proximity_weight",
    ):
        if getattr(args, name) < 0.0:
            raise ValueError("--{} must be non-negative".format(name.replace("_", "-")))


def main():
    args = parse_args()
    validate_args(args)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    log_dir = (
        args.log_dir.expanduser().resolve()
        if args.log_dir is not None
        else REPO_ROOT
        / "logs/simtoolreal"
        / "{}_pregrasp_proximity_scratch".format(timestamp)
    )
    command = build_command(args, log_dir)
    print(
        "Scratch pre-grasp training: RSI {}..{} (100%), episode length {}".format(
            PREGRASP_START_INDEX, PREGRASP_END_INDEX, EPISODE_LENGTH
        )
    )
    print("Log directory: {}".format(log_dir))
    print(" ".join(command))
    if args.dry_run:
        return
    if log_dir.exists():
        raise FileExistsError(
            "log directory already exists; choose a fresh --log-dir: {}".format(
                log_dir
            )
        )
    completed = subprocess.run(command)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
