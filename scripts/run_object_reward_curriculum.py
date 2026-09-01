#!/usr/bin/env python3
"""Run sequential object-reward curricula from a good policy and from scratch."""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from simtoolreal_animrl.cfg import SimToolRealCfg


DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "logs/simtoolreal/2026-08-28_173516_no_object_reward/model_7500.pt"
)
DEFAULT_SCALES = tuple(Decimal(index) / Decimal(10) for index in range(1, 11))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Good no-object policy used to initialize the warm-start chain.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Fresh parent directory for every stage of both chains.",
    )
    parser.add_argument(
        "--iterations-per-stage",
        type=int,
        default=1000,
        help="PPO updates at each reward scale (default: 1000).",
    )
    parser.add_argument(
        "--scales",
        type=Decimal,
        nargs="+",
        default=DEFAULT_SCALES,
        help="Ordered object-reward multipliers (default: 0.1 through 1.0).",
    )
    parser.add_argument(
        "--branches",
        choices=("warm", "scratch"),
        nargs="+",
        default=("warm", "scratch"),
        help="Curriculum chains to execute, in the given order.",
    )
    parser.add_argument(
        "--warm-start-mode",
        choices=("policy", "full"),
        default="policy",
        help=(
            "How the first warm stage loads --checkpoint: 'policy' keeps only "
            "the actor and normalizers; 'full' resumes actor, critic, optimizer, "
            "normalizers, and counters (default: policy)."
        ),
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument(
        "--no-periodic-eval",
        action="store_true",
        help="Disable periodic evaluation in every stage.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print all commands without creating directories or training.",
    )
    return parser.parse_args()


def scale_label(scale):
    return format(scale, "f").rstrip("0").rstrip(".").replace(".", "p")


def decimal_text(value):
    return format(value.normalize(), "f")


def numeric_checkpoints(run_dir):
    checkpoints = []
    for path in run_dir.glob("model_*.pt"):
        try:
            iteration = int(path.stem.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        checkpoints.append((iteration, path))
    return sorted(checkpoints)


def write_manifest(path, manifest):
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_command(args, run_dir, weights, checkpoint, initialize):
    # Preserve a virtualenv symlink: Path.resolve() would turn, for example,
    # ``.venv/bin/python`` into the system interpreter and lose the environment.
    python = args.python.expanduser().absolute()
    command = [
        str(python),
        str(REPO_ROOT / "scripts/train.py"),
        "--log-dir",
        str(run_dir),
        "--iterations",
        str(args.iterations_per_stage),
        "--sim-device",
        args.sim_device,
        "--set",
        "table.surface_below_robot_base_m=0.035",
        "--set",
        "contact.enabled=false",
        "--set",
        "rewards.object_position_weight={}".format(weights["position"]),
        "--set",
        "rewards.object_orientation_weight={}".format(weights["orientation"]),
        "--set",
        "rewards.fingertip_object_distance_weight={}".format(
            weights["proximity"]
        ),
    ]
    if checkpoint is not None:
        command.extend(
            ["--initialize-from" if initialize else "--resume", str(checkpoint)]
        )
    if args.num_envs is not None:
        command.extend(["--num-envs", str(args.num_envs)])
    if args.seed is not None:
        command.extend(["--seed", str(args.seed)])
    if args.save_interval is not None:
        command.extend(["--save-interval", str(args.save_interval)])
    if args.no_periodic_eval:
        command.append("--no-periodic-eval")
    return command


def run_branch(args, output_root, branch, scales, base_weights, manifest):
    previous_checkpoint = (
        args.checkpoint.expanduser().resolve() if branch == "warm" else None
    )
    branch_failed = False
    for stage_number, scale in enumerate(scales, start=1):
        weights = {
            name: decimal_text(value * scale)
            for name, value in base_weights.items()
        }
        run_dir = output_root / branch / "stage_{:02d}_scale_{}".format(
            stage_number, scale_label(scale)
        )
        initialize = (
            branch == "warm"
            and stage_number == 1
            and args.warm_start_mode == "policy"
        )
        command = build_command(
            args, run_dir, weights, previous_checkpoint, initialize
        )
        stage = {
            "branch": branch,
            "stage": stage_number,
            "scale": decimal_text(scale),
            "weights": weights,
            "initialization": (
                "policy_and_normalizers"
                if initialize
                else "full_resume" if previous_checkpoint else "random"
            ),
            "input_checkpoint": (
                str(previous_checkpoint) if previous_checkpoint else None
            ),
            "run_dir": str(run_dir),
            "command": command,
            "status": "planned" if args.dry_run else "running",
        }
        manifest["stages"].append(stage)

        print("\n[{} {}/{}] object reward x{}".format(
            branch, stage_number, len(scales), scale
        ))
        print(" ".join(command))
        if args.dry_run:
            # Use the path that a real preceding stage would eventually emit
            # only as a readable placeholder for subsequent dry-run commands.
            previous_checkpoint = run_dir / "model_<final_iteration>.pt"
            continue

        run_dir.parent.mkdir(parents=True, exist_ok=True)
        if run_dir.exists():
            stage["status"] = "failed"
            stage["error"] = "run directory already exists"
            branch_failed = True
            write_manifest(output_root / "curriculum.json", manifest)
            break

        write_manifest(output_root / "curriculum.json", manifest)
        completed = subprocess.run(command)
        if completed.returncode != 0:
            stage["status"] = "failed"
            stage["returncode"] = completed.returncode
            branch_failed = True
        elif (run_dir / "diverged_model.pt").is_file():
            stage["status"] = "diverged"
            branch_failed = True
        else:
            checkpoints = numeric_checkpoints(run_dir)
            if checkpoints:
                previous_checkpoint = checkpoints[-1][1].resolve()
                stage["status"] = "completed"
                stage["output_checkpoint"] = str(previous_checkpoint)
            else:
                stage["status"] = "failed"
                stage["error"] = "training produced no numeric checkpoint"
                branch_failed = True
        write_manifest(output_root / "curriculum.json", manifest)
        if branch_failed:
            print(
                "Stopping the {} chain: a failed/diverged stage must not feed "
                "the next reward scale.".format(branch),
                file=sys.stderr,
            )
            break
    return not branch_failed


def main():
    args = parse_args()
    if args.iterations_per_stage <= 0:
        raise ValueError("--iterations-per-stage must be positive")
    scales = tuple(args.scales)
    if not scales or any(scale <= 0 for scale in scales):
        raise ValueError("--scales must contain only positive values")
    if any(current <= previous for previous, current in zip(scales, scales[1:])):
        raise ValueError("--scales must be strictly increasing")
    if "warm" in args.branches and not args.checkpoint.expanduser().is_file():
        raise FileNotFoundError(args.checkpoint.expanduser())

    cfg = SimToolRealCfg()
    base_weights = {
        "position": Decimal(str(cfg.rewards.object_position_weight)),
        "orientation": Decimal(str(cfg.rewards.object_orientation_weight)),
        "proximity": Decimal(
            str(cfg.rewards.fingertip_object_distance_weight)
        ),
    }
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else REPO_ROOT
        / "logs/simtoolreal"
        / "{}_object_reward_curriculum".format(timestamp)
    )
    manifest = {
        "base_object_reward_weights": {
            name: decimal_text(value) for name, value in base_weights.items()
        },
        "contact_reward_enabled": False,
        "iterations_per_stage": args.iterations_per_stage,
        "output_root": str(output_root),
        "stages": [],
    }

    print("Output root: {}".format(output_root))
    print("Base object weights: {}".format(manifest["base_object_reward_weights"]))
    if not args.dry_run:
        if output_root.exists():
            raise FileExistsError(
                "output root already exists; choose a fresh --output-root: {}".format(
                    output_root
                )
            )
        output_root.mkdir(parents=True)
        write_manifest(output_root / "curriculum.json", manifest)

    all_succeeded = True
    for branch in args.branches:
        succeeded = run_branch(
            args, output_root, branch, scales, base_weights, manifest
        )
        all_succeeded = all_succeeded and succeeded
    if not args.dry_run:
        write_manifest(output_root / "curriculum.json", manifest)
    if not all_succeeded:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
