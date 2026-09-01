#!/usr/bin/env python3
"""Run independent object-reward hyperparameter experiments."""

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
        help="Checkpoint used independently by every warm-start experiment.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Fresh parent directory for all independent experiments.",
    )
    parser.add_argument(
        "--iterations-per-run",
        "--iterations-per-stage",
        dest="iterations_per_run",
        type=int,
        default=1000,
        help=(
            "PPO updates in each independent experiment (default: 1000). "
            "--iterations-per-stage is retained as a compatibility alias."
        ),
    )
    parser.add_argument(
        "--scales",
        type=Decimal,
        nargs="+",
        default=DEFAULT_SCALES,
        help="Object-reward multipliers to test (default: 0.1 through 1.0).",
    )
    parser.add_argument(
        "--initializations",
        "--branches",
        dest="initializations",
        choices=("warm", "scratch"),
        nargs="+",
        default=("warm", "scratch"),
        help=(
            "Independent initialization variants to test. --branches is "
            "retained as a compatibility alias."
        ),
    )
    parser.add_argument(
        "--warm-start-mode",
        choices=("policy", "full"),
        default="policy",
        help=(
            "How every warm experiment loads --checkpoint: 'policy' keeps only "
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
        help="Disable periodic evaluation in every experiment.",
    )
    video_group = parser.add_mutually_exclusive_group()
    video_group.add_argument(
        "--record-video",
        dest="record_video",
        action="store_true",
        default=None,
        help="Record one real training environment in every sweep run.",
    )
    video_group.add_argument(
        "--no-record-video",
        dest="record_video",
        action="store_false",
        help="Force every experiment to run without a graphics context.",
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
        str(args.iterations_per_run),
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
    if args.record_video is not None:
        command.append(
            "--record-video" if args.record_video else "--no-record-video"
        )
    return command


def run_initialization(
    args, output_root, initialization, scales, base_weights, manifest
):
    source_checkpoint = (
        args.checkpoint.expanduser().resolve()
        if initialization == "warm"
        else None
    )
    all_succeeded = True
    for run_number, scale in enumerate(scales, start=1):
        weights = {
            name: decimal_text(value * scale)
            for name, value in base_weights.items()
        }
        run_dir = output_root / initialization / "run_{:02d}_scale_{}".format(
            run_number, scale_label(scale)
        )
        initialize = (
            initialization == "warm"
            and args.warm_start_mode == "policy"
        )
        command = build_command(
            args, run_dir, weights, source_checkpoint, initialize
        )
        experiment = {
            "initialization": initialization,
            "run": run_number,
            "scale": decimal_text(scale),
            "weights": weights,
            "checkpoint_loading": (
                "policy_and_normalizers"
                if initialize
                else "full_resume" if source_checkpoint else "random"
            ),
            "input_checkpoint": (
                str(source_checkpoint) if source_checkpoint else None
            ),
            "run_dir": str(run_dir),
            "command": command,
            "status": "planned" if args.dry_run else "running",
        }
        manifest["experiments"].append(experiment)

        print("\n[{} {}/{}] independent object reward x{}".format(
            initialization, run_number, len(scales), scale
        ))
        print(" ".join(command))
        if args.dry_run:
            continue

        run_dir.parent.mkdir(parents=True, exist_ok=True)
        if run_dir.exists():
            experiment["status"] = "failed"
            experiment["error"] = "run directory already exists"
            all_succeeded = False
            write_manifest(output_root / "sweep.json", manifest)
            continue

        write_manifest(output_root / "sweep.json", manifest)
        completed = subprocess.run(command)
        if completed.returncode != 0:
            experiment["status"] = "failed"
            experiment["returncode"] = completed.returncode
            all_succeeded = False
        elif (run_dir / "diverged_model.pt").is_file():
            experiment["status"] = "diverged"
            experiment["diverged_checkpoint"] = str(
                (run_dir / "diverged_model.pt").resolve()
            )
            all_succeeded = False
        else:
            checkpoints = numeric_checkpoints(run_dir)
            if checkpoints:
                output_checkpoint = checkpoints[-1][1].resolve()
                experiment["status"] = "completed"
                experiment["output_checkpoint"] = str(output_checkpoint)
            else:
                experiment["status"] = "failed"
                experiment["error"] = "training produced no numeric checkpoint"
                all_succeeded = False
        write_manifest(output_root / "sweep.json", manifest)
        if experiment["status"] != "completed":
            print(
                "The {} x{} experiment {}. Continuing the independent "
                "sweep.".format(
                    initialization, scale, experiment["status"]
                ),
                file=sys.stderr,
            )
    return all_succeeded


def main():
    args = parse_args()
    if args.iterations_per_run <= 0:
        raise ValueError("--iterations-per-run must be positive")
    scales = tuple(args.scales)
    if not scales or any(scale <= 0 for scale in scales):
        raise ValueError("--scales must contain only positive values")
    if len(set(scales)) != len(scales):
        raise ValueError("--scales must not contain duplicates")
    if (
        "warm" in args.initializations
        and not args.checkpoint.expanduser().is_file()
    ):
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
        / "{}_object_reward_sweep".format(timestamp)
    )
    manifest = {
        "base_object_reward_weights": {
            name: decimal_text(value) for name, value in base_weights.items()
        },
        "contact_reward_enabled": False,
        "iterations_per_run": args.iterations_per_run,
        "independent_experiments": True,
        "output_root": str(output_root),
        "experiments": [],
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
        write_manifest(output_root / "sweep.json", manifest)

    all_succeeded = True
    for initialization in args.initializations:
        succeeded = run_initialization(
            args,
            output_root,
            initialization,
            scales,
            base_weights,
            manifest,
        )
        all_succeeded = all_succeeded and succeeded
    if not args.dry_run:
        write_manifest(output_root / "sweep.json", manifest)
    if not all_succeeded:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
