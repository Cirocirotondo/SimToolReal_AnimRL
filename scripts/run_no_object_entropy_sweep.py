#!/usr/bin/env python3
"""Sweep PPO entropy below 0.01 on the scratch no-object-reward task."""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED = 42
REFERENCE_ENTROPY_COEF = Decimal("0.01")
DEFAULT_ENTROPY_COEFFICIENTS = (
    Decimal("0.005"),
    Decimal("0.002"),
    Decimal("0.001"),
    Decimal("0.0005"),
    Decimal("0.0"),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--entropy-coefficients",
        type=Decimal,
        nargs="+",
        default=DEFAULT_ENTROPY_COEFFICIENTS,
        help="Descending entropy coefficients below 0.01.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Common seed for every independent run (default: 42).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=12000,
        help="PPO updates per independent run (default: 12000).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Fresh parent directory for the five run directories.",
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--eval-num-envs", type=int, default=None)
    parser.add_argument(
        "--no-periodic-eval",
        action="store_true",
        help="Disable periodic deterministic evaluation in every run.",
    )
    parser.add_argument(
        "--no-final-eval",
        action="store_true",
        help="Disable final deterministic evaluation in every run.",
    )
    video_group = parser.add_mutually_exclusive_group()
    video_group.add_argument(
        "--record-video",
        dest="record_video",
        action="store_true",
        default=None,
        help="Enable periodic training video in every run.",
    )
    video_group.add_argument(
        "--no-record-video",
        dest="record_video",
        action="store_false",
        help="Force every run onto the no-graphics headless path.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="Apply one common train.py override to every run; repeatable.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print all commands without creating directories or training.",
    )
    return parser.parse_args()


def decimal_text(value):
    return format(value.normalize(), "f")


def coefficient_label(value):
    return decimal_text(value).replace("-", "m").replace(".", "p")


def validate_args(args):
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    if args.num_envs is not None and args.num_envs <= 0:
        raise ValueError("--num-envs must be positive")
    coefficients = tuple(args.entropy_coefficients)
    if not coefficients:
        raise ValueError("--entropy-coefficients must not be empty")
    if any(value < 0 or value >= REFERENCE_ENTROPY_COEF for value in coefficients):
        raise ValueError(
            "--entropy-coefficients must be non-negative and below 0.01"
        )
    if len(set(coefficients)) != len(coefficients):
        raise ValueError("--entropy-coefficients must not contain duplicates")
    if any(left <= right for left, right in zip(coefficients, coefficients[1:])):
        raise ValueError("--entropy-coefficients must be strictly descending")


def write_manifest(path, manifest):
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def numeric_checkpoints(run_dir):
    checkpoints = []
    for path in run_dir.glob("model_*.pt"):
        try:
            iteration = int(path.stem.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        checkpoints.append((iteration, path))
    return sorted(checkpoints)


def build_command(args, run_dir, entropy_coefficient):
    # Do not resolve the virtualenv symlink: that can select the system Python.
    python = args.python.expanduser().absolute()
    command = [
        str(python),
        str(REPO_ROOT / "scripts/train.py"),
        "--log-dir",
        str(run_dir),
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

    # These values define the experiment and therefore take precedence over a
    # repeated common --set supplied by the caller.
    defining_overrides = (
        "contact.enabled=false",
        "rewards.fingertip_object_distance_weight=0",
        "rewards.object_position_weight=0",
        "rewards.object_orientation_weight=0",
        "train.algorithm.entropy_coef={}".format(
            decimal_text(entropy_coefficient)
        ),
    )
    for override in defining_overrides:
        command.extend(("--set", override))
    return command


def planned_runs(args, output_root):
    return [
        {
            "run": run_number,
            "entropy_coefficient": coefficient,
            "run_dir": output_root
            / "run_{:02d}_entropy_{}_seed_{}".format(
                run_number, coefficient_label(coefficient), args.seed
            ),
        }
        for run_number, coefficient in enumerate(
            args.entropy_coefficients, start=1
        )
    ]


def run_sweep(args, output_root, manifest):
    all_succeeded = True
    plans = planned_runs(args, output_root)
    manifest_path = output_root / "entropy_sweep.json"
    for plan in plans:
        coefficient = plan["entropy_coefficient"]
        run_dir = plan["run_dir"]
        command = build_command(args, run_dir, coefficient)
        experiment = {
            "run": plan["run"],
            "seed": args.seed,
            "entropy_coefficient": decimal_text(coefficient),
            "initialization": "random",
            "object_reward_enabled": False,
            "run_dir": str(run_dir),
            "command": command,
            "status": "planned" if args.dry_run else "running",
        }
        manifest["runs"].append(experiment)

        print(
            "\n[run {}/{}] seed {} | entropy_coef={}".format(
                plan["run"], len(plans), args.seed, decimal_text(coefficient)
            )
        )
        print(" ".join(command))
        if args.dry_run:
            continue

        if run_dir.exists():
            experiment["status"] = "failed"
            experiment["error"] = "run directory already exists"
            all_succeeded = False
            write_manifest(manifest_path, manifest)
            continue

        write_manifest(manifest_path, manifest)
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
                experiment["status"] = "completed"
                experiment["output_checkpoint"] = str(
                    checkpoints[-1][1].resolve()
                )
            else:
                experiment["status"] = "failed"
                experiment["error"] = "training produced no numeric checkpoint"
                all_succeeded = False
        write_manifest(manifest_path, manifest)
        if experiment["status"] != "completed":
            print(
                "entropy_coef={} {}. Continuing the independent sweep.".format(
                    decimal_text(coefficient), experiment["status"]
                ),
                file=sys.stderr,
            )
    return all_succeeded


def main():
    args = parse_args()
    validate_args(args)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else REPO_ROOT
        / "logs/simtoolreal"
        / "{}_no_object_entropy_sweep_seed_{}".format(timestamp, args.seed)
    )
    manifest = {
        "independent_runs": True,
        "initialization": "random",
        "object_reward_enabled": False,
        "seed": args.seed,
        "iterations_per_run": args.iterations,
        "reference_entropy_coefficient_already_tested": decimal_text(
            REFERENCE_ENTROPY_COEF
        ),
        "entropy_coefficients": [
            decimal_text(value) for value in args.entropy_coefficients
        ],
        "output_root": str(output_root),
        "runs": [],
    }

    print("Scratch no-object-reward entropy sweep")
    print("Output root: {}".format(output_root))
    print("Fixed seed: {}".format(args.seed))
    print(
        "Entropy coefficients: {}".format(
            " ".join(
                decimal_text(value) for value in args.entropy_coefficients
            )
        )
    )
    if not args.dry_run:
        if output_root.exists():
            raise FileExistsError(
                "output root already exists; choose a fresh --output-root: "
                "{}".format(output_root)
            )
        output_root.mkdir(parents=True)
        write_manifest(output_root / "entropy_sweep.json", manifest)

    all_succeeded = run_sweep(args, output_root, manifest)
    if not args.dry_run:
        write_manifest(output_root / "entropy_sweep.json", manifest)
    if not all_succeeded:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
