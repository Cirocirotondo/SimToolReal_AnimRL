#!/usr/bin/env python3
"""Sweep PPO entropy below 0.01 on the scratch pre-grasp proximity task."""

import argparse
import json
import subprocess
import sys
from argparse import Namespace
from datetime import datetime
from decimal import Decimal
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_pregrasp_proximity as pregrasp  # noqa: E402


DEFAULT_SEED = 43
REFERENCE_ENTROPY_COEF = Decimal("0.01")
DEFAULT_ENTROPY_COEFFICIENTS = (
    Decimal("0.005"),
    Decimal("0.002"),
    Decimal("0.001"),
    Decimal("0.0005"),
    Decimal("0.0"),
)
OBJECT_POSITION_WEIGHT = 0.8
OBJECT_ORIENTATION_WEIGHT = 0.2
PROXIMITY_WEIGHT = 0.2


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
        help="Common seed for every independent run (default: 43).",
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
    pregrasp_args = Namespace(
        iterations=args.iterations,
        seed=args.seed,
        sim_device=args.sim_device,
        num_envs=args.num_envs,
        save_interval=args.save_interval,
        eval_interval=args.eval_interval,
        eval_num_envs=args.eval_num_envs,
        python=args.python,
        object_position_weight=OBJECT_POSITION_WEIGHT,
        object_orientation_weight=OBJECT_ORIENTATION_WEIGHT,
        proximity_weight=PROXIMITY_WEIGHT,
        no_periodic_eval=args.no_periodic_eval,
        no_final_eval=args.no_final_eval,
        record_video=args.record_video,
        overrides=list(args.overrides),
        dry_run=args.dry_run,
    )
    command = pregrasp.build_command(pregrasp_args, run_dir)
    # This is the only setting that changes between the five commands.
    command.extend(
        (
            "--set",
            "train.algorithm.entropy_coef={}".format(
                decimal_text(entropy_coefficient)
            ),
        )
    )
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
            "task": "pregrasp_proximity",
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
        / "{}_pregrasp_proximity_entropy_sweep_seed_{}".format(
            timestamp, args.seed
        )
    )
    manifest = {
        "independent_runs": True,
        "initialization": "random",
        "task": "pregrasp_proximity",
        "seed": args.seed,
        "iterations_per_run": args.iterations,
        "reference_run": str(
            REPO_ROOT
            / "logs/simtoolreal/2026-09-02_120236_pregrasp_proximity_scratch"
        ),
        "reference_entropy_coefficient_already_tested": decimal_text(
            REFERENCE_ENTROPY_COEF
        ),
        "entropy_coefficients": [
            decimal_text(value) for value in args.entropy_coefficients
        ],
        "pregrasp_start_index": pregrasp.PREGRASP_START_INDEX,
        "pregrasp_end_index": pregrasp.PREGRASP_END_INDEX,
        "episode_length": pregrasp.EPISODE_LENGTH,
        "object_position_weight": OBJECT_POSITION_WEIGHT,
        "object_orientation_weight": OBJECT_ORIENTATION_WEIGHT,
        "proximity_weight": PROXIMITY_WEIGHT,
        "contact_reward_enabled": False,
        "object_distance_termination_enabled": True,
        "output_root": str(output_root),
        "runs": [],
    }

    print("Scratch pre-grasp proximity entropy sweep")
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
