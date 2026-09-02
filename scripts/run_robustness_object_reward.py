#!/usr/bin/env python3
"""Run scratch robustness replicas followed by object-reward experiments."""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROBUSTNESS_SEEDS = (42, 43, 44)
DEFAULT_OBJECT_SCALES = (
    Decimal("1.0"),
    Decimal("0.8"),
    Decimal("0.6"),
    Decimal("0.4"),
)
BASE_OBJECT_POSITION_WEIGHT = Decimal("0.8")
BASE_OBJECT_ORIENTATION_WEIGHT = Decimal("0.2")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robustness-seeds",
        type=int,
        nargs="+",
        default=DEFAULT_ROBUSTNESS_SEEDS,
        help="Seeds for no-object-reward replicas (default: 42 43 44).",
    )
    parser.add_argument(
        "--object-scales",
        type=Decimal,
        nargs="+",
        default=DEFAULT_OBJECT_SCALES,
        help="Object-pose reward multipliers (default: 1.0 0.8 0.6 0.4).",
    )
    parser.add_argument(
        "--object-seed",
        type=int,
        default=42,
        help="Common seed for all reward-scale comparisons (default: 42).",
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
        help="Fresh parent directory for all seven run directories.",
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
        help="Disable final evaluation in every run.",
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
        help="Apply the same train.py configuration override to every run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print all commands without creating directories or training.",
    )
    return parser.parse_args()


def decimal_text(value):
    return format(value.normalize(), "f")


def scale_label(scale):
    return format(scale, "f").rstrip("0").rstrip(".").replace(".", "p")


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


def build_command(args, run_dir, seed, object_scale):
    # Keep the virtualenv symlink: resolving .venv/bin/python would silently
    # replace it with the system interpreter and lose installed dependencies.
    python = args.python.expanduser().absolute()
    command = [
        str(python),
        str(REPO_ROOT / "scripts/train.py"),
        "--log-dir",
        str(run_dir),
        "--iterations",
        str(args.iterations),
        "--seed",
        str(seed),
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

    if object_scale is None:
        position_weight = Decimal("0")
        orientation_weight = Decimal("0")
    else:
        position_weight = BASE_OBJECT_POSITION_WEIGHT * object_scale
        orientation_weight = BASE_OBJECT_ORIENTATION_WEIGHT * object_scale
    # Experiment-specific values come last, so a common --set cannot
    # accidentally make the baseline and object-reward groups incomparable.
    command.extend(
        (
            "--set",
            "contact.enabled=false",
            "--set",
            "rewards.fingertip_object_distance_weight=0",
            "--set",
            "rewards.object_position_weight={}".format(
                decimal_text(position_weight)
            ),
            "--set",
            "rewards.object_orientation_weight={}".format(
                decimal_text(orientation_weight)
            ),
        )
    )
    return command, position_weight, orientation_weight


def planned_runs(args, output_root):
    plans = []
    for run_number, seed in enumerate(args.robustness_seeds, start=1):
        plans.append(
            {
                "group": "robustness",
                "run": run_number,
                "seed": seed,
                "object_scale": None,
                "run_dir": output_root
                / "robustness"
                / "run_{:02d}_seed_{}".format(run_number, seed),
            }
        )
    for run_number, scale in enumerate(args.object_scales, start=1):
        plans.append(
            {
                "group": "object_reward",
                "run": run_number,
                "seed": args.object_seed,
                "object_scale": scale,
                "run_dir": output_root
                / "object_reward"
                / "run_{:02d}_scale_{}_seed_{}".format(
                    run_number, scale_label(scale), args.object_seed
                ),
            }
        )
    return plans


def run_experiments(args, output_root, manifest):
    all_succeeded = True
    plans = planned_runs(args, output_root)
    manifest_path = output_root / "experiment_sweep.json"
    for overall_number, plan in enumerate(plans, start=1):
        command, position_weight, orientation_weight = build_command(
            args,
            plan["run_dir"],
            plan["seed"],
            plan["object_scale"],
        )
        experiment = {
            "group": plan["group"],
            "run": plan["run"],
            "seed": plan["seed"],
            "object_scale": (
                decimal_text(plan["object_scale"])
                if plan["object_scale"] is not None
                else None
            ),
            "object_position_weight": decimal_text(position_weight),
            "object_orientation_weight": decimal_text(orientation_weight),
            "fingertip_object_distance_weight": "0",
            "contact_reward_enabled": False,
            "run_dir": str(plan["run_dir"]),
            "command": command,
            "initialization": "random",
            "status": "planned" if args.dry_run else "running",
        }
        manifest["runs"].append(experiment)

        description = (
            "baseline seed {}".format(plan["seed"])
            if plan["object_scale"] is None
            else "object reward x{} seed {}".format(
                plan["object_scale"], plan["seed"]
            )
        )
        print("\n[run {}/{}] {}".format(overall_number, len(plans), description))
        print(" ".join(command))
        if args.dry_run:
            continue

        run_dir = plan["run_dir"]
        run_dir.parent.mkdir(parents=True, exist_ok=True)
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
                "{} {}. Continuing with the remaining independent runs.".format(
                    description, experiment["status"]
                ),
                file=sys.stderr,
            )
    return all_succeeded


def main():
    args = parse_args()
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    robustness_seeds = tuple(args.robustness_seeds)
    if not robustness_seeds or len(set(robustness_seeds)) != len(
        robustness_seeds
    ):
        raise ValueError("--robustness-seeds must be non-empty and unique")
    object_scales = tuple(args.object_scales)
    if not object_scales or any(scale <= 0 for scale in object_scales):
        raise ValueError("--object-scales must contain positive values")
    if len(set(object_scales)) != len(object_scales):
        raise ValueError("--object-scales must not contain duplicates")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else REPO_ROOT
        / "logs/simtoolreal"
        / "{}_robustness_object_reward".format(timestamp)
    )
    manifest = {
        "independent_runs": True,
        "iterations_per_run": args.iterations,
        "robustness_seeds": list(robustness_seeds),
        "object_reward_seed": args.object_seed,
        "object_reward_scales": [
            decimal_text(scale) for scale in object_scales
        ],
        "base_object_position_weight": decimal_text(
            BASE_OBJECT_POSITION_WEIGHT
        ),
        "base_object_orientation_weight": decimal_text(
            BASE_OBJECT_ORIENTATION_WEIGHT
        ),
        "output_root": str(output_root),
        "runs": [],
    }

    print("Output root: {}".format(output_root))
    print(
        "Robustness seeds: {}".format(
            " ".join(str(seed) for seed in robustness_seeds)
        )
    )
    print(
        "Object scales (common seed {}): {}".format(
            args.object_seed,
            " ".join(decimal_text(scale) for scale in object_scales),
        )
    )
    if not args.dry_run:
        if output_root.exists():
            raise FileExistsError(
                "output root already exists; choose a fresh --output-root: "
                "{}".format(output_root)
            )
        output_root.mkdir(parents=True)
        write_manifest(output_root / "experiment_sweep.json", manifest)

    all_succeeded = run_experiments(args, output_root, manifest)
    if not args.dry_run:
        write_manifest(output_root / "experiment_sweep.json", manifest)
    if not all_succeeded:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
