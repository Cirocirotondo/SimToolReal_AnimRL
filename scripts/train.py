#!/usr/bin/env python3
"""Train the UR5e + DG5F motion-imitation policy with AnimRL PPO."""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Preserve Isaac Gym's required import-before-torch ordering.
from simtoolreal_animrl.cfg import (
    SimToolRealCfg,
    SimToolRealTrainCfg,
    config_to_dict,
)
from simtoolreal_animrl.envs.motion_imitation import MotionImitationEnv
from simtoolreal_animrl.runners import (
    PPO,
    SubprocessDeterministicEvaluator,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help="Override the AnimRL production value of 4096 environments.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="PPO updates to run in this invocation (default: config value).",
    )
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--log-root",
        type=Path,
        default=REPO_ROOT / "logs",
        help="Root used for newly created runs.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Use an exact run directory instead of generating one.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume optimizer, networks, normalizers, and counters.",
    )
    parser.add_argument(
        "--start-iteration",
        type=int,
        default=None,
        help="Required only for legacy checkpoints without iteration metadata.",
    )
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--eval-num-envs", type=int, default=None)
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument(
        "--no-periodic-eval",
        action="store_true",
        help="Disable the deterministic fixed/uniform RSI evaluation.",
    )
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument(
        "--no-final-eval",
        dest="final_eval",
        action="store_false",
        help=(
            "Skip the headless evaluation with diagnostic plots that runs "
            "once training finishes."
        ),
    )
    parser.add_argument(
        "--final-eval-rsi-index",
        type=int,
        default=0,
        help=(
            "Reference sample the final evaluation starts from (default: 0, "
            "the whole demonstration)."
        ),
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help=(
            "Override one configuration field, e.g. "
            "--set rewards.velocity_std_rad_per_s=0.3. Repeatable. Prefix the "
            "path with 'train.' to address the training config instead of the "
            "environment config. Unknown fields are rejected."
        ),
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Show the environment (training is headless by default).",
    )
    return parser.parse_args()


def resolve_run_directory(args, train_cfg):
    if args.log_dir is not None:
        return args.log_dir.expanduser().resolve()
    if args.resume is not None:
        return args.resume.expanduser().resolve().parent
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return (
        args.log_root.expanduser().resolve()
        / train_cfg.runner.experiment_name
        / "{}_{}".format(timestamp, train_cfg.runner.run_name)
    )


def resolve_start_iteration(args, checkpoint_infos):
    if args.start_iteration is not None:
        if args.start_iteration < 0:
            raise ValueError("--start-iteration cannot be negative")
        return args.start_iteration
    if args.resume is None:
        return 0
    if isinstance(checkpoint_infos, dict) and "next_iteration" in checkpoint_infos:
        return int(checkpoint_infos["next_iteration"])
    raise ValueError(
        "The resume checkpoint has no iteration metadata; pass "
        "--start-iteration explicitly."
    )


def apply_overrides(env_cfg, train_cfg, overrides):
    """Apply --set PATH=VALUE onto the configs, rejecting unknown fields.

    A typo in a sweep must fail loudly rather than silently training the
    default value, so every path component is checked against the config.
    """
    applied = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError("--set expects PATH=VALUE, got {!r}".format(item))
        path, raw = item.split("=", 1)
        path = path.strip()
        if path.startswith("train."):
            node, remainder = train_cfg, path[len("train."):]
        else:
            node, remainder = env_cfg, path
        parts = [part for part in remainder.split(".") if part]
        if not parts:
            raise ValueError("--set has an empty path in {!r}".format(item))
        for part in parts[:-1]:
            if not hasattr(node, part):
                raise KeyError(
                    "Unknown configuration section {!r} in --set {}".format(
                        part, item
                    )
                )
            node = getattr(node, part)
        leaf = parts[-1]
        if not hasattr(node, leaf):
            raise KeyError(
                "Unknown configuration field {!r} in --set {}".format(leaf, item)
            )
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        setattr(node, leaf, value)
        applied[path] = value
    return applied


def resolve_final_checkpoint(run_dir):
    """Prefer the evaluation-selected checkpoint, else the newest one saved."""
    best = run_dir / "best_model.pt"
    if best.is_file():
        return best
    saved = []
    for path in run_dir.glob("model_*.pt"):
        try:
            saved.append((int(path.stem.split("_")[1]), path))
        except (IndexError, ValueError):
            continue
    return max(saved)[1] if saved else None


def run_final_evaluation(run_dir, args):
    """Evaluate the finished policy in a fresh process and write its plots.

    A separate process is required because Isaac Gym allows one simulation per
    process, and it runs only after the training simulation has been released.
    """
    checkpoint = resolve_final_checkpoint(run_dir)
    if checkpoint is None:
        print("Final evaluation skipped: the run saved no checkpoint")
        return
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "evaluate.py"),
        "--checkpoint",
        str(checkpoint),
        "--config",
        str(run_dir / "config.json"),
        "--rsi-index",
        str(args.final_eval_rsi_index),
        "--sim-device",
        args.sim_device,
        "--num-envs",
        "1",
        "--print-every",
        "0",
    ]
    print("\nFinal evaluation of {} (headless, with plots)".format(checkpoint.name))
    completed = subprocess.run(command)
    if completed.returncode != 0:
        # Training already succeeded and its checkpoints are on disk, so a
        # failed evaluation is reported rather than raised.
        print(
            "Final evaluation failed with exit code {}; the run itself is "
            "unaffected.".format(completed.returncode)
        )


def save_configuration(run_dir, env_cfg, train_cfg, args):
    config_path = run_dir / "config.json"
    if config_path.exists():
        return
    snapshot = {
        "env_cfg": config_to_dict(env_cfg),
        "train_cfg": config_to_dict(train_cfg),
        "runtime": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    with config_path.open("w", encoding="utf-8") as config_file:
        json.dump(snapshot, config_file, indent=2, sort_keys=True)


def main():
    args = parse_args()
    env_cfg = SimToolRealCfg()
    train_cfg = SimToolRealTrainCfg()
    if args.seed is not None:
        env_cfg.seed = int(args.seed)
    if args.num_envs is not None:
        if args.num_envs <= 0:
            raise ValueError("--num-envs must be positive")
        env_cfg.env.num_envs = int(args.num_envs)
    if args.run_name is not None:
        train_cfg.runner.run_name = args.run_name
    if args.no_periodic_eval:
        train_cfg.runner.evaluation_enabled = False
    if args.eval_interval is not None:
        if args.eval_interval <= 0:
            raise ValueError("--eval-interval must be positive")
        train_cfg.runner.evaluation_interval = int(args.eval_interval)
    if args.eval_num_envs is not None:
        if args.eval_num_envs <= 0:
            raise ValueError("--eval-num-envs must be positive")
        train_cfg.runner.evaluation_num_envs = int(args.eval_num_envs)
    if args.eval_seed is not None:
        train_cfg.runner.evaluation_seed = int(args.eval_seed)
    # Applied last so an explicit --set always wins over the flags above.
    applied_overrides = apply_overrides(env_cfg, train_cfg, args.overrides)
    for path, value in applied_overrides.items():
        print("Override: {} = {!r}".format(path, value))

    num_iterations = (
        int(args.iterations)
        if args.iterations is not None
        else int(train_cfg.runner.max_iterations)
    )
    save_interval = (
        int(args.save_interval)
        if args.save_interval is not None
        else int(train_cfg.runner.save_interval)
    )
    if num_iterations <= 0:
        raise ValueError("--iterations must be positive")

    run_dir = resolve_run_directory(args, train_cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_configuration(run_dir, env_cfg, train_cfg, args)
    print("Run directory: {}".format(run_dir))

    env = MotionImitationEnv(
        env_cfg,
        sim_device=args.sim_device,
        headless=not args.viewer,
        num_envs_override=None,
    )
    runner = None
    training_completed = False
    try:
        runner = PPO(env, train_cfg, log_dir=run_dir, device=env.device)
        checkpoint_infos = None
        if args.resume is not None:
            resume_path = args.resume.expanduser().resolve()
            print("Loading checkpoint: {}".format(resume_path))
            checkpoint_infos = runner.load(
                resume_path,
                load_optimizer=True,
                load_normalizers=True,
            )
        start_iteration = resolve_start_iteration(args, checkpoint_infos)
        print(
            "Training {} environments for {} PPO updates, starting at {}".format(
                env.num_envs, num_iterations, start_iteration
            )
        )
        evaluator = None
        if bool(train_cfg.runner.evaluation_enabled):
            evaluator = SubprocessDeterministicEvaluator(
                interval=train_cfg.runner.evaluation_interval,
                num_envs=train_cfg.runner.evaluation_num_envs,
                seed=train_cfg.runner.evaluation_seed,
                fixed_phases=train_cfg.runner.evaluation_fixed_phases,
                sim_device=args.sim_device,
                config_path=run_dir / "config.json",
                run_dir=run_dir,
            )
            print(
                "Periodic deterministic evaluation: {} envs every {} "
                "iterations, plus the final update".format(
                    train_cfg.runner.evaluation_num_envs,
                    train_cfg.runner.evaluation_interval,
                )
            )
        history = runner.learn(
            num_iterations=num_iterations,
            start_iteration=start_iteration,
            checkpoint_dir=run_dir,
            save_interval=save_interval,
            log_interval=args.log_interval,
            metrics_path=run_dir / "metrics.jsonl",
            evaluation_callback=evaluator,
        )
        # A run stopped by the divergence guard has no policy worth replaying,
        # and its last checkpoint is diverged_model.pt rather than a final one.
        diverged = bool(history and history[-1].get("divergence_abort"))
        training_completed = not diverged
    finally:
        if runner is not None:
            runner.close()
        env.close()

    # Outside the try block: the training simulation has to be closed before a
    # second one can start, and an interrupted run should not be evaluated.
    if training_completed and args.final_eval:
        run_final_evaluation(run_dir, args)


if __name__ == "__main__":
    main()
