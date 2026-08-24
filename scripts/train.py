#!/usr/bin/env python3
"""Train the UR5e + DG5F motion-imitation policy with AnimRL PPO."""

import argparse
import json
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
                "iterations".format(
                    train_cfg.runner.evaluation_num_envs,
                    train_cfg.runner.evaluation_interval,
                )
            )
        runner.learn(
            num_iterations=num_iterations,
            start_iteration=start_iteration,
            checkpoint_dir=run_dir,
            save_interval=save_interval,
            log_interval=args.log_interval,
            metrics_path=run_dir / "metrics.jsonl",
            evaluation_callback=evaluator,
        )
    finally:
        if runner is not None:
            runner.close()
        env.close()


if __name__ == "__main__":
    main()
