#!/usr/bin/env python3
"""Internal isolated evaluator used by periodic PPO evaluation."""

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Preserve Isaac Gym's required import-before-torch ordering.
from simtoolreal_animrl.cfg import (
    SimToolRealCfg,
    SimToolRealTrainCfg,
    update_config_from_dict,
)
from simtoolreal_animrl.envs.motion_imitation import MotionImitationEnv
from simtoolreal_animrl.runners import DeterministicEvaluator, PPO


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--fixed-phases", type=float, nargs="+", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as config_file:
        saved = json.load(config_file)
    env_cfg = SimToolRealCfg()
    train_cfg = SimToolRealTrainCfg()
    update_config_from_dict(env_cfg, saved["env_cfg"])
    update_config_from_dict(train_cfg, saved["train_cfg"])
    env_cfg.seed = int(args.seed)
    env_cfg.env.num_envs = int(args.num_envs)
    env_cfg.env.play = True

    env = MotionImitationEnv(
        env_cfg,
        sim_device=args.sim_device,
        headless=True,
        num_envs_override=None,
    )
    runner = None
    try:
        runner = PPO(env, train_cfg, log_dir=None, device=env.device)
        runner.load(
            args.checkpoint,
            load_optimizer=False,
            load_normalizers=True,
        )
        evaluator = DeterministicEvaluator(
            env,
            interval=1,
            seed=args.seed,
            fixed_phases=args.fixed_phases,
        )
        metrics = evaluator(0, runner)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as output_file:
            json.dump(metrics, output_file, indent=2, sort_keys=True)
    finally:
        if runner is not None:
            runner.close()
        env.close()


if __name__ == "__main__":
    main()
