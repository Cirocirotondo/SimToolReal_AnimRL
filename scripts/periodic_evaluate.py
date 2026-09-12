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
from simtoolreal_animrl.runners.evaluation import clamp_evaluation_physx


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--fixed-phases", type=float, nargs="+", required=True)
    parser.add_argument(
        "--object-assist-scale",
        type=float,
        default=0.0,
        help=(
            "Object-assist scale used while evaluating. The default of 0 "
            "measures the unassisted policy, whatever the training schedule "
            "currently applies."
        ),
    )
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
    # This process stands up a second PhysX context beside the live training
    # one, so it takes only the contact-pair budget 64 environments can use.
    clamp_evaluation_physx(env_cfg.sim.physx)
    # Evaluation reports the policy on the nominal robot, for the same reason
    # it runs unassisted: a score that moves with each episode's gain draw and
    # encoder noise cannot rank checkpoints against each other.
    env_cfg.domain_randomization.enabled = False
    if float(args.object_assist_scale) <= 0.0:
        # Not merely scaled to zero: a disabled assist also skips the
        # per-rigid-body force buffers this process would never use.
        env_cfg.object_assist.enabled = False
    else:
        env_cfg.object_assist.schedule = "constant"
        env_cfg.object_assist.initial_scale = float(args.object_assist_scale)

    env = MotionImitationEnv(
        env_cfg,
        sim_device=args.sim_device,
        headless=True,
        num_envs_override=None,
    )
    runner = None
    try:
        runner = PPO(env, train_cfg, log_dir=None, device=env.device)
        checkpoint_infos = runner.load(
            args.checkpoint,
            load_optimizer=False,
            load_normalizers=True,
        )
        evaluation_iteration = int(
            (checkpoint_infos or {}).get("evaluation_iteration", 0)
        )
        arm_action_plot_path = (
            args.checkpoint.parent
            / "eval_arm_actions"
            / "arm_action_per_joint_iter_{:06d}.png".format(
                evaluation_iteration
            )
        )
        evaluator = DeterministicEvaluator(
            env,
            interval=1,
            seed=args.seed,
            fixed_phases=args.fixed_phases,
            arm_action_plot_path=arm_action_plot_path,
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
