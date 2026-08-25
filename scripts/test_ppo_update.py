#!/usr/bin/env python3
"""Run one controlled AnimRL-compatible PPO update against Isaac Gym."""

import argparse
import math
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Preserve Isaac Gym's required import-before-torch ordering.
from simtoolreal_animrl.cfg import SimToolRealCfg, SimToolRealTrainCfg
from simtoolreal_animrl.envs.motion_imitation import MotionImitationEnv
from simtoolreal_animrl.runners import PPO

import torch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--sim-device", default="cuda:0")
    return parser.parse_args()


def clone_parameters(module):
    return [parameter.detach().clone() for parameter in module.parameters()]


def parameter_delta(before, module):
    squared_delta = torch.zeros((), device=before[0].device)
    for old, new in zip(before, module.parameters()):
        squared_delta += (new.detach() - old).square().sum()
    return float(torch.sqrt(squared_delta))


def assert_finite_tensor(name, tensor):
    if not bool(torch.isfinite(tensor).all()):
        raise AssertionError("{} contains NaN or infinity".format(name))


def main():
    args = parse_args()
    env_cfg = SimToolRealCfg()
    train_cfg = SimToolRealTrainCfg()
    train_cfg.runner.record_gif = False
    train_cfg.runner.wandb = False

    env = MotionImitationEnv(
        env_cfg,
        sim_device=args.sim_device,
        headless=True,
        num_envs_override=args.num_envs,
    )
    try:
        ppo = PPO(env, train_cfg, device=env.device)
        if ppo.storage.observations.shape != (
            train_cfg.runner.num_steps_per_env,
            env.num_envs,
            79,
        ):
            raise AssertionError("Unexpected rollout observation shape")
        if ppo.storage.actions.shape[-1] != 26:
            raise AssertionError("Unexpected rollout action shape")

        policy_before = clone_parameters(ppo.policy)
        value_before = clone_parameters(ppo.value)
        configured_lr = float(train_cfg.algorithm.learning_rate)

        rollout = ppo.collect_rollout()
        required_rollout_metrics = {
            "mean_reward",
            "mean_position_reward",
            "mean_velocity_reward",
            "mean_action_rate_reward",
            "mean_rms_position_error",
            "mean_rms_velocity_error",
            "mean_rms_action_rate",
            "mean_hand_position_reward",
            "mean_hand_velocity_reward",
            "mean_hand_action_rate_reward",
            "mean_rms_hand_position_error",
            "max_abs_position_error",
            "action_target_clipped_fraction",
            "mean_abs_action",
            "max_abs_action",
            "done_fraction",
            "early_termination_fraction",
            "timeout_fraction",
        }
        if not required_rollout_metrics.issubset(rollout):
            raise AssertionError("PPO rollout diagnostics are incomplete")
        if not 0.0 <= rollout["action_target_clipped_fraction"] <= 1.0:
            raise AssertionError("Target clipping fraction is outside [0, 1]")
        if ppo.storage.step != train_cfg.runner.num_steps_per_env:
            raise AssertionError("Rollout storage is not full")
        for name in (
            "observations",
            "actions",
            "rewards",
            "values",
            "returns",
            "advantages",
            "actions_log_prob",
            "mu",
            "sigma",
        ):
            assert_finite_tensor(name, getattr(ppo.storage, name))
        advantage_mean = float(ppo.storage.advantages.mean())
        advantage_std = float(ppo.storage.advantages.std())
        if abs(advantage_mean) > 1.0e-5 or abs(advantage_std - 1.0) > 1.0e-4:
            raise AssertionError("GAE advantages were not normalized")

        value_loss, surrogate_loss = ppo.update()
        if not math.isfinite(value_loss) or not math.isfinite(surrogate_loss):
            raise AssertionError("PPO returned a non-finite loss")
        if ppo.storage.step != 0:
            raise AssertionError("PPO did not clear rollout storage")
        if any(
            abs(float(group["lr"]) - configured_lr) > 1.0e-12
            for group in ppo.optimizer.param_groups
        ):
            raise AssertionError("Fixed PPO learning rate changed")

        policy_change = parameter_delta(policy_before, ppo.policy)
        value_change = parameter_delta(value_before, ppo.value)
        if policy_change <= 0.0 or value_change <= 0.0:
            raise AssertionError("PPO update did not change both networks")

        raw_observations = env.get_observations().detach().clone()
        inference_policy = ppo.get_inference_policy(device=env.device)
        with torch.inference_mode():
            actions_before_save = inference_policy(raw_observations).clone()

        with tempfile.TemporaryDirectory(prefix="simtoolreal_animrl_ppo_") as tmp:
            checkpoint = Path(tmp) / "model.pt"
            checkpoint_infos = {"milestone": "single_ppo_update"}
            ppo.save(checkpoint, infos=checkpoint_infos)
            try:
                saved = torch.load(
                    str(checkpoint),
                    map_location=env.device,
                    weights_only=False,
                )
            except TypeError:
                saved = torch.load(str(checkpoint), map_location=env.device)
            expected_keys = {
                "policy_dict",
                "value_dict",
                "optimizer_state_dict",
                "actor_obs_normalizer",
                "critic_obs_normalizer",
                "infos",
            }
            if set(saved) != expected_keys:
                raise AssertionError("Checkpoint schema is not AnimRL-compatible")

            reloaded = PPO(env, train_cfg, device=env.device)
            loaded_infos = reloaded.load(
                checkpoint, load_optimizer=True, load_normalizers=True
            )
            if loaded_infos != checkpoint_infos:
                raise AssertionError("Checkpoint infos did not round-trip")
            reloaded_policy = reloaded.get_inference_policy(device=env.device)
            with torch.inference_mode():
                actions_after_load = reloaded_policy(raw_observations)
            if not bool(
                torch.allclose(
                    actions_before_save,
                    actions_after_load,
                    rtol=0.0,
                    atol=1.0e-7,
                )
            ):
                raise AssertionError("Checkpoint reload changed inference actions")

        print("SINGLE PPO UPDATE TEST PASSED")
        print("  environments              : {}".format(env.num_envs))
        print(
            "  rollout shape             : {}".format(
                tuple(ppo.storage.observations.shape)
            )
        )
        print(
            "  actor/value dimensions    : {} -> {} -> {} / 1".format(
                env.num_obs,
                list(train_cfg.policy.actor_hidden_dims),
                env.num_actions,
            )
        )
        print("  learning epochs/minibatch : 5 / 4")
        print("  fixed learning rate       : {:.1e}".format(configured_lr))
        print("  rollout mean reward       : {:.6f}".format(rollout["mean_reward"]))
        print("  rollout dones             : {}".format(rollout["done_count"]))
        print("  value loss                : {:.6f}".format(value_loss))
        print("  surrogate loss            : {:.6f}".format(surrogate_loss))
        print("  advantage mean/std        : {:.3e} / {:.6f}".format(
            advantage_mean, advantage_std
        ))
        print("  policy parameter delta    : {:.6e}".format(policy_change))
        print("  value parameter delta     : {:.6e}".format(value_change))
        print("  checkpoint reload         : deterministic and compatible")
    finally:
        env.close()


if __name__ == "__main__":
    main()
