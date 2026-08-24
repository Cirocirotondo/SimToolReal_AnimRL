#!/usr/bin/env python3
"""End-to-end headless environment milestone test (no PPO)."""

import argparse
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Import the environment only after its package path is installed. It preserves
# Isaac Gym's required import-before-torch ordering internally.
from simtoolreal_animrl.cfg import SimToolRealCfg
from simtoolreal_animrl.envs.controller import (
    ARM_JOINT_NAMES,
    HAND_PD_DAMPING,
    HAND_PD_STIFFNESS,
)
from simtoolreal_animrl.envs.motion_imitation import MotionImitationEnv


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help="Override AnimRL's production value of 4096 for a quick smoke test.",
    )
    parser.add_argument(
        "--rsi-index",
        type=int,
        default=None,
        help="Use one explicit valid RSI index for every env; default is uniform RSI.",
    )
    parser.add_argument(
        "--sim-device", default="cuda:0", help="Isaac Gym simulation device."
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU PhysX and the CPU tensor pipeline.",
    )
    return parser.parse_args()


def assert_correct_pd_gains(env):
    gains = env.pd_gain_summary()
    np.testing.assert_allclose(
        gains["hand_stiffness"], np.asarray(HAND_PD_STIFFNESS), rtol=0.0, atol=1e-5
    )
    np.testing.assert_allclose(
        gains["hand_damping"], np.asarray(HAND_PD_DAMPING), rtol=0.0, atol=1e-5
    )


def assert_reset_matches_reference(env, reference_index):
    env.gym.refresh_dof_state_tensor(env.sim)
    reference = env.reference.sample(env.reference_index)
    position_error = (env.q - reference.q).abs().max().item()
    velocity_error = (env.dq - reference.dq).abs().max().item()
    if position_error > 1e-6 or velocity_error > 1e-5:
        raise AssertionError(
            "RSI state write failed: max q error={:.3e}, max dq error={:.3e}".format(
                position_error, velocity_error
            )
        )
    if reference_index is not None:
        if not bool((env.reference_index == int(reference_index)).all()):
            raise AssertionError("The requested RSI index was not applied")
    return position_error, velocity_error


def assert_observation_contract(env):
    import torch

    obs = env.get_observations()
    if obs.shape != (env.num_envs, 19):
        raise AssertionError("Expected 19D observations, got {}".format(obs.shape))

    expected_phase = (
        env.reference_index.float() / float(env.reference.last_index)
    ).unsqueeze(1)
    expected = torch.cat(
        (
            env.unscale_arm_positions(env.arm_q),
            env.previous_arm_targets,
            env.arm_dq,
            expected_phase,
        ),
        dim=1,
    )
    if not bool(torch.allclose(obs, expected, rtol=0.0, atol=1e-6)):
        raise AssertionError("The 19D observation blocks are inconsistent")

    if not bool(((obs[:, :6] >= -1.0) & (obs[:, :6] <= 1.0)).all()):
        raise AssertionError("Normalized arm positions escaped [-1, 1]")


def assert_arm_only_objective_contract(env):
    import torch

    metrics = env._compute_reward_and_errors()
    if metrics["q_error"].shape != (env.num_envs, 6):
        raise AssertionError("Position reward is not restricted to the arm")
    if metrics["dq_error"].shape != (env.num_envs, 6):
        raise AssertionError("Velocity reward is not restricted to the arm")
    if metrics["hand_q_error"].shape != (env.num_envs, 20):
        raise AssertionError("Hand diagnostics have an unexpected shape")
    expected_reward = (
        float(env.cfg.rewards.position_weight) * metrics["position_reward"]
        + float(env.cfg.rewards.velocity_weight) * metrics["velocity_reward"]
    )
    if not bool(torch.allclose(env.rew_buf, expected_reward, rtol=0.0, atol=1e-7)):
        raise AssertionError("Reward contains a non-arm contribution")


def assert_ppo_step_contract(env, obs, critic_obs, rewards, dones, extras):
    import torch

    if obs.shape != (env.num_envs, env.num_obs):
        raise AssertionError("PPO observation shape mismatch")
    if critic_obs is not None:
        raise AssertionError("This environment must not expose privileged observations")
    if rewards.shape != (env.num_envs,) or rewards.dtype != torch.float32:
        raise AssertionError("PPO rewards must be float32 with shape (num_envs,)")
    if dones.shape != (env.num_envs,) or dones.dtype != torch.bool:
        raise AssertionError("PPO dones must be bool with shape (num_envs,)")

    required = (
        "time_outs",
        "horizon_time_outs",
        "reference_end",
        "early_termination",
    )
    for name in required:
        values = extras.get(name)
        if values is None or values.shape != (env.num_envs,):
            raise AssertionError("Missing or malformed PPO info {!r}".format(name))
        if values.dtype != torch.bool or values.device != env.device:
            raise AssertionError("PPO info {!r} has wrong dtype/device".format(name))

    if not bool(
        torch.equal(
            extras["time_outs"],
            extras["horizon_time_outs"] | extras["reference_end"],
        )
    ):
        raise AssertionError("time_outs is not horizon OR reference_end")
    if bool((extras["time_outs"] & extras["early_termination"]).any()):
        raise AssertionError("A termination was both timeout and task failure")

    if bool(dones.any()):
        episode = extras.get("episode")
        if episode is None:
            raise AssertionError("Completed episodes did not produce episode statistics")
        required_episode_keys = (
            "return",
            "length",
            "mean_reward",
            "mean_position_reward",
            "mean_velocity_reward",
            "mean_rms_position_error",
            "mean_rms_velocity_error",
            "early_termination_fraction",
            "horizon_fraction",
            "reference_end_fraction",
            "completed_episodes",
        )
        for name in required_episode_keys:
            value = episode.get(name)
            if value is None or value.ndim != 0 or not bool(torch.isfinite(value)):
                raise AssertionError(
                    "Missing, non-scalar, or non-finite episode statistic {!r}".format(
                        name
                    )
                )
        if int(episode["completed_episodes"]) != int(dones.sum()):
            raise AssertionError("Episode completion count does not match dones")
    elif "episode" in extras:
        raise AssertionError("Episode statistics were emitted without a completed episode")


def run_ideal_episode(env, initial_indices):
    import torch

    peak_position_error = 0.0
    peak_velocity_error = 0.0
    minimum_reward = float("inf")
    mean_rewards = []
    reference_end_count = 0
    horizon_count = 0

    # One initial episode per environment. With Cartwheel-style RSI, episodes
    # near the end of the motion are intentionally shorter than the horizon.
    expected_steps = torch.minimum(
        torch.full_like(initial_indices, env.max_episode_length),
        env.reference.last_index - initial_indices,
    )
    pending = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    observed_steps = torch.zeros_like(initial_indices)

    for step in range(1, env.max_episode_length + 1):
        actions, complete_target = env.next_reference_action()
        if not bool(((actions >= -1.0) & (actions <= 1.0)).all()):
            raise AssertionError("Ideal action escaped normalized [-1, 1] bounds")

        obs, critic_obs, rewards, dones, extras = env.step(actions)
        assert_ppo_step_contract(
            env, obs, critic_obs, rewards, dones, extras
        )
        if not bool(torch_isfinite(rewards)):
            raise AssertionError("Reward contains NaN or infinity")
        if bool((rewards < -1e-7).any()) or bool((rewards > 1.0 + 1e-6).any()):
            raise AssertionError("Gaussian weighted reward is outside [0, 1]")

        peak_position_error = max(
            peak_position_error, float(extras["max_abs_position_error"].max())
        )
        peak_velocity_error = max(
            peak_velocity_error, float(extras["rms_velocity_error"].max())
        )
        minimum_reward = min(minimum_reward, float(rewards.min()))
        mean_rewards.append(float(rewards.mean()))

        still_active = pending & ~dones
        if bool(still_active.any()):
            expected_hand_target = complete_target[
                still_active, len(ARM_JOINT_NAMES):
            ]
            actual_hand_target = env.previous_hand_targets[still_active]
            if not bool(
                torch.allclose(
                    actual_hand_target,
                    expected_hand_target,
                    rtol=0.0,
                    atol=1e-6,
                )
            ):
                raise AssertionError(
                    "Hand position targets do not follow the demonstration"
                )

        early_count = int((extras["early_termination"] & pending).sum())
        if early_count:
            failed = (
                extras["early_termination"] & pending
            ).nonzero(as_tuple=False).flatten()
            details = []
            for env_id in failed[:5].tolist():
                joint_index = int(extras["worst_joint_index"][env_id])
                details.append(
                    "env={} start={} ref={} joint={} arm_err={:.4f} hand_err={:.4f}".format(
                        env_id,
                        int(initial_indices[env_id]),
                        int(extras["reference_index"][env_id]),
                        ARM_JOINT_NAMES[joint_index],
                        float(extras["max_abs_arm_position_error"][env_id]),
                        float(extras["max_abs_hand_position_error"][env_id]),
                    )
                )
            raise AssertionError(
                "{} environments terminated early at ideal-reference step {}: {}".format(
                    early_count, step, "; ".join(details)
                )
            )

        expected_now = pending & (expected_steps == step)
        observed_now = pending & dones
        if not bool(torch.equal(expected_now, observed_now)):
            raise AssertionError(
                "RSI episode termination mismatch at step {}: expected {}, got {}".format(
                    step, int(expected_now.sum()), int(observed_now.sum())
                )
            )
        if bool(observed_now.any()) and not bool(
            extras["time_outs"][observed_now].all()
        ):
            raise AssertionError(
                "Reference-end/horizon termination was not classified as a timeout"
            )
        expected_reference_end = observed_now & (
            initial_indices + step >= env.reference.last_index
        )
        expected_horizon = observed_now & (step >= env.max_episode_length)
        if not bool(
            torch.equal(
                extras["reference_end"] & observed_now,
                expected_reference_end,
            )
        ):
            raise AssertionError("Reference-end classification mismatch")
        if not bool(
            torch.equal(
                extras["horizon_time_outs"] & observed_now,
                expected_horizon,
            )
        ):
            raise AssertionError("Horizon classification mismatch")
        reference_end_count += int(expected_reference_end.sum())
        horizon_count += int(expected_horizon.sum())
        observed_steps[observed_now] = step
        pending &= ~observed_now
        if not bool(pending.any()):
            break

    if bool(pending.any()) or not bool(torch.equal(observed_steps, expected_steps)):
        raise AssertionError("Not every initial RSI episode ended at the expected step")

    return {
        "peak_position_error": peak_position_error,
        "peak_velocity_rms_error": peak_velocity_error,
        "minimum_reward": minimum_reward,
        "mean_reward": float(np.mean(mean_rewards)),
        "minimum_episode_steps": int(observed_steps.min()),
        "maximum_episode_steps": int(observed_steps.max()),
        "reference_end_count": reference_end_count,
        "horizon_count": horizon_count,
    }


def torch_isfinite(values):
    # Local import keeps the script's top-level import order unambiguous for
    # Isaac Gym.
    import torch

    return torch.isfinite(values).all()


def assert_early_termination_logic(env):
    import torch

    env.episode_length_buf.zero_()
    env.reference_index.zero_()
    env.termination_violation_steps.zero_()
    q_error = torch.zeros(
        (env.num_envs, env.num_actions), dtype=torch.float32, device=env.device
    )
    q_error[:, 0] = float(env.cfg.termination.arm_position_threshold_rad) + 0.1

    grace = int(env.cfg.termination.grace_steps)
    for count in range(1, grace + 1):
        done, early, timeout = env._compute_termination(q_error)
        if bool(timeout.any()):
            raise AssertionError("Synthetic early-termination test unexpectedly timed out")
        expected = count >= grace
        if bool(early.all()) != expected or bool(done.all()) != expected:
            raise AssertionError(
                "Early termination grace mismatch at violation step {}".format(count)
            )


def assert_early_termination_step_contract(env):
    """Force a task failure and verify that PPO must not bootstrap it."""
    original_arm_threshold = env.cfg.termination.arm_position_threshold_rad
    grace = int(env.cfg.termination.grace_steps)
    env.reset(reference_index=0)

    try:
        # A negative diagnostic threshold makes every finite tracking error a
        # violation without perturbing the simulator state itself.
        env.cfg.termination.arm_position_threshold_rad = -1.0
        for step in range(1, grace + 1):
            actions, _ = env.next_reference_action()
            obs, critic_obs, rewards, dones, extras = env.step(actions)
            assert_ppo_step_contract(
                env, obs, critic_obs, rewards, dones, extras
            )
            if step < grace and bool(dones.any()):
                raise AssertionError("Early termination ignored its grace period")

        if not bool(dones.all()):
            raise AssertionError("Forced task failures did not terminate")
        if bool(extras["time_outs"].any()):
            raise AssertionError("Task failures were incorrectly marked for bootstrap")
        if not bool(extras["early_termination"].all()):
            raise AssertionError("Task failures lack the early-termination flag")
        episode = extras["episode"]
        if abs(float(episode["length"]) - grace) > 1e-6:
            raise AssertionError("Early-termination episode length is wrong")
        if abs(float(episode["early_termination_fraction"]) - 1.0) > 1e-6:
            raise AssertionError("Early-termination episode statistics are wrong")
    finally:
        env.cfg.termination.arm_position_threshold_rad = original_arm_threshold
        env.reset(reference_index=0)


def main():
    args = parse_args()
    cfg = SimToolRealCfg()
    sim_device = args.sim_device
    if args.cpu:
        sim_device = "cpu"
        cfg.sim.use_gpu_pipeline = False
        cfg.sim.physx.use_gpu = False

    env = MotionImitationEnv(
        cfg,
        sim_device=sim_device,
        headless=True,
        num_envs_override=args.num_envs,
    )
    try:
        if args.rsi_index is not None:
            env.reset(reference_index=args.rsi_index)

        initial_indices = env.reference_index.clone()
        reset_q_error, reset_dq_error = assert_reset_matches_reference(
            env, args.rsi_index
        )
        assert_observation_contract(env)
        assert_arm_only_objective_contract(env)
        assert_correct_pd_gains(env)
        metrics = run_ideal_episode(env, initial_indices)
        assert_observation_contract(env)
        assert_early_termination_logic(env)
        assert_early_termination_step_contract(env)

        print("HEADLESS ENVIRONMENT TEST PASSED")
        print("  environments              : {}".format(env.num_envs))
        print("  episode length            : {}".format(env.max_episode_length))
        print("  demonstration samples     : {}".format(env.reference.sample_count))
        print("  measured frequency [Hz]   : {:.6f}".format(env.reference.frequency_hz))
        print(
            "  initial RSI range         : [{}, {}]".format(
                int(initial_indices.min()), int(initial_indices.max())
            )
        )
        print("  reset max q error [rad]   : {:.3e}".format(reset_q_error))
        print("  reset max dq error [rad/s]: {:.3e}".format(reset_dq_error))
        print(
            "  peak tracking error [rad] : {:.6f}".format(
                metrics["peak_position_error"]
            )
        )
        print(
            "  peak velocity RMS [rad/s] : {:.6f}".format(
                metrics["peak_velocity_rms_error"]
            )
        )
        print("  minimum reward            : {:.6f}".format(metrics["minimum_reward"]))
        print("  mean reward               : {:.6f}".format(metrics["mean_reward"]))
        print(
            "  initial episode steps     : [{}, {}]".format(
                metrics["minimum_episode_steps"],
                metrics["maximum_episode_steps"],
            )
        )
        print(
            "  initial termination types : reference_end={}, horizon={}".format(
                metrics["reference_end_count"], metrics["horizon_count"]
            )
        )
        print("  PD gains                  : verified")
        print("  19D observation contract  : verified")
        print("  reference-driven hand     : verified")
        print("  arm-only reward/termination: verified")
        print("  PPO step/info contract     : verified")
        print("  horizon/reference timeout : verified")
        print("  early termination/no boot.: verified")
    finally:
        env.close()


if __name__ == "__main__":
    main()
