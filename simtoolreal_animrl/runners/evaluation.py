"""Deterministic periodic evaluation for motion-imitation PPO."""

from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]


@contextmanager
def preserve_random_state():
    """Keep evaluation and evaluation-environment creation out of train RNGs."""
    torch_state = torch.random.get_rng_state()
    cuda_states = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    numpy_state = np.random.get_state()
    try:
        yield
    finally:
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        np.random.set_state(numpy_state)


class DeterministicEvaluator:
    """Evaluate deterministic policy means on repeatable RSI cohorts."""

    def __init__(self, env, interval, seed, fixed_phases):
        self.env = env
        self.interval = int(interval)
        if self.interval <= 0:
            raise ValueError("Evaluation interval must be positive")

        phases = np.asarray(fixed_phases, dtype=np.float64)
        if phases.ndim != 1 or phases.size == 0:
            raise ValueError("Evaluation fixed phases must be a non-empty list")
        if not np.all(np.isfinite(phases)) or np.any(phases < 0.0) or np.any(
            phases > 1.0
        ):
            raise ValueError("Evaluation fixed phases must lie in [0, 1]")

        max_start = int(self.env.reference.last_index - 1)
        fixed_base = np.rint(phases * max_start).astype(np.int64)
        self.fixed_indices = torch.as_tensor(
            np.resize(fixed_base, self.env.num_envs),
            dtype=torch.long,
            device=self.env.device,
        )
        rng = np.random.RandomState(int(seed))
        self.uniform_indices = torch.as_tensor(
            rng.randint(0, max_start + 1, size=self.env.num_envs),
            dtype=torch.long,
            device=self.env.device,
        )

    def __call__(self, iteration, runner, is_final=False):
        # ``is_final`` marks the last update of the segment, which is always
        # evaluated: a run length that is not a multiple of the interval would
        # otherwise leave its final policy unranked.
        if not is_final and int(iteration) % self.interval != 0:
            return None
        with preserve_random_state():
            runner.eval_mode()
            try:
                fixed = self._evaluate_suite(runner, self.fixed_indices)
                uniform = self._evaluate_suite(runner, self.uniform_indices)
            finally:
                runner.train_mode()

        stats = {}
        for suite_name, suite_stats in (("fixed", fixed), ("uniform", uniform)):
            for name, value in suite_stats.items():
                stats["evaluation_{}_{}".format(suite_name, name)] = value
        stats["evaluation_score"] = 0.5 * (
            fixed["position_score"] + uniform["position_score"]
        )
        return stats

    def _reset_to_indices(self, indices):
        env_ids = torch.arange(
            self.env.num_envs, device=self.env.device, dtype=torch.long
        )
        self.env.reset_idx(env_ids, indices)
        self.env.gym.refresh_dof_state_tensor(self.env.sim)
        self.env.compute_observations()
        return self.env.get_observations()

    def _evaluate_suite(self, runner, reference_indices):
        observations = self._reset_to_indices(reference_indices)
        active = torch.ones(
            self.env.num_envs, dtype=torch.bool, device=self.env.device
        )
        episode_steps = torch.zeros(
            self.env.num_envs, dtype=torch.float32, device=self.env.device
        )
        reward_sum = torch.zeros_like(episode_steps)
        position_reward_sum = torch.zeros_like(episode_steps)
        velocity_reward_sum = torch.zeros_like(episode_steps)
        action_rate_reward_sum = torch.zeros_like(episode_steps)
        hand_position_reward_sum = torch.zeros_like(episode_steps)
        hand_velocity_reward_sum = torch.zeros_like(episode_steps)
        hand_action_rate_reward_sum = torch.zeros_like(episode_steps)
        object_position_reward_sum = torch.zeros_like(episode_steps)
        object_orientation_reward_sum = torch.zeros_like(episode_steps)
        object_position_error_sum = torch.zeros_like(episode_steps)
        object_orientation_error_sum = torch.zeros_like(episode_steps)
        rms_hand_position_error_sum = torch.zeros_like(episode_steps)
        max_hand_position_error = torch.zeros_like(episode_steps)
        rms_position_error_sum = torch.zeros_like(episode_steps)
        rms_action_rate_sum = torch.zeros_like(episode_steps)
        rms_velocity_error_sum = torch.zeros_like(episode_steps)
        max_position_error = torch.zeros_like(episode_steps)
        early = torch.zeros_like(active)
        timeout = torch.zeros_like(active)
        clipped_target_components = 0
        action_components = 0
        abs_action_sum = 0.0
        max_abs_action = 0.0

        with torch.inference_mode():
            for _ in range(self.env.max_episode_length):
                normalized = runner.actor_obs_normalizer(observations)
                actions = runner.policy.act_inference(normalized)
                clipped_target_components += int(
                    (
                        (
                            actions.abs() * self.env.action_scales
                            > self.env.action_target_clip
                        )
                        & active[:, None]
                    ).sum()
                )
                action_components += int(active.sum()) * self.env.num_actions
                abs_action_sum += float((actions.abs() * active[:, None]).sum())
                if bool(active.any()):
                    max_abs_action = max(
                        max_abs_action, float(actions[active].abs().max())
                    )
                observations, _, rewards, dones, infos = self.env.step(actions)

                active_float = active.float()
                episode_steps += active_float
                reward_sum += rewards * active_float
                position_reward_sum += infos["position_reward"] * active_float
                velocity_reward_sum += infos["velocity_reward"] * active_float
                action_rate_reward_sum += (
                    infos["action_rate_reward"] * active_float
                )
                rms_action_rate_sum += infos["rms_action_rate"] * active_float
                hand_position_reward_sum += (
                    infos["hand_position_reward"] * active_float
                )
                hand_velocity_reward_sum += (
                    infos["hand_velocity_reward"] * active_float
                )
                hand_action_rate_reward_sum += (
                    infos["hand_action_rate_reward"] * active_float
                )
                object_position_reward_sum += (
                    infos["object_position_reward"] * active_float
                )
                object_orientation_reward_sum += (
                    infos["object_orientation_reward"] * active_float
                )
                object_position_error_sum += (
                    infos["object_position_error_m"] * active_float
                )
                object_orientation_error_sum += (
                    infos["object_orientation_error_rad"] * active_float
                )
                rms_hand_position_error_sum += (
                    infos["rms_hand_position_error"] * active_float
                )
                max_hand_position_error = torch.maximum(
                    max_hand_position_error,
                    infos["max_abs_hand_position_error"] * active_float,
                )
                rms_position_error_sum += (
                    infos["rms_position_error"] * active_float
                )
                rms_velocity_error_sum += (
                    infos["rms_velocity_error"] * active_float
                )
                max_position_error = torch.maximum(
                    max_position_error,
                    infos["max_abs_arm_position_error"] * active_float,
                )

                completed = dones & active
                early[completed] = infos["early_termination"][completed]
                timeout[completed] = infos["time_outs"][completed]
                active &= ~completed
                if not bool(active.any()):
                    break

        # A horizon should always end every initial episode. Treat anything
        # still active as a failed evaluation rather than silently ignoring it.
        early |= active
        lengths = episode_steps.clamp_min(1.0)
        per_env_position_reward = position_reward_sum / lengths
        per_env_hand_position_reward = hand_position_reward_sum / lengths
        per_env_object_position_reward = object_position_reward_sum / lengths
        per_env_object_orientation_reward = (
            object_orientation_reward_sum / lengths
        )
        object_weight_sum = float(
            self.env.cfg.rewards.object_position_weight
            + self.env.cfg.rewards.object_orientation_weight
        )
        per_env_object_pose_score = (
            float(self.env.cfg.rewards.object_position_weight)
            * per_env_object_position_reward
            + float(self.env.cfg.rewards.object_orientation_weight)
            * per_env_object_orientation_reward
        ) / object_weight_sum
        return {
            "mean_reward": float((reward_sum / lengths).mean()),
            "mean_position_reward": float(per_env_position_reward.mean()),
            "mean_velocity_reward": float(
                (velocity_reward_sum / lengths).mean()
            ),
            "mean_action_rate_reward": float(
                (action_rate_reward_sum / lengths).mean()
            ),
            "mean_rms_action_rate": float(
                (rms_action_rate_sum / lengths).mean()
            ),
            "mean_rms_position_error": float(
                (rms_position_error_sum / lengths).mean()
            ),
            "mean_rms_velocity_error": float(
                (rms_velocity_error_sum / lengths).mean()
            ),
            "mean_hand_position_reward": float(
                per_env_hand_position_reward.mean()
            ),
            "mean_hand_velocity_reward": float(
                (hand_velocity_reward_sum / lengths).mean()
            ),
            "mean_hand_action_rate_reward": float(
                (hand_action_rate_reward_sum / lengths).mean()
            ),
            "mean_object_position_reward": float(
                per_env_object_position_reward.mean()
            ),
            "mean_object_orientation_reward": float(
                per_env_object_orientation_reward.mean()
            ),
            "mean_object_position_error_m": float(
                (object_position_error_sum / lengths).mean()
            ),
            "mean_object_orientation_error_rad": float(
                (object_orientation_error_sum / lengths).mean()
            ),
            "mean_object_pose_score": float(per_env_object_pose_score.mean()),
            "mean_rms_hand_position_error": float(
                (rms_hand_position_error_sum / lengths).mean()
            ),
            "max_abs_hand_position_error": float(max_hand_position_error.max()),
            "max_abs_position_error": float(max_position_error.max()),
            "mean_episode_length": float(episode_steps.mean()),
            "early_termination_fraction": float(early.float().mean()),
            "timeout_fraction": float(timeout.float().mean()),
            "action_target_clipped_fraction": clipped_target_components
            / float(max(action_components, 1)),
            "mean_abs_action": abs_action_sum / float(max(action_components, 1)),
            "max_abs_action": max_abs_action,
            # Best-checkpoint selection gives equal importance to robot pose
            # (arm/hand position) and object pose (its configured 80/20
            # position/orientation mixture). Subtracting failure fraction
            # prevents short failed rollouts from looking artificially good.
            "position_score": float(
                0.5
                * (
                    0.5
                    * (
                        per_env_position_reward.mean()
                        + per_env_hand_position_reward.mean()
                    )
                    + per_env_object_pose_score.mean()
                )
                - early.float().mean()
            ),
        }


class SubprocessDeterministicEvaluator:
    """Run evaluation in an isolated process and leave training PhysX intact."""

    def __init__(
        self,
        interval,
        num_envs,
        seed,
        fixed_phases,
        sim_device,
        config_path,
        run_dir,
    ):
        self.interval = int(interval)
        self.num_envs = int(num_envs)
        self.seed = int(seed)
        self.fixed_phases = [float(value) for value in fixed_phases]
        self.sim_device = str(sim_device)
        self.config_path = Path(config_path).resolve()
        self.run_dir = Path(run_dir).resolve()
        if self.interval <= 0 or self.num_envs <= 0:
            raise ValueError(
                "Evaluation interval and environment count must be positive"
            )

    def __call__(self, iteration, runner, is_final=False):
        # ``is_final`` marks the last update of the segment, which is always
        # evaluated: a run length that is not a multiple of the interval would
        # otherwise leave its final policy unranked.
        if not is_final and int(iteration) % self.interval != 0:
            return None
        checkpoint = self.run_dir / ".periodic_evaluation_model.pt"
        output = self.run_dir / ".periodic_evaluation_metrics.json"
        runner.save(checkpoint, infos={"evaluation_iteration": int(iteration)})
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "periodic_evaluate.py"),
            "--checkpoint",
            str(checkpoint),
            "--config",
            str(self.config_path),
            "--output",
            str(output),
            "--num-envs",
            str(self.num_envs),
            "--seed",
            str(self.seed),
            "--sim-device",
            self.sim_device,
            "--fixed-phases",
        ] + [str(value) for value in self.fixed_phases]
        try:
            subprocess.run(command, check=True)
            with output.open("r", encoding="utf-8") as metrics_file:
                return json.load(metrics_file)
        finally:
            checkpoint.unlink(missing_ok=True)
            output.unlink(missing_ok=True)
