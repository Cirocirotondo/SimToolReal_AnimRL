"""Deterministic periodic evaluation for motion-imitation PPO."""

from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from simtoolreal_animrl.envs.rsi import sample_rsi_indices

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

    def __init__(
        self, env, interval, seed, fixed_phases, arm_action_plot_path=None
    ):
        self.env = env
        self.interval = int(interval)
        self.arm_action_plot_path = (
            Path(arm_action_plot_path)
            if arm_action_plot_path is not None
            else None
        )
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
        # A configured phase may point into the post-grasp region, but periodic
        # evaluation must obey the same automatic-reset ceiling as training.
        fixed_base = np.minimum(fixed_base, self.env.rsi_max_start_index)
        self.fixed_indices = torch.as_tensor(
            np.resize(fixed_base, self.env.num_envs),
            dtype=torch.long,
            device=self.env.device,
        )
        generator = torch.Generator(device=self.env.device)
        generator.manual_seed(int(seed))
        bank = getattr(self.env, "transform_bank", None)
        if bank is not None:
            transform_count = int(bank.transform_count)
            # Evenly cover the bank for the fixed suite; use a second seeded
            # cohort for the configured-RSI suite.  Both remain identical at
            # every checkpoint.
            self.fixed_transform_indices = torch.linspace(
                0, transform_count - 1, self.env.num_envs,
                device=self.env.device,
            ).round().long()
            self.uniform_transform_indices = torch.randint(
                0, transform_count, (self.env.num_envs,),
                device=self.env.device, generator=generator,
            )
            # Environment zero is the stable arm-action diagnostic requested
            # for checkpoint-to-checkpoint comparisons.
            identity = (
                bank.translation[:, :2].square().sum(dim=1)
                + (0.1 * bank.yaw_rad).square()
            ).argmin()
            self.fixed_transform_indices[0] = identity
            if self.arm_action_plot_path is not None:
                self.fixed_indices[0] = min(740, self.env.rsi_max_start_index)
        else:
            # Keep small unit-test/stub environments compatible.
            self.fixed_transform_indices = None
            self.uniform_transform_indices = None
        # Keep the historical "uniform" metric prefix for log compatibility,
        # but this cohort now follows the configured training RSI distribution.
        self.uniform_indices = sample_rsi_indices(
            self.env.num_envs,
            self.env.device,
            self.env.rsi_distribution,
            self.env.rsi_max_start_index,
            self.env.rsi_pregrasp_start_index,
            self.env.rsi_early_probability,
            generator=generator,
        )

    def __call__(self, iteration, runner, is_final=False):
        # ``is_final`` marks the last update of the segment, which is always
        # evaluated: a run length that is not a multiple of the interval would
        # otherwise leave its final policy unranked.
        if not is_final and int(iteration) % self.interval != 0:
            return None
        # Evaluation always measures the unassisted policy: with the object
        # assist still on, a checkpoint would be ranked on a task the crutch is
        # partly solving. getattr keeps the stub environments of the unit tests
        # usable.
        pin_assist_scale = getattr(self.env, "set_object_assist_scale", None)
        assist_scale = getattr(self.env, "object_assist_scale", 0.0)
        with preserve_random_state(), torch.inference_mode():
            runner.eval_mode()
            if pin_assist_scale is not None:
                pin_assist_scale(0.0)
            try:
                plotter = None
                if self.arm_action_plot_path is not None:
                    # Import lazily so metric-only evaluation does not import
                    # matplotlib or select a backend.
                    from simtoolreal_animrl.runners.eval_plotter import (
                        EvaluationPlotter,
                    )

                    plotter = EvaluationPlotter(
                        self.arm_action_plot_path.parent, env_idx=0
                    )
                if plotter is None:
                    if self.fixed_transform_indices is None:
                        fixed = self._evaluate_suite(runner, self.fixed_indices)
                    else:
                        fixed = self._evaluate_suite(
                            runner, self.fixed_indices,
                            self.fixed_transform_indices
                        )
                else:
                    fixed = self._evaluate_suite(
                        runner, self.fixed_indices,
                        self.fixed_transform_indices, plotter=plotter
                    )
                if self.uniform_transform_indices is None:
                    uniform = self._evaluate_suite(runner, self.uniform_indices)
                else:
                    uniform = self._evaluate_suite(
                        runner, self.uniform_indices,
                        self.uniform_transform_indices
                    )
            finally:
                runner.train_mode()
                if pin_assist_scale is not None:
                    pin_assist_scale(assist_scale)

        stats = {}
        for suite_name, suite_stats in (("fixed", fixed), ("uniform", uniform)):
            for name, value in suite_stats.items():
                stats["evaluation_{}_{}".format(suite_name, name)] = value
        stats["evaluation_score"] = 0.5 * (
            fixed["position_score"] + uniform["position_score"]
        )
        return stats

    def _reset_to_indices(self, indices, transform_indices=None):
        env_ids = torch.arange(
            self.env.num_envs, device=self.env.device, dtype=torch.long
        )
        if transform_indices is None:
            self.env.reset_idx(env_ids, indices)
        else:
            self.env.reset_idx(env_ids, indices, transform_indices)
        self.env.gym.refresh_dof_state_tensor(self.env.sim)
        self.env.compute_observations()
        return self.env.get_observations()

    def _evaluate_suite(
        self, runner, reference_indices, transform_indices=None, plotter=None
    ):
        observations = self._reset_to_indices(reference_indices, transform_indices)
        plotter_active = plotter is not None
        if plotter_active:
            plotter.start_episode("episode_00", self.env)
        active = torch.ones(
            self.env.num_envs, dtype=torch.bool, device=self.env.device
        )
        episode_steps = torch.zeros(
            self.env.num_envs, dtype=torch.float32, device=self.env.device
        )
        reward_sum = torch.zeros_like(episode_steps)
        position_reward_sum = torch.zeros_like(episode_steps)
        palm_keypoint_reward_sum = torch.zeros_like(episode_steps)
        fingertip_keypoint_reward_sum = torch.zeros_like(episode_steps)
        velocity_reward_sum = torch.zeros_like(episode_steps)
        action_rate_reward_sum = torch.zeros_like(episode_steps)
        hand_position_reward_sum = torch.zeros_like(episode_steps)
        hand_velocity_reward_sum = torch.zeros_like(episode_steps)
        hand_action_rate_reward_sum = torch.zeros_like(episode_steps)
        object_position_reward_sum = torch.zeros_like(episode_steps)
        object_orientation_reward_sum = torch.zeros_like(episode_steps)
        fingertip_object_distance_reward_sum = torch.zeros_like(episode_steps)
        fingertip_object_distance_sum = torch.zeros_like(episode_steps)
        object_position_error_sum = torch.zeros_like(episode_steps)
        object_orientation_error_sum = torch.zeros_like(episode_steps)
        fingertip_contact_reward_sum = torch.zeros_like(episode_steps)
        fingertip_contact_fraction_sum = torch.zeros_like(episode_steps)
        fingertip_contact_force_sum = torch.zeros_like(episode_steps)
        rms_hand_position_error_sum = torch.zeros_like(episode_steps)
        max_hand_position_error = torch.zeros_like(episode_steps)
        rms_position_error_sum = torch.zeros_like(episode_steps)
        rms_action_rate_sum = torch.zeros_like(episode_steps)
        rms_velocity_error_sum = torch.zeros_like(episode_steps)
        max_position_error = torch.zeros_like(episode_steps)
        initial_object_com_height = self.env.cube_position[:, 2].clone()
        peak_object_com_height = initial_object_com_height.clone()
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
                if plotter_active:
                    plotter.record(
                        self.env,
                        int(episode_steps[plotter.env_idx].item()) + 1,
                        actions,
                        rewards,
                        dones,
                        infos,
                    )
                    if bool(dones[plotter.env_idx]):
                        if bool(infos["early_termination"][plotter.env_idx]):
                            reason = "early termination"
                        elif bool(infos["time_outs"][plotter.env_idx]):
                            reason = "timeout"
                        else:
                            reason = "done"
                        plotter.finalize_arm_action(self.arm_action_plot_path)
                        plotter_active = False

                active_float = active.float()
                episode_steps += active_float
                reward_sum += rewards * active_float
                position_reward_sum += infos["position_reward"] * active_float
                palm_keypoint_reward_sum += (
                    infos["palm_keypoint_reward"] * active_float
                )
                fingertip_keypoint_reward_sum += (
                    infos["fingertip_keypoint_reward"] * active_float
                )
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
                fingertip_object_distance_reward_sum += (
                    infos["fingertip_object_distance_reward"] * active_float
                )
                fingertip_object_distance_sum += (
                    infos["fingertip_object_distance_m"] * active_float
                )
                object_position_error_sum += (
                    infos["object_position_error_m"] * active_float
                )
                object_orientation_error_sum += (
                    infos["object_orientation_error_rad"] * active_float
                )
                fingertip_contact_reward_sum += (
                    infos["fingertip_contact_reward"] * active_float
                )
                fingertip_contact_fraction_sum += (
                    infos["fingertip_contact_fraction"] * active_float
                )
                fingertip_contact_force_sum += (
                    infos["mean_fingertip_contact_force_n"] * active_float
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
                peak_object_com_height = torch.where(
                    active,
                    torch.maximum(
                        peak_object_com_height,
                        infos["object_com_height_m"],
                    ),
                    peak_object_com_height,
                )

                completed = dones & active
                early[completed] = infos["early_termination"][completed]
                timeout[completed] = infos["time_outs"][completed]
                active &= ~completed
                if not bool(active.any()):
                    break

        if plotter_active:
            plotter.finalize_arm_action(self.arm_action_plot_path)

        # A horizon should always end every initial episode. Treat anything
        # still active as a failed evaluation rather than silently ignoring it.
        early |= active
        lengths = episode_steps.clamp_min(1.0)
        per_env_palm_keypoint_reward = palm_keypoint_reward_sum / lengths
        per_env_fingertip_keypoint_reward = fingertip_keypoint_reward_sum / lengths
        per_env_position_reward = position_reward_sum / lengths
        per_env_hand_position_reward = hand_position_reward_sum / lengths
        per_env_object_position_reward = object_position_reward_sum / lengths
        per_env_object_orientation_reward = (
            object_orientation_reward_sum / lengths
        )
        # Zero object weights mean the object terms are switched off, so the
        # weighted mixture would be 0/0. Fall back to a robot-pose-only score
        # instead of letting a NaN reach best-checkpoint selection, where every
        # NaN comparison is False and no checkpoint would ever be saved.
        object_weight_sum = float(
            self.env.cfg.rewards.object_position_weight
            + self.env.cfg.rewards.object_orientation_weight
        )
        object_pose_rewarded = object_weight_sum > 0.0
        if object_pose_rewarded:
            per_env_object_pose_score = (
                float(self.env.cfg.rewards.object_position_weight)
                * per_env_object_position_reward
                + float(self.env.cfg.rewards.object_orientation_weight)
                * per_env_object_orientation_reward
            ) / object_weight_sum
        else:
            per_env_object_pose_score = torch.zeros_like(
                per_env_object_position_reward
            )
        # Score the checkpoint on what the policy is actually asked to do.
        # These were the arm and hand JOINT Gaussians, which now carry weights
        # of 0.06 and 0.05 against the keypoint terms' 0.80 and 0.48 -- so
        # best_model.pt would have been selected by tracking the policy is
        # deliberately not optimising for.
        mean_robot_position_reward = 0.5 * (
            per_env_palm_keypoint_reward.mean()
            + per_env_fingertip_keypoint_reward.mean()
        )
        if object_pose_rewarded:
            mean_pose_score = 0.5 * (
                mean_robot_position_reward + per_env_object_pose_score.mean()
            )
        else:
            mean_pose_score = mean_robot_position_reward
        return {
            "mean_reward": float((reward_sum / lengths).mean()),
            "mean_palm_keypoint_reward": float(
                per_env_palm_keypoint_reward.mean()
            ),
            "mean_fingertip_keypoint_reward": float(
                per_env_fingertip_keypoint_reward.mean()
            ),
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
            "mean_fingertip_object_distance_reward": float(
                (fingertip_object_distance_reward_sum / lengths).mean()
            ),
            "mean_fingertip_object_distance_m": float(
                (fingertip_object_distance_sum / lengths).mean()
            ),
            "mean_object_position_error_m": float(
                (object_position_error_sum / lengths).mean()
            ),
            "mean_object_orientation_error_rad": float(
                (object_orientation_error_sum / lengths).mean()
            ),
            "mean_fingertip_contact_reward": float(
                (fingertip_contact_reward_sum / lengths).mean()
            ),
            "mean_fingertip_contact_fraction": float(
                (fingertip_contact_fraction_sum / lengths).mean()
            ),
            "mean_fingertip_contact_force_n": float(
                (fingertip_contact_force_sum / lengths).mean()
            ),
            "mean_object_pose_score": float(per_env_object_pose_score.mean()),
            "mean_rms_hand_position_error": float(
                (rms_hand_position_error_sum / lengths).mean()
            ),
            "max_abs_hand_position_error": float(max_hand_position_error.max()),
            "max_abs_position_error": float(max_position_error.max()),
            "mean_peak_object_com_height_m": float(
                peak_object_com_height.mean()
            ),
            "max_peak_object_com_height_m": float(
                peak_object_com_height.max()
            ),
            "mean_peak_object_com_lift_m": float(
                (peak_object_com_height - initial_object_com_height).mean()
            ),
            "max_peak_object_com_lift_m": float(
                (peak_object_com_height - initial_object_com_height).max()
            ),
            "mean_episode_length": float(episode_steps.mean()),
            "early_termination_fraction": float(early.float().mean()),
            "timeout_fraction": float(timeout.float().mean()),
            "action_target_clipped_fraction": clipped_target_components
            / float(max(action_components, 1)),
            "mean_abs_action": abs_action_sum / float(max(action_components, 1)),
            "max_abs_action": max_abs_action,
            # Best-checkpoint selection gives equal importance to robot pose
            # (arm/hand position) and object pose (its configured 80/20
            # position/orientation mixture); with the object weights zeroed the
            # robot pose carries the score alone, on the same 0-1 scale rather
            # than a halved one. Subtracting failure fraction prevents short
            # failed rollouts from looking artificially good.
            "position_score": float(
                mean_pose_score - early.float().mean()
            ),
        }


# PhysX sizes its GPU buffers from these two fields, not from the environment
# count, so the 64-environment evaluator would otherwise allocate the whole
# budget the trainer asked for. It runs as a second process beside the live
# training context, and measuring one evaluation showed the card go from 8.4 GB
# to 14.3 GB for the few seconds both were up -- the spike that has been killing
# runs at 500-iteration boundaries.
EVALUATION_MAX_GPU_CONTACT_PAIRS = 1024 * 1024


def clamp_evaluation_physx(physx_cfg, max_contact_pairs=None):
    """Shrink the evaluation process's PhysX GPU budget, never grow it.

    Returns the value it settled on. Clamping rather than assigning keeps a run
    that deliberately configured something smaller from being pushed upward.
    """
    limit = (
        EVALUATION_MAX_GPU_CONTACT_PAIRS
        if max_contact_pairs is None
        else int(max_contact_pairs)
    )
    current = int(physx_cfg.max_gpu_contact_pairs)
    physx_cfg.max_gpu_contact_pairs = min(current, limit)
    return physx_cfg.max_gpu_contact_pairs


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
        # Consecutive failures, so a persistently broken evaluator is visible
        # in the log rather than quietly producing a run with no scores.
        self.consecutive_failures = 0
        self.last_success_iteration = -1
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
                metrics = json.load(metrics_file)
        except (subprocess.CalledProcessError, OSError, ValueError) as error:
            # This process is a second PhysX context on the same card as the
            # training one, which makes it the likelier of the two to be refused
            # GPU memory. Training already holds everything it needs, so a failed
            # evaluation costs this iteration's score rather than the whole run.
            # A missing or truncated metrics file lands here too: an evaluator
            # killed mid-write leaves exactly that.
            self.consecutive_failures += 1
            if self.consecutive_failures >= 3:
                # An unmeasured run is not a cheap loss: dr_asym trained 4800
                # iterations with every evaluation failing, so there was no way
                # to tell whether those iterations helped or hurt. The usual
                # cause is the training process leaving too little GPU memory
                # for the second Isaac Gym process this spawns.
                print(
                    "\n{} EVALUATIONS HAVE FAILED IN A ROW. This run is "
                    "training BLIND -- no unassisted measurement since "
                    "iteration {}. Check GPU memory: reduce "
                    "sim.physx.max_gpu_contact_pairs or the environment "
                    "count.\n".format(
                        self.consecutive_failures, self.last_success_iteration
                    ),
                    file=sys.stderr,
                    flush=True,
                )
            print(
                "Evaluation at iteration {} failed ({}: {}). Training continues; "
                "{} consecutive evaluation(s) have failed.".format(
                    int(iteration),
                    type(error).__name__,
                    error,
                    self.consecutive_failures,
                ),
                file=sys.stderr,
                flush=True,
            )
            return None
        else:
            self.consecutive_failures = 0
            self.last_success_iteration = iteration
            return metrics
        finally:
            checkpoint.unlink(missing_ok=True)
            output.unlink(missing_ok=True)
