"""Minimal AnimRL-compatible PPO with local TensorBoard logging."""

import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from simtoolreal_animrl.runners.modules.normalizer import EmpiricalNormalization
from simtoolreal_animrl.runners.modules.policy import Policy
from simtoolreal_animrl.runners.modules.value import Value
from simtoolreal_animrl.runners.storage.rollout_storage import RolloutStorage


class PPO:
    """AnimRL PPO core with compatible networks and checkpoint schema."""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        self.cfg = train_cfg.runner
        self.alg_cfg = train_cfg.algorithm
        self.policy_cfg = train_cfg.policy
        self.alg_name = train_cfg.algorithm_name
        self.device = torch.device(device)
        self.env = env
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.writer = None
        if self.log_dir is not None and bool(self.cfg.tensorboard):
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(
                log_dir=str(self.log_dir),
                flush_secs=int(self.cfg.tensorboard_flush_secs),
            )

        num_actor_obs = self.env.num_obs
        num_critic_obs = (
            self.env.num_privileged_obs
            if self.env.num_privileged_obs is not None
            else self.env.num_obs
        )
        self.policy = Policy(
            num_obs=num_actor_obs,
            num_actions=self.env.num_actions,
            hidden_dims=self.policy_cfg.actor_hidden_dims,
            activation=self.policy_cfg.activation,
            log_std_init=self.policy_cfg.log_std_init,
            device=self.device,
        ).to(self.device)
        self.value = Value(
            num_obs=num_critic_obs,
            hidden_dims=self.policy_cfg.critic_hidden_dims,
            activation=self.policy_cfg.activation,
            device=self.device,
        ).to(self.device)

        self.normalize_observation = bool(self.cfg.normalize_observation)
        if self.normalize_observation:
            self.actor_obs_normalizer = EmpiricalNormalization(
                shape=num_actor_obs, until=int(1.0e8)
            ).to(self.device)
            self.critic_obs_normalizer = EmpiricalNormalization(
                shape=num_critic_obs, until=int(1.0e8)
            ).to(self.device)
        else:
            self.actor_obs_normalizer = nn.Identity()
            self.critic_obs_normalizer = nn.Identity()

        self.storage = RolloutStorage(
            num_envs=self.env.num_envs,
            num_transitions_per_env=self.cfg.num_steps_per_env,
            num_obs=num_actor_obs,
            num_critic_obs=num_critic_obs,
            num_actions=self.env.num_actions,
            device=self.device,
        )
        self.transition = RolloutStorage.Transition()

        parameters = list(self.policy.parameters()) + list(self.value.parameters())
        self.optimizer = optim.Adam(parameters, lr=self.alg_cfg.learning_rate)
        self.learning_rate = float(self.alg_cfg.learning_rate)
        self.total_timesteps = 0
        self.total_time_s = 0.0
        self.best_evaluation_score = -math.inf
        self.best_evaluation_iteration = -1
        self.divergence_streak = 0
        self.env.reset()

    def collect_rollout(self):
        """Collect one AnimRL rollout and compute normalized GAE returns."""
        if self.storage.step != 0:
            raise RuntimeError("Rollout storage must be empty before collection")
        self.train_mode()

        observations = self.actor_obs_normalizer(self.env.get_observations())
        privileged_observations = self.env.get_privileged_observations()
        if privileged_observations is not None:
            privileged_observations = self.critic_obs_normalizer(
                privileged_observations
            )

        reward_sum = 0.0
        done_count = 0
        timeout_count = 0
        early_termination_count = 0
        episode_count = 0
        episode_sums = {}
        position_reward_sum = 0.0
        velocity_reward_sum = 0.0
        action_rate_reward_sum = 0.0
        hand_position_reward_sum = 0.0
        hand_velocity_reward_sum = 0.0
        hand_action_rate_reward_sum = 0.0
        object_position_reward_sum = 0.0
        object_orientation_reward_sum = 0.0
        object_position_error_sum = 0.0
        object_orientation_error_sum = 0.0
        fingertip_contact_reward_sum = 0.0
        fingertip_contact_fraction_sum = 0.0
        fingertip_contact_force_sum = 0.0
        rms_hand_position_error_sum = 0.0
        rms_position_error_sum = 0.0
        rms_velocity_error_sum = 0.0
        rms_action_rate_sum = 0.0
        max_abs_position_error = 0.0
        clipped_action_target_count = 0
        action_value_count = 0
        abs_action_sum = 0.0
        max_abs_action = 0.0

        with torch.inference_mode():
            for _ in range(int(self.cfg.num_steps_per_env)):
                actor_observations = observations
                critic_observations = (
                    privileged_observations
                    if privileged_observations is not None
                    else actor_observations
                )
                actions, log_prob = self.policy.act_and_log_prob(
                    actor_observations
                )
                clipped_action_target_count += int(
                    (
                        actions.abs() * self.env.action_scales
                        > self.env.action_target_clip
                    ).sum()
                )
                action_value_count += actions.numel()
                abs_action_sum += float(actions.abs().sum())
                max_abs_action = max(max_abs_action, float(actions.abs().max()))
                (
                    next_observations,
                    next_privileged_observations,
                    rewards,
                    dones,
                    infos,
                ) = self.env.step(actions)

                self.process_env_step(
                    actor_observations,
                    actions,
                    log_prob,
                    critic_observations,
                    rewards,
                    dones,
                    infos,
                    bootstrap=bool(self.alg_cfg.bootstrap),
                )

                observations = self.actor_obs_normalizer(next_observations)
                privileged_observations = next_privileged_observations
                if privileged_observations is not None:
                    privileged_observations = self.critic_obs_normalizer(
                        privileged_observations
                    )

                reward_sum += float(rewards.mean())
                position_reward_sum += float(infos["position_reward"].mean())
                velocity_reward_sum += float(infos["velocity_reward"].mean())
                action_rate_reward_sum += float(
                    infos["action_rate_reward"].mean()
                )
                rms_action_rate_sum += float(infos["rms_action_rate"].mean())
                hand_position_reward_sum += float(
                    infos["hand_position_reward"].mean()
                )
                hand_velocity_reward_sum += float(
                    infos["hand_velocity_reward"].mean()
                )
                hand_action_rate_reward_sum += float(
                    infos["hand_action_rate_reward"].mean()
                )
                object_position_reward_sum += float(
                    infos["object_position_reward"].mean()
                )
                object_orientation_reward_sum += float(
                    infos["object_orientation_reward"].mean()
                )
                object_position_error_sum += float(
                    infos["object_position_error_m"].mean()
                )
                object_orientation_error_sum += float(
                    infos["object_orientation_error_rad"].mean()
                )
                fingertip_contact_reward_sum += float(
                    infos["fingertip_contact_reward"].mean()
                )
                fingertip_contact_fraction_sum += float(
                    infos["fingertip_contact_fraction"].mean()
                )
                fingertip_contact_force_sum += float(
                    infos["mean_fingertip_contact_force_n"].mean()
                )
                rms_hand_position_error_sum += float(
                    infos["rms_hand_position_error"].mean()
                )
                rms_position_error_sum += float(
                    infos["rms_position_error"].mean()
                )
                rms_velocity_error_sum += float(
                    infos["rms_velocity_error"].mean()
                )
                max_abs_position_error = max(
                    max_abs_position_error,
                    float(infos["max_abs_arm_position_error"].max()),
                )
                done_count += int(dones.sum())
                timeout_count += int(infos["time_outs"].sum())
                early_termination_count += int(
                    infos["early_termination"].sum()
                )
                if "episode" in infos:
                    episode = infos["episode"]
                    completed = int(episode["completed_episodes"])
                    episode_count += completed
                    for name, value in episode.items():
                        if name == "completed_episodes":
                            continue
                        episode_sums[name] = episode_sums.get(name, 0.0) + (
                            float(value) * completed
                        )

            # This deliberately follows AnimRL's existing rollout convention:
            # the critic observation retained by the final transition supplies
            # the bootstrap value passed to RolloutStorage.compute_returns().
            last_values = self.value(critic_observations).detach()
            self.storage.compute_returns(
                last_values, self.alg_cfg.gamma, self.alg_cfg.lam
            )

        rollout_steps = float(self.cfg.num_steps_per_env)
        transition_count = int(self.cfg.num_steps_per_env * self.env.num_envs)
        result = {
            "mean_reward": reward_sum / rollout_steps,
            "mean_position_reward": position_reward_sum / rollout_steps,
            "mean_velocity_reward": velocity_reward_sum / rollout_steps,
            "mean_action_rate_reward": action_rate_reward_sum / rollout_steps,
            "mean_hand_position_reward": hand_position_reward_sum / rollout_steps,
            "mean_hand_velocity_reward": hand_velocity_reward_sum / rollout_steps,
            "mean_hand_action_rate_reward": (
                hand_action_rate_reward_sum / rollout_steps
            ),
            "mean_object_position_reward": (
                object_position_reward_sum / rollout_steps
            ),
            "mean_object_orientation_reward": (
                object_orientation_reward_sum / rollout_steps
            ),
            "mean_object_position_error_m": (
                object_position_error_sum / rollout_steps
            ),
            "mean_object_orientation_error_rad": (
                object_orientation_error_sum / rollout_steps
            ),
            "mean_fingertip_contact_reward": (
                fingertip_contact_reward_sum / rollout_steps
            ),
            "mean_fingertip_contact_fraction": (
                fingertip_contact_fraction_sum / rollout_steps
            ),
            "mean_fingertip_contact_force_n": (
                fingertip_contact_force_sum / rollout_steps
            ),
            "mean_rms_hand_position_error": (
                rms_hand_position_error_sum / rollout_steps
            ),
            "mean_rms_position_error": rms_position_error_sum / rollout_steps,
            "mean_rms_velocity_error": rms_velocity_error_sum / rollout_steps,
            "mean_rms_action_rate": rms_action_rate_sum / rollout_steps,
            "max_abs_position_error": max_abs_position_error,
            "action_target_clipped_fraction": (
                clipped_action_target_count / float(max(action_value_count, 1))
            ),
            "mean_abs_action": abs_action_sum / float(max(action_value_count, 1)),
            "max_abs_action": max_abs_action,
            "done_count": done_count,
            "timeout_count": timeout_count,
            "early_termination_count": early_termination_count,
            "done_fraction": done_count / float(transition_count),
            "timeout_fraction": timeout_count / float(transition_count),
            "early_termination_fraction": (
                early_termination_count / float(transition_count)
            ),
            "episode_count": episode_count,
        }
        if episode_count > 0:
            for name, total in episode_sums.items():
                result["episode_{}".format(name)] = total / episode_count
        return result

    def process_env_step(
        self,
        observations,
        actions,
        log_prob,
        critic_observations,
        rewards,
        dones,
        infos,
        bootstrap=True,
    ):
        self.transition.observations = observations.detach()
        self.transition.critic_observations = critic_observations.detach()
        self.transition.actions = actions.detach()
        self.transition.actions_log_prob = log_prob.detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        self.transition.values = self.value(critic_observations).detach()

        if bootstrap:
            self.transition.rewards += self.alg_cfg.gamma * torch.squeeze(
                self.transition.values
                * infos["time_outs"].unsqueeze(1).to(self.device),
                1,
            )
        self.storage.add_transitions(self.transition)
        self.transition.clear()

    def update(self):
        """Run the configured PPO epochs/minibatches and clear the rollout."""
        if self.storage.step != self.storage.num_transitions_per_env:
            raise RuntimeError("PPO update requires a complete rollout")

        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        generator = self.storage.mini_batch_generator(
            self.alg_cfg.num_mini_batches,
            self.alg_cfg.num_learning_epochs,
        )
        num_updates = (
            self.alg_cfg.num_learning_epochs * self.alg_cfg.num_mini_batches
        )

        for sample in generator:
            (
                observations,
                critic_observations,
                actions,
                target_values,
                advantages,
                returns,
                old_actions_log_prob,
                old_mu,
                old_sigma,
            ) = sample

            self.policy.act_and_log_prob(observations)
            actions_log_prob = self.policy.distribution.log_prob(actions)
            values = self.value(critic_observations)
            mu = self.policy.action_mean
            sigma = self.policy.action_std
            entropy = self.policy.entropy

            if (
                self.alg_cfg.desired_kl is not None
                and self.alg_cfg.schedule == "adaptive"
            ):
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma / old_sigma + 1.0e-5)
                        + (
                            torch.square(old_sigma)
                            + torch.square(old_mu - mu)
                        )
                        / (2.0 * torch.square(sigma))
                        - 0.5,
                        dim=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if kl_mean > self.alg_cfg.desired_kl * 2.0:
                        self.learning_rate = max(
                            1.0e-5, self.learning_rate / 1.5
                        )
                    elif self.alg_cfg.desired_kl / 2.0 > kl_mean > 0.0:
                        self.learning_rate = min(
                            1.0e-2, self.learning_rate * 1.5
                        )
                    for group in self.optimizer.param_groups:
                        group["lr"] = self.learning_rate

            ratio = torch.exp(
                actions_log_prob - torch.squeeze(old_actions_log_prob)
            )
            surrogate = -torch.squeeze(advantages) * ratio
            surrogate_clipped = -torch.squeeze(advantages) * torch.clamp(
                ratio,
                1.0 - self.alg_cfg.clip_param,
                1.0 + self.alg_cfg.clip_param,
            )
            surrogate_loss = torch.max(
                surrogate, surrogate_clipped
            ).mean()

            if self.alg_cfg.use_clipped_value_loss:
                value_clipped = target_values + (values - target_values).clamp(
                    -self.alg_cfg.clip_param, self.alg_cfg.clip_param
                )
                value_losses = (values - returns).pow(2)
                value_losses_clipped = (value_clipped - returns).pow(2)
                value_loss = torch.max(
                    value_losses, value_losses_clipped
                ).mean()
            else:
                value_loss = (returns - values).pow(2).mean()

            loss = (
                self.alg_cfg.surrogate_coef * surrogate_loss
                + self.alg_cfg.value_loss_coef * value_loss
                - self.alg_cfg.entropy_coef * entropy.mean()
            )
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.policy.parameters()) + list(self.value.parameters()),
                self.alg_cfg.max_grad_norm,
            )
            self.optimizer.step()

            mean_value_loss += float(value_loss.item())
            mean_surrogate_loss += float(surrogate_loss.item())

        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        self.storage.clear()
        return mean_value_loss, mean_surrogate_loss

    def learn(
        self,
        num_iterations,
        start_iteration=0,
        checkpoint_dir=None,
        save_interval=None,
        log_interval=1,
        metrics_path=None,
        evaluation_callback=None,
    ):
        """Run a finite training segment and write AnimRL-compatible models.

        ``num_iterations`` is the number of PPO updates performed by this
        invocation. ``start_iteration`` is the global iteration number used
        for logs and checkpoint names, which makes resumed runs explicit.
        """
        num_iterations = int(num_iterations)
        start_iteration = int(start_iteration)
        log_interval = int(log_interval)
        if num_iterations <= 0:
            raise ValueError("num_iterations must be positive")
        if start_iteration < 0:
            raise ValueError("start_iteration cannot be negative")
        if log_interval <= 0:
            raise ValueError("log_interval must be positive")

        checkpoint_dir = (
            Path(checkpoint_dir) if checkpoint_dir is not None else None
        )
        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if save_interval is None:
            save_interval = int(self.cfg.save_interval)
        save_interval = int(save_interval)
        if save_interval <= 0:
            raise ValueError("save_interval must be positive")
        metrics_path = Path(metrics_path) if metrics_path is not None else None
        if metrics_path is not None:
            metrics_path.parent.mkdir(parents=True, exist_ok=True)

        history = []
        end_iteration = start_iteration + num_iterations
        for iteration in range(start_iteration, end_iteration):
            iteration_start = time.perf_counter()
            collection_start = iteration_start
            rollout = self.collect_rollout()
            collection_time = time.perf_counter() - collection_start

            learning_start = time.perf_counter()
            value_loss, surrogate_loss = self.update()
            learning_time = time.perf_counter() - learning_start
            iteration_time = time.perf_counter() - iteration_start

            iteration_timesteps = int(
                self.cfg.num_steps_per_env * self.env.num_envs
            )
            self.total_timesteps += iteration_timesteps
            self.total_time_s += iteration_time
            stats = dict(rollout)
            stats.update(
                {
                    "iteration": iteration,
                    "next_iteration": iteration + 1,
                    "value_loss": float(value_loss),
                    "surrogate_loss": float(surrogate_loss),
                    "mean_action_std": float(self.policy.action_std.mean()),
                    "learning_rate": float(self.learning_rate),
                    "collection_time_s": collection_time,
                    "learning_time_s": learning_time,
                    "iteration_time_s": iteration_time,
                    "total_timesteps": self.total_timesteps,
                    "total_time_s": self.total_time_s,
                    "fps": iteration_timesteps / max(iteration_time, 1.0e-12),
                }
            )
            abort_reason = self._divergence_reason(stats)
            if abort_reason is not None:
                stats["divergence_abort"] = 1.0

            if evaluation_callback is not None:
                evaluation_start = time.perf_counter()
                # An aborting iteration counts as final so the run still gets
                # its last evaluation and can keep the best checkpoint.
                evaluation_stats = evaluation_callback(
                    iteration,
                    self,
                    is_final=(
                        iteration == end_iteration - 1
                        or abort_reason is not None
                    ),
                )
                evaluation_time = time.perf_counter() - evaluation_start
                if evaluation_stats is not None:
                    stats.update(evaluation_stats)
                    stats["evaluation_time_s"] = evaluation_time
                    score = float(stats["evaluation_score"])
                    is_best = score > self.best_evaluation_score
                    if is_best:
                        self.best_evaluation_score = score
                        self.best_evaluation_iteration = iteration
                    stats["evaluation_best_score"] = float(
                        self.best_evaluation_score
                    )
                    stats["evaluation_is_best"] = float(is_best)
                    if is_best and checkpoint_dir is not None:
                        self.save(
                            checkpoint_dir / "best_model.pt",
                            infos=self._checkpoint_infos(stats),
                        )
            history.append(stats)

            if metrics_path is not None:
                with metrics_path.open("a", encoding="utf-8") as metrics_file:
                    metrics_file.write(json.dumps(stats, sort_keys=True) + "\n")
            self._write_tensorboard(stats)
            if iteration % log_interval == 0:
                self._print_iteration(stats, end_iteration)
            if checkpoint_dir is not None and iteration % save_interval == 0:
                self.save(
                    checkpoint_dir / "model_{}.pt".format(iteration),
                    infos=self._checkpoint_infos(stats),
                )

            # Placed last so the triggering iteration is fully written to the
            # history, metrics file and tensorboard before the run stops.
            if abort_reason is not None:
                if checkpoint_dir is not None:
                    self.save(
                        checkpoint_dir / "diverged_model.pt",
                        infos=self._checkpoint_infos(stats),
                    )
                print(
                    "Aborting training at iteration {}: {}. Best evaluation "
                    "score {:.4f} at iteration {}.".format(
                        iteration,
                        abort_reason,
                        self.best_evaluation_score,
                        self.best_evaluation_iteration,
                    )
                )
                return history

        if checkpoint_dir is not None:
            final_stats = history[-1]
            self.save(
                checkpoint_dir / "model_{}.pt".format(end_iteration),
                infos=self._checkpoint_infos(final_stats),
            )
        return history

    def _write_tensorboard(self, stats):
        if self.writer is None:
            return
        iteration = int(stats["iteration"])
        tags = {
            "Train/mean_step_reward": "mean_reward",
            "Reward/position": "mean_position_reward",
            "Reward/velocity": "mean_velocity_reward",
            "Reward/action_rate": "mean_action_rate_reward",
            "Reward/hand_position": "mean_hand_position_reward",
            "Reward/hand_velocity": "mean_hand_velocity_reward",
            "Reward/hand_action_rate": "mean_hand_action_rate_reward",
            "Reward/object_position": "mean_object_position_reward",
            "Reward/object_orientation": "mean_object_orientation_reward",
            "Reward/fingertip_contact": "mean_fingertip_contact_reward",
            "Contact/fingertip_fraction": "mean_fingertip_contact_fraction",
            "Contact/mean_fingertip_force_n": (
                "mean_fingertip_contact_force_n"
            ),
            "Tracking/object_position_error_m": (
                "mean_object_position_error_m"
            ),
            "Tracking/object_orientation_error_rad": (
                "mean_object_orientation_error_rad"
            ),
            "Tracking/rms_hand_position_error": "mean_rms_hand_position_error",
            "Tracking/rms_arm_position_error": "mean_rms_position_error",
            "Tracking/rms_arm_velocity_error": "mean_rms_velocity_error",
            "Tracking/max_abs_arm_position_error": "max_abs_position_error",
            "Policy/rms_action_rate": "mean_rms_action_rate",
            "Policy/mean_action_std": "mean_action_std",
            "Policy/action_target_clipped_fraction": (
                "action_target_clipped_fraction"
            ),
            "Policy/mean_abs_action": "mean_abs_action",
            "Policy/max_abs_action": "max_abs_action",
            "Loss/value": "value_loss",
            "Loss/surrogate": "surrogate_loss",
            "Termination/done_fraction": "done_fraction",
            "Termination/early_fraction": "early_termination_fraction",
            "Termination/timeout_fraction": "timeout_fraction",
            "Perf/fps": "fps",
            "Perf/collection_time_s": "collection_time_s",
            "Perf/learning_time_s": "learning_time_s",
            "Perf/iteration_time_s": "iteration_time_s",
            "Perf/total_timesteps": "total_timesteps",
            "Learn/learning_rate": "learning_rate",
            # This counts the episodes that ended during the rollout, so the
            # explicit name reads as what it measures.
            "Episode/count_terminations": "episode_count",
        }
        for tag, key in tags.items():
            self.writer.add_scalar(tag, stats[key], iteration)
        for key, value in stats.items():
            if key == "episode_count":
                continue  # Already logged above under its explicit tag.
            if key.startswith("episode_"):
                self.writer.add_scalar(
                    "Episode/{}".format(key[len("episode_"):]),
                    value,
                    iteration,
                )
            elif key.startswith("evaluation_"):
                self.writer.add_scalar(
                    "Evaluation/{}".format(key[len("evaluation_"):]),
                    value,
                    iteration,
                )

    def _divergence_reason(self, stats):
        """Name the divergence this iteration shows, or None if it is healthy.

        Checked before the evaluation callback runs so an aborting run can
        still request its final evaluation and keep the best checkpoint.
        """
        if not bool(getattr(self.cfg, "abort_on_divergence", False)):
            return None

        reasons = []
        # NaN first: a NaN action std would silently pass every ``>`` test
        # below, so without this the guard would not fire on the worst case.
        for name in ("mean_action_std", "value_loss", "surrogate_loss",
                     "mean_reward"):
            value = stats.get(name)
            if value is not None and not math.isfinite(float(value)):
                reasons.append("{} is not finite ({})".format(name, value))

        threshold = float(self.cfg.abort_action_std)
        action_std = float(stats["mean_action_std"])
        if action_std > threshold:
            reasons.append(
                "mean_action_std {:.4g} > {:.4g}".format(action_std, threshold)
            )

        threshold = float(self.cfg.abort_action_target_clipped_fraction)
        clipped = float(stats["action_target_clipped_fraction"])
        if clipped > threshold:
            reasons.append(
                "action_target_clipped_fraction {:.4g} > {:.4g}".format(
                    clipped, threshold
                )
            )

        if not reasons:
            self.divergence_streak = 0
            return None

        self.divergence_streak += 1
        patience = max(1, int(self.cfg.abort_patience))
        if self.divergence_streak < patience:
            return None
        return "{} (for {} consecutive iterations)".format(
            "; ".join(reasons), self.divergence_streak
        )

    def _checkpoint_infos(self, stats):
        infos = {
            "iteration": int(stats["iteration"]),
            "next_iteration": int(stats["next_iteration"]),
            "total_timesteps": int(self.total_timesteps),
            "total_time_s": float(self.total_time_s),
            "best_evaluation_score": float(self.best_evaluation_score),
            "best_evaluation_iteration": int(self.best_evaluation_iteration),
        }
        if self.normalize_observation:
            infos["actor_normalizer_count"] = int(
                self.actor_obs_normalizer.count
            )
            infos["critic_normalizer_count"] = int(
                self.critic_obs_normalizer.count
            )
        return infos

    @staticmethod
    def _print_iteration(stats, end_iteration):
        episode_return = stats.get("episode_return")
        episode_length = stats.get("episode_length")
        episode_text = "n/a"
        if episode_return is not None:
            episode_text = "return={:.4f}, length={:.1f}".format(
                episode_return, episode_length
            )
        print(
            "Iteration {}/{} | reward={:.4f} | {} | "
            "value_loss={:.4f} | surrogate_loss={:.4f} | "
            "std={:.3f} | target_clipped={:.3f} | fps={:.0f} | "
            "dones={} (early={}, timeout={})".format(
                stats["iteration"],
                end_iteration - 1,
                stats["mean_reward"],
                episode_text,
                stats["value_loss"],
                stats["surrogate_loss"],
                stats["mean_action_std"],
                stats["action_target_clipped_fraction"],
                stats["fps"],
                stats["done_count"],
                stats["early_termination_count"],
                stats["timeout_count"],
            )
        )
        if "evaluation_score" in stats:
            print(
                "  Evaluation | score={:.4f} | arm_pos(fixed/sampled)="
                "{:.4f}/{:.4f} | hand_pos(fixed/sampled)={:.4f}/{:.4f} | "
                "object_pos(fixed/sampled)={:.4f}/{:.4f} | "
                "object_rot(fixed/sampled)={:.4f}/{:.4f} | "
                "early(fixed/sampled)={:.3f}/{:.3f}{}".format(
                    stats["evaluation_score"],
                    stats["evaluation_fixed_mean_position_reward"],
                    stats["evaluation_uniform_mean_position_reward"],
                    stats["evaluation_fixed_mean_hand_position_reward"],
                    stats["evaluation_uniform_mean_hand_position_reward"],
                    stats["evaluation_fixed_mean_object_position_reward"],
                    stats["evaluation_uniform_mean_object_position_reward"],
                    stats["evaluation_fixed_mean_object_orientation_reward"],
                    stats["evaluation_uniform_mean_object_orientation_reward"],
                    stats["evaluation_fixed_early_termination_fraction"],
                    stats["evaluation_uniform_early_termination_fraction"],
                    " | new best" if stats["evaluation_is_best"] else "",
                )
            )

    def save(self, path, infos=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_dict = {
            "policy_dict": self.policy.state_dict(),
            "value_dict": self.value.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "infos": infos,
        }
        if self.normalize_observation:
            save_dict["actor_obs_normalizer"] = (
                self.actor_obs_normalizer.state_dict()
            )
            save_dict["critic_obs_normalizer"] = (
                self.critic_obs_normalizer.state_dict()
            )
        torch.save(save_dict, str(path))

    def load(self, path, load_optimizer=False, load_normalizers=True):
        try:
            loaded = torch.load(
                str(path), map_location=self.device, weights_only=False
            )
        except TypeError:
            # PyTorch versions predating ``weights_only`` remain supported.
            loaded = torch.load(str(path), map_location=self.device)
        self.policy.load_state_dict(loaded["policy_dict"])
        self.value.load_state_dict(loaded["value_dict"])
        if load_optimizer:
            self.optimizer.load_state_dict(loaded["optimizer_state_dict"])
        if load_normalizers and self.normalize_observation:
            self.actor_obs_normalizer.load_state_dict(
                loaded["actor_obs_normalizer"]
            )
            self.critic_obs_normalizer.load_state_dict(
                loaded["critic_obs_normalizer"]
            )
        infos = loaded["infos"]
        if isinstance(infos, dict):
            self.total_timesteps = int(infos.get("total_timesteps", 0))
            self.total_time_s = float(infos.get("total_time_s", 0.0))
            self.best_evaluation_score = float(
                infos.get("best_evaluation_score", -math.inf)
            )
            self.best_evaluation_iteration = int(
                infos.get("best_evaluation_iteration", -1)
            )
            if load_normalizers and self.normalize_observation:
                self.actor_obs_normalizer.count = int(
                    infos.get("actor_normalizer_count", 0)
                )
                self.critic_obs_normalizer.count = int(
                    infos.get("critic_normalizer_count", 0)
                )
        return infos

    def get_inference_policy(self, device=None):
        self.eval_mode()
        if device is not None:
            self.policy.to(device)
            self.actor_obs_normalizer.to(device)

        def inference_policy(observations):
            return self.policy.act_inference(
                self.actor_obs_normalizer(observations)
            )

        return inference_policy

    def train_mode(self):
        self.policy.train()
        self.value.train()
        if self.normalize_observation:
            self.actor_obs_normalizer.train()
            self.critic_obs_normalizer.train()

    def eval_mode(self):
        self.policy.eval()
        self.value.eval()
        if self.normalize_observation:
            self.actor_obs_normalizer.eval()
            self.critic_obs_normalizer.eval()

    def close(self):
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None
