#!/usr/bin/env python3
"""Evaluate an AnimRL checkpoint with deterministic mean actions."""

import argparse
import json
import os
import sys
from pathlib import Path


def _configure_isaac_gym_graphics_environment():
    """Prevent ROS/Gazebo libraries from shadowing NVIDIA graphics libs."""
    if os.environ.get("ISAACGYM_PRESERVE_ROS_GRAPHICS_PATH") != "1":
        library_path = os.environ.get("LD_LIBRARY_PATH", "")
        kept_paths = [
            entry
            for entry in library_path.split(":")
            if "/opt/ros/" not in entry and "/gazebo" not in entry.lower()
        ]
        if kept_paths:
            os.environ["LD_LIBRARY_PATH"] = ":".join(kept_paths)
        else:
            os.environ.pop("LD_LIBRARY_PATH", None)
    nvidia_icd = "/usr/share/vulkan/icd.d/nvidia_icd.json"
    if os.path.isfile(nvidia_icd):
        os.environ.setdefault("VK_ICD_FILENAMES", nvidia_icd)
    os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")


_configure_isaac_gym_graphics_environment()

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Preserve Isaac Gym's required import-before-torch ordering.
from simtoolreal_animrl.cfg import (
    SimToolRealCfg,
    SimToolRealTrainCfg,
    update_config_from_dict,
)
from simtoolreal_animrl.envs.motion_imitation import MotionImitationEnv
from simtoolreal_animrl.runners import PPO
from simtoolreal_animrl.runners.eval_plotter import EvaluationPlotter

import torch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Defaults to config.json next to the checkpoint.",
    )
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)
    rsi = parser.add_mutually_exclusive_group()
    rsi.add_argument(
        "--rsi-index",
        type=int,
        default=0,
        help=(
            "Reference sample the episode starts from (default: 0). "
            "Playback always continues to the final sample."
        ),
    )
    rsi.add_argument(
        "--sampled-rsi",
        "--uniform-rsi",
        dest="sampled_rsi",
        action="store_true",
        help=(
            "Use the same seeded configured RSI distribution as training. "
            "--uniform-rsi is retained as a compatibility alias."
        ),
    )
    parser.add_argument(
        "--viewer", action="store_true", help="Open the Isaac Gym viewer."
    )
    parser.add_argument(
        "--no-ghost",
        dest="ghost",
        action="store_false",
        help=(
            "Hide the green reference robot shown beside the policy robot. "
            "The ghost needs --viewer and is never built headless."
        ),
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=30,
        help="Print rollout diagnostics every N steps; 0 disables them.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON output path (default: next to the checkpoint).",
    )
    parser.add_argument(
        "--no-plots",
        dest="plots",
        action="store_false",
        help="Skip the per-episode diagnostic figures.",
    )
    parser.add_argument(
        "--contact-forces",
        action="store_true",
        help=(
            "Force PhysX contact reporting on for this evaluation so the "
            "per-fingertip force figure is produced even when the run was "
            "trained with contact.enabled=false. The contact reward is "
            "zeroed, so the measured return is unaffected."
        ),
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help=(
            "Directory for the diagnostic figures "
            "(default: eval_plots/ next to the checkpoint)."
        ),
    )
    return parser.parse_args()


def load_saved_configuration(config_path):
    env_cfg = SimToolRealCfg()
    train_cfg = SimToolRealTrainCfg()
    if config_path is None:
        return env_cfg, train_cfg
    with config_path.open("r", encoding="utf-8") as config_file:
        saved = json.load(config_file)
    if "env_cfg" not in saved or "train_cfg" not in saved:
        raise ValueError(
            "Configuration must contain env_cfg and train_cfg sections"
        )
    update_config_from_dict(env_cfg, saved["env_cfg"])
    update_config_from_dict(train_cfg, saved["train_cfg"])
    return env_cfg, train_cfg


def scalar(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu())
    return float(value)


def termination_reason(infos, env_idx=0):
    """Name why env `env_idx` stopped, for the diagnostic figure titles."""
    if bool(infos["early_termination"][env_idx]):
        return "early_termination"
    if bool(infos["reference_end"][env_idx]):
        return "reference_end"
    if bool(infos["horizon_time_outs"][env_idx]):
        return "horizon"
    return "done"


def main():
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError("Checkpoint not found: {}".format(checkpoint))
    config_path = (
        args.config.expanduser().resolve()
        if args.config is not None
        else checkpoint.parent / "config.json"
    )
    if not config_path.is_file():
        raise FileNotFoundError(
            "Training configuration not found: {}. Pass --config explicitly."
            .format(config_path)
        )
    if args.num_envs <= 0:
        raise ValueError("--num-envs must be positive")
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.print_every < 0:
        raise ValueError("--print-every cannot be negative")

    env_cfg, train_cfg = load_saved_configuration(config_path)
    env_cfg.seed = int(args.seed)
    env_cfg.env.num_envs = int(args.num_envs)
    env_cfg.env.play = True
    env_cfg.viewer.enable_viewer = bool(args.viewer)
    env_cfg.viewer.camera_position = [-1.0, -1.0, 1.5]
    env_cfg.viewer.camera_lookat = [0.0, 0.6, 0.75]
    env_cfg.viewer.reference_ghost = bool(args.viewer and args.ghost)
    # Playback always runs from the chosen RSI index to the final reference
    # sample, so the tracking threshold must not cut the episode short. The
    # threshold is still reported: see max_abs_position_error below.
    env_cfg.termination.enabled = False
    if args.contact_forces:
        # Acquiring the contact tensor is a construction-time decision, but the
        # reward must not change: with reward_per_finger at zero the contact
        # term contributes nothing and the return stays comparable to training.
        env_cfg.contact.enabled = True
        env_cfg.contact.reward_per_finger = 0.0
    if int(env_cfg.env.num_observations) != 114:
        raise ValueError("This evaluator requires the 114D observation contract")
    if int(env_cfg.env.num_actions) != 26:
        raise ValueError("This evaluator requires 26 AnimRL residual actions")

    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else checkpoint.parent / "eval_{}.json".format(checkpoint.stem)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_dir = (
        args.plot_dir.expanduser().resolve()
        if args.plot_dir is not None
        else checkpoint.parent / "eval_plots"
    )

    env = MotionImitationEnv(
        env_cfg,
        sim_device=args.sim_device,
        headless=not args.viewer,
        num_envs_override=None,
    )
    try:
        max_start = int(env.reference.last_index - 1)
        if not args.sampled_rsi and not 0 <= int(args.rsi_index) <= max_start:
            raise ValueError(
                "--rsi-index must lie in [0, {}], got {}".format(
                    max_start, args.rsi_index
                )
            )
        # Starting at reference index k needs last_index - k transitions to
        # reach the final sample; last_index covers the earliest possible
        # start, and reference-end termination closes each episode on its own
        # last transition. This replaces the saved training horizon.
        env.max_episode_length = int(env.reference.last_index)
        env.cfg.env.episode_length = env.max_episode_length

        runner = PPO(env, train_cfg, log_dir=None, device=env.device)
        checkpoint_infos = runner.load(
            checkpoint, load_optimizer=False, load_normalizers=True
        )
        policy = runner.get_inference_policy(device=env.device)

        fixed_rsi = None if args.sampled_rsi else int(args.rsi_index)
        if fixed_rsi is not None:
            env.reset(reference_index=fixed_rsi)
        else:
            env.reset()

        observations = env.get_observations()
        completed_episodes = 0
        total_steps = 0
        total_reward = 0.0
        done_count = 0
        early_count = 0
        timeout_count = 0
        peak_position_error = 0.0
        peak_hand_position_error = 0.0
        peak_object_com_height = -float("inf")
        peak_object_com_lift = -float("inf")
        episode_weight = 0
        episode_totals = {}
        trajectory = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "reference_indices": [],
            "max_abs_position_errors": [],
            "object_com_heights_m": [],
            "object_com_lifts_m": [],
        }
        plotter = EvaluationPlotter(plot_dir) if args.plots else None
        plot_paths = {}
        if plotter is not None:
            plotter.start_episode("episode_{:02d}".format(completed_episodes), env)

        print("Checkpoint: {}".format(checkpoint))
        print("Configuration: {}".format(config_path))
        print(
            "Evaluating deterministic mean actions: {} env(s), RSI={}".format(
                env.num_envs,
                "uniform" if fixed_rsi is None else fixed_rsi,
            )
        )
        if fixed_rsi is None:
            print(
                "Playing to reference sample {} from every uniform start; "
                "early termination disabled.".format(env.reference.last_index)
            )
        else:
            transitions = int(env.reference.last_index) - fixed_rsi
            print(
                "Playing samples {} to {}: {} transitions ({:.2f} s), early "
                "termination disabled.".format(
                    fixed_rsi,
                    env.reference.last_index,
                    transitions,
                    transitions * env.dt,
                )
            )
        with torch.inference_mode():
            while completed_episodes < args.episodes:
                actions = policy(observations)
                (
                    observations,
                    _,
                    rewards,
                    dones,
                    infos,
                ) = env.step(actions)
                total_steps += 1
                total_reward += scalar(rewards.mean())
                step_done_count = int(dones.sum())
                done_count += step_done_count
                early_count += int(infos["early_termination"].sum())
                timeout_count += int(infos["time_outs"].sum())
                # Early termination no longer stops playback, so the tracking
                # threshold is reported instead of enforced.
                peak_position_error = max(
                    peak_position_error,
                    scalar(infos["max_abs_position_error"].max()),
                )
                peak_hand_position_error = max(
                    peak_hand_position_error,
                    scalar(infos["max_abs_hand_position_error"].max()),
                )
                peak_object_com_height = max(
                    peak_object_com_height,
                    scalar(infos["object_com_height_m"].max()),
                )
                peak_object_com_lift = max(
                    peak_object_com_lift,
                    scalar(infos["object_com_lift_m"].max()),
                )

                trajectory["observations"].append(
                    observations[0].detach().cpu().tolist()
                )
                trajectory["actions"].append(
                    actions[0].detach().cpu().tolist()
                )
                trajectory["rewards"].append(float(rewards[0]))
                trajectory["dones"].append(bool(dones[0]))
                trajectory["reference_indices"].append(
                    int(infos["reference_index"][0])
                )
                trajectory["max_abs_position_errors"].append(
                    float(infos["max_abs_position_error"][0])
                )
                trajectory["object_com_heights_m"].append(
                    float(infos["object_com_height_m"][0])
                )
                trajectory["object_com_lifts_m"].append(
                    float(infos["object_com_lift_m"][0])
                )
                if plotter is not None:
                    plotter.record(
                        env, total_steps, actions, rewards, dones, infos
                    )

                if "episode" in infos:
                    episode = infos["episode"]
                    completed = int(episode["completed_episodes"])
                    completed_episodes += completed
                    episode_weight += completed
                    for name, value in episode.items():
                        if name == "completed_episodes":
                            continue
                        episode_totals[name] = episode_totals.get(name, 0.0) + (
                            scalar(value) * completed
                        )
                    if plotter is not None:
                        plot_paths = plotter.finalize(termination_reason(infos))
                        if completed_episodes < args.episodes:
                            plotter.start_episode(
                                "episode_{:02d}".format(completed_episodes), env
                            )
                    if fixed_rsi is not None and completed_episodes < args.episodes:
                        env.reset(reference_index=fixed_rsi)
                        observations = env.get_observations()

                if args.print_every > 0 and total_steps % args.print_every == 0:
                    print(
                        "step {:5d} | reward {:.6f} | max q error {:.6f} rad "
                        "| completed {}".format(
                            total_steps,
                            scalar(rewards.mean()),
                            scalar(infos["max_abs_position_error"].max()),
                            completed_episodes,
                        )
                    )
                if args.viewer and env.viewer_closed():
                    print("Viewer closed before the requested episodes completed")
                    break

        if plotter is not None and completed_episodes < args.episodes:
            # The loop broke out early (viewer closed); keep whatever the
            # partial episode recorded rather than discarding it.
            plot_paths = plotter.finalize("stopped") or plot_paths

        episode_metrics = {
            name: total / episode_weight
            for name, total in episode_totals.items()
        } if episode_weight else {}
        result = {
            "checkpoint": str(checkpoint),
            "config": str(config_path),
            "checkpoint_infos": checkpoint_infos,
            "deterministic": True,
            "seed": int(args.seed),
            "num_envs": env.num_envs,
            "rsi": "uniform" if fixed_rsi is None else fixed_rsi,
            "final_reference_index": int(env.reference.last_index),
            "requested_episodes": int(args.episodes),
            "completed_episodes": completed_episodes,
            "environment_steps": total_steps,
            "mean_step_reward": total_reward / max(total_steps, 1),
            "done_count": done_count,
            "early_termination_count": early_count,
            "timeout_count": timeout_count,
            "peak_position_error": peak_position_error,
            "peak_hand_position_error": peak_hand_position_error,
            "peak_object_com_height_m": peak_object_com_height,
            "peak_object_com_lift_m": peak_object_com_lift,
            "termination_threshold": float(
                env_cfg.termination.arm_position_threshold_rad
            ),
            "hand_termination_threshold": float(
                env_cfg.termination.hand_position_threshold_rad
            ),
            "exceeded_termination_threshold": bool(
                peak_position_error
                > float(env_cfg.termination.arm_position_threshold_rad)
            ),
            "exceeded_hand_termination_threshold": bool(
                peak_hand_position_error
                > float(env_cfg.termination.hand_position_threshold_rad)
            ),
            "episode_metrics": episode_metrics,
            "trajectory_env_0": trajectory,
            "plot_paths": plot_paths,
        }
        with output_path.open("w", encoding="utf-8") as output_file:
            json.dump(result, output_file, indent=2, sort_keys=True)

        print("Evaluation complete")
        print("  completed episodes : {}".format(completed_episodes))
        print("  environment steps  : {}".format(total_steps))
        print("  mean step reward   : {:.6f}".format(result["mean_step_reward"]))
        print("  peak |q error| arm : {:.6f} rad (threshold {:.2f}{})".format(
            peak_position_error,
            result["termination_threshold"],
            ", EXCEEDED" if result["exceeded_termination_threshold"] else "",
        ))
        print("  peak |q error| hand: {:.6f} rad (threshold {:.2f}{})".format(
            peak_hand_position_error,
            result["hand_termination_threshold"],
            ", EXCEEDED" if result["exceeded_hand_termination_threshold"] else "",
        ))
        print("  peak cube COM z    : {:.6f} m".format(peak_object_com_height))
        print("  peak cube COM lift : {:.6f} m".format(peak_object_com_lift))
        if episode_metrics:
            print("  mean return        : {:.6f}".format(
                episode_metrics["return"]
            ))
            print("  mean episode length: {:.2f}".format(
                episode_metrics["length"]
            ))
        print("  output             : {}".format(output_path))
        if plot_paths.get("episode_dir"):
            print("  plots              : {}".format(plot_paths["episode_dir"]))
    finally:
        env.close()


if __name__ == "__main__":
    main()
