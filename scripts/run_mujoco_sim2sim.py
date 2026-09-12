#!/usr/bin/env python3
"""Run a blind AnimRL motion-imitation checkpoint in MuJoCo."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from simtoolreal_animrl.envs.demonstration import JointDemonstration60Hz
from simtoolreal_animrl.sim2sim.mujoco_sim import (
    AnimRLMujocoSim,
    MujocoSceneConfig,
)
from simtoolreal_animrl.sim2sim.observation import (
    QuaternionContinuity,
    actions_to_position_targets,
    build_observation,
    smoothstep01,
)
from simtoolreal_animrl.sim2sim.policy import (
    AnimRLInferencePolicy,
    load_saved_run,
)


DEFAULT_CHECKPOINT = REPO_ROOT / (
    "logs/simtoolreal/2026-09-07_003258_pg830_blind512_n256/best_model.pt"
)

# Effective Isaac Gym position-drive gains used to train adapt_sigma. The hand
# Kp values already include control.hand_stiffness_scale=0.5; damping was not
# scaled. Keep these values local so sim2sim does not depend on training code.
TRAINING_ARM_KP = (1000.0, 1000.0, 1000.0, 200.0, 200.0, 100.0)
TRAINING_ARM_KD = (100.0, 100.0, 100.0, 10.0, 10.0, 10.0)
TRAINING_HAND_KP = (
    21.4859, 200.0, 21.4859, 21.4859,
    21.4859, 21.4859, 21.4859, 21.4859,
    21.4859, 21.4859, 21.4859, 21.4859,
    21.4859, 21.4859, 21.4859, 21.4859,
    21.4859, 21.4859, 21.4859, 21.4859,
)
TRAINING_HAND_KD = (
    0.1, 0.9475, 0.3012, 0.1821,
    0.7523, 0.4126, 0.2856, 0.1365,
    0.7587, 0.4126, 0.2856, 0.1365,
    0.7274, 0.4126, 0.2856, 0.1365,
    0.2662, 0.4796, 0.3012, 0.1821,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Defaults to config.json beside the checkpoint.",
    )
    parser.add_argument("--device", default="cpu", help="Actor device: cpu or cuda.")
    parser.add_argument("--rsi-index", type=int, default=0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Maximum control steps; 0 runs to the end of the demonstration.",
    )
    parser.add_argument(
        "--sim-dt",
        type=float,
        default=1.0 / 600.0,
        help="MuJoCo physics timestep. The policy remains at 60 Hz.",
    )
    parser.add_argument("--arm-kp", type=float, default=300.0)
    parser.add_argument("--arm-kv", type=float, default=20.0)
    parser.add_argument("--hand-kp", type=float, default=5.0)
    parser.add_argument("--hand-kv", type=float, default=0.25)
    parser.add_argument(
        "--startup-ramp-seconds", type=float, default=1.0,
        help="Smoothly ramp targets from the RSI pose (default: 1.0 s).",
    )
    parser.add_argument(
        "--startup-policy-blend-seconds", type=float, default=1.0,
        help="Blend from reference to learned actions at startup (default: 1.0 s).",
    )
    parser.add_argument(
        "--training-pd-gains",
        action="store_true",
        help=(
            "Use the hardcoded per-joint Isaac Gym Kp/Kd values used to train "
            "the adapt_sigma policy. This "
            "takes precedence over --arm-kp/--arm-kv/--hand-kp/--hand-kv."
        ),
    )
    parser.add_argument(
        "--training-arm-pd-gains",
        dest="training_arm_pd_gains",
        action="store_true",
        default=True,
        help=(
            "Use the per-joint Isaac Gym arm Kp/Kd used in training while "
            "keeping the conservative scalar hand gains."
        ),
    )
    parser.add_argument(
        "--scalar-arm-pd-gains",
        dest="training_arm_pd_gains",
        action="store_false",
        help="Use --arm-kp/--arm-kv instead of the training arm gains.",
    )
    parser.add_argument(
        "--contact-settle-seconds",
        type=float,
        default=0.1,
        help=(
            "At an RSI reset with existing hand/cube contact, keep the cube "
            "fixed for this simulated duration while the fingers settle. "
            "Use 0 to disable (default: 0.1)."
        ),
    )
    parser.add_argument(
        "--start-delay-seconds",
        type=float,
        default=2.0,
        help="Viewer pause before the policy starts moving (default: 2.0).",
    )
    parser.add_argument(
        "--no-reference-ghost",
        "--no-ghost",
        dest="reference_ghost",
        action="store_false",
        help="Hide the green kinematic demonstration robot.",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Output folder for final PNG/NPZ diagnostics.",
    )
    parser.add_argument(
        "--no-plots",
        dest="plots",
        action="store_false",
        help="Do not create the final rollout graphs.",
    )
    parser.add_argument(
        "--no-show-plots",
        action="store_true",
        help="Save final graphs without opening their windows.",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-realtime", action="store_true")
    parser.add_argument(
        "--continuous-quaternions",
        action="store_true",
        help=(
            "Unwrap the two legacy 108-D quaternion signs against the "
            "preceding observation."
        ),
    )
    parser.add_argument(
        "--continuous-quaternions-until-seconds",
        type=float,
        default=None,
        help=(
            "Apply sign unwrapping only before this absolute demonstration "
            "time; requires --continuous-quaternions."
        ),
    )
    parser.add_argument("--smooth-quaternion-transition", action="store_true")
    parser.add_argument(
        "--quaternion-transition-duration-seconds", type=float, default=1.0,
    )
    parser.add_argument("--print-every", type=int, default=60)
    return parser.parse_args()


def resolve_demo_path(repo_root: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    return path if path.is_absolute() else repo_root / path


def main() -> None:
    args = parse_args()
    if args.max_steps < 0:
        raise ValueError("--max-steps cannot be negative")
    if args.sim_dt <= 0.0:
        raise ValueError("--sim-dt must be positive")
    if args.print_every < 0:
        raise ValueError("--print-every cannot be negative")
    if args.contact_settle_seconds < 0.0:
        raise ValueError("--contact-settle-seconds cannot be negative")
    if args.start_delay_seconds < 0.0:
        raise ValueError("--start-delay-seconds cannot be negative")
    if args.startup_ramp_seconds < 0.0:
        raise ValueError("--startup-ramp-seconds cannot be negative")
    if args.startup_policy_blend_seconds < 0.0:
        raise ValueError("--startup-policy-blend-seconds cannot be negative")
    if (
        args.continuous_quaternions_until_seconds is not None
        and args.continuous_quaternions_until_seconds < 0.0
    ):
        raise ValueError("--continuous-quaternions-until-seconds cannot be negative")
    if (
        args.continuous_quaternions_until_seconds is not None
        and not args.continuous_quaternions
    ):
        raise ValueError(
            "--continuous-quaternions-until-seconds requires "
            "--continuous-quaternions"
        )
    if args.smooth_quaternion_transition and args.continuous_quaternions:
        raise ValueError(
            "--smooth-quaternion-transition and --continuous-quaternions "
            "are mutually exclusive"
        )
    if args.quaternion_transition_duration_seconds <= 0.0:
        raise ValueError("--quaternion-transition-duration-seconds must be positive")

    run = load_saved_run(args.checkpoint, args.config)
    env_cfg = run.env_cfg
    demo_path = resolve_demo_path(run.repo_root, env_cfg["motion"]["file"])
    reference = JointDemonstration60Hz.load(
        demo_path,
        device="cpu",
        expected_hz=float(env_cfg["motion"]["frequency_hz"]),
    )
    if not 0 <= args.rsi_index < reference.last_index:
        raise ValueError(
            "--rsi-index must lie in [0, {}]".format(reference.last_index - 1)
        )

    actor = AnimRLInferencePolicy(run, device=args.device)
    control_cfg = env_cfg["control"]
    training_joint_kp = None
    training_joint_kd = None
    if args.training_pd_gains:
        training_joint_kp = TRAINING_ARM_KP + TRAINING_HAND_KP
        training_joint_kd = TRAINING_ARM_KD + TRAINING_HAND_KD
    elif args.training_arm_pd_gains:
        training_joint_kp = TRAINING_ARM_KP + (args.hand_kp,) * 20
        training_joint_kd = TRAINING_ARM_KD + (args.hand_kv,) * 20
    scene_config = MujocoSceneConfig.from_saved_config(
        run.repo_root,
        env_cfg,
        sim_dt=args.sim_dt,
        enable_viewer=not args.headless,
        arm_kp=args.arm_kp,
        arm_kv=args.arm_kv,
        hand_kp=args.hand_kp,
        hand_kv=args.hand_kv,
        joint_kp=training_joint_kp,
        joint_kv=training_joint_kd,
        enable_reference_ghost=args.reference_ghost,
    )
    defaults = np.asarray(
        env_cfg["init_state"]["default_arm_joint_angles"]
        + env_cfg["init_state"]["default_hand_joint_angles"],
        dtype=np.float64,
    )
    control_dt = 1.0 / float(env_cfg["motion"]["frequency_hz"])
    start_index = int(args.rsi_index)
    action_scales = np.concatenate(
        (
            np.full(6, float(control_cfg["scale_joint_target"])),
            np.full(20, float(control_cfg["scale_hand_joint_target"])),
        )
    )
    trace = {
        "reference_indices": [],
        "policy_actions": [],
        "reference_actions": [],
        "action_deltas": [],
        "actual_joint_positions": [],
        "applied_position_targets": [],
        "raw_position_targets": [],
        "reference_joint_positions": [],
    }

    with AnimRLMujocoSim(scene_config) as sim:
        sample = reference.sample(np_to_long_tensor(start_index))
        sim.reset(
            sample.q[0].numpy(),
            sample.dq[0].numpy(),
            sample.cube_pose[0].numpy(),
            sample.cube_linear_velocity[0].numpy(),
            sample.cube_angular_velocity[0].numpy(),
        )
        settling = sim.settle_robot_cube_contacts(args.contact_settle_seconds)
        sim.set_reference_ghost(sample.q[0].numpy())
        sim.sync_viewer()
        previous_targets = sample.q[0].numpy().astype(np.float64)
        startup_q = previous_targets.copy()
        previous_action = (previous_targets - defaults) / action_scales
        reference_index = start_index
        steps = 0
        if args.training_pd_gains:
            pd_description = "hardcoded adapt_sigma training per-joint gains"
        elif args.training_arm_pd_gains:
            pd_description = (
                "training per-joint arm gains, hand={}/{}"
            ).format(args.hand_kp, args.hand_kv)
        else:
            pd_description = "arm={}/{}, hand={}/{}".format(
                args.arm_kp, args.arm_kv, args.hand_kp, args.hand_kv
            )
        print(
            "MuJoCo AnimRL sim2sim: checkpoint={}, observations/actions=108/26, "
            "RSI={}, control=60 Hz, physics={:.1f} Hz, viewer={}, "
            "PD {}".format(
                run.checkpoint_path,
                start_index,
                1.0 / args.sim_dt,
                not args.headless,
                pd_description,
            )
        )
        if settling["steps"]:
            print(
                "RSI contact settling: steps={steps}, contacts={contacts_before}"
                "->{contacts_after}, min_distance={minimum_distance_before_m:.4f}"
                "->{minimum_distance_after_m:.4f} m, max_joint_shift="
                "{max_joint_displacement_rad:.3f} rad".format(**settling)
            )
        if not args.headless and args.start_delay_seconds:
            print(
                "Starting movement in {:.1f} seconds...".format(
                    args.start_delay_seconds
                )
            )
            deadline = time.perf_counter() + args.start_delay_seconds
            while sim.viewer_is_running() and time.perf_counter() < deadline:
                sim.sync_viewer()
                time.sleep(min(0.01, max(0.0, deadline - time.perf_counter())))

        quaternion_continuity = (
            QuaternionContinuity()
            if args.continuous_quaternions or args.smooth_quaternion_transition
            else None
        )
        quaternion_transition_started_at = None
        while reference_index < reference.last_index and sim.viewer_is_running():
            if args.max_steps and steps >= args.max_steps:
                break
            started = time.perf_counter()
            phase = reference_index / float(reference.last_index)
            canonical_observation = build_observation(
                sim.get_state(),
                previous_targets,
                phase,
                sim.joint_lower_limits,
                sim.joint_upper_limits,
            )
            continuity_active = (
                quaternion_continuity is not None
                and args.continuous_quaternions
                and (
                    args.continuous_quaternions_until_seconds is None
                    or reference_index * control_dt
                    < args.continuous_quaternions_until_seconds
                )
            )
            observation = canonical_observation
            if continuity_active:
                observation = quaternion_continuity.apply(observation)
            if args.smooth_quaternion_transition:
                continuous_observation = quaternion_continuity.apply(
                    canonical_observation
                )
                demo_seconds = reference_index * control_dt
                if (
                    quaternion_transition_started_at is None
                    and any(quaternion_continuity.last_flipped)
                ):
                    quaternion_transition_started_at = demo_seconds
                    print(
                        "Quaternion action transition triggered at {:.3f} s".format(
                            demo_seconds
                        )
                    )
                progress = (
                    0.0
                    if quaternion_transition_started_at is None
                    else (demo_seconds - quaternion_transition_started_at)
                    / args.quaternion_transition_duration_seconds
                )
                canonical_weight = smoothstep01(progress)
                if canonical_weight <= 0.0:
                    actions = actor(continuous_observation)
                elif canonical_weight >= 1.0:
                    actions = actor(canonical_observation)
                else:
                    actions = (
                        (1.0 - canonical_weight) * actor(continuous_observation)
                        + canonical_weight * actor(canonical_observation)
                    )
            else:
                actions = actor(observation)
            if args.startup_policy_blend_seconds > 0.0:
                startup_policy_progress = (
                    steps * control_dt / args.startup_policy_blend_seconds
                )
                if startup_policy_progress < 1.0:
                    current_reference = reference.sample(
                        np_to_long_tensor(reference_index)
                    )
                    reference_actions = (
                        current_reference.q[0].numpy() - defaults
                    ) / action_scales
                    policy_weight = smoothstep01(startup_policy_progress)
                    actions = (
                        (1.0 - policy_weight) * reference_actions
                        + policy_weight * actions
                    )
            targets = actions_to_position_targets(
                actions,
                defaults,
                arm_scale=float(control_cfg["scale_joint_target"]),
                hand_scale=float(control_cfg["scale_hand_joint_target"]),
                residual_clip=float(control_cfg["clip_joint_target"]),
            )
            if args.startup_ramp_seconds > 0.0:
                startup_progress = (
                    (steps + 1) * control_dt / args.startup_ramp_seconds
                )
                if startup_progress < 1.0:
                    startup_weight = smoothstep01(startup_progress)
                    targets = (
                        (1.0 - startup_weight) * startup_q
                        + startup_weight * targets
                    )
            next_reference = reference.sample(
                np_to_long_tensor(reference_index + 1)
            )
            sim.set_reference_ghost(next_reference.q[0].numpy())
            sim.set_position_targets(targets)
            sim.step_for(control_dt)
            previous_targets = targets
            reference_index += 1
            steps += 1
            state = sim.get_state()
            expected_q = next_reference.q[0].numpy()
            trace["reference_indices"].append(reference_index)
            trace["policy_actions"].append(actions.copy())
            trace["reference_actions"].append(
                (expected_q - defaults) / action_scales
            )
            trace["action_deltas"].append(actions - previous_action)
            trace["actual_joint_positions"].append(
                state["joint_positions"].copy()
            )
            trace["applied_position_targets"].append(
                np.clip(
                    targets,
                    sim.joint_lower_limits,
                    sim.joint_upper_limits,
                )
            )
            trace["raw_position_targets"].append(targets.copy())
            trace["reference_joint_positions"].append(expected_q.copy())
            previous_action = actions.copy()

            if args.print_every and (
                steps == 1 or steps % args.print_every == 0
            ):
                expected_cube, _, _, _ = sim.reference_cube_state_to_world(
                    next_reference.cube_pose[0].numpy(),
                    next_reference.cube_linear_velocity[0].numpy(),
                    next_reference.cube_angular_velocity[0].numpy(),
                )
                joint_errors = np.abs(
                    state["joint_positions"] - expected_q
                )
                arm_error = np.max(joint_errors[:6])
                hand_error = np.max(joint_errors[6:])
                cube_error = np.linalg.norm(
                    state["cube_position_world"] - expected_cube
                )
                print(
                    "step={:4d} ref={:4d} phase={:.3f} "
                    "max|arm-ref|={:.3f} rad max|hand-ref|={:.3f} rad "
                    "cube_pos_error={:.3f} m".format(
                        steps,
                        reference_index,
                        reference_index / float(reference.last_index),
                        arm_error,
                        hand_error,
                        cube_error,
                    )
                )

            if not args.no_realtime:
                remaining = control_dt - (time.perf_counter() - started)
                if remaining > 0.0:
                    time.sleep(remaining)

        state = sim.get_state()
        print(
            "Finished after {} control steps at reference index {}. "
            "Cube position: {}".format(
                steps,
                reference_index,
                np.round(state["cube_position_world"], 4).tolist(),
            )
        )

    if args.plots and trace["reference_indices"]:
        from simtoolreal_animrl.sim2sim.plotting import save_rollout_plots

        plot_dir = args.plot_dir or (
            run.checkpoint_path.parent
            / "sim2sim_plots"
            / "rsi_{:04d}".format(start_index)
        )
        paths = save_rollout_plots(
            plot_dir,
            trace,
            show=not args.headless and not args.no_show_plots,
        )
        print("Saved sim2sim diagnostics to {}".format(Path(plot_dir).resolve()))
        for name, path in paths.items():
            print("  {}: {}".format(name, path))


def np_to_long_tensor(index: int):
    # Keep torch out of module import ordering until the local package has been
    # added to sys.path. The demonstration API expects a 1-D torch index.
    import torch

    return torch.tensor([int(index)], dtype=torch.long)


if __name__ == "__main__":
    main()
