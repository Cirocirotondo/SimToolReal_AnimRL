#!/usr/bin/env python3
"""Isaac Gym check for the annealed object-assist wrench (no PPO).

The robot is frozen at its RSI pose while the demonstration keeps moving, so
the physical cube can only follow the demonstrated cube if the external assist
wrench is actually reaching PhysX. The same run then repeats at scale zero,
which must reproduce the unassisted environment exactly.
"""

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Preserve Isaac Gym's required import-before-torch ordering.
from simtoolreal_animrl.cfg import SimToolRealCfg
from simtoolreal_animrl.envs.motion_imitation import MotionImitationEnv

import torch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument(
        "--rsi-index",
        type=int,
        default=900,
        help="Frozen start frame; the default is inside the lift.",
    )
    parser.add_argument("--steps", type=int, default=90)
    parser.add_argument("--sim-device", default="cuda:0")
    return parser.parse_args()


def hold_from(env, scale, rsi_index, steps):
    """Freeze the robot at its RSI pose and report the cube's tracking error."""
    env.set_object_assist_scale(scale)
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    env.reset_idx(env_ids, torch.full_like(env_ids, int(rsi_index)))
    env.compute_observations()
    # reset_idx seeds the action that reproduces the RSI pose, so repeating it
    # holds the robot still while the reference walks away from it.
    actions = env.actions.clone()
    errors = []
    forces = []
    torques = []
    for _ in range(int(steps)):
        _, _, _, _, extras = env.step(actions)
        errors.append(float(extras["object_position_error_m"].mean()))
        forces.append(float(extras["object_assist_force_n"].mean()))
        torques.append(float(extras["object_assist_torque_nm"].mean()))
    return errors, forces, torques


def main():
    args = parse_args()
    cfg = SimToolRealCfg()
    cfg.env.num_envs = int(args.num_envs)
    # The frozen robot violates the tracking thresholds within a few steps.
    cfg.termination.enabled = False
    cfg.object_assist.enabled = True

    env = MotionImitationEnv(cfg, sim_device=args.sim_device, headless=True)
    try:
        settings = env.object_assist_settings
        print(
            "Assist schedule: {} -> {} between iterations {} and {}".format(
                settings.initial_scale,
                settings.final_scale,
                settings.start_iteration,
                settings.end_iteration,
            )
        )
        for iteration in (
            settings.start_iteration,
            (settings.start_iteration + settings.end_iteration) // 2,
            settings.end_iteration,
        ):
            print(
                "  scale({}) = {:.3f}".format(
                    iteration, env.set_training_iteration(iteration)
                )
            )
        if env.rigid_body_forces.shape != (
            env.num_envs * env.rigid_body_state.shape[1],
            3,
        ):
            raise AssertionError("The assist force buffer has the wrong shape")

        assisted, forces, torques = hold_from(
            env, 1.0, args.rsi_index, args.steps
        )
        unassisted, zero_forces, zero_torques = hold_from(
            env, 0.0, args.rsi_index, args.steps
        )
        print(
            "\nRSI {}, {} steps with the robot frozen".format(
                args.rsi_index, args.steps
            )
        )
        print(
            "  assist 1.0: cube error {:.4f} -> {:.4f} m, mean wrench "
            "{:.2f} N / {:.4f} Nm".format(
                assisted[0],
                assisted[-1],
                sum(forces) / len(forces),
                sum(torques) / len(torques),
            )
        )
        print(
            "  assist 0.0: cube error {:.4f} -> {:.4f} m, mean wrench "
            "{:.2f} N / {:.4f} Nm".format(
                unassisted[0],
                unassisted[-1],
                sum(zero_forces) / len(zero_forces),
                sum(zero_torques) / len(zero_torques),
            )
        )
        if max(zero_forces) != 0.0 or max(zero_torques) != 0.0:
            raise AssertionError("A zero assist scale still applied a wrench")
        if assisted[-1] >= unassisted[-1]:
            raise AssertionError(
                "The assist did not improve the cube's pose tracking"
            )
        print("\nObject-assist environment test passed")
    finally:
        env.close()


if __name__ == "__main__":
    main()
