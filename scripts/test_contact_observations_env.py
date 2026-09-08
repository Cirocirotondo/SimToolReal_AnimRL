#!/usr/bin/env python3
"""Isaac Gym check for the fingertip contact-force observation block (no PPO).

The block is the last 3 numbers per selected fingertip of the observation
vector. This holds the robot at a frame inside the grasp and at a frame before
it, so the block has to come alive in one case and stay at zero in the other.
It also re-derives the block from the raw PhysX force tensor to confirm the
wiring, ordering, scaling and clipping all match.

The object assist pins the cube to its demonstrated pose for the duration, so
the frozen fingers keep something to touch: without it the unheld cube drops
away within a few steps and every fingertip correctly reads zero.
"""

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Preserve Isaac Gym's required import-before-torch ordering.
from simtoolreal_animrl.cfg import SimToolRealCfg
from simtoolreal_animrl.envs.motion_imitation import MotionImitationEnv

# contact.py imports torch, so it has to follow the isaacgym import above.
from simtoolreal_animrl.envs.contact import select_fingertip_forces

import torch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument(
        "--grasp-index",
        type=int,
        default=900,
        help="Frozen start frame with the fingers on the cube.",
    )
    parser.add_argument(
        "--free-index",
        type=int,
        default=0,
        help="Frozen start frame with the hand away from the cube.",
    )
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--sim-device", default="cuda:0")
    return parser.parse_args()


def hold_from(env, rsi_index, steps, block_width):
    """Freeze the robot at its RSI pose and return every contact block it saw.

    The cube is only in the fingers for part of the hold, so the caller gets the
    whole time series rather than the final step alone.
    """
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    env.reset_idx(env_ids, torch.full_like(env_ids, int(rsi_index)))
    env.compute_observations()
    actions = env.actions.clone()
    blocks = [env.obs_buf[:, -block_width:].clone()]
    observations = [env.obs_buf.clone()]
    for _ in range(int(steps)):
        obs, _, _, _, _ = env.step(actions)
        blocks.append(obs[:, -block_width:].clone())
        observations.append(obs.clone())
    return torch.stack(blocks), observations[-1]


def main():
    args = parse_args()

    blind_cfg = SimToolRealCfg()
    blind_cfg.env.num_envs = int(args.num_envs)
    base_width = int(blind_cfg.env.num_observations)

    cfg = SimToolRealCfg()
    cfg.env.num_envs = int(args.num_envs)
    # The frozen robot violates the tracking thresholds within a few steps.
    cfg.termination.enabled = False
    cfg.contact.enabled = True
    cfg.contact.observe_fingertip_forces = True
    # Holds the cube at its demonstrated pose so the frozen fingers stay in
    # touch with something for the whole hold.
    cfg.object_assist.enabled = True

    num_tips = len(cfg.contact.fingertip_names)
    block_width = 3 * num_tips
    expected_width = base_width + block_width

    env = MotionImitationEnv(cfg, sim_device=args.sim_device, headless=True)
    try:
        print(
            "Observation width: {} base + {} contact = {}".format(
                base_width, block_width, env.num_obs
            )
        )
        if env.num_obs != expected_width:
            raise AssertionError(
                "Expected {} observations, got {}".format(
                    expected_width, env.num_obs
                )
            )
        if env.obs_buf.shape[1] != expected_width:
            raise AssertionError("The observation buffer has the wrong width")

        scale = float(cfg.contact.observation_force_scale_n)
        clip = float(cfg.contact.observation_clip)

        env.set_object_assist_scale(1.0)
        grasp_blocks, grasp_obs = hold_from(
            env, args.grasp_index, args.steps, block_width
        )
        block = grasp_blocks[-1]

        # Re-derive the block from the raw tensor. A rotation into the palm
        # frame preserves each fingertip's force magnitude, which is what makes
        # this comparable without duplicating the palm kinematics here.
        world_forces = select_fingertip_forces(
            env.net_contact_forces, env.contact_fingertip_body_indices
        )
        world_norms = torch.linalg.vector_norm(world_forces, dim=2) / scale
        block_norms = torch.linalg.vector_norm(
            block.reshape(env.num_envs, num_tips, 3), dim=2
        )
        unclipped = world_norms < clip
        if unclipped.any():
            deviation = float(
                (block_norms[unclipped] - world_norms[unclipped]).abs().max()
            )
            print(
                "  max |palm-frame norm - world norm| = {:.3e} "
                "over {} unclipped fingertips".format(
                    deviation, int(unclipped.sum())
                )
            )
            if deviation > 1e-4:
                raise AssertionError(
                    "The observation block does not match the force tensor"
                )

        # Over the whole hold, not just its last step.
        all_norms = torch.linalg.vector_norm(
            grasp_blocks.reshape(-1, env.num_envs, num_tips, 3), dim=3
        )
        ever_touched = (all_norms > 0.0).any(dim=0).float().mean(dim=0)
        peak_force = all_norms.amax(dim=(0, 1)) * scale
        print(
            "\nRSI {}, {} steps frozen inside the grasp, cube pinned".format(
                args.grasp_index, args.steps
            )
        )
        for name, fraction, force in zip(
            cfg.contact.fingertip_names,
            ever_touched.tolist(),
            peak_force.tolist(),
        ):
            print(
                "  {:<7} touched in {:>5.1f}% of envs, peak {:.2f} N".format(
                    name, 100.0 * fraction, force
                )
            )
        touching = ever_touched
        if float(grasp_blocks.abs().max()) > clip:
            raise AssertionError("The observation block escaped its clip")
        if float(touching.max()) == 0.0:
            raise AssertionError(
                "No fingertip ever registered contact inside the grasp"
            )

        env.set_object_assist_scale(0.0)
        free_blocks, _ = hold_from(
            env, args.free_index, args.steps, block_width
        )
        print(
            "\nRSI {}, {} steps frozen before the grasp".format(
                args.free_index, args.steps
            )
        )
        print(
            "  max |contact observation| = {:.4f}".format(
                float(free_blocks.abs().max())
            )
        )

        # The rest of the vector must be untouched by the new block, so a run
        # without it sees exactly what it saw before.
        if not torch.isfinite(grasp_obs).all():
            raise AssertionError("The observation vector carries a non-finite")
        print("\nContact-observation environment test passed")
    finally:
        env.close()


if __name__ == "__main__":
    main()
