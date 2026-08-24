#!/usr/bin/env python3
"""View the training environment driven by ideal demonstration actions."""

import argparse
import os
import sys
from pathlib import Path


def _configure_isaac_gym_graphics_environment() -> None:
    """Prevent ROS/Gazebo libraries from shadowing NVIDIA graphics libraries.
        If you don't do it, whenever you try to run the Isaac Gym viewer, you will get a segmentation fault."""

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

# MotionImitationEnv preserves Isaac Gym's required import-before-torch order.
from simtoolreal_animrl.cfg import SimToolRealCfg
from simtoolreal_animrl.envs.motion_imitation import MotionImitationEnv


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of training environments to display (default: 1).",
    )
    parser.add_argument(
        "--rsi-index",
        type=int,
        default=None,
        help=(
            "Force the episode to begin at this demonstration sample; "
            "by default the environment uses the same uniform RSI as training."
        ),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=0,
        help="Stop after this many episodes; 0 runs until the viewer is closed.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=30,
        help="Print tracking diagnostics every N steps; 0 disables them.",
    )
    parser.add_argument(
        "--sim-device", default="cuda:0", help="Isaac Gym simulation device."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SimToolRealCfg()
    env = MotionImitationEnv(
        cfg,
        sim_device=args.sim_device,
        headless=False,
        num_envs_override=args.num_envs
    )
    completed_episodes = 0
    total_steps = 0

    try:
        if args.rsi_index is not None:
            env.reset(reference_index=args.rsi_index)

        print(
            "Loaded the training environment: {} env(s), {}-step episodes, "
            "uniform RSI={}.".format(
                env.num_envs,
                env.max_episode_length,
                args.rsi_index is None,
            )
        )
        print("The policy output is replaced by the next demonstration action.")

        # Show the RSI state before applying the first action.
        env.render(sync_frame_time=False)
        while not env.viewer_closed():
            actions_ideal, _ = env.next_reference_action()
            _, _, rewards, dones, extras = env.step(actions_ideal)
            total_steps += 1

            if args.print_every > 0 and total_steps % args.print_every == 0:
                print(
                    "step {:5d} | ref {:4d} | reward {:.6f} | "
                    "max q error {:.6f} rad".format(
                        total_steps,
                        int(extras["reference_index"][0]),
                        float(rewards.mean()),
                        float(extras["max_abs_position_error"].max()),
                    )
                )

            if bool(dones.any()):
                completed_episodes += 1
                print(
                    "episode {} completed | timeout={} | early termination={}".format(
                        completed_episodes,
                        int(extras["time_outs"].sum()),
                        int(extras["early_termination"].sum()),
                    )
                )
                if args.episodes > 0 and completed_episodes >= args.episodes:
                    break

                # step() performs the same automatic uniform-RSI reset that the
                # training runner will see. An explicit RSI is reapplied only
                # when the user requested deterministic playback.
                if args.rsi_index is not None:
                    env.reset(reference_index=args.rsi_index)
    finally:
        env.close()


if __name__ == "__main__":
    main()
