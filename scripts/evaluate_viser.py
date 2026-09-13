#!/usr/bin/env python3
"""Interactive measured evaluation of a checkpoint, driven from a Viser GUI.

Isaac Gym runs headless in this process; Viser renders what it reports. The
cuboid's planar transform and the RSI start frame are chosen in the browser,
each ``Run`` performs one measured rollout with deterministic mean actions, and
the rollout is both streamed live and kept for frame-by-frame scrubbing.

The cuboid is placed at the exact continuous transform requested while the arm
reference comes from its nearest bank entry -- the same approximation training
makes, so what is measured here is what the policy was trained against. The
residual between the two is reported rather than hidden.

Run it with the project's Viser/Isaac Gym interpreter:

    /home/duplo/simone/SimToolReal/.venv/bin/python scripts/evaluate_viser.py \
        --checkpoint logs/simtoolreal/<run>/model_8500.pt
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# Isaac Gym must be imported before torch, and evaluate.py owns the graphics
# environment fix-ups the headless simulator needs.
from evaluate import (  # noqa: E402
    _configure_isaac_gym_graphics_environment,
    load_saved_configuration,
)

_configure_isaac_gym_graphics_environment()

from demo_viewer_viser import (  # noqa: E402
    ROBOT_POSITION_WORLD,
    ROBOT_WXYZ_WORLD,
    install_path_is_relative_to_backport,
)
from simtoolreal_animrl.cfg import SimToolRealCfg  # noqa: E402
from simtoolreal_animrl.envs.motion_imitation import MotionImitationEnv  # noqa: E402
from simtoolreal_animrl.envs.transform_bank import (  # noqa: E402
    nearest_transform_indices,
)
from simtoolreal_animrl.runners import PPO  # noqa: E402

import torch  # noqa: E402

DEFAULT_URDF = REPO_ROOT / (
    "assets/urdf/ur5e_delto_description/ur5e_right_dg5f_mount_60deg.urdf"
)
CUBE_DIMENSIONS = (0.15, 0.05, 0.05)
# Lift used to call one environment a success, matching the demonstration's own
# clearance rather than any threshold the reward uses.
SUCCESS_LIFT_M = 0.05

# How far a placement may sit from its serving bank reference before it is worth
# a warning. These are the 99th percentiles of the residual training itself
# lives with, measured over 20k uniform draws from the configured training range
# against banks/stage1.pt: median 16.5 mm / 5.4 deg, p95 30.7 mm / 15.8 deg,
# p99 37.5 mm / 20.4 deg, max 48.8 mm / 29.9 deg. A threshold tighter than this
# fires on ordinary in-distribution placements and means nothing.
TRAINING_TRANSLATION_RESIDUAL_P99_M = 0.0375
TRAINING_YAW_RESIDUAL_P99_DEG = 20.4
TRAINING_TRANSLATION_RESIDUAL_MEDIAN_M = 0.0165
TRAINING_YAW_RESIDUAL_MEDIAN_DEG = 5.4

# The demonstration's own per-step hand action-delta MSE, 99th percentile,
# measured over demonstrations/..._stable_grasp.npz with scale_hand_joint_target
# = 0.15: median 5.2e-05, p95 2.5e-03, p99 4.7e-03, max 3.1e-02. It is the
# yardstick for "as smooth as the demonstration"; the jitter plot draws it.
DEMONSTRATION_HAND_ACTION_RATE_P99_MSE = 4.669e-03

FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
FINGER_COLORS = ("#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd")


def quaternion_xyzw_to_wxyz(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    return quaternion[..., [3, 0, 1, 2]]


def render_diagnostics(frames, table_top_z, fingertip_std_m, palm_std_m):
    """Six per-step diagnostics for one recorded rollout, as a PNG array.

    Every panel draws the reference beside the policy, because each of these
    questions is about a gap rather than a level: fingers below the table, two
    fingertips closer than the demonstration ever put them, and a command that
    changes faster step to step than the demonstration's does.
    """
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    import imageio.v3 as iio

    steps = np.asarray([frame.step for frame in frames])
    series = {
        name: np.asarray([frame.metrics[name] for frame in frames])
        for name in frames[0].metrics
    }
    fingertip_z = np.stack([frame.fingertip_z for frame in frames])
    reference_fingertip_z = np.stack(
        [frame.reference_fingertip_z for frame in frames]
    )

    figure, axes = plt.subplots(3, 2, figsize=(13.0, 10.5), dpi=100)
    figure.subplots_adjust(hspace=0.42, wspace=0.22)

    axis = axes[0][0]
    axis.plot(steps, series["palm_keypoint_error_m"], color="#1f77b4",
              label="palm")
    axis.plot(steps, series["fingertip_keypoint_error_m"], color="#2ca02c",
              label="fingertips")
    axis.axhline(palm_std_m, color="#1f77b4", ls=":", lw=1.0,
                 label="palm sigma {:.3f} m".format(palm_std_m))
    axis.axhline(fingertip_std_m, color="#2ca02c", ls=":", lw=1.0,
                 label="fingertip sigma {:.3f} m".format(fingertip_std_m))
    axis.set_title("Keypoint tracking error (cuboid frame)")
    axis.set_ylabel("m")
    axis.legend(fontsize=7)

    axis = axes[0][1]
    for index, (name, color) in enumerate(zip(FINGER_NAMES, FINGER_COLORS)):
        axis.plot(steps, 1e3 * (fingertip_z[:, index] - table_top_z),
                  color=color, lw=1.0, label=name)
        axis.plot(steps, 1e3 * (reference_fingertip_z[:, index] - table_top_z),
                  color=color, lw=0.8, ls="--", alpha=0.45)
    axis.axhline(0.0, color="black", lw=1.2)
    axis.fill_between(steps, 1e3 * (fingertip_z.min(axis=1) - table_top_z),
                      0.0,
                      where=(fingertip_z.min(axis=1) < table_top_z),
                      color="#d62728", alpha=0.15)
    axis.set_title("Fingertip height above the table top (dashed: reference)")
    axis.set_ylabel("mm")
    axis.legend(fontsize=7, ncol=5)

    axis = axes[1][0]
    axis.plot(steps, 1e3 * series["index_middle_m"], color="#1f77b4",
              label="policy")
    axis.plot(steps, 1e3 * series["reference_index_middle_m"], color="#7f7f7f",
              ls="--", label="reference")
    reference_floor = 1e3 * series["reference_index_middle_m"].min()
    axis.axhline(reference_floor, color="#d62728", ls=":", lw=1.0,
                 label="reference minimum {:.0f} mm".format(reference_floor))
    axis.set_title("Index-middle fingertip separation")
    axis.set_ylabel("mm")
    axis.legend(fontsize=7)

    axis = axes[1][1]
    axis.semilogy(steps, np.maximum(series["rms_hand_action_rate"] ** 2, 1e-12),
                  color="#1f77b4", lw=0.8, label="policy")
    axis.axhline(DEMONSTRATION_HAND_ACTION_RATE_P99_MSE, color="#2ca02c",
                 ls="--", label="demonstration p99")
    axis.axhline(10.0 * DEMONSTRATION_HAND_ACTION_RATE_P99_MSE, color="#ff7f0e",
                 ls=":", label="10x demonstration")
    axis.set_title("Hand action rate (jitter), per-step MSE")
    axis.legend(fontsize=7)

    axis = axes[2][0]
    axis.plot(steps, np.rad2deg(series["rms_hand_position_error"]),
              color="#9467bd", label="hand joints")
    axis.plot(steps, np.rad2deg(series["palm_tilt_error_rad"]), color="#8c564b",
              label="palm tilt")
    axis.set_title("Joint-space tracking and palm tilt")
    axis.set_ylabel("deg")
    axis.set_xlabel("step")
    axis.legend(fontsize=7)

    axis = axes[2][1]
    axis.plot(steps, series["object_com_lift_m"], color="#1f77b4", label="lift")
    axis.plot(steps, series["object_position_error_m"], color="#d62728",
              label="cube position error")
    axis.set_title("Cuboid")
    axis.set_ylabel("m")
    axis.set_xlabel("step")
    axis.legend(fontsize=7)

    for row in axes:
        for cell in row:
            cell.grid(alpha=0.25)

    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(figure)
    buffer.seek(0)
    return iio.imread(buffer, extension=".png")


def summarise_diagnostics(frames, table_top_z):
    """The scalars the plots are there to make arguable, as a dict."""
    fingertip_z = np.stack([frame.fingertip_z for frame in frames])
    reference_fingertip_z = np.stack(
        [frame.reference_fingertip_z for frame in frames]
    )
    spread = np.asarray([frame.metrics["index_middle_m"] for frame in frames])
    reference_spread = np.asarray(
        [frame.metrics["reference_index_middle_m"] for frame in frames]
    )
    rate_mse = np.asarray(
        [frame.metrics["rms_hand_action_rate"] for frame in frames]
    ) ** 2
    depth = table_top_z - fingertip_z
    summary = {
        "table_top_z_m": float(table_top_z),
        "fingers": {},
        "index_middle_min_m": float(spread.min()),
        "reference_index_middle_min_m": float(reference_spread.min()),
        "hand_action_rate_mse_median": float(np.median(rate_mse)),
        "hand_action_rate_mse_p95": float(np.percentile(rate_mse, 95)),
        "hand_action_rate_vs_demonstration_p99": float(
            np.median(rate_mse) / DEMONSTRATION_HAND_ACTION_RATE_P99_MSE
        ),
        "mean_palm_keypoint_error_m": float(
            np.mean([f.metrics["palm_keypoint_error_m"] for f in frames])
        ),
        "mean_fingertip_keypoint_error_m": float(
            np.mean([f.metrics["fingertip_keypoint_error_m"] for f in frames])
        ),
        "mean_hand_position_error_rad": float(
            np.mean([f.metrics["rms_hand_position_error"] for f in frames])
        ),
        "mean_palm_tilt_error_deg": float(
            np.rad2deg(
                np.mean([f.metrics["palm_tilt_error_rad"] for f in frames])
            )
        ),
    }
    for index, name in enumerate(FINGER_NAMES):
        summary["fingers"][name] = {
            "max_penetration_mm": float(1e3 * depth[:, index].max()),
            "fraction_of_steps_below_table": float(
                (depth[:, index] > 0.0).mean()
            ),
            "reference_min_clearance_mm": float(
                1e3 * (reference_fingertip_z[:, index] - table_top_z).min()
            ),
        }
    return summary


class RolloutFrame:
    """One recorded simulator step, everything the GUI needs to redraw it."""

    __slots__ = (
        "step",
        "reference_index",
        "q",
        "cube_position",
        "cube_wxyz",
        "reference_q",
        "reference_cube_position",
        "reference_cube_wxyz",
        "fingertip_z",
        "reference_fingertip_z",
        "metrics",
    )

    def __init__(self, **fields) -> None:
        for name, value in fields.items():
            setattr(self, name, value)


class EvaluationViewer:
    def __init__(self, args: argparse.Namespace) -> None:
        install_path_is_relative_to_backport()
        try:
            import viser
            from viser.extras import ViserUrdf
        except ImportError as exc:  # pragma: no cover - environment guard
            raise RuntimeError(
                "This evaluator needs viser. Run it with "
                "/home/duplo/simone/SimToolReal/.venv/bin/python."
            ) from exc

        self.args = args
        self.checkpoint = args.checkpoint.expanduser().resolve()
        config_path = (
            args.config.expanduser().resolve()
            if args.config is not None
            else self.checkpoint.parent / "config.json"
        )
        if not config_path.is_file():
            raise FileNotFoundError(
                "Configuration not found: {}".format(config_path)
            )
        self.config_path = config_path
        self._build_environment()
        self._build_policy()

        self.lock = threading.Lock()
        self.frames: List[RolloutFrame] = []
        self.running = False
        self.stop_requested = False
        self.playing = False
        self.frame = 0
        self.last_tick = time.monotonic()
        self.final_metrics: Dict[str, float] = {}

        self.server = viser.ViserServer(host="0.0.0.0", port=args.port)
        self.port = int(self.server.get_port())
        self.server.scene.add_grid(
            "/ground", width=3.0, height=3.0, cell_size=0.1, position=(0, 0, 0)
        )
        table_size = np.asarray((0.75, 0.75, 0.30))
        table_center = np.asarray(
            (0.0, 0.0, 0.55 - 0.035 - table_size[2] / 2.0)
        )
        self.server.scene.add_box(
            "/table",
            dimensions=table_size,
            position=table_center,
            color=(209, 143, 89),
            opacity=0.9,
        )
        self.server.scene.add_box(
            "/reference_table",
            dimensions=table_size,
            position=table_center + self.ghost_offset,
            color=(209, 143, 89),
            opacity=0.35,
        )
        self.server.scene.add_frame(
            "/robot",
            position=ROBOT_POSITION_WORLD,
            wxyz=ROBOT_WXYZ_WORLD,
            show_axes=False,
        )
        self.server.scene.add_frame(
            "/reference_robot",
            position=ROBOT_POSITION_WORLD + self.ghost_offset,
            wxyz=ROBOT_WXYZ_WORLD,
            show_axes=False,
        )
        urdf = args.urdf.expanduser().resolve()
        self.robot = ViserUrdf(self.server, urdf, root_node_name="/robot")
        ghost_color = tuple(
            float(value) for value in self.env_cfg.viewer.reference_ghost_color
        )
        self.reference_robot = ViserUrdf(
            self.server,
            urdf,
            root_node_name="/reference_robot",
            mesh_color_override=ghost_color + (0.45,),
        )
        self.cube = self.server.scene.add_box(
            "/cube",
            dimensions=CUBE_DIMENSIONS,
            color=(65, 115, 210),
            opacity=0.95,
            side="double",
        )
        self.reference_cube = self.server.scene.add_box(
            "/reference_cube",
            dimensions=CUBE_DIMENSIONS,
            color=(65, 115, 210),
            opacity=0.35,
            side="double",
        )
        self._check_urdf_joint_order()
        self._build_gui()

        @self.server.on_client_connect
        def _(client):
            client.camera.position = (1.2, -1.3, 1.3)
            client.camera.look_at = (
                float(self.ghost_offset[0]) / 2.0,
                0.3,
                0.6,
            )

        self._show_reset_pose()

    # ------------------------------------------------------------------ setup

    def _build_environment(self) -> None:
        args = self.args
        env_cfg, train_cfg = load_saved_configuration(self.config_path)
        env_cfg.seed = int(args.seed)
        env_cfg.env.num_envs = int(args.num_envs)
        env_cfg.env.play = True
        env_cfg.viewer.enable_viewer = False
        # Viser draws the reference itself, so the simulator needs no ghost
        # actor: one fewer articulation per environment to step.
        env_cfg.viewer.reference_ghost = False
        env_cfg.viewer.training_camera_enabled = False
        # Terminations are the measurement here, not an obstacle to playback,
        # so they stay on and the GUI reports where each environment died. The
        # checkbox can switch this off to see what the policy would have done
        # past the failure.
        env_cfg.termination.enabled = True
        if float(args.object_assist_scale) <= 0.0:
            env_cfg.object_assist.enabled = False
        else:
            env_cfg.object_assist.schedule = "constant"
            env_cfg.object_assist.initial_scale = float(args.object_assist_scale)
        if int(env_cfg.env.num_observations) != int(
            SimToolRealCfg.env.num_observations
        ):
            raise ValueError(
                "This checkpoint was trained with a {}D observation but the "
                "current environment builds {}D".format(
                    env_cfg.env.num_observations,
                    SimToolRealCfg.env.num_observations,
                )
            )
        self.env_cfg = env_cfg
        self.train_cfg = train_cfg
        self.env = MotionImitationEnv(
            env_cfg,
            sim_device=args.sim_device,
            headless=True,
            num_envs_override=None,
        )
        self.env.max_episode_length = int(self.env.reference.last_index)
        self.env.cfg.env.episode_length = self.env.max_episode_length
        self.ghost_offset = np.asarray(
            [float(value) for value in env_cfg.viewer.reference_ghost_offset],
            dtype=np.float64,
        )
        self.table_top_z = float(env_cfg.init_state.pos[2]) - float(
            env_cfg.table.surface_below_robot_base_m
        )
        bank = self.env.transform_bank
        self.bank_translation = bank.translation.detach().cpu().numpy()
        self.bank_yaw_deg = np.rad2deg(bank.yaw_rad.detach().cpu().numpy())

    def _build_policy(self) -> None:
        runner = PPO(self.env, self.train_cfg, log_dir=None, device=self.env.device)
        self.checkpoint_infos = runner.load(
            self.checkpoint, load_optimizer=False, load_normalizers=True
        )
        self.policy = runner.get_inference_policy(device=self.env.device)

    def _check_urdf_joint_order(self) -> None:
        """The Viser robots are driven with demonstration-order joint vectors."""
        from simtoolreal_animrl.envs.controller import (
            ARM_JOINT_NAMES,
            HAND_JOINT_NAMES,
        )

        expected = tuple(ARM_JOINT_NAMES) + tuple(HAND_JOINT_NAMES)
        actual = tuple(self.robot.get_actuated_joint_names())
        if actual != expected:
            raise ValueError(
                "URDF actuated-joint order does not match demonstration order:\n"
                "  urdf: {}\n  demo: {}".format(actual, expected)
            )

    # -------------------------------------------------------------------- gui

    def _build_gui(self) -> None:
        gui = self.server.gui
        randomization = self.env_cfg.object_randomization
        gui.add_markdown(
            "# Measured evaluation\n"
            "`{}`\n\n"
            "Solid: the policy. Translucent, to the side: the retargeted "
            "reference it is scored against.".format(self.checkpoint.name)
        )
        with gui.add_folder("Cuboid placement", expand_by_default=True):
            self.xy = gui.add_vector3(
                "Position offset (m)",
                initial_value=(0.0, 0.0, 0.0),
                min=(-0.20, -0.20, 0.0),
                max=(0.20, 0.20, 0.0),
                step=0.005,
            )
            self.yaw = gui.add_slider(
                "Yaw (deg)",
                min=float(min(self.bank_yaw_deg.min(), -45.0)),
                max=float(max(self.bank_yaw_deg.max(), 90.0)),
                step=0.5,
                initial_value=0.0,
            )
            reset_transform = gui.add_button("Reset placement")
            reset_transform.on_click(self._reset_transform)
            self.placement_readout = gui.add_markdown("")
            self.xy.on_update(self._update_placement_readout)
            self.yaw.on_update(self._update_placement_readout)
        with gui.add_folder("Episode start", expand_by_default=True):
            self.start_frame = gui.add_slider(
                "RSI start frame",
                min=0,
                max=int(self.env.reference.last_index) - 1,
                step=1,
                initial_value=0,
            )
            self.start_frame.on_update(self._update_placement_readout)
            self.keep_going = gui.add_checkbox(
                "Ignore terminations (play to the end)", initial_value=False
            )
            self.stream = gui.add_checkbox("Stream while running", initial_value=True)
        with gui.add_folder("Rollout", expand_by_default=True):
            self.run_button = gui.add_button("Run", color="green")
            self.run_button.on_click(self._start_run)
            self.stop_button = gui.add_button("Stop", color="red", disabled=True)
            self.stop_button.on_click(self._request_stop)
            self.status = gui.add_markdown("**Status:** idle.")
        with gui.add_folder("Playback", expand_by_default=True):
            self.frame_slider = gui.add_slider(
                "Frame",
                min=0,
                max=int(self.env.reference.last_index),
                step=1,
                initial_value=0,
            )
            self.frame_slider.on_update(self._set_frame)
            self.play_button = gui.add_button("Play / pause")
            self.play_button.on_click(self._toggle_play)
            self.speed = gui.add_slider(
                "Speed", min=0.1, max=3.0, step=0.1, initial_value=1.0
            )
            self.loop = gui.add_checkbox("Loop", initial_value=True)
        with gui.add_folder("Measurements", expand_by_default=True):
            self.frame_readout = gui.add_markdown("**No rollout recorded yet.**")
            self.cohort_readout = gui.add_markdown("")
        with gui.add_folder("Diagnostics", expand_by_default=True):
            self.hand_readout = gui.add_markdown(
                "Run a rollout to measure the hand."
            )
            self.diagnostics_image = None
            self.diagnostics_note = gui.add_markdown("")
        self._update_placement_readout()
        print(
            "Training range: x [{:+.3f}, {:+.3f}] m, y [{:+.3f}, {:+.3f}] m, "
            "yaw [{:+.1f}, {:+.1f}] deg".format(
                randomization.translation_x_min_m,
                randomization.translation_x_max_m,
                randomization.translation_y_min_m,
                randomization.translation_y_max_m,
                randomization.yaw_min_deg,
                randomization.yaw_max_deg,
            )
        )

    def _reset_transform(self, _) -> None:
        self.xy.value = (0.0, 0.0, 0.0)
        self.yaw.value = 0.0

    def _requested_transform(self):
        offset = np.asarray(self.xy.value, dtype=np.float64)
        return (
            np.asarray((offset[0], offset[1], 0.0)),
            float(np.deg2rad(float(self.yaw.value))),
        )

    def _nearest_bank_entry(self, translation: np.ndarray, yaw_rad: float):
        index = int(
            nearest_transform_indices(
                torch.as_tensor(translation, dtype=torch.float32).unsqueeze(0),
                torch.as_tensor([yaw_rad], dtype=torch.float32),
                torch.as_tensor(self.bank_translation, dtype=torch.float32),
                torch.as_tensor(
                    np.deg2rad(self.bank_yaw_deg), dtype=torch.float32
                ),
                float(
                    self.env_cfg.object_randomization.nearest_yaw_lever_arm_m
                ),
            )[0]
        )
        bank_translation = self.bank_translation[index]
        bank_yaw_deg = float(self.bank_yaw_deg[index])
        translation_residual_m = float(
            np.linalg.norm(bank_translation[:2] - translation[:2])
        )
        yaw_residual_deg = float(
            (bank_yaw_deg - np.rad2deg(yaw_rad) + 180.0) % 360.0 - 180.0
        )
        return index, bank_translation, bank_yaw_deg, translation_residual_m, yaw_residual_deg

    def _update_placement_readout(self, _=None) -> None:
        translation, yaw_rad = self._requested_transform()
        (
            index,
            bank_translation,
            bank_yaw_deg,
            translation_residual_m,
            yaw_residual_deg,
        ) = self._nearest_bank_entry(translation, yaw_rad)
        randomization = self.env_cfg.object_randomization
        inside = (
            randomization.translation_x_min_m <= translation[0] <= randomization.translation_x_max_m
            and randomization.translation_y_min_m <= translation[1] <= randomization.translation_y_max_m
            and randomization.yaw_min_deg <= np.rad2deg(yaw_rad) <= randomization.yaw_max_deg
        )
        warning = ""
        if (
            translation_residual_m > TRAINING_TRANSLATION_RESIDUAL_P99_M
            or abs(yaw_residual_deg) > TRAINING_YAW_RESIDUAL_P99_DEG
        ):
            warning = (
                "  \n⚠️ This placement sits further from its serving reference "
                "than 99% of training episodes did, so the arm reference "
                "describes it less well than anything the policy was trained "
                "on. The bank only admits transforms whose whole clip solves, "
                "so beyond its coverage the residual grows without bound."
            )
        self.placement_readout.content = (
            "**Requested cuboid transform:** x={:+.3f}, y={:+.3f} m, "
            "yaw={:+.1f}°  \n"
            "**Serving bank reference #{}:** x={:+.3f}, y={:+.3f} m, "
            "yaw={:+.1f}°  \n"
            "**Nearest-bank residual:** {:.1f} mm, {:+.2f}° "
            "(training median {:.1f} mm, {:.1f}°)  \n"
            "**Training distribution:** {}{}".format(
                translation[0],
                translation[1],
                np.rad2deg(yaw_rad),
                index,
                bank_translation[0],
                bank_translation[1],
                bank_yaw_deg,
                1e3 * translation_residual_m,
                yaw_residual_deg,
                1e3 * TRAINING_TRANSLATION_RESIDUAL_MEDIAN_M,
                TRAINING_YAW_RESIDUAL_MEDIAN_DEG,
                "inside" if inside else "**outside** (extrapolation)",
                warning,
            )
        )

    def _set_frame(self, event) -> None:
        with self.lock:
            self.frame = int(event.target.value)
            self.last_tick = time.monotonic()
            available = len(self.frames)
        if available:
            self._publish(min(self.frame, available - 1))

    def _toggle_play(self, _) -> None:
        with self.lock:
            self.playing = not self.playing
            self.last_tick = time.monotonic()

    def _request_stop(self, _) -> None:
        with self.lock:
            self.stop_requested = True

    # ----------------------------------------------------------------- rollout

    def _start_run(self, _) -> None:
        with self.lock:
            if self.running:
                return
            self.running = True
            self.stop_requested = False
            self.playing = False
            self.frames = []
        translation, yaw_rad = self._requested_transform()
        start_index = int(self.start_frame.value)
        self.run_button.disabled = True
        self.stop_button.disabled = False
        self.status.content = "**Status:** rollout running…"
        threading.Thread(
            target=self._rollout,
            args=(translation, yaw_rad, start_index),
            daemon=True,
        ).start()

    def _reference_state(self, env_index: int = 0):
        """The bank reference the policy is scored against, at this step."""
        env = self.env
        env_ids = torch.tensor([env_index], device=env.device, dtype=torch.long)
        sample = env.transform_bank.sample(
            env.transform_index[env_ids], env.reference_index[env_ids]
        )
        root_state = env._cube_reference_root_states(sample, env_ids)
        return (
            sample.q[0].detach().cpu().numpy().astype(np.float64),
            root_state[0, 0:3].detach().cpu().numpy().astype(np.float64),
            quaternion_xyzw_to_wxyz(
                root_state[0, 3:7].detach().cpu().numpy().astype(np.float64)
            ),
        )

    def _fingertip_heights(self):
        """Policy and reference fingertip world z for environment 0.

        The reference fingertips come from the bank's cube-frame keypoints
        mapped through the cuboid's *measured* pose -- the same anchor the
        reward uses, so "where the fingers should be" is read exactly as the
        reward reads it, with no forward kinematics of its own.
        """
        from simtoolreal_animrl.envs.keypoints import split_palm_and_fingertips
        from simtoolreal_animrl.envs.motion_imitation import _quat_rotate

        env = self.env
        actual = env._fingertip_positions_world()[0]
        reference_keypoints = env.transform_bank.keypoints_at(env.reference_index)
        _, reference_fingertips = split_palm_and_fingertips(reference_keypoints)
        orientation = env.canonical_cube_orientation()[0]
        reference_world = env.cube_position[0].unsqueeze(0) + _quat_rotate(
            orientation.unsqueeze(0).expand(reference_fingertips.shape[1], 4),
            reference_fingertips[0],
        )
        return (
            actual[:, 2].detach().cpu().numpy().astype(np.float64),
            reference_world[:, 2].detach().cpu().numpy().astype(np.float64),
            float((actual[1] - actual[2]).norm()),
            float((reference_world[1] - reference_world[2]).norm()),
        )

    def _capture(self, step: int, metrics: Dict[str, float]) -> RolloutFrame:
        env = self.env
        reference_q, reference_cube_position, reference_cube_wxyz = (
            self._reference_state()
        )
        fingertip_z, reference_fingertip_z, spread, reference_spread = (
            self._fingertip_heights()
        )
        if metrics:
            metrics["index_middle_m"] = spread
            metrics["reference_index_middle_m"] = reference_spread
        return RolloutFrame(
            step=step,
            reference_index=int(env.reference_index[0]),
            q=env.q[0].detach().cpu().numpy().astype(np.float64),
            cube_position=env.cube_position[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64),
            cube_wxyz=quaternion_xyzw_to_wxyz(
                env.cube_orientation[0].detach().cpu().numpy().astype(np.float64)
            ),
            reference_q=reference_q,
            reference_cube_position=reference_cube_position,
            reference_cube_wxyz=reference_cube_wxyz,
            fingertip_z=fingertip_z,
            reference_fingertip_z=reference_fingertip_z,
            metrics=metrics,
        )

    def _rollout(
        self, translation: np.ndarray, yaw_rad: float, start_index: int
    ) -> None:
        env = self.env
        started = time.monotonic()
        try:
            with self.lock:
                # Owned here, not by the caller: a stale buffer would relabel
                # the previous rollout's frames as this one's.
                self.frames = []
                self.frame = 0
            self.final_metrics = {}
            env.cfg.termination.enabled = not bool(self.keep_going.value)
            env.reset(
                reference_index=start_index,
                translation_xy=(float(translation[0]), float(translation[1])),
                yaw_rad=float(yaw_rad),
            )
            observations = env.get_observations()
            horizon = int(env.reference.last_index) - int(start_index)
            alive = torch.ones(
                env.num_envs, dtype=torch.bool, device=env.device
            )
            peak_lift = torch.zeros(env.num_envs, device=env.device)
            last_lift = torch.zeros(env.num_envs, device=env.device)
            initial_lift = None
            palm_error_sum = torch.zeros(env.num_envs, device=env.device)
            tilt_error_sum = torch.zeros(env.num_envs, device=env.device)
            alive_steps = torch.zeros(env.num_envs, device=env.device)
            death_step = torch.full(
                (env.num_envs,), -1, dtype=torch.long, device=env.device
            )
            death_reason = ["alive"] * env.num_envs
            last_publish = 0.0
            recorded_env_zero = True

            with torch.inference_mode():
                for step in range(1, horizon + 1):
                    with self.lock:
                        if self.stop_requested:
                            break
                        streaming = bool(self.stream.value)
                    actions = self.policy(observations)
                    observations, _, rewards, dones, infos = env.step(actions)

                    if initial_lift is None:
                        # Starting mid-clip hands the policy a cube that is
                        # already off the table, so peak lift alone says
                        # nothing; the lift it inherited is reported beside it.
                        initial_lift = infos["object_com_lift_m"].clone()
                    peak_lift = torch.where(
                        alive,
                        torch.maximum(peak_lift, infos["object_com_lift_m"]),
                        peak_lift,
                    )
                    last_lift = torch.where(
                        alive, infos["object_com_lift_m"], last_lift
                    )
                    palm_error_sum += torch.where(
                        alive,
                        infos["palm_keypoint_error_m"],
                        torch.zeros_like(palm_error_sum),
                    )
                    tilt_error_sum += torch.where(
                        alive,
                        infos["palm_tilt_error_rad"],
                        torch.zeros_like(tilt_error_sum),
                    )
                    alive_steps += alive.float()

                    newly_done = dones & alive
                    if bool(newly_done.any()):
                        for index in newly_done.nonzero(as_tuple=False).flatten():
                            index = int(index)
                            death_step[index] = step
                            death_reason[index] = self._termination_reason(
                                infos, index
                            )
                        alive = alive & ~dones

                    if recorded_env_zero:
                        if bool(dones[0]):
                            # step() resets a finished environment before it
                            # returns, so by now environment 0 already holds the
                            # NEXT episode's RSI pose: this step's pose is gone
                            # and recording it would label the new episode's
                            # reset as this episode's final frame. The metrics
                            # are cloned pre-reset, so they are kept as the
                            # termination reading instead.
                            self.final_metrics = self._frame_metrics(
                                infos, rewards
                            )
                            recorded_env_zero = False
                        else:
                            frame = self._capture(
                                step, self._frame_metrics(infos, rewards)
                            )
                            with self.lock:
                                self.frames.append(frame)
                                available = len(self.frames)
                            now = time.monotonic()
                            if streaming and now - last_publish > 1.0 / 30.0:
                                last_publish = now
                                self._publish(available - 1)
                                self.frame_slider.value = available - 1
                    if not bool(alive.any()) and not recorded_env_zero:
                        break

            elapsed = time.monotonic() - started
            self._finish(
                elapsed,
                start_index,
                horizon,
                alive,
                initial_lift,
                last_lift,
                peak_lift,
                palm_error_sum,
                tilt_error_sum,
                alive_steps,
                death_step,
                death_reason,
            )
        except Exception as exc:  # pragma: no cover - surfaced in the browser
            self.status.content = "## Rollout error\n`{}: {}`".format(
                type(exc).__name__, exc
            )
            raise
        finally:
            with self.lock:
                self.running = False
                self.stop_requested = False
            self.run_button.disabled = False
            self.stop_button.disabled = True

    @staticmethod
    def _termination_reason(infos, index: int) -> str:
        if bool(infos["time_outs"][index]):
            return "time out"
        reasons = []
        if bool(infos["arm_threshold_violation"][index]):
            reasons.append("palm keypoint")
        if bool(infos["hand_threshold_violation"][index]):
            reasons.append("hand joints")
        if bool(infos["object_threshold_violation"][index]):
            reasons.append("object")
        return " + ".join(reasons) if reasons else "unattributed"

    @staticmethod
    def _frame_metrics(infos, rewards) -> Dict[str, float]:
        return {
            "reward": float(rewards[0]),
            "palm_keypoint_error_m": float(infos["palm_keypoint_error_m"][0]),
            "fingertip_keypoint_error_m": float(
                infos["fingertip_keypoint_error_m"][0]
            ),
            "palm_tilt_error_rad": float(infos["palm_tilt_error_rad"][0]),
            "object_com_lift_m": float(infos["object_com_lift_m"][0]),
            "object_position_error_m": float(infos["object_position_error_m"][0]),
            "ik_residual_norm": float(infos["ik_residual_norm"][0]),
            "rms_hand_position_error": float(
                infos["rms_hand_position_error"][0]
            ),
            "rms_hand_action_rate": float(infos["rms_hand_action_rate"][0]),
            "arm_joint_delta_clipped": float(
                infos["arm_joint_delta_clipped"][0]
            ),
        }

    def _finish(
        self,
        elapsed: float,
        start_index: int,
        horizon: int,
        alive,
        initial_lift,
        last_lift,
        peak_lift,
        palm_error_sum,
        tilt_error_sum,
        alive_steps,
        death_step,
        death_reason,
    ) -> None:
        with self.lock:
            recorded = len(self.frames)
        steps = alive_steps.clamp_min(1.0)
        mean_palm_error = (palm_error_sum / steps).detach().cpu().numpy()
        mean_tilt_error = (tilt_error_sum / steps).detach().cpu().numpy()
        lift = peak_lift.detach().cpu().numpy()
        final_lift = last_lift.detach().cpu().numpy()
        inherited_lift = (
            float(initial_lift.mean()) if initial_lift is not None else 0.0
        )
        survived = alive.detach().cpu().numpy()
        deaths = death_step.detach().cpu().numpy()
        holding = int(np.sum(final_lift >= SUCCESS_LIFT_M))
        reasons: Dict[str, int] = {}
        for index, reason in enumerate(death_reason):
            if not survived[index]:
                reasons[reason] = reasons.get(reason, 0) + 1
        reason_text = (
            ", ".join(
                "{} x{}".format(name, count)
                for name, count in sorted(
                    reasons.items(), key=lambda item: -item[1]
                )
            )
            or "none"
        )
        died = deaths[deaths >= 0]
        termination_note = ""
        if self.final_metrics:
            termination_note = (
                "  \nEnvironment 0 ended at palm keypoint error "
                "{:.3f} m, palm tilt {:.1f}°, cube lift {:.3f} m. Its pose at "
                "that step is not recoverable: Isaac Gym resets a finished "
                "environment inside step().".format(
                    self.final_metrics["palm_keypoint_error_m"],
                    np.rad2deg(self.final_metrics["palm_tilt_error_rad"]),
                    self.final_metrics["object_com_lift_m"],
                )
            )
        self.status.content = (
            "**Status:** rollout complete in {:.1f} s "
            "({} recorded frames for environment 0).{}".format(
                elapsed, recorded, termination_note
            )
        )
        self.cohort_readout.content = (
            "## Cohort of {} environments\n"
            "Start frame **{}** → {} ({} transitions).  \n"
            "**Survived to the end:** {}/{}  \n"
            "**Still holding the cube ≥ {:.0f} cm at their last step:** "
            "{}/{}  \n"
            "**Cube lift inherited from the start pose:** {:.3f} m  \n"
            "**Peak cube lift:** mean {:.3f} m, best {:.3f} m, worst {:.3f} m  \n"
            "**Final cube lift:** mean {:.3f} m  \n"
            "**Mean palm keypoint error:** {:.3f} m (worst environment "
            "{:.3f} m)  \n"
            "**Mean palm tilt error:** {:.1f}° (worst {:.1f}°)  \n"
            "**Terminations:** {}  \n"
            "**First termination at step:** {}".format(
                self.env.num_envs,
                start_index,
                int(self.env.reference.last_index),
                horizon,
                int(survived.sum()),
                self.env.num_envs,
                100.0 * SUCCESS_LIFT_M,
                holding,
                self.env.num_envs,
                inherited_lift,
                float(lift.mean()),
                float(lift.max()),
                float(lift.min()),
                float(final_lift.mean()),
                float(mean_palm_error.mean()),
                float(mean_palm_error.max()),
                float(np.rad2deg(mean_tilt_error.mean())),
                float(np.rad2deg(mean_tilt_error.max())),
                reason_text,
                int(died.min()) if died.size else "—",
            )
        )
        if recorded:
            self._publish(recorded - 1)
            self.frame_slider.value = recorded - 1
            self._build_diagnostics(start_index)

    def _build_diagnostics(self, start_index: int) -> None:
        """Render the per-step figure and write it, with its scalars, to disk."""
        with self.lock:
            frames = list(self.frames)
        if not frames or not frames[0].metrics:
            return
        rewards_cfg = self.env_cfg.rewards
        try:
            image = render_diagnostics(
                frames,
                self.table_top_z,
                float(rewards_cfg.fingertip_keypoint_std_m),
                float(rewards_cfg.palm_keypoint_std_m),
            )
        except Exception as exc:  # pragma: no cover - surfaced in the browser
            self.diagnostics_note.content = "Plotting failed: `{}: {}`".format(
                type(exc).__name__, exc
            )
            return
        if self.diagnostics_image is None:
            self.diagnostics_image = self.server.gui.add_image(
                image, label="Rollout diagnostics"
            )
        else:
            self.diagnostics_image.image = image

        summary = summarise_diagnostics(frames, self.table_top_z)
        penetrating = {
            name: values
            for name, values in summary["fingers"].items()
            if values["max_penetration_mm"] > 0.0
        }
        if penetrating:
            table_line = ", ".join(
                "{} {:.0f} mm ({:.0f}% of steps)".format(
                    name,
                    values["max_penetration_mm"],
                    100.0 * values["fraction_of_steps_below_table"],
                )
                for name, values in penetrating.items()
            )
        else:
            table_line = "none"
        self.hand_readout.content = (
            "**Fingertips below the table:** {}  \n"
            "**Closest index-middle separation:** {:.1f} mm "
            "(reference never below {:.1f} mm)  \n"
            "**Hand action rate:** {:.0f}x the demonstration's p99 (median), "
            "{:.0f}x (p95)  \n"
            "**Mean hand joint error:** {:.1f}°  \n"
            "**Mean palm / fingertip keypoint error:** {:.3f} / {:.3f} m".format(
                table_line,
                1e3 * summary["index_middle_min_m"],
                1e3 * summary["reference_index_middle_min_m"],
                summary["hand_action_rate_vs_demonstration_p99"],
                summary["hand_action_rate_mse_p95"]
                / DEMONSTRATION_HAND_ACTION_RATE_P99_MSE,
                np.rad2deg(summary["mean_hand_position_error_rad"]),
                summary["mean_palm_keypoint_error_m"],
                summary["mean_fingertip_keypoint_error_m"],
            )
        )

        translation, yaw_rad = self._requested_transform()
        stem = "viser_{}_x{:+.0f}_y{:+.0f}_yaw{:+.0f}_rsi{}".format(
            self.checkpoint.stem,
            1e3 * translation[0],
            1e3 * translation[1],
            np.rad2deg(yaw_rad),
            start_index,
        )
        directory = self.checkpoint.parent / "eval_viser"
        directory.mkdir(parents=True, exist_ok=True)
        import imageio.v3 as iio

        image_path = directory / (stem + ".png")
        iio.imwrite(str(image_path), image)
        summary.update(
            {
                "checkpoint": str(self.checkpoint),
                "requested_translation_m": [
                    float(translation[0]),
                    float(translation[1]),
                ],
                "requested_yaw_deg": float(np.rad2deg(yaw_rad)),
                "rsi_start_index": int(start_index),
                "recorded_steps": len(frames),
                "cohort": self.cohort_readout.content,
            }
        )
        summary_path = directory / (stem + ".json")
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        self.diagnostics_note.content = (
            "Saved `{}` and `{}`.".format(image_path.name, summary_path.name)
        )
        print("diagnostics: {}".format(image_path))

    # --------------------------------------------------------------- rendering

    def _show_reset_pose(self) -> None:
        """Draw the reset state so the scene is populated before the first run."""
        translation, yaw_rad = self._requested_transform()
        self.env.reset(
            reference_index=int(self.start_frame.value),
            translation_xy=(float(translation[0]), float(translation[1])),
            yaw_rad=float(yaw_rad),
        )
        with self.lock:
            self.frames = [self._capture(0, {})]
        self._publish(0)

    def _publish(self, index: int) -> None:
        with self.lock:
            if not self.frames:
                return
            frame = self.frames[min(index, len(self.frames) - 1)]
            total = len(self.frames)
        self.robot.update_cfg(frame.q)
        self.cube.position = frame.cube_position
        self.cube.wxyz = frame.cube_wxyz
        self.reference_robot.update_cfg(frame.reference_q)
        self.reference_cube.position = frame.reference_cube_position + self.ghost_offset
        self.reference_cube.wxyz = frame.reference_cube_wxyz
        metrics = frame.metrics
        if not metrics:
            self.frame_readout.content = (
                "**Reset pose shown.** Reference sample {}. Press **Run** to "
                "measure a rollout.".format(frame.reference_index)
            )
            return
        self.frame_readout.content = (
            "**Step {}/{}** (reference sample {})  \n"
            "**Reward:** {:.4f}  \n"
            "**Palm keypoint error:** {:.4f} m  \n"
            "**Fingertip keypoint error:** {:.4f} m  \n"
            "**Palm tilt error:** {:.2f}°  \n"
            "**Cube lift:** {:.4f} m  \n"
            "**Cube position error:** {:.4f} m  \n"
            "**IK residual:** {:.2e}  \n"
            "**IK clamp saturation:** {:.2f}".format(
                frame.step,
                total,
                frame.reference_index,
                metrics["reward"],
                metrics["palm_keypoint_error_m"],
                metrics["fingertip_keypoint_error_m"],
                np.rad2deg(metrics["palm_tilt_error_rad"]),
                metrics["object_com_lift_m"],
                metrics["object_position_error_m"],
                metrics["ik_residual_norm"],
                metrics["arm_joint_delta_clipped"],
            )
        )

    # -------------------------------------------------------------------- loop

    def run(self) -> None:
        print("Open http://localhost:{}".format(self.port))
        try:
            while True:
                time.sleep(1.0 / 120.0)
                with self.lock:
                    if self.running or not self.playing or not self.frames:
                        continue
                    now = time.monotonic()
                    advance = int(
                        (now - self.last_tick) * 60.0 * float(self.speed.value)
                    )
                    if advance <= 0:
                        continue
                    total = len(self.frames)
                    next_frame = self.frame + advance
                    if next_frame >= total:
                        if self.loop.value:
                            next_frame %= total
                        else:
                            next_frame = total - 1
                            self.playing = False
                    self.last_tick = now
                self.frame_slider.value = int(next_frame)
        except KeyboardInterrupt:
            print("Evaluator stopped.")
        finally:
            self.env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Defaults to config.json next to the checkpoint.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=16,
        help=(
            "Environments per rollout. They all receive the chosen placement "
            "and start frame, so the spread across them is the stochasticity "
            "the run was trained with; environment 0 is the one rendered."
        ),
    )
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument(
        "--object-assist-scale",
        type=float,
        default=0.0,
        help=(
            "Object-assist scale for this evaluation. The default of 0 "
            "measures the policy on the unassisted task."
        ),
    )
    return parser.parse_args()


def main() -> None:
    EvaluationViewer(parse_args()).run()


if __name__ == "__main__":
    main()
