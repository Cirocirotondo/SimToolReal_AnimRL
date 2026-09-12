#!/usr/bin/env python3
"""Interactive Viser viewer for the recorded robot/cuboid demonstration.

The robot and palm trajectory always show the recorded demonstration.  The
object controls apply one planar rigid transform to the complete recorded
cuboid trajectory, so the object keeps its demonstrated lift after its reset
position and yaw are changed.
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
DEFAULT_DEMO = REPO_ROOT / "demonstrations" / (
    "demo_20260727_152551_335339_60hz_cube_collision_resolved_stable_grasp.npz"
)
DEFAULT_URDF = REPO_ROOT / "assets/urdf/ur5e_delto_description/ur5e_right_dg5f_mount_60deg.urdf"
DEFAULT_FEASIBILITY_GRID = REPO_ROOT / "banks/feasibility_grid.npz"

# These are the frame conventions used by MotionImitationEnv.
ROBOT_POSITION_WORLD = np.asarray((0.0, 0.6, 0.55), dtype=np.float64)
ROBOT_WXYZ_WORLD = np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64)
DEMO_POSITION_AXIS_SIGN = np.asarray((-1.0, -1.0, 1.0), dtype=np.float64)


def install_path_is_relative_to_backport() -> None:
    """Provide the Python 3.9 Path API required by Viser's HTTP server."""
    if hasattr(Path, "is_relative_to"):
        return

    def _is_relative_to(self: Path, *other: Path) -> bool:
        try:
            self.relative_to(*other)
            return True
        except ValueError:
            return False

    Path.is_relative_to = _is_relative_to  # type: ignore[attr-defined]


def _normalize_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm <= 1e-12):
        raise ValueError("Quaternion must have nonzero norm")
    return q / norm


def demo_cube_pose_to_world(pose_xyzw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Match MotionImitationEnv's recorded UR-base to world conversion."""
    pose = np.asarray(pose_xyzw, dtype=np.float64)
    if pose.shape != (7,):
        raise ValueError("Expected cube pose with shape (7,)")
    x, y, z, w = _normalize_xyzw(pose[3:7])
    world_xyzw = _normalize_xyzw(np.asarray((-y, x, w, -z)))
    world_position = ROBOT_POSITION_WORLD + pose[:3] * DEMO_POSITION_AXIS_SIGN
    return world_position, world_xyzw[[3, 0, 1, 2]]


def yaw_wxyz(yaw_rad: float) -> np.ndarray:
    return np.asarray(
        (np.cos(yaw_rad / 2.0), 0.0, 0.0, np.sin(yaw_rad / 2.0)),
        dtype=np.float64,
    )


def multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return _normalize_xyzw(
        np.asarray(
            (
                lw * rw - lx * rx - ly * ry - lz * rz,
                lw * rx + lx * rw + ly * rz - lz * ry,
                lw * ry - lx * rz + ly * rw + lz * rx,
                lw * rz + lx * ry - ly * rx + lz * rw,
            )
        )
    )


def rotate_z(points: np.ndarray, yaw_rad: float) -> np.ndarray:
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    rotation = np.asarray(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))
    return np.asarray(points) @ rotation.T


def transformed_cube_pose(
    positions: np.ndarray,
    orientations_wxyz: np.ndarray,
    frame: int,
    translation_xy: np.ndarray,
    yaw_offset_rad: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply a reset-centred planar transform to one cube trajectory frame."""
    initial = positions[0]
    relative = positions[frame] - initial
    position = initial + rotate_z(relative, yaw_offset_rad)
    position[:2] += np.asarray(translation_xy, dtype=np.float64)
    orientation = multiply_wxyz(yaw_wxyz(yaw_offset_rad), orientations_wxyz[frame])
    return position, orientation


def transform_points_about_cube_start(
    points: np.ndarray,
    cube_start: np.ndarray,
    translation_xy: np.ndarray,
    yaw_offset_rad: float,
) -> np.ndarray:
    """Apply the cuboid's planar rigid transform to world-space points."""
    transformed = cube_start + rotate_z(points - cube_start, yaw_offset_rad)
    transformed = np.asarray(transformed, dtype=np.float64)
    transformed[..., :2] += np.asarray(translation_xy, dtype=np.float64)
    return transformed


def load_demo(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(str(path)) as archive:
        q = np.concatenate(
            (archive["arm_q"], archive["hand_q_measured"]), axis=1
        ).astype(np.float64)
        cube_pose = np.asarray(archive["cube_pose"], dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 26 or cube_pose.shape != (len(q), 7):
        raise ValueError("Expected an N x 26 demonstration and an N x 7 cube pose")
    return q, cube_pose


def palm_trajectory_world(urdf_path: Path, q: np.ndarray) -> np.ndarray:
    """Compute the recorded rl_dg_palm origins using the URDF itself."""
    import yourdfpy

    model = yourdfpy.URDF.load(
        urdf_path,
        build_scene_graph=True,
        load_meshes=False,
        load_collision_meshes=False,
    )
    if tuple(model.actuated_joint_names) != tuple(
        (
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
            *(f"rj_dg_{finger}_{joint}" for finger in range(1, 6) for joint in range(1, 5)),
        )
    ):
        raise ValueError("URDF actuated-joint order does not match the demonstration")
    trajectory = np.empty((len(q), 3), dtype=np.float64)
    for index, configuration in enumerate(q):
        model.update_cfg(configuration)
        palm_in_base = model.get_transform("rl_dg_palm", "base_link")
        trajectory[index] = ROBOT_POSITION_WORLD + palm_in_base[:3, 3]
    return trajectory


def check_robot_collisions(
    urdf_path: Path, joint_trajectory: np.ndarray
) -> Tuple[dict, dict]:
    """Return hand-arm and arm-arm penetrating link pairs along a trajectory.

    MuJoCo compiles the URDF collision meshes and performs a kinematic forward
    pass only; no dynamics are simulated. Parent-child collisions are excluded
    by MuJoCo. The two known wrist/hand mesh intersections are ignored exactly
    as they are by ``configure_asset_wrist_collision_filters()``.
    """
    import mujoco

    urdf_path = Path(urdf_path).resolve()
    text = urdf_path.read_text(encoding="utf-8")
    if "<mujoco>" not in text:
        text = re.sub(
            r'(<robot\s+name="[^"]+">)',
            r'\1\n  <mujoco><compiler strippath="false"/></mujoco>',
            text,
            count=1,
        )

    def absolute_mesh_path(match) -> str:
        filename = match.group(1)
        if filename.startswith("urdf/"):
            path = REPO_ROOT / "assets" / filename
        else:
            path = urdf_path.parent / filename
        return f'filename="{path.resolve()}"'

    text = re.sub(r'filename="([^"]+)"', absolute_mesh_path, text)
    geometry_counts = {"visual": 0, "collision": 0}

    def name_geometry(match) -> str:
        kind = match.group(1)
        index = geometry_counts[kind]
        geometry_counts[kind] += 1
        return f'<{kind} name="{kind}_{index}">'

    text = re.sub(r"<(visual|collision)>", name_geometry, text)
    ignored = {
        frozenset(("wrist_3_link", "rl_dg_1_2")),
        frozenset(("wrist_3_link", "rl_dg_4_2")),
    }
    reports = {"hand_arm": {}, "arm_arm": {}}

    with tempfile.TemporaryDirectory(prefix="animrl_collision_") as directory:
        compatible = Path(directory) / "robot.urdf"
        compatible.write_text(text, encoding="utf-8")
        model = mujoco.MjModel.from_xml_path(str(compatible))
        data = mujoco.MjData(model)
        if model.nq != joint_trajectory.shape[1]:
            raise ValueError(
                f"MuJoCo model has {model.nq} coordinates, expected "
                f"{joint_trajectory.shape[1]}"
            )
        model.geom_contype[:] = 1
        model.geom_conaffinity[:] = 1

        for frame, configuration in enumerate(joint_trajectory):
            data.qpos[:] = configuration
            mujoco.mj_forward(model, data)
            frame_pairs = set()
            for contact_index in range(data.ncon):
                contact = data.contact[contact_index]
                if float(contact.dist) >= -1e-6:
                    continue
                body_a = model.body(
                    int(model.geom_bodyid[int(contact.geom1)])
                ).name
                body_b = model.body(
                    int(model.geom_bodyid[int(contact.geom2)])
                ).name
                pair_set = frozenset((body_a, body_b))
                if len(pair_set) != 2 or "world" in pair_set or pair_set in ignored:
                    continue
                hand_a = body_a.startswith("rl_dg_")
                hand_b = body_b.startswith("rl_dg_")
                if hand_a == hand_b:
                    # Hand-hand is outside the requested diagnostic; arm-arm
                    # requires both links to be outside the DG5F subtree.
                    category = "arm_arm" if not hand_a else None
                else:
                    category = "hand_arm"
                if category is None:
                    continue
                pair = tuple(sorted((body_a, body_b)))
                key = (category, pair)
                report = reports[category].setdefault(
                    pair,
                    {
                        "first_frame": frame,
                        "deepest_frame": frame,
                        "depth_m": -float(contact.dist),
                        "frames": 0,
                    },
                )
                depth = -float(contact.dist)
                if depth > report["depth_m"]:
                    report["depth_m"] = depth
                    report["deepest_frame"] = frame
                if key not in frame_pairs:
                    report["frames"] += 1
                    frame_pairs.add(key)
    return reports["hand_arm"], reports["arm_arm"]


def format_collision_report(label: str, reports: dict) -> str:
    if not reports:
        return f"**{label}:** nessuna"
    lines = [f"**{label}:** {len(reports)} coppie"]
    ordered = sorted(reports.items(), key=lambda item: -item[1]["depth_m"])
    for pair, report in ordered[:5]:
        lines.append(
            f"- `{pair[0]}` ↔ `{pair[1]}`: profondità max "
            f"{1e3 * report['depth_m']:.2f} mm al frame "
            f"{report['deepest_frame']} ({report['frames']} frame)"
        )
    if len(ordered) > 5:
        lines.append(f"- …e altre {len(ordered) - 5} coppie")
    return "  \n".join(lines)


class DemoViewer:
    def __init__(
        self, demo: Path, urdf: Path, feasibility_grid: Path, port: int
    ) -> None:
        # The project Viser environment uses Python 3.8, while Viser's static
        # HTTP handler calls Path.is_relative_to(), introduced in Python 3.9.
        # This must run before the first browser request reaches the server.
        install_path_is_relative_to_backport()
        try:
            import viser
            from viser.extras import ViserUrdf
        except ImportError as exc:
            raise RuntimeError(
                "This viewer needs viser. Run it with /home/simone/.venv/bin/python."
            ) from exc

        self.viser = viser
        self.frame_demo_path = demo
        self.q, cube_demo = load_demo(demo)
        self.display_q = self.q.copy()
        converted = [demo_cube_pose_to_world(pose) for pose in cube_demo]
        self.cube_positions = np.stack([item[0] for item in converted])
        self.cube_orientations = np.stack([item[1] for item in converted])
        self.palm_positions = palm_trajectory_world(urdf, self.q)
        self.urdf_path = urdf
        self._load_feasibility_grid(feasibility_grid)
        self.ik_running = False
        self.collision_running = False
        self.ik_solution_current = False
        self.frame = 0
        self.translation_xy = np.zeros(2, dtype=np.float64)
        self.yaw_offset_rad = 0.0
        self.playing = False
        self.last_tick = time.monotonic()
        self.lock = threading.Lock()

        self.server = viser.ViserServer(host="0.0.0.0", port=port)
        self.port = int(self.server.get_port())
        self.server.scene.add_grid(
            "/ground", width=2.0, height=2.0, cell_size=0.1, position=(0, 0, 0)
        )
        table_size = np.asarray((0.75, 0.75, 0.30))
        table_center = np.asarray((0.0, 0.0, 0.55 - 0.035 - table_size[2] / 2.0))
        self.server.scene.add_box(
            "/table", dimensions=table_size, position=table_center,
            color=(209, 143, 89), opacity=0.9,
        )
        self.server.scene.add_frame(
            "/robot", position=ROBOT_POSITION_WORLD, wxyz=ROBOT_WXYZ_WORLD,
            show_axes=False,
        )
        self.robot = ViserUrdf(self.server, urdf, root_node_name="/robot")
        self.cube = self.server.scene.add_box(
            "/cube", dimensions=(0.15, 0.05, 0.05), color=(65, 115, 210),
            opacity=0.95, side="double",
        )
        self.trajectory = self.server.scene.add_spline_catmull_rom(
            "/palm_trajectory", points=self.palm_positions, line_width=3.0,
            color=(240, 70, 55), segments=max(100, len(self.q)),
        )
        self.palm_marker = self.server.scene.add_frame(
            "/palm_position", axes_length=0.04, axes_radius=0.002,
        )
        self._build_gui(demo)

        @self.server.on_client_connect
        def _(client):
            client.camera.position = (1.0, -1.1, 1.1)
            client.camera.look_at = (0.0, 0.0, 0.55)

        self._render()

    def _load_feasibility_grid(self, path: Path) -> None:
        path = Path(path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Feasibility grid not found: {path}")
        with np.load(str(path)) as archive:
            required = {
                "feasible", "translation_x_m", "translation_y_m", "yaw_deg"
            }
            missing = required.difference(archive.files)
            if missing:
                raise ValueError(
                    f"Feasibility grid is missing fields: {sorted(missing)}"
                )
            self.feasibility = np.asarray(archive["feasible"], dtype=bool)
            self.feasibility_x = np.asarray(archive["translation_x_m"], dtype=float)
            self.feasibility_y = np.asarray(archive["translation_y_m"], dtype=float)
            self.feasibility_yaw_deg = np.asarray(archive["yaw_deg"], dtype=float)
        expected = (
            len(self.feasibility_x),
            len(self.feasibility_y),
            len(self.feasibility_yaw_deg),
        )
        if self.feasibility.shape != expected:
            raise ValueError(
                f"Feasibility tensor has shape {self.feasibility.shape}, expected {expected}"
            )
        if any(np.any(np.diff(axis) <= 0.0) for axis in (
            self.feasibility_x, self.feasibility_y, self.feasibility_yaw_deg
        )):
            raise ValueError("Feasibility-grid axes must be strictly increasing")

    def _build_gui(self, demo: Path) -> None:
        gui = self.server.gui
        gui.add_markdown(
            "# Demonstration viewer\n"
            f"`{demo.name}`\n\n"
            "Red: recorded palm trajectory. Blue: transformed recorded cuboid."
        )
        with gui.add_folder("Playback", expand_by_default=True):
            self.frame_slider = gui.add_slider(
                "Frame", min=0, max=len(self.q) - 1, step=1, initial_value=0
            )
            self.frame_slider.on_update(self._set_frame)
            self.play_button = gui.add_button("Play / pause")
            self.play_button.on_click(self._toggle_play)
            self.speed = gui.add_slider(
                "Speed", min=0.1, max=3.0, step=0.1, initial_value=1.0
            )
            self.loop = gui.add_checkbox("Loop", initial_value=True)
            reset_playback = gui.add_button("Reset playback")
            reset_playback.on_click(lambda _: self._set_frame_value(0))
        with gui.add_folder("Cuboid transform", expand_by_default=True):
            self.xy = gui.add_vector3(
                "Position offset (m)", initial_value=(0.0, 0.0, 0.0),
                min=(-0.3, -0.3, 0.0), max=(0.3, 0.3, 0.0), step=0.005,
            )
            self.xy.on_update(self._set_position_offset)
            self.yaw = gui.add_slider(
                "Cuboid yaw offset (deg)", min=-180.0, max=180.0, step=1.0,
                initial_value=0.0,
            )
            self.yaw.on_update(self._set_yaw_offset)
            reset_cube = gui.add_button("Reset cuboid transform")
            reset_cube.on_click(self._reset_cube_transform)
            self.cube_readout = gui.add_markdown("")
        with gui.add_folder("Inverse kinematics", expand_by_default=True):
            self.force_preferred_branch = gui.add_checkbox(
                "Force inverse branch",
                initial_value=False,
            )
            self.force_preferred_branch.on_update(self._set_ik_mode)
            self.solve_ik_button = gui.add_button(
                "Calcola IK sull'intera demo", color="green"
            )
            self.solve_ik_button.on_click(self._start_ik)
            self.ik_status = gui.add_markdown(
                "**Stato:** traiettoria originale; IK non ancora calcolata."
            )
            self.collision_button = gui.add_button(
                "Controlla collisioni", disabled=True, color="blue"
            )
            self.collision_button.on_click(self._start_collision_check)
            self.collision_status = gui.add_markdown(
                "**Collisioni:** calcola prima l'IK."
            )
        with gui.add_folder("Feasibility map", expand_by_default=True):
            yaw_axis = self.feasibility_yaw_deg
            self.feasibility_yaw = gui.add_slider(
                "Feasibility slice yaw (does not move cuboid)",
                min=float(yaw_axis[0]),
                max=float(yaw_axis[-1]),
                step=float(np.min(np.diff(yaw_axis))),
                initial_value=float(yaw_axis[np.abs(yaw_axis).argmin()]),
            )
            self.feasibility_yaw.on_update(self._update_feasibility_map)
            self.feasibility_image = gui.add_image(
                self._feasibility_map_image(0.0), label="x–y feasibility"
            )
            self.feasibility_readout = gui.add_markdown("")
            self._update_feasibility_map()
        with gui.add_folder("Display", expand_by_default=False):
            show_path = gui.add_checkbox("Show palm trajectory", initial_value=True)
            show_path.on_update(lambda event: setattr(self.trajectory, "visible", event.target.value))

    def _set_frame(self, event) -> None:
        with self.lock:
            self.frame = int(event.target.value)
            self.last_tick = time.monotonic()
        self._render()

    def _set_frame_value(self, frame: int) -> None:
        self.frame_slider.value = int(frame)

    def _toggle_play(self, _) -> None:
        with self.lock:
            self.playing = not self.playing
            self.last_tick = time.monotonic()

    def _invalidate_transformed_solution(self) -> None:
        """Shared UI bookkeeping after either transform component changes."""
        if not self.ik_running:
            self.ik_status.content = (
                "**Stato:** trasformazione modificata; premi "
                "**Calcola IK sull'intera demo**."
            )
        self.collision_button.disabled = True
        self.collision_status.content = "**Collisioni:** IK da ricalcolare."
        self._render()

    def _set_position_offset(self, _) -> None:
        value = np.asarray(self.xy.value, dtype=np.float64)
        with self.lock:
            self.translation_xy[:] = value[:2]
            self.ik_solution_current = False
        self._invalidate_transformed_solution()

    def _set_yaw_offset(self, _) -> None:
        with self.lock:
            self.yaw_offset_rad = np.deg2rad(float(self.yaw.value))
            self.ik_solution_current = False
        self._invalidate_transformed_solution()

    def _set_ik_mode(self, _) -> None:
        with self.lock:
            self.ik_solution_current = False
        self._invalidate_transformed_solution()

    def _reset_cube_transform(self, _) -> None:
        self.xy.value = (0.0, 0.0, 0.0)
        self.yaw.value = 0.0

    def _feasibility_map_image(self, requested_yaw_deg: float) -> np.ndarray:
        """Render one stored yaw slice with metric x/y axes."""
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        from matplotlib.colors import ListedColormap

        yaw_index = int(
            np.abs(self.feasibility_yaw_deg - requested_yaw_deg).argmin()
        )
        values = self.feasibility[:, :, yaw_index].T.astype(np.uint8)
        fig, axis = plt.subplots(figsize=(5.2, 5.0), dpi=110)
        half_x = 0.5 * np.min(np.diff(self.feasibility_x))
        half_y = 0.5 * np.min(np.diff(self.feasibility_y))
        extent = (
            self.feasibility_x[0] - half_x,
            self.feasibility_x[-1] + half_x,
            self.feasibility_y[0] - half_y,
            self.feasibility_y[-1] + half_y,
        )
        axis.imshow(
            values,
            origin="lower",
            extent=extent,
            interpolation="nearest",
            cmap=ListedColormap(("#d93636", "#29a34a")),
            vmin=0,
            vmax=1,
            aspect="equal",
        )
        axis.set_xlabel("x translation [m]")
        axis.set_ylabel("y translation [m]")
        axis.set_title(f"Yaw = {self.feasibility_yaw_deg[yaw_index]:+.1f}°")
        axis.set_xticks(self.feasibility_x[::2])
        axis.set_yticks(self.feasibility_y[::2])
        axis.tick_params(labelsize=7)
        axis.grid(color="white", linewidth=0.35, alpha=0.65)
        fig.tight_layout()
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight")
        plt.close(fig)
        buffer.seek(0)
        import imageio.v3 as iio
        return iio.imread(buffer, extension=".png")

    def _update_feasibility_map(self, _=None) -> None:
        requested = float(self.feasibility_yaw.value)
        index = int(np.abs(self.feasibility_yaw_deg - requested).argmin())
        actual = float(self.feasibility_yaw_deg[index])
        self.feasibility_image.image = self._feasibility_map_image(actual)
        count = int(self.feasibility[:, :, index].sum())
        total = self.feasibility.shape[0] * self.feasibility.shape[1]
        self.feasibility_readout.content = (
            f"**Stored yaw:** {actual:+.1f}°  \n"
            f"**Feasible cells:** {count}/{total} ({100.0 * count / total:.1f}%)  \n"
            "Green = feasible; red = infeasible."
        )

    def _start_ik(self, _) -> None:
        with self.lock:
            if self.ik_running:
                return
            self.ik_running = True
            translation = self.translation_xy.copy()
            yaw = self.yaw_offset_rad
            force_preferred = bool(self.force_preferred_branch.value)
        self.solve_ik_button.disabled = True
        self.ik_status.content = (
            "**Stato:** calcolo IK di tutti i 1.108 frame in corso…"
        )
        threading.Thread(
            target=self._solve_ik,
            args=(translation, yaw, force_preferred),
            daemon=True,
        ).start()

    def _solve_ik(
        self, translation: np.ndarray, yaw: float, force_preferred: bool
    ) -> None:
        """Retarget the complete arm clip and publish feasibility evidence."""
        import torch

        from simtoolreal_animrl.envs.retarget import (
            PalmKinematics,
            cube_pose_to_base_frame,
            retarget_clip_preferred_branch,
            retarget_clip_translation_continuation,
        )
        from simtoolreal_animrl.envs.transform_bank import (
            ARM_JOINT_VELOCITY_LIMIT_RAD_S,
        )

        started = time.monotonic()
        try:
            kinematics = PalmKinematics(self.urdf_path, device="cpu")
            demo_arm_q = torch.as_tensor(self.q[:, :6], dtype=torch.float64)
            # Recover the recorded cube pose in base_link coordinates from its
            # original archive representation, avoiding world-frame offsets.
            _, cube_demo = load_demo(Path(self.frame_demo_path))
            pivot = cube_pose_to_base_frame(
                torch.as_tensor(cube_demo[0], dtype=torch.float64)
            )[:3]
            translation_3d = torch.tensor(
                [[translation[0], translation[1], 0.0]], dtype=torch.float64
            )
            solve = (
                retarget_clip_preferred_branch
                if force_preferred
                else retarget_clip_translation_continuation
            )
            solve_arguments = (
                kinematics,
                demo_arm_q,
                torch.tensor([yaw], dtype=torch.float64),
                translation_3d,
                pivot,
            )
            result = solve(*solve_arguments)
            arm_q = result.arm_q[:, 0]
            position = result.position_residual_m[:, 0]
            rotation = result.rotation_residual_rad[:, 0]
            margin = result.limit_margin_rad[:, 0]
            speed = torch.zeros_like(position)
            speed[1:] = (arm_q[1:] - arm_q[:-1]).abs().amax(dim=1) * 60.0

            max_position = float(position.max())
            max_rotation = float(rotation.max())
            min_margin = float(margin.min())
            max_speed = float(speed.max())
            position_frame = int(position.argmax())
            rotation_frame = int(rotation.argmax())
            margin_frame = int(margin.argmin())
            speed_frame = int(speed.argmax())
            kinematically_feasible = (
                max_position <= 1e-3
                and max_rotation <= 1e-2
                and min_margin >= 0.05
                and max_speed <= 0.5 * ARM_JOINT_VELOCITY_LIMIT_RAD_S
            )

            transformed_q = self.q.copy()
            transformed_q[:, :6] = arm_q.cpu().numpy()
            # Finger joints (columns 6:26) deliberately remain byte-for-byte
            # equal to the recorded demonstration.
            with self.lock:
                unchanged = np.allclose(translation, self.translation_xy) and np.isclose(
                    yaw, self.yaw_offset_rad
                ) and force_preferred == bool(self.force_preferred_branch.value)
                if unchanged:
                    self.display_q = transformed_q
                    self.ik_solution_current = True
            elapsed = time.monotonic() - started
            label = (
                "✅ IK CINEMATICAMENTE FATTIBILE"
                if kinematically_feasible
                else "❌ IK NON FATTIBILE"
            )
            stale = "" if unchanged else (
                "  \n⚠️ I controlli sono cambiati durante il calcolo: risultato non applicato."
            )
            self.ik_status.content = (
                f"## {label}\n"
                f"Calcolo completo in {elapsed:.1f} s.  \n"
                f"Errore posizione massimo: **{1e3 * max_position:.2f} mm** "
                f"(frame {position_frame})  \n"
                f"Errore rotazione massimo: **{np.rad2deg(max_rotation):.3f}°** "
                f"(frame {rotation_frame})  \n"
                f"Margine minimo dai limiti: **{min_margin:.3f} rad** "
                f"(frame {margin_frame})  \n"
                f"Velocità massima: **{max_speed:.3f} rad/s** "
                f"(frame {speed_frame}; soglia {0.5 * ARM_JOINT_VELOCITY_LIMIT_RAD_S:.3f})"
                f"{stale}"
            )
            if unchanged:
                self.collision_button.disabled = False
                self.collision_status.content = (
                    "**Collisioni:** non ancora controllate per questa IK."
                )
                self._render()
        except Exception as exc:
            self.ik_status.content = f"## Errore IK\n`{type(exc).__name__}: {exc}`"
        finally:
            with self.lock:
                self.ik_running = False
            self.solve_ik_button.disabled = False

    def _start_collision_check(self, _) -> None:
        with self.lock:
            if self.collision_running or not self.ik_solution_current:
                return
            self.collision_running = True
            trajectory = self.display_q.copy()
        self.collision_button.disabled = True
        self.collision_status.content = (
            "**Collisioni:** controllo in background; il playback può continuare…"
        )
        threading.Thread(
            target=self._check_collisions,
            args=(trajectory,),
            daemon=True,
        ).start()

    def _check_collisions(self, trajectory: np.ndarray) -> None:
        started = time.monotonic()
        try:
            hand_arm, arm_arm = check_robot_collisions(
                self.urdf_path, trajectory
            )
            elapsed = time.monotonic() - started
            collision_free = not hand_arm and not arm_arm
            label = (
                "✅ NESSUNA COLLISIONE"
                if collision_free
                else "❌ COLLISIONI RILEVATE"
            )
            self.collision_status.content = (
                f"## {label}\n"
                f"Controllo completo in {elapsed:.1f} s.  \n"
                f"{format_collision_report('Collisioni mano–braccio', hand_arm)}  \n"
                f"{format_collision_report('Collisioni braccio–braccio', arm_arm)}"
            )
        except Exception as exc:
            self.collision_status.content = (
                f"## Errore collisioni\n`{type(exc).__name__}: {exc}`"
            )
        finally:
            with self.lock:
                self.collision_running = False
                current = self.ik_solution_current
            self.collision_button.disabled = not current

    def _render(self) -> None:
        with self.lock:
            frame = self.frame
            translation = self.translation_xy.copy()
            yaw = self.yaw_offset_rad
        position, orientation = transformed_cube_pose(
            self.cube_positions, self.cube_orientations, frame, translation, yaw
        )
        transformed_palm_positions = transform_points_about_cube_start(
            self.palm_positions,
            self.cube_positions[0],
            translation,
            yaw,
        )
        self.robot.update_cfg(self.display_q[frame])
        self.cube.position = position
        self.cube.wxyz = orientation
        self.trajectory.points = transformed_palm_positions
        self.palm_marker.position = transformed_palm_positions[frame]
        self.cube_readout.content = (
            f"**Frame:** {frame}/{len(self.q) - 1}  \n"
            f"**Requested offset:** x={translation[0]:+.3f}, "
            f"y={translation[1]:+.3f} m  \n"
            f"**Absolute cuboid xyz (world):** ({position[0]:+.3f}, "
            f"{position[1]:+.3f}, {position[2]:+.3f}) m  \n"
            f"**Requested cuboid yaw offset:** {np.rad2deg(yaw):+.1f}°  \n"
            "*The feasibility-slice yaw below only changes the map.*"
        )

    def run(self) -> None:
        print(f"Open http://localhost:{self.port}")
        try:
            while True:
                time.sleep(1.0 / 120.0)
                with self.lock:
                    if not self.playing:
                        continue
                    now = time.monotonic()
                    elapsed = now - self.last_tick
                    advance = int(elapsed * 60.0 * float(self.speed.value))
                    if advance <= 0:
                        continue
                    next_frame = self.frame + advance
                    if next_frame >= len(self.q):
                        if self.loop.value:
                            next_frame %= len(self.q)
                        else:
                            next_frame = len(self.q) - 1
                            self.playing = False
                    self.last_tick = now
                self._set_frame_value(next_frame)
        except KeyboardInterrupt:
            print("Viewer stopped.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", type=Path, default=DEFAULT_DEMO)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument(
        "--feasibility-grid", type=Path, default=DEFAULT_FEASIBILITY_GRID
    )
    parser.add_argument("--port", type=int, default=8080)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    DemoViewer(
        args.demo.expanduser().resolve(),
        args.urdf.expanduser().resolve(),
        args.feasibility_grid.expanduser().resolve(),
        args.port,
    ).run()


if __name__ == "__main__":
    main()
