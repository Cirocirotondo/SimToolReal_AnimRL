"""Per-episode reward and tracking diagnostics for AnimRL evaluation.

Mirrors the structure of dextoolbench's RewardEpisodePlotter /
MotionDiagnosticPlotter: every episode produces one folder holding a compressed
npz of the raw series plus the PNGs derived from it, so a saved episode can be
re-plotted later without Isaac Gym.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# Joint names are read off the environment in start_episode rather than
# imported from envs.controller: that module pulls in isaacgym, and importing
# it from this package would put isaacgym after torch, which Isaac Gym rejects.


# (info key, cfg.rewards weight attribute). ``None`` denotes the optional
# contact weight, which lives in cfg.contact. Order fixes the legend order and
# must stay aligned with MotionImitationEnv._compute_reward_and_errors.
REWARD_TERMS: Tuple[Tuple[str, Optional[str]], ...] = (
    ("position_reward", "position_arm_weight"),
    ("velocity_reward", "velocity_arm_weight"),
    ("action_rate_reward", "action_rate_arm_weight"),
    ("hand_position_reward", "position_hand_weight"),
    ("hand_velocity_reward", "velocity_hand_weight"),
    ("hand_action_rate_reward", "action_rate_hand_weight"),
    ("object_position_reward", "object_position_weight"),
    ("object_orientation_reward", "object_orientation_weight"),
    (
        "fingertip_object_distance_reward",
        "fingertip_object_distance_weight",
    ),
    ("fingertip_contact_reward", None),
)

# Scalar per-step diagnostics copied straight out of the step extras.
SCALAR_INFO_KEYS: Tuple[str, ...] = (
    "rms_position_error",
    "rms_velocity_error",
    "rms_action_rate",
    "rms_hand_position_error",
    "rms_hand_velocity_error",
    "rms_hand_action_rate",
    "max_abs_position_error",
    "max_abs_hand_position_error",
    "object_position_error_m",
    "object_orientation_error_rad",
    "object_com_height_m",
    "object_com_lift_m",
    "fingertip_object_distance_m",
    "fingertip_contact_fraction",
    "mean_fingertip_contact_force_n",
    "proximity_active",
)


def _use_agg():
    """Saved-only plots never need a GUI.

    Isaac Gym evaluation runs alongside Qt libraries that clash with
    matplotlib's Qt backend, so force Agg before pyplot is imported.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _scalar(value: Any, env_idx: int = 0) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value).reshape(-1)
    if array.size == 0:
        return float("nan")
    return float(array[min(env_idx, array.size - 1)])


def _vector(value: Any, env_idx: int = 0) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 1:
        return array
    return array[min(env_idx, array.shape[0] - 1)]


def _short_joint_name(name: str) -> str:
    suffix = "_joint"
    if name.endswith(suffix) and len(name) > len(suffix):
        return name[: -len(suffix)]
    return name


def _finish_axis(ax, ylabel: str, legend: bool = True) -> None:
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if legend:
        ax.legend(loc="upper right", fontsize=7, ncol=2)


def _joint_colors(count: int) -> List[Any]:
    """One stable colour per joint.

    Matplotlib advances its cycle per plot call, so drawing reference and
    actual as two calls would give the same joint two colours. Assigning by
    index keeps a joint one colour across every panel, leaving linestyle to
    separate reference from actual.
    """
    import matplotlib.pyplot as plt

    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return [cycle[index % len(cycle)] for index in range(count)]


def _plot_per_joint(ax, time_s, values, prefix, labels=(), **kwargs):
    colors = None if "color" in kwargs else _joint_colors(values.shape[1])
    for index in range(values.shape[1]):
        style = dict(kwargs)
        if colors is not None:
            style["color"] = colors[index]
        name = labels[index] if index < len(labels) else str(index)
        ax.plot(time_s, values[:, index], label="{}_{}".format(prefix, name), **style)


class EvaluationPlotter:
    """Record one evaluation episode and write its npz + diagnostic PNGs."""

    def __init__(self, save_dir, *, env_idx: int = 0) -> None:
        self.save_dir = Path(save_dir)
        self.env_idx = int(env_idx)
        self._slug = "episode"
        self._dt = 1.0
        self._weights: Dict[str, float] = {}
        self._position_threshold: Optional[float] = None
        self._hand_position_threshold: Optional[float] = None
        self._reference_initial_object_com_height_m = 0.0
        self._action_scale = 1.0
        self._hand_action_scale = 1.0
        self._num_arm_dofs = 0
        self._num_hand_dofs = 0
        self._arm_names: Tuple[str, ...] = ()
        self._hand_names: Tuple[str, ...] = ()
        self._fingertip_names: Tuple[str, ...] = ()
        self._contact_fingertip_names: Tuple[str, ...] = ()
        self._contact_enabled = False
        self._contact_force_threshold_n = 0.0
        self._proximity_fingertip_names: Tuple[str, ...] = ()
        self._proximity_std_m = 0.0
        self._proximity_weight = 0.0
        self._records: Dict[str, List[Any]] = {}

    def start_episode(self, slug: str, env) -> None:
        self._slug = str(slug)
        self._dt = float(env.dt)
        self._action_scale = float(env.action_scale)
        self._hand_action_scale = float(env.hand_action_scale)
        # The policy now drives all 26 joints, so the split comes from the
        # environment's own arm/hand views rather than from the action width:
        # one 26-panel figure would be unreadable.
        self._num_arm_dofs = int(env.arm_q.shape[1])
        self._num_hand_dofs = int(env.hand_q.shape[1])
        names = tuple(_short_joint_name(name) for name in env.JOINT_NAMES)
        self._arm_names = names[: self._num_arm_dofs]
        self._hand_names = names[self._num_arm_dofs:]
        self._weights = {
            info_key: (
                float(env.cfg.contact.reward_per_finger)
                if weight_attribute is None
                else float(getattr(env.cfg.rewards, weight_attribute))
            )
            for info_key, weight_attribute in REWARD_TERMS
        }
        self._fingertip_names = tuple(env.FINGERTIP_NAMES)
        self._contact_fingertip_names = tuple(env.contact_fingertip_names)
        self._contact_enabled = bool(env.contact_enabled)
        self._contact_force_threshold_n = float(env.contact_force_threshold_n)
        self._proximity_fingertip_names = tuple(env.proximity_fingertip_names)
        self._proximity_std_m = float(env.proximity_std_m)
        self._proximity_weight = float(env.proximity_weight)
        enabled = bool(env.cfg.termination.enabled)
        self._position_threshold = (
            float(env.cfg.termination.arm_position_threshold_rad)
            if enabled
            else None
        )
        self._hand_position_threshold = (
            float(env.cfg.termination.hand_position_threshold_rad)
            if enabled
            else None
        )
        initial_reference = env.reference.sample(
            env.reference_index[self.env_idx].reshape(1)
        )
        self._reference_initial_object_com_height_m = _scalar(
            env._cube_reference_root_states(initial_reference)[:, 2], 0
        )
        self._records.clear()

    def _append(self, key: str, value: Any) -> None:
        self._records.setdefault(key, []).append(value)

    def record(self, env, step: int, actions, rewards, dones, infos) -> None:
        """Record one post-step transition.

        `infos` carries pre-reset clones, so every scalar diagnostic stays exact
        even on the step that terminates the episode. The measured joint state
        does not: the environment auto-resets inside step(), so it is recorded
        as NaN on a done step rather than showing the post-reset pose.
        """
        env_idx = self.env_idx
        reference_index = int(_scalar(infos["reference_index"], env_idx))
        done = bool(_scalar(dones, env_idx))

        self._append("steps", int(step))
        self._append("time_s", float(step) * self._dt)
        self._append("reference_index", reference_index)
        self._append("reward", _scalar(rewards, env_idx))
        for info_key, _ in REWARD_TERMS:
            self._append(info_key, _scalar(infos.get(info_key), env_idx))
        for key in SCALAR_INFO_KEYS:
            self._append(key, _scalar(infos.get(key), env_idx))
        self._append("worst_joint_index", _scalar(infos.get("worst_joint_index"), env_idx))
        fingertip_force = infos.get("fingertip_force_n")
        self._append(
            "fingertip_force_n",
            _vector(fingertip_force, env_idx)
            if fingertip_force is not None
            else np.full(len(self._fingertip_names), np.nan),
        )
        proximity_distance = infos.get("fingertip_object_distance_per_finger_m")
        self._append(
            "fingertip_object_distance_per_finger_m",
            _vector(proximity_distance, env_idx)
            if proximity_distance is not None
            else np.full(len(self._proximity_fingertip_names), np.nan),
        )

        arm = self._num_arm_dofs
        action = _vector(actions, env_idx)
        self._append("action", action[:arm])
        self._append("hand_action", action[arm:])
        # Recomputed from the action rather than read back from the environment:
        # reset_idx() overwrites the stored targets of a terminated env.
        target = _vector(env.scale_actions(
            torch.as_tensor(action, dtype=torch.float32, device=env.device).unsqueeze(0)
        ), 0)
        self._append("arm_target_q", target[:arm])
        self._append("hand_target_q", target[arm:])

        indices = torch.as_tensor(
            [reference_index], dtype=torch.long, device=env.device
        )
        reference = env.reference.sample(indices)
        reference_cube_root_state = env._cube_reference_root_states(reference)
        self._append(
            "reference_object_com_height_m",
            _scalar(reference_cube_root_state[:, 2], 0),
        )
        # The action that would have landed exactly on this reference sample.
        # step() advances reference_index before publishing it, so the index
        # recorded here is the one the action applied this step was aiming at,
        # and inverting the residual mapping on it gives the ideal action.
        ideal_action = _vector(env.positions_to_actions(reference.q), 0)
        self._append("reference_action", ideal_action[:arm])
        self._append("reference_hand_action", ideal_action[arm:])
        self._append("reference_arm_q", _vector(reference.q[:, :arm], 0))
        self._append("reference_arm_dq", _vector(reference.dq[:, :arm], 0))
        self._append("reference_hand_q", _vector(reference.q[:, arm:], 0))
        self._append("reference_hand_dq", _vector(reference.dq[:, arm:], 0))

        if done:
            nan_arm = np.full(arm, np.nan)
            nan_hand = np.full(self._num_hand_dofs, np.nan)
            self._append("actual_arm_q", nan_arm)
            self._append("actual_arm_dq", nan_arm)
            self._append("actual_hand_q", nan_hand)
            self._append("actual_hand_dq", nan_hand)
        else:
            self._append("actual_arm_q", _vector(env.arm_q, env_idx))
            self._append("actual_arm_dq", _vector(env.arm_dq, env_idx))
            self._append("actual_hand_q", _vector(env.hand_q, env_idx))
            self._append("actual_hand_dq", _vector(env.hand_dq, env_idx))

    def _arrays(self) -> Dict[str, np.ndarray]:
        arrays = {
            key: np.asarray(values, dtype=np.float64)
            for key, values in self._records.items()
        }
        if not arrays:
            return arrays
        arrays["steps"] = arrays["steps"].astype(np.int64)
        arrays["reference_index"] = arrays["reference_index"].astype(np.int64)
        arrays["hand_position_error"] = (
            arrays["actual_hand_q"] - arrays["reference_hand_q"]
        )
        arrays["hand_velocity_error"] = (
            arrays["actual_hand_dq"] - arrays["reference_hand_dq"]
        )
        hand_delta = np.diff(arrays["hand_action"], axis=0)
        arrays["hand_action_delta"] = np.vstack(
            (np.full((1, arrays["hand_action"].shape[1]), np.nan), hand_delta)
        )
        arrays["arm_position_error"] = (
            arrays["actual_arm_q"] - arrays["reference_arm_q"]
        )
        arrays["arm_velocity_error"] = (
            arrays["actual_arm_dq"] - arrays["reference_arm_dq"]
        )
        # a_t - a_{t-1}, the quantity the action-rate reward term regularizes.
        # The first entry has no predecessor inside the recorded window.
        action_delta = np.diff(arrays["action"], axis=0)
        arrays["action_delta"] = np.vstack(
            (np.full((1, arrays["action"].shape[1]), np.nan), action_delta)
        )
        for info_key, _ in REWARD_TERMS:
            arrays["weighted_" + info_key] = (
                self._weights.get(info_key, 0.0) * arrays[info_key]
            )
        for key in ("reward",) + tuple(
            "weighted_" + info_key for info_key, _ in REWARD_TERMS
        ):
            arrays["cumulative_" + key] = np.cumsum(arrays[key])
        return arrays

    def finalize(self, reason: str = "done") -> Dict[str, str]:
        arrays = self._arrays()
        if not arrays:
            return {}

        episode_dir = self.save_dir / self._slug
        episode_dir.mkdir(parents=True, exist_ok=True)
        per_term_dir = episode_dir / "per_term"
        per_term_dir.mkdir(parents=True, exist_ok=True)
        for stale in per_term_dir.glob("*.png"):
            stale.unlink()

        npz_path = episode_dir / "evaluation_log.npz"
        np.savez_compressed(
            npz_path,
            **arrays,
            joint_names=np.asarray(self._arm_names + self._hand_names),
            fingertip_names=np.asarray(self._fingertip_names),
            contact_fingertip_names=np.asarray(self._contact_fingertip_names),
            contact_force_threshold_n=np.asarray(
                self._contact_force_threshold_n
            ),
            proximity_fingertip_names=np.asarray(
                self._proximity_fingertip_names
            ),
            proximity_std_m=np.asarray(self._proximity_std_m),
            reward_weights=np.asarray(
                [self._weights.get(key, 0.0) for key, _ in REWARD_TERMS]
            ),
            control_dt=np.asarray(self._dt),
            termination_reason=np.asarray(reason),
        )

        paths = {"episode_dir": str(episode_dir), "npz": str(npz_path)}
        try:
            plt = _use_agg()
        except ImportError:
            print("[eval-plot] matplotlib missing; saved .npz only.", flush=True)
            return paths

        paths.update(self._save_overview(plt, episode_dir, arrays, reason))
        paths.update(self._save_reward_terms(plt, episode_dir, arrays))
        paths.update(self._save_tracking_errors(plt, episode_dir, arrays))
        paths.update(self._save_object_height(plt, episode_dir, arrays))
        paths.update(self._save_fingertip_forces(plt, episode_dir, arrays))
        paths.update(self._save_fingertip_proximity(plt, episode_dir, arrays))
        paths.update(self._save_joint_tracking(plt, episode_dir, arrays, "arm"))
        paths.update(self._save_joint_tracking(plt, episode_dir, arrays, "hand"))
        paths.update(self._save_action_per_joint(plt, episode_dir, arrays, "arm"))
        paths.update(self._save_action_per_joint(plt, episode_dir, arrays, "hand"))
        paths.update(self._save_per_term(plt, per_term_dir, arrays))

        print(
            "[eval-plot] Episode folder: {} ({} figures)".format(
                episode_dir, len(list(episode_dir.rglob("*.png")))
            ),
            flush=True,
        )
        return paths

    def _save_overview(self, plt, episode_dir, data, reason) -> Dict[str, str]:
        time_s = data["time_s"]
        fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)

        for info_key, _ in REWARD_TERMS:
            axes[0].plot(
                time_s,
                data["weighted_" + info_key],
                label="{} (w={:g})".format(info_key, self._weights.get(info_key, 0.0)),
                linewidth=1.3,
            )
        axes[0].plot(
            time_s, data["reward"], label="total", color="black", linewidth=2.0
        )
        upper = sum(self._weights.values())
        axes[0].axhline(upper, color="0.6", linestyle=":", linewidth=0.8)
        _finish_axis(axes[0], "Per-step reward")

        axes[1].plot(
            time_s, data["cumulative_reward"], color="black", linewidth=1.8, label="return"
        )
        _finish_axis(axes[1], "Cumulative return")

        axes[2].plot(
            time_s, data["max_abs_position_error"], label="max |arm q error|", linewidth=1.3
        )
        axes[2].plot(
            time_s, data["rms_position_error"], label="RMS arm q error", linewidth=1.3
        )
        if self._position_threshold is not None:
            axes[2].axhline(
                self._position_threshold,
                color="red",
                linestyle="--",
                linewidth=1.0,
                label="termination threshold",
            )
        _finish_axis(axes[2], "Position error [rad]")
        axes[2].set_xlabel("Episode time [s]")

        fig.suptitle(
            "Evaluation overview — {} (ended: {}, {} steps, return {:.3f})".format(
                self._slug, reason, data["steps"].size, data["cumulative_reward"][-1]
            )
        )
        fig.tight_layout()
        path = episode_dir / "overview.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return {"overview_png": str(path)}

    def _save_reward_terms(self, plt, episode_dir, data) -> Dict[str, str]:
        time_s = data["time_s"]
        fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
        for info_key, _ in REWARD_TERMS:
            axes[0].plot(time_s, data[info_key], label=info_key, linewidth=1.3)
            axes[1].plot(
                time_s, data["weighted_" + info_key], label=info_key, linewidth=1.3
            )
            axes[2].plot(
                time_s,
                data["cumulative_weighted_" + info_key],
                label=info_key,
                linewidth=1.3,
            )
        axes[2].plot(
            time_s,
            data["cumulative_reward"],
            label="total",
            color="black",
            linestyle="--",
            linewidth=1.5,
        )
        _finish_axis(axes[0], "Unweighted term (0-1)")
        _finish_axis(axes[1], "Weighted contribution")
        _finish_axis(axes[2], "Cumulative contribution")
        axes[2].set_xlabel("Episode time [s]")
        fig.suptitle("Reward terms — {}".format(self._slug))
        fig.tight_layout()
        path = episode_dir / "reward_terms.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return {"reward_terms_png": str(path)}

    def _save_tracking_errors(self, plt, episode_dir, data) -> Dict[str, str]:
        time_s = data["time_s"]
        fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)

        axes[0].plot(time_s, data["rms_position_error"], label="arm RMS", linewidth=1.3)
        axes[0].plot(
            time_s, data["max_abs_position_error"], label="arm max abs", linewidth=1.3
        )
        axes[0].plot(
            time_s,
            data["rms_hand_position_error"],
            label="hand RMS",
            linewidth=1.3,
        )
        axes[0].plot(
            time_s,
            data["max_abs_hand_position_error"],
            label="hand max abs",
            linewidth=1.0,
            alpha=0.7,
        )
        for threshold, label in (
            (self._position_threshold, "arm threshold"),
            (self._hand_position_threshold, "hand threshold"),
        ):
            if threshold is not None:
                axes[0].axhline(
                    threshold,
                    color="red",
                    linestyle="--",
                    linewidth=1.0,
                    label=label,
                )
        _finish_axis(axes[0], "Position error [rad]")

        axes[1].plot(time_s, data["rms_velocity_error"], label="arm RMS", linewidth=1.3)
        axes[1].plot(
            time_s, data["rms_hand_velocity_error"], label="hand RMS", linewidth=1.3
        )
        _finish_axis(axes[1], "Velocity error [rad/s]")

        axes[2].plot(
            time_s,
            data["rms_action_rate"],
            label="arm RMS |delta(a) - delta(a_demo)|",
            linewidth=1.3,
        )
        axes[2].plot(
            time_s,
            data["rms_hand_action_rate"],
            label="hand RMS |delta(a) - delta(a_demo)|",
            linewidth=1.3,
        )
        _finish_axis(axes[2], "Action-delta tracking error")
        axes[2].set_xlabel("Episode time [s]")

        fig.suptitle("Tracking errors — {}".format(self._slug))
        fig.tight_layout()
        path = episode_dir / "tracking_errors.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return {"tracking_errors_png": str(path)}

    def _save_object_height(self, plt, episode_dir, data) -> Dict[str, str]:
        """Plot cube COM height and lift without turning them into rewards."""
        time_s = data["time_s"]
        reference_lift = (
            data["reference_object_com_height_m"]
            - self._reference_initial_object_com_height_m
        )
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        axes[0].plot(
            time_s,
            data["object_com_height_m"],
            label="physical cube",
            linewidth=1.7,
        )
        axes[0].plot(
            time_s,
            data["reference_object_com_height_m"],
            label="demonstration",
            linestyle="--",
            linewidth=1.4,
        )
        _finish_axis(axes[0], "Cube COM world z [m]")

        axes[1].plot(
            time_s,
            data["object_com_lift_m"],
            label="physical cube",
            linewidth=1.7,
        )
        axes[1].plot(
            time_s,
            reference_lift,
            label="demonstration",
            linestyle="--",
            linewidth=1.4,
        )
        axes[1].axhline(0.0, color="0.5", linestyle=":", linewidth=0.8)
        _finish_axis(axes[1], "Lift from episode start [m]")
        axes[1].set_xlabel("Episode time [s]")
        fig.suptitle(
            "Cube centre-of-mass height — {} (peak {:.4f} m, lift {:.4f} m)".format(
                self._slug,
                np.nanmax(data["object_com_height_m"]),
                np.nanmax(data["object_com_lift_m"]),
            )
        )
        fig.tight_layout()
        path = episode_dir / "object_com_height.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return {"object_com_height_png": str(path)}

    def _save_fingertip_forces(self, plt, episode_dir, data) -> Dict[str, str]:
        """Per-fingertip net contact force, which the averaged metric hides.

        ``mean_fingertip_contact_force_n`` averages over the selected fingers,
        including those touching nothing, so one finger pressing hard and three
        pressing lightly look identical. These curves separate them.
        """
        forces = data.get("fingertip_force_n")
        if forces is None or forces.ndim != 2 or forces.shape[1] == 0:
            return {}
        if not self._contact_enabled:
            # Without contact reporting the tensor is never acquired and every
            # value is a hard zero; a flat figure would read as "no contact".
            return {}

        time_s = data["time_s"]
        names = self._fingertip_names
        selected = set(self._contact_fingertip_names)
        threshold = self._contact_force_threshold_n
        colors = _joint_colors(forces.shape[1])

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        for index in range(forces.shape[1]):
            name = names[index] if index < len(names) else str(index)
            in_reward = name in selected
            axes[0].plot(
                time_s,
                forces[:, index],
                label="{}{}".format(name, "" if in_reward else " (not rewarded)"),
                color=colors[index],
                linewidth=1.6 if in_reward else 1.1,
                linestyle="-" if in_reward else "--",
                alpha=1.0 if in_reward else 0.65,
            )
        axes[0].axhline(
            threshold,
            color="0.4",
            linestyle=":",
            linewidth=1.0,
            label="threshold {:g} N".format(threshold),
        )
        _finish_axis(axes[0], "Net contact force [N]")
        # Headroom so the legend never sits on top of the tallest curve.
        finite = forces[np.isfinite(forces)]
        if finite.size:
            axes[0].set_ylim(
                min(0.0, float(finite.min())), float(finite.max()) * 1.35 + 0.1
            )

        # One filled band per rewarded finger: a raster reads far better than
        # overlapping step lines when several fingers touch at once.
        rewarded = [
            (index, name)
            for index, name in enumerate(names[: forces.shape[1]])
            if name in selected
        ]
        for row, (index, name) in enumerate(rewarded):
            contact = forces[:, index] > threshold
            axes[1].fill_between(
                time_s,
                row,
                row + 0.8,
                where=contact,
                color=colors[index],
                alpha=0.85,
                step="post",
                linewidth=0,
            )
            axes[1].axhline(row, color="0.85", linewidth=0.8)
        axes[1].set_yticks([row + 0.4 for row in range(len(rewarded))])
        axes[1].set_yticklabels([name for _, name in rewarded])
        axes[1].set_ylim(-0.1, max(len(rewarded), 1))
        _finish_axis(axes[1], "Above threshold", legend=False)
        axes[1].set_xlabel("Episode time [s]")

        peak = np.nanmax(forces) if np.isfinite(forces).any() else float("nan")
        fig.suptitle(
            "Fingertip net contact force \u2014 {} (peak {:.3f} N, "
            "rewarded fingers: {})".format(
                self._slug, peak, ", ".join(self._contact_fingertip_names) or "none"
            )
        )
        fig.tight_layout()
        path = episode_dir / "fingertip_forces.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return {"fingertip_forces_png": str(path)}

    def _save_fingertip_proximity(self, plt, episode_dir, data) -> Dict[str, str]:
        """Per-finger surface distance behind the proximity reward.

        The reward averages one Gaussian per selected finger, so the mean
        distance cannot tell an evenly closing hand from a single finger
        reaching the cube alone. These curves can.
        """
        distances = data.get("fingertip_object_distance_per_finger_m")
        if distances is None or distances.ndim != 2 or distances.shape[1] == 0:
            return {}
        if not np.isfinite(distances).any():
            return {}

        time_s = data["time_s"]
        names = self._proximity_fingertip_names
        std_m = self._proximity_std_m
        colors = _joint_colors(distances.shape[1])
        active = data.get("proximity_active")

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # The term is gated off before the pre-grasp window; without this band
        # the flat zero reward there reads as "far from the cube".
        if active is not None and np.isfinite(active).any():
            axes[0].fill_between(
                time_s,
                0.0,
                1.0,
                where=active <= 0.0,
                transform=axes[0].get_xaxis_transform(),
                color="0.85",
                alpha=0.45,
                linewidth=0,
                label="reward gated off",
            )

        for index in range(distances.shape[1]):
            name = names[index] if index < len(names) else str(index)
            axes[0].plot(
                time_s,
                distances[:, index],
                label=name,
                color=colors[index],
                linewidth=1.6,
            )
        axes[0].plot(
            time_s,
            data["fingertip_object_distance_m"],
            label="mean (reward input)",
            color="0.2",
            linestyle="--",
            linewidth=1.3,
        )
        for multiple in (1.0, 2.0):
            axes[0].axhline(
                multiple * std_m,
                color="0.5",
                linestyle=":",
                linewidth=0.9,
                label="{:g}\u03c3 = {:.3f} m".format(multiple, multiple * std_m),
            )
        axes[0].axhline(0.0, color="0.75", linewidth=0.8)
        # The gated-off band spans the axes vertically, which would otherwise
        # autoscale this panel to 1.0 and squash centimetre-scale distances.
        finite = distances[np.isfinite(distances)]
        upper = max(float(finite.max()) if finite.size else 0.0, 2.0 * std_m)
        axes[0].set_ylim(-0.02 * upper, upper * 1.25)
        _finish_axis(axes[0], "Surface distance to cube [m]")

        # Distance zero means touching the cuboid surface; the Gaussian is the
        # exact term the reward uses, so plotting both makes the shaping legible.
        per_finger_reward = np.exp(
            -np.square(distances) / (2.0 * max(std_m, 1e-12) ** 2)
        )
        for index in range(distances.shape[1]):
            name = names[index] if index < len(names) else str(index)
            axes[1].plot(
                time_s,
                per_finger_reward[:, index],
                label=name,
                color=colors[index],
                linewidth=1.3,
                alpha=0.8,
            )
        axes[1].plot(
            time_s,
            data["fingertip_object_distance_reward"],
            label="term (gated)",
            color="0.2",
            linewidth=1.8,
        )
        axes[1].plot(
            time_s,
            data["weighted_fingertip_object_distance_reward"],
            label="weighted (w={:g})".format(self._proximity_weight),
            color="0.2",
            linestyle="--",
            linewidth=1.3,
        )
        axes[1].set_ylim(-0.02, 1.05)
        _finish_axis(axes[1], "Proximity reward")
        axes[1].set_xlabel("Episode time [s]")

        closest = np.nanmin(distances) if np.isfinite(distances).any() else np.nan
        fig.suptitle(
            "Fingertip-to-cube proximity \u2014 {} (closest {:.4f} m, "
            "\u03c3 {:.3f} m, fingers: {})".format(
                self._slug, closest, std_m, ", ".join(names) or "none"
            )
        )
        fig.tight_layout()
        path = episode_dir / "fingertip_proximity.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return {"fingertip_proximity_png": str(path)}

    def _save_joint_tracking(self, plt, episode_dir, data, group) -> Dict[str, str]:
        """One tracking figure per joint block; 26 joints in one is unreadable."""
        time_s = data["time_s"]
        prefix = "arm" if group == "arm" else "hand"
        names = self._arm_names if group == "arm" else self._hand_names
        threshold = (
            self._position_threshold
            if group == "arm"
            else self._hand_position_threshold
        )
        fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)

        _plot_per_joint(
            axes[0], time_s, data["reference_%s_q" % prefix], "reference", names
        )
        _plot_per_joint(
            axes[0], time_s, data["actual_%s_q" % prefix], "actual", names,
            linestyle="--",
        )
        _finish_axis(axes[0], "Joint position [rad]")

        _plot_per_joint(
            axes[1], time_s, data["%s_position_error" % prefix], "error", names
        )
        if threshold is not None:
            for sign in (-1.0, 1.0):
                axes[1].axhline(
                    sign * threshold,
                    color="red",
                    linestyle="--",
                    linewidth=0.9,
                )
        _finish_axis(axes[1], "Position error [rad]")

        # Reference and measured velocity get their own panels: a chattering
        # policy swings the measured velocity an order of magnitude wider than
        # the demonstration, which would flatten the reference into the axis.
        _plot_per_joint(
            axes[2], time_s, data["reference_%s_dq" % prefix], "reference", names
        )
        _finish_axis(axes[2], "Reference velocity [rad/s]")

        _plot_per_joint(
            axes[3], time_s, data["actual_%s_dq" % prefix], "actual", names
        )
        _finish_axis(axes[3], "Measured velocity [rad/s]")

        _plot_per_joint(
            axes[4], time_s, data["%s_target_q" % prefix], "target", names
        )
        _plot_per_joint(
            axes[4], time_s, data["actual_arm_q"], "actual", self._arm_names,
            linestyle="--",
        )
        _finish_axis(axes[4], "PD target vs measured [rad]")
        axes[4].set_xlabel("Episode time [s]")

        fig.suptitle("{} joint tracking — {}".format(group.capitalize(), self._slug))
        fig.tight_layout()
        path = episode_dir / "{}_joint_tracking.png".format(group)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return {"{}_joint_tracking_png".format(group): str(path)}

    def _save_action_per_joint(self, plt, episode_dir, data, group) -> Dict[str, str]:
        """One panel per joint of the block, to expose per-joint chatter."""
        time_s = data["time_s"]
        action = data["action"] if group == "arm" else data["hand_action"]
        delta = (
            data["action_delta"] if group == "arm" else data["hand_action_delta"]
        )
        ideal = (
            data["reference_action"]
            if group == "arm"
            else data["reference_hand_action"]
        )
        names = self._arm_names if group == "arm" else self._hand_names
        scale = self._action_scale if group == "arm" else self._hand_action_scale
        count = action.shape[1]

        fig, axes = plt.subplots(count, 1, figsize=(14, 2.3 * count), sharex=True)
        axes = np.atleast_1d(axes)
        for index in range(count):
            ax = axes[index]
            trace = action[:, index]
            ideal_trace = ideal[:, index]
            ax.plot(time_s, trace, label="action", linewidth=1.1)
            # The open-loop action that reproduces the demonstration exactly:
            # the gap to it is the residual the policy is adding on top.
            ax.plot(
                time_s,
                ideal_trace,
                label="ideal (demo)",
                linewidth=1.1,
                linestyle="--",
                color="0.25",
            )
            ax.plot(
                time_s,
                delta[:, index],
                label="a_t - a_(t-1)",
                linewidth=0.9,
                alpha=0.75,
            )
            ax.axhline(0.0, color="0.5", linestyle=":", linewidth=0.8)

            sign = np.sign(trace)
            flip_rate = (
                float(np.mean(sign[1:] != sign[:-1])) if trace.size > 1 else 0.0
            )
            finite_delta = delta[1:, index]
            rms_delta = (
                float(np.sqrt(np.nanmean(finite_delta ** 2)))
                if finite_delta.size
                else 0.0
            )
            gap = trace - ideal_trace
            rms_gap = (
                float(np.sqrt(np.nanmean(gap ** 2))) if gap.size else 0.0
            )
            ax.set_title(
                "{}    sign-flip rate {:.3f}    RMS delta {:.4f}    "
                "RMS vs ideal {:.3f}    ±1 -> {:+.3f} rad residual".format(
                    names[index] if index < len(names) else index,
                    flip_rate,
                    rms_delta,
                    rms_gap,
                    scale,
                ),
                fontsize=9,
                loc="left",
            )
            _finish_axis(ax, "Action")

        axes[-1].set_xlabel("Episode time [s]")
        fig.suptitle(
            "{} actions per joint — {}".format(group.capitalize(), self._slug)
        )
        fig.tight_layout()
        path = episode_dir / "{}_action_per_joint.png".format(group)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return {"{}_action_per_joint_png".format(group): str(path)}

    def _save_per_term(self, plt, per_term_dir, data) -> Dict[str, str]:
        """One PNG per reward term: per-step on top, cumulative below."""
        time_s = data["time_s"]
        paths: Dict[str, str] = {}
        terms: Sequence[str] = [key for key, _ in REWARD_TERMS] + ["reward"]
        for key in terms:
            weighted = "weighted_" + key if key != "reward" else "reward"
            cumulative = "cumulative_" + weighted
            fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
            axes[0].plot(time_s, data[key], color="C0", linewidth=1.4, label=key)
            if weighted != key:
                axes[0].plot(
                    time_s,
                    data[weighted],
                    color="C2",
                    linewidth=1.4,
                    linestyle="--",
                    label="weighted",
                )
            _finish_axis(axes[0], "Per-step")
            axes[1].plot(
                time_s, data[cumulative], color="C1", linewidth=1.4, label="cumulative"
            )
            _finish_axis(axes[1], "Cumulative")
            axes[1].set_xlabel("Episode time [s]")
            fig.suptitle("{} — {}".format(self._slug, key))
            fig.tight_layout()
            path = per_term_dir / "{}.png".format(key)
            fig.savefig(path, dpi=150)
            plt.close(fig)
            paths["per_term/{}".format(key)] = str(path)
        return paths
