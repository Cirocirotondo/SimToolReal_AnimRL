"""Saved and interactive diagnostics for one MuJoCo sim2sim rollout."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .constants import ACTION_DIM, ARM_JOINT_NAMES, HAND_JOINT_NAMES


TRACE_KEYS = (
    "reference_indices",
    "policy_actions",
    "reference_actions",
    "action_deltas",
    "actual_joint_positions",
    "applied_position_targets",
    "raw_position_targets",
    "reference_joint_positions",
)


def _validated_arrays(trace: Mapping[str, Sequence]) -> dict[str, np.ndarray]:
    missing = sorted(set(TRACE_KEYS) - set(trace))
    if missing:
        raise ValueError("Rollout trace is missing fields: {}".format(missing))
    arrays = {key: np.asarray(trace[key]) for key in TRACE_KEYS}
    frames = arrays["reference_indices"]
    if frames.ndim != 1 or frames.size == 0:
        raise ValueError("reference_indices must be a non-empty 1-D array")
    for key in TRACE_KEYS[1:]:
        if arrays[key].shape != (frames.size, ACTION_DIM):
            raise ValueError(
                "{} has shape {}, expected ({}, {})".format(
                    key, arrays[key].shape, frames.size, ACTION_DIM
                )
            )
        if not np.all(np.isfinite(arrays[key])):
            raise ValueError("{} contains non-finite values".format(key))
    return arrays


def _plot_joint_grid(
    plt,
    frames: np.ndarray,
    series: Sequence[tuple[str, np.ndarray, str]],
    joint_names: Sequence[str],
    title: str,
    ylabel: str,
    rows: int,
    columns: int,
):
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.2 * columns, 2.25 * rows),
        sharex=True,
        layout="constrained",
    )
    axes = np.asarray(axes).reshape(-1)
    for joint_index, name in enumerate(joint_names):
        axis = axes[joint_index]
        for label, values, style in series:
            axis.plot(
                frames,
                values[:, joint_index],
                style,
                linewidth=1.15,
                label=label,
            )
        axis.set_title(name.replace("_joint", ""), fontsize=9)
        axis.grid(True, alpha=0.28)
        axis.set_ylabel(ylabel, fontsize=8)
        if joint_index == 0:
            axis.legend(loc="best", fontsize=7)
    for axis in axes[len(joint_names) :]:
        axis.set_visible(False)
    figure.suptitle(title)
    figure.supxlabel("demonstration frame")
    return figure


def save_rollout_plots(
    output_dir: Path,
    trace: Mapping[str, Sequence],
    *,
    show: bool,
) -> dict[str, Path]:
    """Write raw rollout data and four per-joint figures, then optionally show."""
    arrays = _validated_arrays(trace)
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not show:
        import matplotlib

        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    frames = arrays["reference_indices"]
    action_series = (
        ("policy action $a_t$", arrays["policy_actions"], "-"),
        ("target/reference action", arrays["reference_actions"], "--"),
        ("$a_t-a_{t-1}$", arrays["action_deltas"], ":"),
    )
    joint_series = (
        ("measured q", arrays["actual_joint_positions"], "-"),
        ("applied PD target", arrays["applied_position_targets"], "--"),
        ("demonstration q", arrays["reference_joint_positions"], ":"),
    )
    figures = {
        "arm_actions": _plot_joint_grid(
            plt,
            frames,
            tuple(
                (label, values[:, :6], style)
                for label, values, style in action_series
            ),
            ARM_JOINT_NAMES,
            "Arm actions: policy, reference, and temporal delta",
            "action",
            3,
            2,
        ),
        "hand_actions": _plot_joint_grid(
            plt,
            frames,
            tuple(
                (label, values[:, 6:], style)
                for label, values, style in action_series
            ),
            HAND_JOINT_NAMES,
            "Hand actions: policy, reference, and temporal delta",
            "action",
            5,
            4,
        ),
        "arm_joint_tracking": _plot_joint_grid(
            plt,
            frames,
            tuple(
                (label, values[:, :6], style)
                for label, values, style in joint_series
            ),
            ARM_JOINT_NAMES,
            "Arm joint tracking",
            "angle [rad]",
            3,
            2,
        ),
        "hand_joint_tracking": _plot_joint_grid(
            plt,
            frames,
            tuple(
                (label, values[:, 6:], style)
                for label, values, style in joint_series
            ),
            HAND_JOINT_NAMES,
            "Hand joint tracking",
            "angle [rad]",
            5,
            4,
        ),
    }
    paths = {"data": output_dir / "rollout_data.npz"}
    np.savez_compressed(paths["data"], **arrays)
    for name, figure in figures.items():
        path = output_dir / "{}.png".format(name)
        figure.savefig(path, dpi=160)
        paths[name] = path
        try:
            figure.canvas.manager.set_window_title(name.replace("_", " "))
        except AttributeError:
            pass
    if show:
        plt.show()
    plt.close("all")
    return paths
