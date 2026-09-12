#!/usr/bin/env python3
"""Build the phone-friendly training dashboard from the local run logs.

The dashboard is a single self-contained HTML file: metric series, run
configuration diffs and the recorded videos are all inlined, so it can be
published as an Artifact and read from a phone with nothing else running.

    python scripts/build_training_dashboard.py
    python scripts/build_training_dashboard.py --featured 4 --videos-per-run 6

The companion template is scripts/dashboard_template.html.
"""

import argparse
import base64
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from simtoolreal_animrl.cfg import (  # noqa: E402  (needs the path above)
    SimToolRealCfg,
    SimToolRealTrainCfg,
    config_to_dict,
)

TEMPLATE = Path(__file__).resolve().parent / "dashboard_template.html"
DATA_MARKER = "__DASHBOARD_DATA__"

# One series per line chart. Every entry is (key, label, unit, "up" or "down"
# for the direction that means progress).
SERIES = [
    ("mean_reward", "Mean step reward", "", "up"),
    ("evaluation_score", "Evaluation score", "", "up"),
    ("episode_return", "Episode return", "", "up"),
    ("episode_length", "Episode length", "steps", "up"),
    ("episode_mean_object_position_error_m", "Cube position error", "m", "down"),
    ("episode_max_peak_object_com_lift_m", "Best cube lift", "m", "up"),
    ("episode_mean_fingertip_object_distance_m", "Fingertip-to-cube", "m", "down"),
    ("episode_early_termination_fraction", "Early termination", "frac", "down"),
    ("mean_action_std", "Action std", "", "flat"),
    ("object_assist_scale", "Object-assist scale", "", "flat"),
    # Both evaluation cohorts, because they routinely disagree and only one of
    # them is about the policy. "uniform" follows the training RSI
    # distribution; "fixed" replays four phases of the demonstration, three of
    # which start outside the window pregrasp_mixture actually trains, so its
    # early-termination fraction measures coverage rather than competence --
    # and evaluation_score subtracts that fraction outright.
    (
        "evaluation_uniform_early_termination_fraction",
        "Early termination (trained starts)",
        "frac",
        "down",
    ),
    (
        "evaluation_fixed_early_termination_fraction",
        "Early termination (fixed starts)",
        "frac",
        "down",
    ),
    (
        "evaluation_uniform_mean_peak_object_com_lift_m",
        "Cube lift (trained starts)",
        "m",
        "up",
    ),
    # Scale-free, so it is the one smoothness number comparable across runs
    # with different reward sigmas.
    (
        "evaluation_fixed_mean_rms_action_rate",
        "Arm action rate (vibration)",
        "",
        "down",
    ),
    # The hand is where the chatter starts: it twitches and the arm rings.
    (
        "evaluation_fixed_mean_rms_hand_action_rate",
        "Hand action rate (vibration)",
        "",
        "down",
    ),
]
SERIES_KEYS = [key for key, _, _, _ in SERIES]

# Evaluation figures inlined per featured run, as (file stem, label, caption).
# The per-joint action plots are the ones that show the commanded action, the
# open-loop action that would reproduce the demonstration, and the step-to-step
# difference on the same axes, which is what exposes chatter.
EVAL_FIGURES = [
    ("arm_action_per_joint", "Arm action per joint",
     "action vs ideal (demo) vs a_t - a_(t-1), one panel per arm joint"),
    ("hand_action_per_joint", "Hand action per joint",
     "action vs ideal (demo) vs a_t - a_(t-1), one panel per hand joint"),
]

# Fields whose value is per-run noise rather than an experiment choice.
IGNORED_CONFIG_PATHS = {
    "env.num_envs",
    "env.play",
    "env.debug",
    "seed",
    "viewer.enable_viewer",
    "viewer.camera_position",
    "viewer.camera_lookat",
    "viewer.training_camera_enabled",
    "viewer.reference_ghost",
    "train.runner.run_name",
    "train.runner.experiment_name",
    "train.runner.record_video",
    "train.runner.record_gif",
    "train.runner.max_iterations",
    "train.runner.wandb",
    "train.runner.wandb_group",
    "train.runner.tensorboard",
    "train.runner.tensorboard_flush_secs",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-root",
        type=Path,
        default=REPO_ROOT / "logs" / "simtoolreal",
        help="Directory holding the run directories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "logs" / "dashboard" / "training_dashboard.html",
        help="HTML file to write.",
    )
    parser.add_argument(
        "--featured",
        type=int,
        default=5,
        help="Most recent runs that get full charts, videos and config diff.",
    )
    parser.add_argument(
        "--videos-per-run",
        type=int,
        default=5,
        help="Recorded clips embedded per featured run (newest first).",
    )
    parser.add_argument(
        "--eval-videos-per-run",
        type=int,
        default=2,
        help="Evaluation replays embedded per featured run (newest first).",
    )
    parser.add_argument(
        "--eval-video-width",
        type=int,
        default=560,
        help="Width the embedded evaluation replays are re-encoded to.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=240,
        help="Points per series after downsampling.",
    )
    parser.add_argument(
        "--live-minutes",
        type=float,
        default=6.0,
        help="A run whose metrics changed this recently counts as live.",
    )
    parser.add_argument(
        "--video-width",
        type=int,
        default=480,
        help="Width the embedded clips are re-encoded to.",
    )
    parser.add_argument(
        "--video-crf",
        type=int,
        default=28,
        help="x264 quality for the embedded clips (higher is smaller).",
    )
    parser.add_argument(
        "--max-video-mb",
        type=float,
        default=7.0,
        help="Total budget for embedded video, before base64 expansion.",
    )
    parser.add_argument(
        "--figure-width",
        type=int,
        default=1100,
        help="Width the embedded evaluation plots are re-encoded to.",
    )
    parser.add_argument(
        "--max-figure-mb",
        type=float,
        default=5.0,
        help="Total budget for embedded evaluation plots, before base64.",
    )
    parser.add_argument(
        "--no-videos",
        dest="videos",
        action="store_false",
        help="Skip video embedding entirely (much faster).",
    )
    parser.add_argument(
        "--since",
        default=None,
        help="Only include runs started on or after this YYYY-MM-DD date.",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------
# Discovery


def discover_runs(log_root, since=None):
    """Return every directory that holds both a config and a metrics file."""
    runs = []
    for config_path in sorted(log_root.glob("**/config.json")):
        run_dir = config_path.parent
        metrics_path = run_dir / "metrics.jsonl"
        if not metrics_path.is_file() or metrics_path.stat().st_size == 0:
            continue
        started = run_started_at(run_dir)
        if since is not None and started is not None and started < since:
            continue
        runs.append(run_dir)
    return runs


def run_started_at(run_dir):
    """Parse the leading YYYY-MM-DD[_HHMMSS] stamp the run directories carry."""
    for part in (run_dir.name, run_dir.parent.name):
        match = re.search(r"(\d{4}-\d{2}-\d{2})(?:_(\d{6}))?", part)
        if match:
            return match.group(1)
    return None


def started_timestamp(run_dir):
    """Sort key: the directory stamp, falling back to the config's mtime."""
    match = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{6})", str(run_dir))
    if match:
        try:
            return datetime.strptime(
                match.group(1) + match.group(2), "%Y-%m-%d%H%M%S"
            ).timestamp()
        except ValueError:
            pass
    return (run_dir / "config.json").stat().st_mtime


# --------------------------------------------------------------------------
# Metrics


def count_lines(path):
    total = 0
    with path.open("rb") as handle:
        for _ in handle:
            total += 1
    return total


def load_metrics(path, max_points):
    """Downsample metrics.jsonl into the series the dashboard draws.

    Evaluation rows are sparse and always kept: they are the only unassisted
    measurement of the policy and would otherwise be sampled away.
    """
    total = count_lines(path)
    stride = max(1, total // max(1, max_points))
    series = {key: [] for key in SERIES_KEYS}
    last = None
    first = None
    recent_iteration_times = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for index, line in enumerate(handle):
            is_eval = '"evaluation_score"' in line
            if not (is_eval or index % stride == 0 or index == total - 1):
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            iteration = row.get("iteration")
            if iteration is None:
                continue
            if first is None:
                first = row
            last = row
            for key in SERIES_KEYS:
                value = row.get(key)
                if isinstance(value, (int, float)) and math.isfinite(value):
                    series[key].append([int(iteration), round(float(value), 6)])
            duration = row.get("iteration_time_s")
            if isinstance(duration, (int, float)):
                recent_iteration_times.append(float(duration))
    series = {key: values for key, values in series.items() if values}
    return {
        "series": series,
        "first": first or {},
        "last": last or {},
        "rows": total,
        "seconds_per_iteration": (
            sum(recent_iteration_times[-40:]) / len(recent_iteration_times[-40:])
            if recent_iteration_times
            else None
        ),
    }


def trend(points, window=0.25):
    """Signed change of a series' tail, as (delta, relative_delta)."""
    if len(points) < 6:
        return None, None
    tail = max(3, int(len(points) * window))
    early = [value for _, value in points[-2 * tail:-tail]] or [points[0][1]]
    late = [value for _, value in points[-tail:]]
    early_mean = sum(early) / len(early)
    late_mean = sum(late) / len(late)
    delta = late_mean - early_mean
    scale = max(abs(early_mean), 1e-9)
    return delta, delta / scale


# --------------------------------------------------------------------------
# Configuration


def flatten(values, prefix=""):
    flat = {}
    for key, value in values.items():
        path = "{}.{}".format(prefix, key) if prefix else key
        if isinstance(value, dict):
            flat.update(flatten(value, path))
        else:
            flat[path] = value
    return flat


def default_flat_config():
    env = flatten(config_to_dict(SimToolRealCfg()))
    train = flatten(config_to_dict(SimToolRealTrainCfg()), "train")
    env.update(train)
    return env


def config_difference(saved, defaults):
    """Everything this run set differently from the repository defaults."""
    flat = flatten(saved.get("env_cfg", {}))
    flat.update(flatten(saved.get("train_cfg", {}), "train"))
    rows = []
    for path in sorted(set(flat) | set(defaults)):
        if path in IGNORED_CONFIG_PATHS:
            continue
        mine = flat.get(path, "<absent>")
        theirs = defaults.get(path, "<absent>")
        if mine == theirs:
            continue
        if isinstance(mine, float) and isinstance(theirs, float):
            if math.isclose(mine, theirs, rel_tol=1e-9, abs_tol=1e-12):
                continue
        rows.append({"path": path, "value": mine, "default": theirs})
    return rows


def reconstruct_command(runtime):
    """Rebuild the train.py invocation from the runtime block config.json saved."""
    parts = ["python scripts/train.py"]
    simple = [
        ("iterations", "--iterations"),
        ("num_envs", "--num-envs"),
        ("seed", "--seed"),
        ("run_name", "--run-name"),
        ("save_interval", "--save-interval"),
        ("eval_interval", "--eval-interval"),
        ("eval_num_envs", "--eval-num-envs"),
        ("resume", "--resume"),
        ("log_dir", "--log-dir"),
    ]
    for key, flag in simple:
        value = runtime.get(key)
        if value not in (None, ""):
            parts.append("{} {}".format(flag, value))
    if runtime.get("record_video"):
        parts.append("--record-video")
    if runtime.get("no_periodic_eval"):
        parts.append("--no-periodic-eval")
    if runtime.get("object_assist"):
        parts.append("--object-assist")
    for override in runtime.get("overrides") or []:
        parts.append("--set {}".format(override))
    return " ".join(parts)


# --------------------------------------------------------------------------
# Videos and figures


def encode_videos(run_dir, count, width, crf, budget_bytes, kind="training"):
    """Re-encode the newest clips small enough to inline as data URIs.

    ``training`` clips are the 10 s rollouts the trainer records every few
    hundred updates, named by iteration. ``evaluation`` clips are the full
    deterministic replays evaluate.py writes, which carry the green reference
    robot beside the policy robot, and are ordered by mtime instead.
    """
    video_dir = run_dir / ("videos" if kind == "training" else "eval_videos")
    if not video_dir.is_dir() or count <= 0:
        return [], 0
    order = (
        iteration_of
        if kind == "training"
        else (lambda path: path.stat().st_mtime)
    )
    clips = sorted(video_dir.glob("*.mp4"), key=order, reverse=True)[:count]
    encoded = []
    used = 0
    with tempfile.TemporaryDirectory() as workspace:
        target = Path(workspace) / "clip.mp4"
        for clip in sorted(clips, key=order):
            if used >= budget_bytes:
                break
            command = [
                "ffmpeg", "-v", "error", "-y", "-i", str(clip),
                "-vf", "scale={}:-2".format(int(width)),
                "-r", "24",
                "-c:v", "libx264", "-profile:v", "baseline", "-level", "3.0",
                "-pix_fmt", "yuv420p", "-crf", str(int(crf)),
                "-movflags", "+faststart", "-an", str(target),
            ]
            try:
                subprocess.run(command, check=True)
            except (OSError, subprocess.CalledProcessError):
                continue
            payload = target.read_bytes()
            used += len(payload)
            encoded.append(
                {
                    "iteration": iteration_of(clip),
                    "name": clip.stem,
                    "bytes": len(payload),
                    "src": "data:video/mp4;base64,"
                    + base64.b64encode(payload).decode("ascii"),
                }
            )
    return encoded, used


def iteration_of(path):
    match = re.search(r"iteration_(\d+)", path.stem)
    return int(match.group(1)) if match else -1


def encode_eval_figure(run_dir, stem, width):
    """Inline one evaluation plot by file stem, scaled and re-encoded as JPEG.

    These plots are one tall panel per joint -- the hand one is 20 -- so the
    source PNG runs to megabytes and has to be recompressed to survive inlining.
    """
    candidates = sorted(run_dir.glob("eval_plots/**/{}.png".format(stem)))
    if not candidates:
        return None
    source = candidates[-1]
    with tempfile.TemporaryDirectory() as workspace:
        target = Path(workspace) / "{}.jpg".format(stem)
        command = [
            "ffmpeg", "-v", "error", "-y", "-i", str(source),
            "-vf", "scale={}:-2".format(int(width)),
            "-q:v", "7", str(target),
        ]
        try:
            subprocess.run(command, check=True)
        except (OSError, subprocess.CalledProcessError):
            return None
        payload = target.read_bytes()
    return {
        "name": str(source.relative_to(run_dir)),
        "bytes": len(payload),
        "src": "data:image/jpeg;base64,"
        + base64.b64encode(payload).decode("ascii"),
    }


def encode_eval_figures(run_dir, width, budget_bytes):
    """Inline the per-joint action plots for one run, inside a byte budget."""
    figures = []
    used = 0
    for stem, label, caption in EVAL_FIGURES:
        if used >= budget_bytes:
            break
        figure = encode_eval_figure(run_dir, stem, width)
        if figure is None:
            continue
        if used + figure["bytes"] > budget_bytes and figures:
            break
        figure["label"] = label
        figure["caption"] = caption
        figures.append(figure)
        used += figure["bytes"]
    return figures, used


def encode_overview_figure(run_dir, width=880):
    """Inline the final evaluation overview plot, if the run produced one."""
    candidates = sorted(run_dir.glob("eval_plots/**/overview.png"))
    if not candidates:
        return None
    source = candidates[-1]
    with tempfile.TemporaryDirectory() as workspace:
        target = Path(workspace) / "overview.jpg"
        command = [
            "ffmpeg", "-v", "error", "-y", "-i", str(source),
            "-vf", "scale={}:-2".format(int(width)),
            "-q:v", "6", str(target),
        ]
        try:
            subprocess.run(command, check=True)
        except (OSError, subprocess.CalledProcessError):
            return None
        payload = target.read_bytes()
    return {
        "name": str(source.relative_to(run_dir)),
        "src": "data:image/jpeg;base64,"
        + base64.b64encode(payload).decode("ascii"),
    }


# --------------------------------------------------------------------------
# Status and findings


def running_training_commands():
    """The train.py processes alive on this machine right now."""
    try:
        output = subprocess.run(
            ["ps", "-eo", "pid,etimes,args"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    processes = []
    for line in output.splitlines()[1:]:
        if "scripts/train.py" not in line or "ps -eo" in line:
            continue
        pid, elapsed, args = line.strip().split(None, 2)
        processes.append(
            {"pid": int(pid), "elapsed_s": int(elapsed), "command": args}
        )
    return processes


def gpu_status():
    query = "name,utilization.gpu,memory.used,memory.total,temperature.gpu"
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=" + query, "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    fields = [item.strip() for item in output.splitlines()[0].split(",")]
    if len(fields) != 5:
        return None
    return {
        "name": fields[0],
        "utilization": fields[1],
        "memory_used": fields[2],
        "memory_total": fields[3],
        "temperature_c": fields[4],
    }


def classify(run, live_seconds):
    last = run["last"]
    age = time.time() - run["updated_at"]
    if last.get("divergence_abort"):
        return "diverged"
    if age <= live_seconds:
        return "live"
    if run["planned_iterations"] and run["iteration"] + 1 >= run[
        "planned_iterations"
    ]:
        return "finished"
    return "stopped"


def findings(run):
    """Short plain-language readings of how the run is behaving."""
    notes = []
    series = run["series"]
    if run["status"] == "diverged":
        notes.append(
            {
                "level": "critical",
                "text": "Aborted by the divergence guard. The action std or the "
                "target clipping ran away, so the policy is not recoverable.",
            }
        )
    reward = series.get("mean_reward", [])
    delta, relative = trend(reward)
    if delta is not None:
        if relative > 0.02:
            notes.append(
                {
                    "level": "good",
                    "text": "Reward still climbing: {:+.3f} over the last "
                    "quarter of the run.".format(delta),
                }
            )
        elif relative < -0.02:
            notes.append(
                {
                    "level": "warning",
                    "text": "Reward is falling: {:+.3f} over the last quarter "
                    "of the run.".format(delta),
                }
            )
        else:
            notes.append(
                {
                    "level": "neutral",
                    "text": "Reward has plateaued ({:+.3f} over the last "
                    "quarter).".format(delta),
                }
            )
    lift = series.get("episode_max_peak_object_com_lift_m", [])
    if lift:
        best = max(value for _, value in lift)
        if best < 0.02:
            notes.append(
                {
                    "level": "warning",
                    "text": "The cube is never really lifted: best peak is "
                    "{:.0f} mm above its reset height.".format(best * 1000.0),
                }
            )
        else:
            notes.append(
                {
                    "level": "good",
                    "text": "Best cube lift {:.0f} mm above the reset "
                    "height.".format(best * 1000.0),
                }
            )
    early = series.get("episode_early_termination_fraction", [])
    if early:
        latest = early[-1][1]
        if latest > 0.9:
            notes.append(
                {
                    "level": "warning",
                    "text": "{:.0f}% of episodes still end on a tracking "
                    "failure rather than reaching the horizon.".format(
                        latest * 100.0
                    ),
                }
            )
    std = series.get("mean_action_std", [])
    if std and std[-1][1] > 4.0:
        notes.append(
            {
                "level": "warning",
                "text": "Action std is {:.2f} and growing; the divergence "
                "guard fires at 15.".format(std[-1][1]),
            }
        )
    assist = series.get("object_assist_scale", [])
    if assist:
        notes.append(
            {
                "level": "neutral",
                "text": "Object assist is at scale {:.2f}; the policy carries "
                "the rest of the load.".format(assist[-1][1]),
            }
        )
    return notes


# --------------------------------------------------------------------------
# Assembly


def build_run(run_dir, defaults, args, featured, budget_bytes, figure_budget_bytes=0):
    with (run_dir / "config.json").open("r", encoding="utf-8") as handle:
        saved = json.load(handle)
    metrics_path = run_dir / "metrics.jsonl"
    metrics = load_metrics(
        metrics_path, args.max_points if featured else 60
    )
    last = metrics["last"]
    runtime = saved.get("runtime", {})
    runner = saved.get("train_cfg", {}).get("runner", {})
    planned = runtime.get("iterations") or runner.get("max_iterations")
    iteration = int(last.get("iteration", 0))
    checkpoints = sorted(
        (path.name for path in run_dir.glob("model_*.pt")),
        key=lambda name: int(re.findall(r"\d+", name)[0]),
    )
    run = {
        "id": str(run_dir.relative_to(args.log_root)),
        "name": run_dir.name,
        "group": (
            str(run_dir.parent.relative_to(args.log_root))
            if run_dir.parent != args.log_root
            else None
        ),
        "path": str(run_dir),
        "started": run_started_at(run_dir),
        "started_at": started_timestamp(run_dir),
        "updated_at": metrics_path.stat().st_mtime,
        "iteration": iteration,
        "planned_iterations": int(planned) if planned else None,
        "series": metrics["series"],
        "last": last,
        "seconds_per_iteration": metrics["seconds_per_iteration"],
        "seed": saved.get("env_cfg", {}).get("seed"),
        "num_envs": saved.get("env_cfg", {}).get("env", {}).get("num_envs"),
        "episode_length": saved.get("env_cfg", {})
        .get("env", {})
        .get("episode_length"),
        "demonstration": Path(
            str(saved.get("env_cfg", {}).get("motion", {}).get("file", ""))
        ).name,
        "command": reconstruct_command(runtime),
        "overrides": runtime.get("overrides") or [],
        "differences": config_difference(saved, defaults),
        "checkpoints": len(checkpoints),
        "best_checkpoint": (run_dir / "best_model.pt").is_file(),
        "featured": featured,
        "videos": [],
        "video_bytes": 0,
        "eval_figures": [],
        "figure_bytes": 0,
        "overview": None,
        "total_time_s": last.get("total_time_s"),
        "total_timesteps": last.get("total_timesteps"),
    }
    run["eval_videos"] = []
    run["status"] = classify(run, args.live_minutes * 60.0)
    run["findings"] = findings(run) if featured else []
    if featured and args.videos:
        videos, used = encode_videos(
            run_dir,
            args.videos_per_run,
            args.video_width,
            args.video_crf,
            budget_bytes,
        )
        run["videos"] = videos
        run["video_bytes"] = used
        # The evaluation replay is the one that shows the reference robot, so
        # it is worth its bytes even when the training clips already fit.
        replays, replay_bytes = encode_videos(
            run_dir,
            args.eval_videos_per_run,
            args.eval_video_width,
            args.video_crf,
            max(0, budget_bytes - used),
            kind="evaluation",
        )
        run["eval_videos"] = replays
        run["video_bytes"] += replay_bytes
        if not videos and not replays:
            run["overview"] = encode_overview_figure(run_dir)
    if featured and figure_budget_bytes > 0:
        figures, figure_used = encode_eval_figures(
            run_dir, args.figure_width, figure_budget_bytes
        )
        run["eval_figures"] = figures
        run["figure_bytes"] = figure_used
    return run


def main():
    args = parse_args()
    log_root = args.log_root.expanduser().resolve()
    args.log_root = log_root
    if not log_root.is_dir():
        raise SystemExit("No such log root: {}".format(log_root))
    if args.videos and shutil.which("ffmpeg") is None:
        print("ffmpeg not found; embedding no videos")
        args.videos = False

    defaults = default_flat_config()
    run_dirs = discover_runs(log_root, args.since)
    if not run_dirs:
        raise SystemExit("No runs with metrics found under {}".format(log_root))
    run_dirs.sort(key=started_timestamp, reverse=True)
    print("Found {} runs under {}".format(len(run_dirs), log_root))

    budget = args.max_video_mb * 1024 * 1024
    figure_budget = args.max_figure_mb * 1024 * 1024
    runs = []
    for index, run_dir in enumerate(run_dirs):
        featured = index < args.featured
        try:
            run = build_run(run_dir, defaults, args, featured, budget, figure_budget)
        except Exception as error:  # a broken run must not sink the dashboard
            print("  skipped {}: {}".format(run_dir.name, error))
            continue
        budget -= run["video_bytes"]
        figure_budget -= run["figure_bytes"]
        runs.append(run)
        print(
            "  {:<56} it {:>6} {:<9} {} clips {} plots".format(
                run["id"][:56],
                run["iteration"],
                run["status"],
                len(run["videos"]),
                len(run["eval_figures"]),
            )
        )

    # A live run always leads, then the most recently touched.
    runs.sort(
        key=lambda run: (run["status"] != "live", -run["updated_at"])
    )
    payload = {
        "generated_at": time.time(),
        "generated_iso": datetime.now().astimezone().isoformat(timespec="seconds"),
        "host": subprocess.run(
            ["hostname"], capture_output=True, text=True
        ).stdout.strip(),
        "log_root": str(log_root),
        "repo": str(REPO_ROOT),
        "gpu": gpu_status(),
        "processes": running_training_commands(),
        "series_meta": [
            {"key": key, "label": label, "unit": unit, "better": better}
            for key, label, unit, better in SERIES
        ],
        "runs": runs,
    }

    template = TEMPLATE.read_text(encoding="utf-8")
    if DATA_MARKER not in template:
        raise SystemExit("The template lost its {} marker".format(DATA_MARKER))
    html = template.replace(
        DATA_MARKER, json.dumps(payload, separators=(",", ":"))
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(
        "\nWrote {} ({:.1f} MB, {} runs, {} embedded clips)".format(
            args.output,
            args.output.stat().st_size / 1024 / 1024,
            len(runs),
            sum(len(run["videos"]) for run in runs),
        )
    )


if __name__ == "__main__":
    main()
