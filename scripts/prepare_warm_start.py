#!/usr/bin/env python3
"""Stage a checkpoint so a warm start can resume from it without overwriting it.

    python scripts/prepare_warm_start.py \\
        --checkpoint logs/simtoolreal/<parent>/best_model.pt \\
        --run-name smooth_warm2

``--resume`` writes into the directory holding the checkpoint it was given, so
resuming directly from a finished run would overwrite that run. This creates a
fresh run directory, copies the checkpoint into it, and -- the point of the
script -- leaves a sidecar naming the run the weights actually came from, which
``train.py`` folds into the new run's config.json as its lineage.

Prints the path to pass to --resume.
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from simtoolreal_animrl.runners.lineage import SIDECAR_NAME  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--log-root",
        type=Path,
        default=REPO_ROOT / "logs" / "simtoolreal",
        help="Directory the new run directory is created in.",
    )
    return parser.parse_args()


def checkpoint_infos(path):
    """Read the checkpoint's metadata, without needing a GPU or isaacgym."""
    try:
        import torch
    except ImportError:
        return {}
    try:
        loaded = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        loaded = torch.load(str(path), map_location="cpu")
    except Exception:
        return {}
    infos = loaded.get("infos") if isinstance(loaded, dict) else None
    return infos if isinstance(infos, dict) else {}


def main():
    args = parse_args()
    source = args.checkpoint.expanduser().resolve()
    if not source.is_file():
        raise SystemExit("No such checkpoint: {}".format(source))

    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = args.log_root.expanduser().resolve() / "{}_{}".format(
        stamp, args.run_name
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    target = run_dir / source.name
    shutil.copy2(source, target)

    infos = checkpoint_infos(source)
    sidecar = {
        "parent_run": source.parent.name,
        "parent_checkpoint": str(source),
        "copied_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    if "iteration" in infos:
        sidecar["parent_iteration"] = infos["iteration"]
    if "best_evaluation_score" in infos:
        sidecar["parent_best_evaluation_score"] = infos["best_evaluation_score"]
    with (run_dir / SIDECAR_NAME).open("w", encoding="utf-8") as handle:
        json.dump(sidecar, handle, indent=2, sort_keys=True)

    print("Staged {} from {}".format(target, sidecar["parent_run"]))
    print(target)


if __name__ == "__main__":
    main()
