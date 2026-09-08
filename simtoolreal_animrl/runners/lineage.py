"""Where a run's starting weights came from.

A warm start uses ``--resume``, which writes into the checkpoint's own
directory, so the checkpoint has to be copied into the new run first. That copy
is what breaks the chain: ``runtime.resume`` then points at a file inside the
new run and says nothing about which run actually produced those weights.

``scripts/prepare_warm_start.py`` leaves a sidecar beside the copy naming the
origin, and ``resolve_lineage`` folds it into the record ``train.py`` saves, so
a run's ancestry can be read back from its own config.json.
"""

import json
from pathlib import Path


SIDECAR_NAME = "checkpoint_source.json"


def read_sidecar(checkpoint_path):
    """Return the sidecar written beside a copied checkpoint, or None."""
    sidecar = Path(checkpoint_path).resolve().parent / SIDECAR_NAME
    if not sidecar.is_file():
        return None
    try:
        with sidecar.open("r", encoding="utf-8") as handle:
            content = json.load(handle)
    except (OSError, ValueError):
        return None
    return content if isinstance(content, dict) else None


def resolve_lineage(resume_path, checkpoint_infos=None):
    """Describe the starting point of a run, as far as it can be established.

    ``resume_path`` is what the user passed to --resume; ``checkpoint_infos`` is
    the ``infos`` dict inside that checkpoint when it has already been read.
    Returns None for a run that started from random weights, so a scratch run
    records no lineage rather than an empty one.
    """
    if resume_path is None:
        return None
    checkpoint = Path(resume_path)
    lineage = {
        "resume_checkpoint": str(checkpoint),
        "checkpoint_name": checkpoint.name,
        # Correct for a direct resume, and overwritten below by the sidecar when
        # the checkpoint was copied out of another run.
        "parent_run": checkpoint.resolve().parent.name,
        "copied": False,
    }
    sidecar = read_sidecar(checkpoint)
    if sidecar is not None:
        lineage["copied"] = True
        for key in (
            "parent_run",
            "parent_checkpoint",
            "parent_iteration",
            "parent_best_evaluation_score",
            "copied_at",
        ):
            if key in sidecar:
                lineage[key] = sidecar[key]
    if isinstance(checkpoint_infos, dict):
        for source, target in (
            ("iteration", "checkpoint_iteration"),
            ("next_iteration", "checkpoint_next_iteration"),
            ("best_evaluation_score", "checkpoint_best_evaluation_score"),
            ("total_timesteps", "checkpoint_total_timesteps"),
        ):
            if source in checkpoint_infos:
                lineage[target] = checkpoint_infos[source]
    return lineage


def ancestry(run_dir, log_root=None, limit=20):
    """Walk config.json lineage backwards, newest first, to the scratch run.

    Stops at the first run whose config records no lineage, and guards against
    a cycle or a missing ancestor rather than raising.
    """
    run_dir = Path(run_dir)
    log_root = Path(log_root) if log_root is not None else run_dir.parent
    chain = []
    seen = set()
    current = run_dir
    for _ in range(int(limit)):
        if current is None or current.name in seen:
            break
        seen.add(current.name)
        chain.append(current.name)
        config = current / "config.json"
        if not config.is_file():
            break
        try:
            with config.open("r", encoding="utf-8") as handle:
                saved = json.load(handle)
        except (OSError, ValueError):
            break
        lineage = saved.get("lineage")
        if not isinstance(lineage, dict):
            break
        parent = lineage.get("parent_run")
        if not parent:
            break
        current = log_root / parent
    return chain
