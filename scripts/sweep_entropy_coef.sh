#!/usr/bin/env bash
# Run four grasp-training experiments sequentially from the same checkpoint.
#
# Every run uses the current repository configuration (including pre-grasp RSI
# and fingertip-object proximity shaping), changes only PPO entropy_coef, and
# saves into a fresh directory. A failed run is reported but does not prevent
# the remaining entropy values from being tested.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/simone/.venv/bin/python}"
CHECKPOINT="${CHECKPOINT:-$REPO_ROOT/logs/simtoolreal/2026-08-28_173516_no_object_reward/model_7500.pt}"
SWEEP_ID="${SWEEP_ID:-$(date +%Y-%m-%d_%H%M%S)_entropy_sweep}"
RUN_ROOT="$REPO_ROOT/logs/simtoolreal"
ENTROPY_VALUES=(0.01 0.005 0.002 0.001)

if [ ! -x "$PYTHON" ]; then
    echo "Python interpreter is not executable: $PYTHON" >&2
    exit 2
fi
if [ ! -f "$CHECKPOINT" ]; then
    echo "Checkpoint not found: $CHECKPOINT" >&2
    exit 2
fi

mkdir -p "$RUN_ROOT"
echo "Sequential entropy sweep: ${ENTROPY_VALUES[*]}"
echo "Checkpoint: $CHECKPOINT"
echo "Sweep id:   $SWEEP_ID"

failures=()
for entropy_coef in "${ENTROPY_VALUES[@]}"; do
    entropy_label="${entropy_coef/./p}"
    run_dir="$RUN_ROOT/${SWEEP_ID}_entropy_${entropy_label}"

    echo
    echo "============================================================"
    echo " entropy_coef=$entropy_coef"
    echo " run_dir=$run_dir"
    echo " started=$(date --iso-8601=seconds)"
    echo "============================================================"

    if [ -e "$run_dir" ]; then
        echo "Run directory already exists; skipping to avoid mixing logs." >&2
        failures+=("$entropy_coef:directory-exists")
        continue
    fi

    if ! "$PYTHON" "$REPO_ROOT/scripts/train.py" \
        --resume "$CHECKPOINT" \
        --log-dir "$run_dir" \
        --set "table.surface_below_robot_base_m=0.035" \
        --set "train.algorithm.entropy_coef=$entropy_coef"; then
        echo "Training failed for entropy_coef=$entropy_coef; continuing." >&2
        failures+=("$entropy_coef:training-failed")
    fi
done

echo
if [ "${#failures[@]}" -ne 0 ]; then
    echo "Entropy sweep completed with failures: ${failures[*]}" >&2
    exit 1
fi
echo "Entropy sweep completed successfully."
