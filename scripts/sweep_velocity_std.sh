#!/usr/bin/env bash
# Sequential sweep over the velocity reward std.
#
#   ./scripts/sweep_velocity_std.sh                 # default values below
#   ./scripts/sweep_velocity_std.sh 0.2 0.3 0.5     # explicit values
#
# Each value trains in its own run directory named velstd_<value>. Extra
# train.py flags can be passed through the SWEEP_ARGS environment variable:
#
#   SWEEP_ARGS="--iterations 1500 --num-envs 2048" ./scripts/sweep_velocity_std.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/simone/.venv/bin/python3}"
VALUES=("${@:-}")
if [ -z "${VALUES[0]}" ]; then
    VALUES=(0.15 0.3 0.5 1.0 2.0)
fi

echo "Sweeping rewards.velocity_std_rad_per_s over: ${VALUES[*]}"
echo

for value in "${VALUES[@]}"; do
    run_name="velstd_${value}"
    echo "=============================================================="
    echo " ${run_name}   ($(date +%H:%M:%S))"
    echo "=============================================================="
    # Keep going if one configuration diverges, so the sweep still covers
    # the remaining values.
    if ! "$PYTHON" "$REPO_ROOT/scripts/train.py" \
        --run-name "$run_name" \
        --set "rewards.velocity_std_rad_per_s=${value}" \
        ${SWEEP_ARGS:-}; then
        echo "!! ${run_name} failed; continuing with the next value" >&2
    fi
    echo
done

echo "Sweep complete. Runs are under $REPO_ROOT/logs/simtoolreal/"
