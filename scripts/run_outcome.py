#!/usr/bin/env python3
"""Classify how a finished training run ended, and whether to restart it.

    python scripts/run_outcome.py --run-name pg830_blind --want 12000

Prints one shell-friendly line:

    status=crashed reached=5451 want=12000 lift=0.2994 contact=0.4902 verdict=retry

`status` says what happened, `verdict` says what the queue should do:

    done   the run finished its schedule
    retry  it stopped early while still worth the GPU, so relaunch it smaller
    skip   it stopped early and the numbers say another attempt is not worth it

Deliberately conservative: anything it cannot read confidently comes back as
`retry`, because losing a run that was working costs far more than one wasted
relaunch.
"""

import argparse
import json
import glob
import os


# A crashed run that got this far has had a fair chance to show progress, so
# its metrics can be trusted to decide whether another attempt is worthwhile.
JUDGEMENT_ITERATION = 3000
# Peak cube lift below this is the cube never leaving the table.
MINIMUM_LIFT_M = 0.02


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--want", type=int, required=True)
    parser.add_argument("--log-root", default="logs/simtoolreal")
    return parser.parse_args()


def newest_run_dir(log_root, run_name):
    matches = sorted(glob.glob(os.path.join(log_root, "*_" + run_name)))
    return matches[-1] if matches else None


def read_metrics(run_dir):
    path = os.path.join(run_dir, "metrics.jsonl")
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return []
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def mean_tail(rows, key, count=300):
    values = [r[key] for r in rows[-count:] if key in r]
    return sum(values) / len(values) if values else float("nan")


def main():
    args = parse_args()
    run_dir = newest_run_dir(args.log_root, args.run_name)
    if run_dir is None:
        print("status=missing reached=0 want=%d lift=nan contact=nan verdict=retry"
              % args.want)
        return

    rows = read_metrics(run_dir)
    reached = rows[-1].get("iteration", 0) if rows else 0
    lift = mean_tail(rows, "episode_max_peak_object_com_lift_m")
    contact = mean_tail(rows, "mean_fingertip_contact_fraction")
    diverged = os.path.isfile(os.path.join(run_dir, "diverged_model.pt"))

    if reached >= args.want - 1:
        status, verdict = "complete", "done"
    elif diverged:
        # The divergence guard fired: the policy blew up, which a smaller batch
        # does not fix. Worth a different hyperparameter, not another attempt.
        status, verdict = "diverged", "skip"
    elif reached >= JUDGEMENT_ITERATION and not (lift >= MINIMUM_LIFT_M):
        # Far enough in to have shown something, and the cube never left the
        # table: relaunching the same configuration buys nothing.
        status, verdict = "crashed", "skip"
    else:
        status, verdict = "crashed", "retry"

    print("status=%s reached=%d want=%d lift=%.4f contact=%.4f verdict=%s"
          % (status, reached, args.want, lift, contact, verdict))


if __name__ == "__main__":
    main()
