#!/usr/bin/env bash
# Run queued trainings back to back, one at a time, restarting the ones that
# crash and abandoning the ones that are not worth another attempt.
#
#   scripts/run_queue.sh <pid-to-wait-for|0> <queue-dir> <logfile>
#
# The queue directory holds one *.cmd file per job, each a single command line
# containing a unique --run-name and an --iterations count. Jobs run in filename
# order and leave a marker beside themselves when they are settled:
#
#   <job>.done     finished its schedule
#   <job>.skipped  stopped early and scripts/run_outcome.py judged it not worth
#                  another attempt (diverged, or the cube never left the table)
#   <job>.failed   kept crashing until the attempts ran out
#
# A crash is anything that ends the process before its last iteration without
# the divergence guard firing -- in practice the PhysX GPU faults this workload
# hits once the fingers really grip the cube. Each retry halves the environment
# count, which is the lever that reduces contact pairs and GPU memory together.
#
# Each training is launched in its own session so it outlives this runner and
# the shell that started it. The runner then waits on the launched process, so
# killing the runner stops the chain without touching the run in flight.
set -u

wait_pid="$1"
queue_dir="$2"
log="$3"
cd /home/simone/simtoolreal_animrl

# Attempt 1 runs the command as written; each further attempt halves num_envs.
MAX_ATTEMPTS=3
DEFAULT_NUM_ENVS=4096

say() { echo "[$(date -Is)] $*" >> "${log}"; }

# Anchored at the start of the command line so this only matches a real
# interpreter running the trainer, never this runner or the shell above it,
# both of which carry the command in their own argv.
trainer_running() {
  pgrep -f "^[^ ]*python[0-9.]* .*scripts/train\.py" > /dev/null
}

if [ "${wait_pid}" != "0" ]; then
  say "waiting for pid ${wait_pid}"
  while kill -0 "${wait_pid}" 2>/dev/null; do sleep 30; done
  say "pid ${wait_pid} exited"
fi

for job in $(ls "${queue_dir}"/*.cmd 2>/dev/null | sort); do
  name="$(basename "${job}")"
  if [ -e "${job}.done" ] || [ -e "${job}.skipped" ] || [ -e "${job}.failed" ]; then
    say "skipping ${name} (already settled)"
    continue
  fi

  base_command="$(grep -v '^[[:space:]]*#' "${job}" | grep -v '^[[:space:]]*$')"
  base_run_name="$(sed -n 's/.*--run-name[= ]\([^ ]*\).*/\1/p' <<< "${base_command}")"
  want="$(sed -n 's/.*--iterations[= ]\([0-9]*\).*/\1/p' <<< "${base_command}")"
  if [ -z "${base_run_name}" ] || [ -z "${want}" ]; then
    say "ABORTED: ${name} needs both --run-name and --iterations"
    exit 1
  fi

  attempt=1
  while [ "${attempt}" -le "${MAX_ATTEMPTS}" ]; do
    command_line="${base_command}"
    run_name="${base_run_name}"
    if [ "${attempt}" -gt 1 ]; then
      # Halve the environments per retry, starting from whatever the command
      # asked for, and give the attempt its own run name so the run directory
      # and the pid lookup below stay unambiguous.
      current_envs="$(sed -n 's/.*--num-envs[= ]\([0-9]*\).*/\1/p' <<< "${base_command}")"
      current_envs="${current_envs:-${DEFAULT_NUM_ENVS}}"
      for _ in $(seq 2 "${attempt}"); do
        current_envs=$((current_envs / 2))
      done
      run_name="${base_run_name}_n${current_envs}"
      command_line="$(sed "s/--num-envs[= ][0-9]*//" <<< "${command_line}")"
      command_line="$(sed "s/--run-name[= ]${base_run_name}/--run-name ${run_name}/" <<< "${command_line}")"
      command_line="${command_line} --num-envs ${current_envs}"
      say "retry ${attempt}/${MAX_ATTEMPTS} for ${name} at ${current_envs} environments"
    fi

    # The previous run needs a moment to flush its final evaluation and video.
    sleep 60
    if trainer_running; then
      say "ABORTED: another scripts/train.py is running"
      exit 1
    fi

    say "launching ${run_name}: ${command_line}"
    setsid nohup ${command_line} >> "${log}" 2>&1 &

    # setsid re-execs, so the shell's $! is not the trainer. The run name is
    # unique per attempt, which makes it the reliable handle.
    job_pid=""
    for _ in $(seq 1 60); do
      sleep 5
      # Anchored on the interpreter, like trainer_running above: an unanchored
      # match also hits the shell that launched this queue, whose argv can carry
      # the whole command text, and waiting on that never returns.
      job_pid="$(pgrep -f "^[^ ]*python[0-9.]* .*scripts/train\.py .*--run-name ${run_name}( |\$)" | head -1)"
      [ -n "${job_pid}" ] && break
    done
    if [ -z "${job_pid}" ]; then
      say "${run_name} did not come up within 5 minutes"
      attempt=$((attempt + 1))
      continue
    fi

    say "${run_name} running as pid ${job_pid}"
    while kill -0 "${job_pid}" 2>/dev/null; do sleep 30; done

    outcome="$(/home/simone/.venv/bin/python scripts/run_outcome.py \
      --run-name "${run_name}" --want "${want}" 2>/dev/null)"
    say "${run_name} ended: ${outcome:-unreadable}"
    verdict="$(sed -n 's/.*verdict=\([a-z]*\).*/\1/p' <<< "${outcome}")"

    case "${verdict}" in
      done)
        touch "${job}.done"
        break
        ;;
      skip)
        say "${name} abandoned: the numbers do not justify another attempt"
        touch "${job}.skipped"
        break
        ;;
      *)
        # Unreadable outcomes land here too, deliberately: a wasted relaunch is
        # cheaper than dropping a run that was working.
        attempt=$((attempt + 1))
        if [ "${attempt}" -gt "${MAX_ATTEMPTS}" ]; then
          say "${name} FAILED: out of attempts"
          touch "${job}.failed"
        fi
        ;;
    esac
  done
done

say "queue empty"
