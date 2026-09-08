#!/usr/bin/env bash
# Wait for a training process to exit, then launch the next one detached.
#
#   scripts/run_after.sh <pid-to-wait-for> <logfile> <command...>
#
# The launched command is put in its own session so it outlives this waiter and
# the shell that started it. Nothing is launched if some other train.py is still
# alive when the wait ends, so a manual start cannot end up racing this one.
set -u

pid="$1"; shift
log="$1"; shift
cd /home/simone/simtoolreal_animrl

echo "[$(date -Is)] waiting for pid ${pid}" >> "${log}"
while kill -0 "${pid}" 2>/dev/null; do sleep 30; done
echo "[$(date -Is)] pid ${pid} exited" >> "${log}"

# The finished process needs a moment to flush its final evaluation and video.
sleep 60

# Anchored at the start of the command line so this only ever matches a real
# interpreter running the trainer. An unanchored match also hits this waiter and
# the shell that launched it, since both carry the command in their own argv.
if pgrep -f "^[^ ]*python[0-9.]* .*scripts/train\.py" > /dev/null; then
  echo "[$(date -Is)] ABORTED: another scripts/train.py is running" >> "${log}"
  exit 1
fi

echo "[$(date -Is)] launching: $*" >> "${log}"
setsid nohup "$@" >> "${log}" 2>&1 &
echo "[$(date -Is)] launched pid $!" >> "${log}"
