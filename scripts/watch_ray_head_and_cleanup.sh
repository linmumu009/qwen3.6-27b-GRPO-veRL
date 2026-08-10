#!/usr/bin/env bash
set -u

RAY_ADDRESS="${RAY_ADDRESS:-192.168.202.5:26379}"
INITIAL_GRACE_SECONDS="${INITIAL_GRACE_SECONDS:-300}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-60}"
MAX_CONSECUTIVE_FAILURES="${MAX_CONSECUTIVE_FAILURES:-3}"
LOG_FILE="${LOG_FILE:-/workspace/llin-verl-grpo/runs/ray_cleanup_watcher.log}"

sleep "${INITIAL_GRACE_SECONDS}"
failures=0
while true; do
  if ray status --address "${RAY_ADDRESS}" >/dev/null 2>&1; then
    failures=0
  else
    failures=$((failures + 1))
    printf '%s ray_head_unreachable failures=%s\n' \
      "$(date --iso-8601=seconds)" "${failures}" >> "${LOG_FILE}"
    if (( failures >= MAX_CONSECUTIVE_FAILURES )); then
      ray stop --force >> "${LOG_FILE}" 2>&1
      printf '%s rollout_ray_stopped\n' "$(date --iso-8601=seconds)" >> "${LOG_FILE}"
      exit 0
    fi
  fi
  sleep "${CHECK_INTERVAL_SECONDS}"
done
