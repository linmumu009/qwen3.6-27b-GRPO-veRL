#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/llin-verl-grpo}"
LOCAL_ARM_DIR="${LOCAL_ARM_DIR:?LOCAL_ARM_DIR is required}"
REMOTE_ARM_DIR="${REMOTE_ARM_DIR:?REMOTE_ARM_DIR is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
RAY_ADDRESS="${RAY_ADDRESS:-192.168.202.5:26379}"
REMOTE_RESOURCE="${REMOTE_RESOURCE:-llin_rollout_m06}"
FINALIZER_TIMEOUT_SECONDS="${FINALIZER_TIMEOUT_SECONDS:-172800}"
FINALIZER_POLL_SECONDS="${FINALIZER_POLL_SECONDS:-30}"

mkdir -p "${OUTPUT_DIR}"
if [[ -s "${OUTPUT_DIR}/finalizer.pid" ]] && kill -0 "$(<"${OUTPUT_DIR}/finalizer.pid")" 2>/dev/null; then
  printf 'finalizer already active: %s\n' "${OUTPUT_DIR}" >&2
  exit 2
fi

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
nohup bash -lc '
  set +e
  python3 "${PROJECT_ROOT}/scripts/finalize_multisandbox_dwh_dual_server.py" \
    --ray-address "${RAY_ADDRESS}" \
    --local-arm-dir "${LOCAL_ARM_DIR}" \
    --remote-arm-dir "${REMOTE_ARM_DIR}" \
    --remote-resource "${REMOTE_RESOURCE}" \
    --output-dir "${OUTPUT_DIR}" \
    --timeout-seconds "${FINALIZER_TIMEOUT_SECONDS}" \
    --poll-seconds "${FINALIZER_POLL_SECONDS}" \
    > "${OUTPUT_DIR}/finalizer.log" 2>&1
  code=$?
  printf "%s\n" "${code}" > "${OUTPUT_DIR}/finalizer_exit_code"
  date -Iseconds > "${OUTPUT_DIR}/finalizer_finished_at"
  exit "${code}"
' >/dev/null 2>&1 &
printf '%s\n' "$!" > "${OUTPUT_DIR}/finalizer.pid"
date -Iseconds > "${OUTPUT_DIR}/finalizer_started_at"
printf 'launched dual finalizer pid=%s output=%s\n' "$!" "${OUTPUT_DIR}"
