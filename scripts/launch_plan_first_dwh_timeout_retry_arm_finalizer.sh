#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/llin-verl-grpo}"
ORIGINAL_DATASET="${ORIGINAL_DATASET:?ORIGINAL_DATASET is required}"
ORIGINAL_SHARDS_DIR="${ORIGINAL_SHARDS_DIR:?ORIGINAL_SHARDS_DIR is required}"
RETRY_DATASET="${RETRY_DATASET:?RETRY_DATASET is required}"
RETRY_RUN_DIR="${RETRY_RUN_DIR:?RETRY_RUN_DIR is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"

export PROJECT_ROOT ORIGINAL_DATASET ORIGINAL_SHARDS_DIR RETRY_DATASET RETRY_RUN_DIR OUTPUT_DIR
mkdir -p "${OUTPUT_DIR}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
nohup bash -lc '
  set +e
  python3 "${PROJECT_ROOT}/scripts/plan_first_dwh_timeout_retry.py" finalize-arm \
    --original-dataset "${ORIGINAL_DATASET}" \
    --original-shards-dir "${ORIGINAL_SHARDS_DIR}" \
    --retry-dataset "${RETRY_DATASET}" \
    --retry-run-dir "${RETRY_RUN_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    > "${OUTPUT_DIR}/retry_arm_finalizer.log" 2>&1
  code=$?
  printf "%s\n" "${code}" > "${OUTPUT_DIR}/exit_code"
  date -Iseconds > "${OUTPUT_DIR}/finished_at"
  exit "${code}"
' >/dev/null 2>&1 &
printf '%s\n' "$!" > "${OUTPUT_DIR}/finalizer.pid"
date -Iseconds > "${OUTPUT_DIR}/started_at"
printf 'launched timeout retry arm finalizer pid=%s\n' "$!"
