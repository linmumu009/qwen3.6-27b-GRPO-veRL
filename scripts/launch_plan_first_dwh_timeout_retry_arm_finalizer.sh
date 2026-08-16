#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/llin-verl-grpo}"
ORIGINAL_DATASET="${ORIGINAL_DATASET:?ORIGINAL_DATASET is required}"
ORIGINAL_SHARDS_DIR="${ORIGINAL_SHARDS_DIR:?ORIGINAL_SHARDS_DIR is required}"
RETRY_DATASET="${RETRY_DATASET:?RETRY_DATASET is required}"
RETRY_RUN_DIR="${RETRY_RUN_DIR:?RETRY_RUN_DIR is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
EXPECTED_TASKS="${EXPECTED_TASKS:-300}"
SAMPLES_PER_TASK="${SAMPLES_PER_TASK:-8}"

export PROJECT_ROOT ORIGINAL_DATASET ORIGINAL_SHARDS_DIR RETRY_DATASET RETRY_RUN_DIR OUTPUT_DIR
export EXPECTED_TASKS SAMPLES_PER_TASK
mkdir -p "${OUTPUT_DIR}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
nohup bash -c '
  set +e
  python3 "${PROJECT_ROOT}/scripts/plan_first_dwh_timeout_retry.py" finalize-arm \
    --original-dataset "${ORIGINAL_DATASET}" \
    --original-shards-dir "${ORIGINAL_SHARDS_DIR}" \
    --retry-dataset "${RETRY_DATASET}" \
    --retry-run-dir "${RETRY_RUN_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --expected-tasks "${EXPECTED_TASKS}" \
    --samples-per-task "${SAMPLES_PER_TASK}" \
    > "${OUTPUT_DIR}/retry_arm_finalizer.log" 2>&1
  code=$?
  printf "%s\n" "${code}" > "${OUTPUT_DIR}/exit_code"
  date -Iseconds > "${OUTPUT_DIR}/finished_at"
  exit "${code}"
' >/dev/null 2>&1 &
printf '%s\n' "$!" > "${OUTPUT_DIR}/finalizer.pid"
date -Iseconds > "${OUTPUT_DIR}/started_at"
printf 'launched timeout retry arm finalizer pid=%s\n' "$!"
