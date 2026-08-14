#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/llin-verl-grpo}"
DATASET="${DATASET:?DATASET is required}"
NATIVE_ARM_DIR="${NATIVE_ARM_DIR:?NATIVE_ARM_DIR is required}"
STEP120_ARM_DIR="${STEP120_ARM_DIR:?STEP120_ARM_DIR is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
RAY_ADDRESS="${RAY_ADDRESS:-192.168.202.5:26379}"
REMOTE_RESOURCE="${REMOTE_RESOURCE:-llin_rollout_m06}"

export PROJECT_ROOT DATASET NATIVE_ARM_DIR STEP120_ARM_DIR OUTPUT_DIR RAY_ADDRESS REMOTE_RESOURCE
mkdir -p "${OUTPUT_DIR}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
nohup bash -lc '
  set +e
  python3 "${PROJECT_ROOT}/scripts/finalize_plan_first_dwh_model_comparison.py" \
    --ray-address "${RAY_ADDRESS}" \
    --dataset "${DATASET}" \
    --native-arm-dir "${NATIVE_ARM_DIR}" \
    --step120-arm-dir "${STEP120_ARM_DIR}" \
    --remote-resource "${REMOTE_RESOURCE}" \
    --output-dir "${OUTPUT_DIR}" \
    > "${OUTPUT_DIR}/finalizer.log" 2>&1
  code=$?
  printf "%s\n" "${code}" > "${OUTPUT_DIR}/finalizer_exit_code"
  date -Iseconds > "${OUTPUT_DIR}/finalizer_finished_at"
  exit "${code}"
' >/dev/null 2>&1 &
printf '%s\n' "$!" > "${OUTPUT_DIR}/finalizer.pid"
date -Iseconds > "${OUTPUT_DIR}/finalizer_started_at"
printf 'launched plan-first model comparison finalizer pid=%s\n' "$!"
