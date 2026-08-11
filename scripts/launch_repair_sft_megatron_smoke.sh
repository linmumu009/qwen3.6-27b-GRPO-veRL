#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/llin-verl-grpo}"
RUN_NAME="${RUN_NAME:-llin-repair-sft-megatron-smoke-20260811-01}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/runs/${RUN_NAME}}"
mkdir -p "${OUTPUT_DIR}"

date --iso-8601=seconds > "${OUTPUT_DIR}/started_at"
set +e
bash "${PROJECT_ROOT}/scripts/run_repair_sft_megatron_smoke.sh" "$@" \
  > "${OUTPUT_DIR}/driver.log" 2>&1
status=$?
set -e
printf '%s\n' "${status}" > "${OUTPUT_DIR}/exit_code"
date --iso-8601=seconds > "${OUTPUT_DIR}/finished_at"
exit "${status}"
