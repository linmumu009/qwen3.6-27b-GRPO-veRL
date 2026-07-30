#!/usr/bin/env bash
set -u

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/llin-verl-grpo}"
RUN_NAME="${RUN_NAME:-pi-grpo-megatron-tp4-pp2-cp2-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${PROJECT_ROOT}/runs/${RUN_NAME}"

mkdir -p "${RUN_DIR}"
printf '%s\n' "$$" > "${RUN_DIR}/driver.pid"
date --iso-8601=seconds > "${RUN_DIR}/started_at"

set +e
RUN_NAME="${RUN_NAME}" \
OUTPUT_DIR="${RUN_DIR}" \
bash "${PROJECT_ROOT}/scripts/run_pi_grpo_megatron_tp4_pp2_cp2.sh" \
  > "${RUN_DIR}/driver.log" 2>&1
exit_code=$?
set -e

printf '%s\n' "${exit_code}" > "${RUN_DIR}/exit_code"
date --iso-8601=seconds > "${RUN_DIR}/finished_at"
exit "${exit_code}"
