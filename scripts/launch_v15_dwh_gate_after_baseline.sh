#!/usr/bin/env bash
set -u

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/llin-verl-grpo}"
BASELINE_RUN_NAME="${BASELINE_RUN_NAME:-llin-v15-dwh-full277-frozen-val20-20260804-01}"
TARGET_RUN_NAME="${TARGET_RUN_NAME:-llin-v15-dwh-bossreward-5step-20260804-01}"
POLL_SECONDS="${POLL_SECONDS:-60}"
COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-60}"
BASELINE_DIR="${PROJECT_ROOT}/runs/${BASELINE_RUN_NAME}"
TARGET_DIR="${PROJECT_ROOT}/runs/${TARGET_RUN_NAME}"
SUPERVISOR_DIR="${TARGET_DIR}.supervisor"

mkdir -p "${SUPERVISOR_DIR}"
date --iso-8601=seconds > "${SUPERVISOR_DIR}/started_at"
printf '%s\n' "${BASELINE_RUN_NAME}" > "${SUPERVISOR_DIR}/baseline_run_name"
printf '%s\n' "${TARGET_RUN_NAME}" > "${SUPERVISOR_DIR}/target_run_name"

while [[ ! -f "${BASELINE_DIR}/exit_code" ]]; do
  sleep "${POLL_SECONDS}"
done

baseline_exit="$(tr -d '[:space:]' < "${BASELINE_DIR}/exit_code")"
printf '%s\n' "${baseline_exit}" > "${SUPERVISOR_DIR}/baseline_exit_code"
if [[ "${baseline_exit}" != "0" ]]; then
  printf 'baseline_failed\n' > "${SUPERVISOR_DIR}/status"
  date --iso-8601=seconds > "${SUPERVISOR_DIR}/finished_at"
  exit 3
fi

sleep "${COOLDOWN_SECONDS}"
printf 'launching_5step\n' > "${SUPERVISOR_DIR}/status"

set +e
RUN_NAME="${TARGET_RUN_NAME}" \
TOTAL_TRAINING_STEPS=5 \
EVAL_FREQ=5 \
SAVE_FREQ=5 \
bash "${PROJECT_ROOT}/scripts/launch_pi_formal_50step.sh"
target_exit=$?
set -e

printf '%s\n' "${target_exit}" > "${SUPERVISOR_DIR}/target_exit_code"
if [[ "${target_exit}" == "0" ]]; then
  printf 'completed\n' > "${SUPERVISOR_DIR}/status"
else
  printf 'target_failed\n' > "${SUPERVISOR_DIR}/status"
fi
date --iso-8601=seconds > "${SUPERVISOR_DIR}/finished_at"
exit "${target_exit}"
