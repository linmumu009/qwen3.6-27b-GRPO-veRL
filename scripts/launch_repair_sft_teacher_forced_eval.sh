#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/llin-verl-grpo}"
RUN_NAME="${RUN_NAME:?RUN_NAME is required}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/runs/${RUN_NAME}}"
export PROJECT_ROOT RUN_NAME OUTPUT_DIR
mkdir -p "${OUTPUT_DIR}"

if [[ -f "${OUTPUT_DIR}/exit_code" ]]; then
  printf 'run already has exit_code: %s\n' "${OUTPUT_DIR}" >&2
  exit 2
fi

date -Iseconds > "${OUTPUT_DIR}/started_at"
nohup bash -lc '
  set +e
  bash "${PROJECT_ROOT}/scripts/run_repair_sft_teacher_forced_eval.sh" > "${OUTPUT_DIR}/driver.log" 2>&1
  code=$?
  printf "%s\n" "${code}" > "${OUTPUT_DIR}/exit_code"
  date -Iseconds > "${OUTPUT_DIR}/finished_at"
  exit "${code}"
' >/dev/null 2>&1 &
printf '%s\n' "$!" > "${OUTPUT_DIR}/pid"
printf 'launched %s pid=%s\n' "${RUN_NAME}" "$!"
