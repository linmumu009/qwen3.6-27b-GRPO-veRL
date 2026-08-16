#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/llin-verl-grpo}"
RUN_NAME="${RUN_NAME:?RUN_NAME is required}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/runs/${RUN_NAME}}"
MODEL="${MODEL:?MODEL is required}"
MODEL_LABEL="${MODEL_LABEL:-step120}"
POLICY_STEP="${POLICY_STEP:-120}"
DATASET="${DATASET:?DATASET is required}"
EXPECTED_TASKS="${EXPECTED_TASKS:-281}"
SAMPLES_PER_TASK="${SAMPLES_PER_TASK:-8}"
TASK_BATCH_SIZE="${TASK_BATCH_SIZE:-32}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
AGENT_WORKERS="${AGENT_WORKERS:-16}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-4096}"
MAX_RESPONSE_TOKENS="${MAX_RESPONSE_TOKENS:-45056}"
MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-49152}"
TRAJECTORY_TIMEOUT_SECONDS="${TRAJECTORY_TIMEOUT_SECONDS:-900}"
ROLLING_ADMISSION="${ROLLING_ADMISSION:-0}"
ROLLING_WINDOW_TRAJECTORIES="${ROLLING_WINDOW_TRAJECTORIES:-0}"
RAY_ADDRESS="${RAY_ADDRESS:-192.168.202.5:26379}"
ROLLOUT_RESOURCE="${ROLLOUT_RESOURCE:?ROLLOUT_RESOURCE is required}"
ANALYZE_ON_SUCCESS="${ANALYZE_ON_SUCCESS:-1}"
MONITOR_NPU="${MONITOR_NPU:-1}"
MONITOR_ROLE="${MONITOR_ROLE:-standalone_rollout}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-5}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
RETRY_DELAY_SECONDS="${RETRY_DELAY_SECONDS:-60}"

export PROJECT_ROOT RUN_NAME OUTPUT_DIR MODEL MODEL_LABEL POLICY_STEP DATASET EXPECTED_TASKS
export SAMPLES_PER_TASK TASK_BATCH_SIZE MAX_NUM_SEQS AGENT_WORKERS
export MAX_NUM_BATCHED_TOKENS GPU_MEMORY_UTILIZATION
export MAX_PROMPT_TOKENS MAX_RESPONSE_TOKENS MAX_CONTEXT_TOKENS
export TRAJECTORY_TIMEOUT_SECONDS MAX_ATTEMPTS RETRY_DELAY_SECONDS
export ROLLING_ADMISSION ROLLING_WINDOW_TRAJECTORIES
export RAY_ADDRESS ROLLOUT_RESOURCE
export ANALYZE_ON_SUCCESS MONITOR_NPU MONITOR_ROLE MONITOR_INTERVAL
export LLIN_PIN_RAY_ROLES=1
export LLIN_ROLLOUT_RESOURCE="${ROLLOUT_RESOURCE}"
export PYTHONPATH="/vllm:${PROJECT_ROOT}/runtime:${PROJECT_ROOT}:${PYTHONPATH:-}"

mkdir -p "${OUTPUT_DIR}"
if [[ -e "${OUTPUT_DIR}/exit_code" ]]; then
  printf 'run already completed or failed: %s\n' "${OUTPUT_DIR}" >&2
  exit 2
fi
if [[ -s "${OUTPUT_DIR}/driver.pid" ]] && kill -0 "$(<"${OUTPUT_DIR}/driver.pid")" 2>/dev/null; then
  printf 'run already active: %s\n' "${OUTPUT_DIR}" >&2
  exit 2
fi

date -Iseconds > "${OUTPUT_DIR}/started_at"
nohup bash -c '
  set +e
  attempt=1
  code=1
  : > "${OUTPUT_DIR}/driver.log"
  while (( attempt <= MAX_ATTEMPTS )); do
    printf "attempt=%s/%s started_at=%s\n" "${attempt}" "${MAX_ATTEMPTS}" "$(date -Iseconds)" \
      >> "${OUTPUT_DIR}/driver.log"
    python3 "${PROJECT_ROOT}/scripts/run_runtime_parity_verl_standalone.py" \
      --project-root "${PROJECT_ROOT}" \
      --model "${MODEL}" \
      --model-label "${MODEL_LABEL}" \
      --policy-step "${POLICY_STEP}" \
      --dataset "${DATASET}" \
      --output-dir "${OUTPUT_DIR}" \
      --ray-address "${RAY_ADDRESS}" \
      --expected-tasks "${EXPECTED_TASKS}" \
      --samples-per-task "${SAMPLES_PER_TASK}" \
      --task-batch-size "${TASK_BATCH_SIZE}" \
      --max-num-seqs "${MAX_NUM_SEQS}" \
      --agent-workers "${AGENT_WORKERS}" \
      --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
      --max-prompt-tokens "${MAX_PROMPT_TOKENS}" \
      --max-response-tokens "${MAX_RESPONSE_TOKENS}" \
      --max-context-tokens "${MAX_CONTEXT_TOKENS}" \
      --trajectory-timeout-seconds "${TRAJECTORY_TIMEOUT_SECONDS}" \
      --rolling-admission "${ROLLING_ADMISSION}" \
      --rolling-window-trajectories "${ROLLING_WINDOW_TRAJECTORIES}" \
      >> "${OUTPUT_DIR}/driver.log" 2>&1
    code=$?
    printf "attempt=%s exit_code=%s finished_at=%s\n" "${attempt}" "${code}" "$(date -Iseconds)" \
      >> "${OUTPUT_DIR}/driver.log"
    [[ "${code}" == "0" ]] && break
    attempt=$((attempt + 1))
    (( attempt <= MAX_ATTEMPTS )) && sleep "${RETRY_DELAY_SECONDS}"
  done
  if [[ "${code}" == "0" && "${ANALYZE_ON_SUCCESS}" == "1" ]]; then
    python3 "${PROJECT_ROOT}/scripts/analyze_multisandbox_dwh_rollout.py" \
      --dataset "${DATASET}" \
      --shards-dir "${OUTPUT_DIR}/shards" \
      --output-dir "${OUTPUT_DIR}/outcomes" \
      --expected-tasks "${EXPECTED_TASKS}" \
      --samples-per-task "${SAMPLES_PER_TASK}" \
      >> "${OUTPUT_DIR}/driver.log" 2>&1
    code=$?
  fi
  printf "%s\n" "${code}" > "${OUTPUT_DIR}/exit_code"
  date -Iseconds > "${OUTPUT_DIR}/finished_at"
  exit "${code}"
' >/dev/null 2>&1 &
printf '%s\n' "$!" > "${OUTPUT_DIR}/driver.pid"
if [[ "${MONITOR_NPU}" == "1" ]]; then
  nohup python3 "${PROJECT_ROOT}/scripts/monitor_npu_utilization.py" \
    --output "${OUTPUT_DIR}/npu_utilization.csv" \
    --role "${MONITOR_ROLE}" \
    --until-file "${OUTPUT_DIR}/exit_code" \
    --interval "${MONITOR_INTERVAL}" \
    --first-card 0 \
    --num-cards 8 \
    > "${OUTPUT_DIR}/npu_monitor.log" 2>&1 &
  printf '%s\n' "$!" > "${OUTPUT_DIR}/npu_monitor.pid"
fi
printf 'launched %s pid=%s resource=%s max_num_seqs=%s\n' \
  "${RUN_NAME}" "$!" "${ROLLOUT_RESOURCE}" "${MAX_NUM_SEQS}"
