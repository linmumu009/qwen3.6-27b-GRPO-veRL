#!/usr/bin/env bash
set -Eeuo pipefail

HOST_PROJECT_ROOT="${HOST_PROJECT_ROOT:-/data3/llin/qwen3.6-27b-verl-grpo}"
CONTAINER_PROJECT_ROOT="${CONTAINER_PROJECT_ROOT:-/workspace/llin-verl-grpo}"
RUN_NAME="${RUN_NAME:-qwen38-prefix-curriculum-frontier-canary5-20260830-01}"
RUN_HOST="${HOST_PROJECT_ROOT}/runs/${RUN_NAME}"
RUN_CONTAINER="${CONTAINER_PROJECT_ROOT}/runs/${RUN_NAME}"
RUNTIME_HOST="${RUN_HOST}/runtime"
RUNTIME_CONTAINER="${RUN_CONTAINER}/runtime"
TRAINER_CONTAINER="${TRAINER_CONTAINER:-llin-verl-qwen38-canary-m05-20260822}"
ROLLOUT_CONTAINER="${ROLLOUT_CONTAINER:-llin-verl-qwen38-canary-m06-20260822}"
ROLLOUT_HOST="${ROLLOUT_HOST:-192.168.202.4}"
RAY_ADDRESS="${RAY_ADDRESS:-192.168.202.5:36379}"
MODEL_PATH="/models/Qwen3.8-27B"
PACKAGE_HOST="${HOST_PROJECT_ROOT}/runs/prefix-state-curriculum-grpo-v1-pi27b-pass55-20260830-05"
PACKAGE_CONTAINER="${CONTAINER_PROJECT_ROOT}/runs/prefix-state-curriculum-grpo-v1-pi27b-pass55-20260830-05"
FORMAL_CONTAINER="${CONTAINER_PROJECT_ROOT}/runs/qwen38-27b-pi-two-stage-screen-20260828-02/private/api/formal_mixed_candidates.sensitive.jsonl"
EXPECTED_SAFE_SHA256="7fcace126d13b0ded74bada98075ec64994f1dea800218f6bf3c9ce0f82788bf"
EXPECTED_MANIFEST_SHA256="3668fc60118a7f0371f7eb304881907d49eea31738daf8347e441d7ee723b37d"
EXPECTED_ALL_STATES_SHA256="99129edb21b64dfa9786eb56f413c813e6ecc9de6ba5b1da95f5709cb8d28c4e"
EXPECTED_CONFIG_SHA256="191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab"
EXPECTED_MODEL_COMPOUND_SHA256="e2c3b44e4e198e94fcd74903983fc8997f8e504a21575e397f9d59db1cc2fc8f"

mkdir -p "${RUN_HOST}/audit" "${RUN_HOST}/private" "${RUN_HOST}/frontier"
chmod 700 "${RUN_HOST}" "${RUN_HOST}/audit" "${RUN_HOST}/private" "${RUN_HOST}/frontier"
exec 9>"${HOST_PROJECT_ROOT}/runs/.qwen38-prefix-curriculum-canary5.lock"
flock -n 9 || { printf 'another prefix curriculum canary owns the lock\n' >&2; exit 9; }
printf '%s\n' "$$" > "${RUN_HOST}/supervisor.pid"
date -Iseconds > "${RUN_HOST}/started_at"
printf 'asset_gates\n' > "${RUN_HOST}/state"

ray_started=false
cleanup() {
  local code=$?
  set +e
  if [[ "${ray_started}" == true ]]; then
    docker exec "${TRAINER_CONTAINER}" ray stop --force >> "${RUN_HOST}/audit/ray_cleanup.log" 2>&1
    ssh -o BatchMode=yes "root@${ROLLOUT_HOST}" \
      "docker exec '${ROLLOUT_CONTAINER}' ray stop --force" >> "${RUN_HOST}/audit/ray_cleanup.log" 2>&1
  fi
  printf '%s\n' "${code}" > "${RUN_HOST}/exit_code"
  date -Iseconds > "${RUN_HOST}/finished_at"
  if (( code == 0 )); then
    printf 'canary_complete_full_training_locked\n' > "${RUN_HOST}/state"
  elif [[ "$(cat "${RUN_HOST}/state" 2>/dev/null)" != frontier_gate_failed* ]]; then
    printf 'failed_full_training_locked\n' > "${RUN_HOST}/state"
  fi
  set -e
}
trap cleanup EXIT

npu_process_pattern='^\|[[:space:]]*[0-9]+[[:space:]]+[0-9]+[[:space:]]+\|[[:space:]]*[0-9]+[[:space:]]+\|[[:space:]]*[[:alnum:]_]'
local_npu="$(npu-smi info)"
remote_npu="$(ssh -o BatchMode=yes "root@${ROLLOUT_HOST}" npu-smi info)"
! grep -Eq "${npu_process_pattern}" <<< "${local_npu}"
! grep -Eq "${npu_process_pattern}" <<< "${remote_npu}"
docker inspect "${TRAINER_CONTAINER}" >/dev/null
ssh -o BatchMode=yes "root@${ROLLOUT_HOST}" "docker inspect '${ROLLOUT_CONTAINER}'" >/dev/null
! ss -ltn | grep -q ':36379 '
! ssh -o BatchMode=yes "root@${ROLLOUT_HOST}" "ss -ltn | grep -q ':36379 '"

[[ "$(sha256sum "${PACKAGE_HOST}/safe_summary.json" | awk '{print $1}')" == "${EXPECTED_SAFE_SHA256}" ]]
[[ "$(sha256sum "${PACKAGE_HOST}/curriculum_manifest.safe.json" | awk '{print $1}')" == "${EXPECTED_MANIFEST_SHA256}" ]]
[[ "$(sha256sum "${PACKAGE_HOST}/private/all_states.sensitive.parquet" | awk '{print $1}')" == "${EXPECTED_ALL_STATES_SHA256}" ]]

compound_local="$(docker exec "${TRAINER_CONTAINER}" bash -lc "LC_ALL=C sha256sum '${MODEL_PATH}'/model-*-of-00018.safetensors | sha256sum | cut -d' ' -f1")"
compound_remote="$(ssh -o BatchMode=yes "root@${ROLLOUT_HOST}" "docker exec '${ROLLOUT_CONTAINER}' bash -lc \"LC_ALL=C sha256sum '${MODEL_PATH}'/model-*-of-00018.safetensors | sha256sum | cut -d' ' -f1\"")"
config_local="$(docker exec "${TRAINER_CONTAINER}" sha256sum "${MODEL_PATH}/config.json" | awk '{print $1}')"
config_remote="$(ssh -o BatchMode=yes "root@${ROLLOUT_HOST}" "docker exec '${ROLLOUT_CONTAINER}' sha256sum '${MODEL_PATH}/config.json'" | awk '{print $1}')"
[[ "${compound_local}" == "${EXPECTED_MODEL_COMPOUND_SHA256}" && "${compound_remote}" == "${EXPECTED_MODEL_COMPOUND_SHA256}" ]]
[[ "${config_local}" == "${EXPECTED_CONFIG_SHA256}" && "${config_remote}" == "${EXPECTED_CONFIG_SHA256}" ]]

cat > "${RUN_HOST}/audit/asset_gate.safe.json" <<EOF
{"npu_idle_m05":true,"npu_idle_m06":true,"ray_36379_absent_before":true,"model_config_sha256":"${config_local}","model_compound_sha256":"${compound_local}","curriculum_safe_sha256":"${EXPECTED_SAFE_SHA256}","curriculum_manifest_sha256":"${EXPECTED_MANIFEST_SHA256}","all_states_sha256":"${EXPECTED_ALL_STATES_SHA256}","api_requests":0,"full_training_allowed":false}
EOF

printf 'freezing_runtime\n' > "${RUN_HOST}/state"
mkdir -p "${RUNTIME_HOST}"
if [[ -f "${RUNTIME_HOST}/.llin_runtime_commit" ]]; then
  runtime_commit="$(cat "${RUNTIME_HOST}/.llin_runtime_commit")"
  [[ "${runtime_commit}" =~ ^[0-9a-f]{40}$ ]]
else
  git -C "${HOST_PROJECT_ROOT}" fetch origin main
  git -C "${HOST_PROJECT_ROOT}" archive origin/main | tar -x -C "${RUNTIME_HOST}"
  runtime_commit="$(git -C "${HOST_PROJECT_ROOT}" rev-parse origin/main)"
fi
printf '%s\n' "${runtime_commit}" > "${RUN_HOST}/audit/runtime_commit.safe.txt"
chmod -R go-rwx "${RUNTIME_HOST}"
ssh -o BatchMode=yes "root@${ROLLOUT_HOST}" \
  "mkdir -p '${RUN_HOST}/private' && chmod 700 '${RUN_HOST}' '${RUN_HOST}/private'"
scp -pr "${RUNTIME_HOST}" "root@${ROLLOUT_HOST}:${RUN_HOST}/"

printf 'container_cpu_gates\n' > "${RUN_HOST}/state"
docker exec "${TRAINER_CONTAINER}" env PYTHONPATH="${RUNTIME_CONTAINER}" python3 -m pytest \
  "${RUNTIME_CONTAINER}/tests/test_prefix_state_curriculum.py" \
  "${RUNTIME_CONTAINER}/tests/test_tiered_query_cost_reward.py" \
  "${RUNTIME_CONTAINER}/tests/test_qwen38_approved43_outcome_launcher.py::test_skipped_batch_keeps_optimizer_policy_and_adam_but_reopens_rollout_window" \
  "${RUNTIME_CONTAINER}/tests/test_qwen38_approved43_outcome_launcher.py::test_mixed_group_updates_and_advances_policy_while_stale_group_fails_closed" \
  -q --basetemp=/tmp/llin-prefix-curriculum-gate > "${RUN_HOST}/audit/container_cpu_gate.log" 2>&1

printf 'preparing_runtime_data\n' > "${RUN_HOST}/state"
docker exec "${TRAINER_CONTAINER}" env PYTHONPATH="${RUNTIME_CONTAINER}" python3 \
  "${RUNTIME_CONTAINER}/scripts/prepare_prefix_state_curriculum_runtime.py" \
  --package-root "${PACKAGE_CONTAINER}" --formal-tasks "${FORMAL_CONTAINER}" \
  --output-root "${RUN_CONTAINER}/prepared" --database-root "${RUN_CONTAINER}/private/pi_sandbox" \
  > "${RUN_HOST}/audit/prepare_runtime.log"

docker exec "${TRAINER_CONTAINER}" env PYTHONPATH="${RUNTIME_CONTAINER}" python3 \
  "${RUNTIME_CONTAINER}/scripts/stage_bound_pi_sandbox.py" \
  --dataset "${RUN_CONTAINER}/prepared/private/all_ready.runtime.sensitive.parquet" \
  --source-root /pi_sandbox --output-root "${RUN_CONTAINER}/private/pi_sandbox" \
  --safe-summary "${RUN_CONTAINER}/audit/bound_pi_sandbox.safe.json" \
  > "${RUN_HOST}/audit/stage_bound_pi_sandbox.log"
scp -pr "${RUN_HOST}/private/pi_sandbox" "root@${ROLLOUT_HOST}:${RUN_HOST}/private/"
scp -pr "${RUN_HOST}/prepared" "root@${ROLLOUT_HOST}:${RUN_HOST}/"
local_db="$(cd "${RUN_HOST}/private/pi_sandbox" && find . -name logistics.sqlite -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')"
remote_db="$(ssh -o BatchMode=yes "root@${ROLLOUT_HOST}" "cd '${RUN_HOST}/private/pi_sandbox' && find . -name logistics.sqlite -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1")"
[[ "${local_db}" == "${remote_db}" ]]

printf 'starting_isolated_ray\n' > "${RUN_HOST}/state"
docker exec "${TRAINER_CONTAINER}" env PROJECT_ROOT="${RUNTIME_CONTAINER}" RAY_HEAD_PORT=36379 \
  PI_AGENT_SANDBOX_LOWER="${RUN_CONTAINER}/private/pi_sandbox" PI_AGENT_TOKENIZER_PATH="${MODEL_PATH}" \
  RAY_TEMP_DIR=/tmp/q38-prefix-curriculum-ray-m05 \
  bash "${RUNTIME_CONTAINER}/scripts/start_ray_qwen38_smoke_m05.sh" > "${RUN_HOST}/audit/ray_m05.log" 2>&1
ray_started=true
ssh -o BatchMode=yes "root@${ROLLOUT_HOST}" \
  "docker exec '${ROLLOUT_CONTAINER}' env PROJECT_ROOT='${RUNTIME_CONTAINER}' RAY_HEAD_ADDRESS='${RAY_ADDRESS}' PI_AGENT_SANDBOX_LOWER='${RUN_CONTAINER}/private/pi_sandbox' PI_AGENT_TOKENIZER_PATH='${MODEL_PATH}' bash '${RUNTIME_CONTAINER}/scripts/start_ray_qwen38_smoke_m06.sh'" \
  > "${RUN_HOST}/audit/ray_m06.log" 2>&1
docker exec "${TRAINER_CONTAINER}" ray status --address="${RAY_ADDRESS}" > "${RUN_HOST}/audit/ray_status.safe.txt"

printf 'frontier_round_00\n' > "${RUN_HOST}/state"
current_data="${RUN_CONTAINER}/prepared/private/frontier_endpoints_10x2.runtime.sensitive.parquet"
validation_args=()
frontier_passed=false
for round in $(seq 0 7); do
  round_name="$(printf '%02d' "${round}")"
  round_container="${RUN_CONTAINER}/frontier/round-${round_name}"
  round_host="${RUN_HOST}/frontier/round-${round_name}"
  mkdir -p "${round_host}/validation"
  chmod 700 "${round_host}" "${round_host}/validation"
  printf 'frontier_round_%s_rollout\n' "${round_name}" > "${RUN_HOST}/state"
  docker exec "${TRAINER_CONTAINER}" env \
    PROJECT_ROOT="${RUNTIME_CONTAINER}" MODEL_PATH="${MODEL_PATH}" DATA_FILE="${current_data}" \
    RUN_NAME="${RUN_NAME}-frontier-${round_name}" OUTPUT_DIR="${round_container}" \
    VALIDATION_DIR="${round_container}/validation" RAY_ADDRESS="${RAY_ADDRESS}" \
    PI_AGENT_SANDBOX_LOWER="${RUN_CONTAINER}/private/pi_sandbox" \
    bash "${RUNTIME_CONTAINER}/scripts/run_pi_qwen38_prefix_frontier_v1.sh" \
    > "${round_host}/driver.log" 2>&1
  validation_file="${round_host}/validation/0.jsonl"
  [[ -s "${validation_file}" ]]
  chmod 600 "${validation_file}"
  validation_args+=(--validation "${round_container}/validation/0.jsonl")
  analysis_container="${RUN_CONTAINER}/frontier/analysis-${round_name}"
  analysis_host="${RUN_HOST}/frontier/analysis-${round_name}"
  mkdir -p "${analysis_host}"
  chmod 700 "${analysis_host}"
  printf 'frontier_round_%s_gate\n' "${round_name}" > "${RUN_HOST}/state"
  docker exec "${TRAINER_CONTAINER}" env PYTHONPATH="${RUNTIME_CONTAINER}" python3 \
    "${RUNTIME_CONTAINER}/scripts/analyze_prefix_frontier.py" \
    --ladders "${RUN_CONTAINER}/prepared/private/representative_ladders.runtime.sensitive.parquet" \
    "${validation_args[@]}" --output-root "${analysis_container}" \
    > "${analysis_host}/analyze.log"
  frontier_passed="$(python3 -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["frontier_gate_passed"]).lower())' "${analysis_host}/frontier_gate.safe.json")"
  if [[ "${frontier_passed}" == true ]]; then
    cp "${analysis_host}/frontier_gate.safe.json" "${RUN_HOST}/frontier_gate.safe.json"
    cp "${analysis_host}/accepted_frontier.runtime.sensitive.parquet" "${RUN_HOST}/private/accepted_frontier.runtime.sensitive.parquet"
    printf 'frontier_gate_passed\n' > "${RUN_HOST}/state"
    break
  fi
  next_count="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["next_round_state_count"])' "${analysis_host}/frontier_gate.safe.json")"
  if (( next_count == 0 )); then
    printf 'frontier_gate_failed_no_remaining_states\n' > "${RUN_HOST}/state"
    exit 5
  fi
  current_data="${analysis_container}/next_round.runtime.sensitive.parquet"
done
[[ "${frontier_passed}" == true ]] || { printf 'frontier_gate_failed_round_cap\n' > "${RUN_HOST}/state"; exit 6; }

printf 'preparing_canary_schedule\n' > "${RUN_HOST}/state"
docker exec "${TRAINER_CONTAINER}" env PYTHONPATH="${RUNTIME_CONTAINER}" python3 \
  "${RUNTIME_CONTAINER}/scripts/prepare_prefix_canary_schedule.py" \
  --frontier "${RUN_CONTAINER}/private/accepted_frontier.runtime.sensitive.parquet" \
  --ladders "${RUN_CONTAINER}/prepared/private/representative_ladders.runtime.sensitive.parquet" \
  --output "${RUN_CONTAINER}/private/canary20.runtime.sensitive.parquet" \
  --safe-summary "${RUN_CONTAINER}/audit/canary20.safe.json" > "${RUN_HOST}/audit/prepare_canary.log"
scp -p "${RUN_HOST}/private/canary20.runtime.sensitive.parquet" \
  "root@${ROLLOUT_HOST}:${RUN_HOST}/private/canary20.runtime.sensitive.parquet"

printf 'canary_training_active\n' > "${RUN_HOST}/state"
docker exec "${TRAINER_CONTAINER}" env \
  PROJECT_ROOT="${RUNTIME_CONTAINER}" MODEL_PATH="${MODEL_PATH}" \
  TRAIN_FILE="${RUN_CONTAINER}/private/canary20.runtime.sensitive.parquet" \
  SEALED_FILE="${RUN_CONTAINER}/prepared/private/heldout_endpoints_4x2.runtime.sensitive.parquet" \
  RUN_NAME="${RUN_NAME}" OUTPUT_DIR="${RUN_CONTAINER}" RAY_ADDRESS="${RAY_ADDRESS}" \
  PI_AGENT_SANDBOX_LOWER="${RUN_CONTAINER}/private/pi_sandbox" \
  LLIN_CANARY_AUDIT_DIR="${RUN_CONTAINER}/private/parameter_audit" \
  bash "${RUNTIME_CONTAINER}/scripts/run_pi_qwen38_prefix_curriculum_canary5_v1.sh" \
  > "${RUN_HOST}/training.log" 2>&1

compound_local_after="$(docker exec "${TRAINER_CONTAINER}" bash -lc "LC_ALL=C sha256sum '${MODEL_PATH}'/model-*-of-00018.safetensors | sha256sum | cut -d' ' -f1")"
compound_remote_after="$(ssh -o BatchMode=yes "root@${ROLLOUT_HOST}" "docker exec '${ROLLOUT_CONTAINER}' bash -lc \"LC_ALL=C sha256sum '${MODEL_PATH}'/model-*-of-00018.safetensors | sha256sum | cut -d' ' -f1\"")"
[[ "${compound_local_after}" == "${EXPECTED_MODEL_COMPOUND_SHA256}" && "${compound_remote_after}" == "${EXPECTED_MODEL_COMPOUND_SHA256}" ]]
printf 'canary_complete_full_training_locked\n' > "${RUN_HOST}/state"
