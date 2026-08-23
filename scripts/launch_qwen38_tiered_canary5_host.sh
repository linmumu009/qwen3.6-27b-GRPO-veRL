#!/usr/bin/env bash
set -Eeuo pipefail

HOST_PROJECT_ROOT="${HOST_PROJECT_ROOT:-/data3/llin/qwen3.6-27b-verl-grpo}"
CONTAINER_PROJECT_ROOT="${CONTAINER_PROJECT_ROOT:-/workspace/llin-verl-grpo}"
RUN_NAME="${RUN_NAME:-llin-qwen38-approved43-tiered-v1-canary5-20260822-01}"
RUN_HOST="${HOST_PROJECT_ROOT}/runs/${RUN_NAME}"
RUN_CONTAINER="${CONTAINER_PROJECT_ROOT}/runs/${RUN_NAME}"
RUNTIME_CONTAINER="${RUN_CONTAINER}/runtime"
TRAINER_CONTAINER="${TRAINER_CONTAINER:-llin-verl-qwen38-canary-m05-20260822}"
ROLLOUT_CONTAINER="${ROLLOUT_CONTAINER:-llin-verl-qwen38-canary-m06-20260822}"
ROLLOUT_HOST="${ROLLOUT_HOST:-192.168.202.4}"
RAY_ADDRESS="${RAY_ADDRESS:-192.168.202.5:36379}"
MODEL_PATH="/models/Qwen3.8-27B"
SOURCE_RUN_CONTAINER="${CONTAINER_PROJECT_ROOT}/runs/llin-v15-codex-model2-100-step120-8x-20260821-01"
PACKAGE_HOST="${HOST_PROJECT_ROOT}/runs/llin-v15-codex-model2-100-step120-8x-20260821-01/grpo_readiness_audit_20260822-05"
PACKAGE_CONTAINER="${SOURCE_RUN_CONTAINER}/grpo_readiness_audit_20260822-05"
TASKS_CONTAINER="${SOURCE_RUN_CONTAINER}/data/tasks.jsonl"
RAW100_CONTAINER="${SOURCE_RUN_CONTAINER}/data/rollout_100.sensitive.parquet"
FROZEN96_RUN_CONTAINER="${CONTAINER_PROJECT_ROOT}/runs/llin-qwen38-approved43-tiered-v1-canary5-20260823-03"
EXPECTED_CONFIG_SHA256="191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab"
EXPECTED_MODEL_COMPOUND_SHA256="e2c3b44e4e198e94fcd74903983fc8997f8e504a21575e397f9d59db1cc2fc8f"
EXPECTED_APPROVED_SHA256="d86b53d906806b150d43a508dce9b0dd6d05105c07e03961e8e7bf9439ccd944"
EXPECTED_MANIFEST_SHA256="1426bc09a3dbaf4709fd89227790603afb7a2bf11beeba80946057d490e0f424"
EXPECTED_RAW100_SHA256="c0befda32166340bf68e6b948a1e8fcc6f8f0887d7a5f38a4e6b1051b8f9f7af"

mkdir -p "${RUN_HOST}/audit" "${RUN_HOST}/live_patch_backup" "${RUN_HOST}/private"
chmod 700 "${RUN_HOST}" "${RUN_HOST}/audit" "${RUN_HOST}/live_patch_backup" "${RUN_HOST}/private"
exec 9>"${HOST_PROJECT_ROOT}/runs/.qwen38-tiered-canary5.lock"
flock -n 9 || { printf 'another Qwen3.8 tiered canary holds the lock\n' >&2; exit 9; }
printf '%s\n' "$$" > "${RUN_HOST}/supervisor.pid"
date -Iseconds > "${RUN_HOST}/started_at"
printf 'initializing\n' > "${RUN_HOST}/state"

ray_started=false
cleanup() {
  local code=$?
  set +e
  if [[ "${ray_started}" == "true" ]]; then
    docker exec "${TRAINER_CONTAINER}" ray stop --force >> "${RUN_HOST}/ray_cleanup.log" 2>&1
    ssh -o BatchMode=yes "root@${ROLLOUT_HOST}" \
      "docker exec '${ROLLOUT_CONTAINER}' ray stop --force" >> "${RUN_HOST}/ray_cleanup.log" 2>&1
  fi
  printf '%s\n' "${code}" > "${RUN_HOST}/exit_code"
  date -Iseconds > "${RUN_HOST}/finished_at"
  if (( code == 0 )); then
    printf 'canary_complete_pending_review\n' > "${RUN_HOST}/state"
  else
    printf 'failed_full_training_locked\n' > "${RUN_HOST}/state"
  fi
  set -e
}
trap cleanup EXIT

npu_process_pattern='^\|[[:space:]]*[0-9]+[[:space:]]+[0-9]+[[:space:]]+\|[[:space:]]*[0-9]+[[:space:]]+\|[[:space:]]*[[:alnum:]_]'
assert_idle() {
  local local_info remote_info
  local_info="$(npu-smi info)"
  remote_info="$(ssh -o BatchMode=yes "root@${ROLLOUT_HOST}" npu-smi info)"
  [[ -n "${local_info}" && -n "${remote_info}" ]]
  ! grep -Eq "${npu_process_pattern}" <<< "${local_info}"
  ! grep -Eq "${npu_process_pattern}" <<< "${remote_info}"
}

compound_hash_local() {
  docker exec "${TRAINER_CONTAINER}" bash -lc \
    "LC_ALL=C sha256sum '${MODEL_PATH}'/model-*-of-00018.safetensors | sha256sum | cut -d' ' -f1"
}

compound_hash_remote() {
  ssh -o BatchMode=yes "root@${ROLLOUT_HOST}" \
    "docker exec '${ROLLOUT_CONTAINER}' bash -lc \"LC_ALL=C sha256sum '${MODEL_PATH}'/model-*-of-00018.safetensors | sha256sum | cut -d' ' -f1\""
}

printf 'asset_gates\n' > "${RUN_HOST}/state"
assert_idle
docker inspect "${TRAINER_CONTAINER}" >/dev/null
ssh -o BatchMode=yes "root@${ROLLOUT_HOST}" "docker inspect '${ROLLOUT_CONTAINER}'" >/dev/null
[[ "$(sha256sum "${PACKAGE_HOST}/private/grpo_approved43.sensitive.parquet" | cut -d' ' -f1)" == "${EXPECTED_APPROVED_SHA256}" ]]
[[ "$(sha256sum "${PACKAGE_HOST}/private/grpo_approved43_manifest.sensitive.jsonl" | cut -d' ' -f1)" == "${EXPECTED_MANIFEST_SHA256}" ]]
[[ "$(sha256sum "${HOST_PROJECT_ROOT}/runs/llin-v15-codex-model2-100-step120-8x-20260821-01/data/rollout_100.sensitive.parquet" | cut -d' ' -f1)" == "${EXPECTED_RAW100_SHA256}" ]]
local_config="$(docker exec "${TRAINER_CONTAINER}" sha256sum "${MODEL_PATH}/config.json" | cut -d' ' -f1)"
remote_config="$(ssh -o BatchMode=yes "root@${ROLLOUT_HOST}" "docker exec '${ROLLOUT_CONTAINER}' sha256sum '${MODEL_PATH}/config.json'" | cut -d' ' -f1)"
[[ "${local_config}" == "${EXPECTED_CONFIG_SHA256}" && "${remote_config}" == "${EXPECTED_CONFIG_SHA256}" ]]
model_hash_local_before="$(compound_hash_local)"
model_hash_remote_before="$(compound_hash_remote)"
[[ "${model_hash_local_before}" == "${EXPECTED_MODEL_COMPOUND_SHA256}" ]]
[[ "${model_hash_remote_before}" == "${EXPECTED_MODEL_COMPOUND_SHA256}" ]]
cat > "${RUN_HOST}/audit/asset_gate.safe.json" <<EOF
{
  "approved43_parquet_sha256": "${EXPECTED_APPROVED_SHA256}",
  "approved43_manifest_sha256": "${EXPECTED_MANIFEST_SHA256}",
  "raw100_sha256": "${EXPECTED_RAW100_SHA256}",
  "model_config_sha256_m05": "${local_config}",
  "model_config_sha256_m06": "${remote_config}",
  "model_compound_sha256_m05_before": "${model_hash_local_before}",
  "model_compound_sha256_m06_before": "${model_hash_remote_before}",
  "model_compound_method": "LC_ALL=C sha256sum absolute sorted glob lines then sha256",
  "npu_idle_before": true,
  "qwen36_or_historical_checkpoint_reused": false,
  "full_training_allowed": false
}
EOF

printf 'container_cpu_gates\n' > "${RUN_HOST}/state"
docker exec "${TRAINER_CONTAINER}" env PYTHONPATH="${RUNTIME_CONTAINER}" python3 -m pytest \
  "${RUNTIME_CONTAINER}/tests/test_tiered_query_cost_reward.py" \
  "${RUNTIME_CONTAINER}/tests/test_pi_tool_contract.py" \
  "${RUNTIME_CONTAINER}/tests/test_qwen38_approved43_outcome_launcher.py::test_tiered_canary_launcher_freezes_actual_update_contract" \
  "${RUNTIME_CONTAINER}/tests/test_qwen38_approved43_outcome_launcher.py::test_prepare_tiered_sealed8_is_disjoint_and_balanced" \
  "${RUNTIME_CONTAINER}/tests/test_qwen38_approved43_outcome_launcher.py::test_prepare_tiered_canary_alternates_ten_numeric_ten_table" \
  "${RUNTIME_CONTAINER}/tests/test_v15_mixed21_strict_training.py::test_group_gate_masks_all_uniform_groups_and_skips_empty_optimizer_batch" \
  "${RUNTIME_CONTAINER}/tests/test_v15_mixed21_strict_training.py::test_group_gate_rejects_incomplete_eight_sample_group" \
  "${RUNTIME_CONTAINER}/tests/test_qwen38_approved43_outcome_launcher.py::test_runtime_gate_consumes_success_without_legacy_acc" \
  "${RUNTIME_CONTAINER}/tests/test_qwen38_approved43_outcome_launcher.py::test_actual_optimizer_patch_covers_parent_and_fully_async_versioning" \
  "${RUNTIME_CONTAINER}/tests/test_qwen38_approved43_outcome_launcher.py::test_skipped_batch_keeps_optimizer_policy_and_adam_but_reopens_rollout_window" \
  "${RUNTIME_CONTAINER}/tests/test_qwen38_approved43_outcome_launcher.py::test_mixed_group_updates_and_advances_policy_while_stale_group_fails_closed" \
  -q --basetemp=/tmp/llin-canary-host-gate > "${RUN_HOST}/audit/container_cpu_gate.log" 2>&1

printf 'backing_up_live_verl\n' > "${RUN_HOST}/state"
docker cp "${TRAINER_CONTAINER}:/verl/verl/experimental/separation/ray_trainer.py" "${RUN_HOST}/live_patch_backup/m05.ray_trainer.py"
docker cp "${TRAINER_CONTAINER}:/verl/verl/experimental/fully_async_policy/fully_async_trainer.py" "${RUN_HOST}/live_patch_backup/m05.fully_async_trainer.py"
ssh -o BatchMode=yes "root@${ROLLOUT_HOST}" \
  "docker exec '${ROLLOUT_CONTAINER}' cat /verl/verl/experimental/agent_loop/agent_loop.py" \
  > "${RUN_HOST}/live_patch_backup/m06.agent_loop.py"
chmod 600 "${RUN_HOST}/live_patch_backup/"*.py
sha256sum "${RUN_HOST}/live_patch_backup/"*.py > "${RUN_HOST}/live_patch_backup/pre_patch.sha256"

printf 'staging_rollout_data\n' > "${RUN_HOST}/state"
docker exec "${TRAINER_CONTAINER}" env PYTHONPATH="${RUNTIME_CONTAINER}" python3 \
  "${RUNTIME_CONTAINER}/scripts/prepare_qwen38_tiered_canary_data.py" \
  --approved43 "${PACKAGE_CONTAINER}/private/grpo_approved43.sensitive.parquet" \
  --manifest "${PACKAGE_CONTAINER}/private/grpo_approved43_manifest.sensitive.jsonl" \
  --tasks "${TASKS_CONTAINER}" \
  --output "${RUN_CONTAINER}/private/canary20.sensitive.parquet" \
  --safe-summary "${RUN_CONTAINER}/canary20.safe.json" \
  --database-root "${RUN_CONTAINER}/private/pi_sandbox" \
  > "${RUN_HOST}/audit/prestage_canary20.log"
docker exec "${TRAINER_CONTAINER}" env PYTHONPATH="${RUNTIME_CONTAINER}" python3 \
  "${RUNTIME_CONTAINER}/scripts/prepare_qwen38_tiered_canary_sealed8.py" \
  --approved43 "${PACKAGE_CONTAINER}/private/grpo_approved43.sensitive.parquet" \
  --raw100 "${RAW100_CONTAINER}" \
  --output "${RUN_CONTAINER}/private/sealed8.sensitive.parquet" \
  --safe-summary "${RUN_CONTAINER}/sealed8.safe.json" \
  --tasks "${TASKS_CONTAINER}" \
  --database-root "${RUN_CONTAINER}/private/pi_sandbox" \
  > "${RUN_HOST}/audit/prestage_sealed8.log"
chmod 600 "${RUN_HOST}/private/canary20.sensitive.parquet" "${RUN_HOST}/private/sealed8.sensitive.parquet"
ssh -o BatchMode=yes "root@${ROLLOUT_HOST}" \
  "mkdir -p '${RUN_HOST}/private' '${RUN_HOST}/audit' && chmod 700 '${RUN_HOST}' '${RUN_HOST}/private' '${RUN_HOST}/audit'"
scp -p \
  "${RUN_HOST}/private/canary20.sensitive.parquet" \
  "${RUN_HOST}/private/sealed8.sensitive.parquet" \
  "root@${ROLLOUT_HOST}:${RUN_HOST}/private/"
ssh -o BatchMode=yes "root@${ROLLOUT_HOST}" \
  "chmod 600 '${RUN_HOST}/private/canary20.sensitive.parquet' '${RUN_HOST}/private/sealed8.sensitive.parquet'"
train_sha_local="$(sha256sum "${RUN_HOST}/private/canary20.sensitive.parquet" | cut -d' ' -f1)"
sealed_sha_local="$(sha256sum "${RUN_HOST}/private/sealed8.sensitive.parquet" | cut -d' ' -f1)"
train_sha_remote="$(ssh -o BatchMode=yes "root@${ROLLOUT_HOST}" "sha256sum '${RUN_HOST}/private/canary20.sensitive.parquet'" | cut -d' ' -f1)"
sealed_sha_remote="$(ssh -o BatchMode=yes "root@${ROLLOUT_HOST}" "sha256sum '${RUN_HOST}/private/sealed8.sensitive.parquet'" | cut -d' ' -f1)"
[[ "${train_sha_local}" == "${train_sha_remote}" && "${sealed_sha_local}" == "${sealed_sha_remote}" ]]
cat > "${RUN_HOST}/audit/rollout_data_staging.safe.json" <<EOF
{
  "canary20_sha256_m05": "${train_sha_local}",
  "canary20_sha256_m06": "${train_sha_remote}",
  "sealed8_sha256_m05": "${sealed_sha_local}",
  "sealed8_sha256_m06": "${sealed_sha_remote}",
  "private_mode": "0600",
  "cross_host_identical": true
}
EOF

printf 'staging_bound_pi_sandbox\n' > "${RUN_HOST}/state"
docker exec "${TRAINER_CONTAINER}" env PYTHONPATH="${RUNTIME_CONTAINER}" python3 \
  "${RUNTIME_CONTAINER}/scripts/stage_bound_pi_sandbox.py" \
  --dataset "${PACKAGE_CONTAINER}/private/grpo_approved43.sensitive.parquet" \
  --dataset "${RUN_CONTAINER}/private/sealed8.sensitive.parquet" \
  --source-root /pi_sandbox \
  --output-root "${RUN_CONTAINER}/private/pi_sandbox" \
  --safe-summary "${RUN_CONTAINER}/audit/bound_pi_sandbox.safe.json" \
  > "${RUN_HOST}/audit/stage_bound_pi_sandbox.log"
scp -pr "${RUN_HOST}/private/pi_sandbox" "root@${ROLLOUT_HOST}:${RUN_HOST}/private/"
scp -p "${RUN_HOST}/audit/bound_pi_sandbox.safe.json" \
  "root@${ROLLOUT_HOST}:${RUN_HOST}/audit/bound_pi_sandbox.safe.json"
local_database_compound="$({
  cd "${RUN_HOST}/private/pi_sandbox"
  find . -name logistics.sqlite -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum
} | sha256sum | cut -d' ' -f1)"
remote_database_compound="$(ssh -o BatchMode=yes "root@${ROLLOUT_HOST}" \
  "cd '${RUN_HOST}/private/pi_sandbox' && find . -name logistics.sqlite -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1")"
[[ "${local_database_compound}" == "${remote_database_compound}" ]]
local_database_count="$(find "${RUN_HOST}/private/pi_sandbox" -name logistics.sqlite -type f | wc -l)"
remote_database_count="$(ssh -o BatchMode=yes "root@${ROLLOUT_HOST}" \
  "find '${RUN_HOST}/private/pi_sandbox' -name logistics.sqlite -type f | wc -l")"
expected_database_count="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["database_files"])' \
  "${RUN_HOST}/audit/bound_pi_sandbox.safe.json")"
[[ "${local_database_count}" == "${remote_database_count}" \
  && "${local_database_count}" == "${expected_database_count}" \
  && "${local_database_count}" -ge 1 ]]
cat > "${RUN_HOST}/audit/bound_pi_sandbox_cross_host.safe.json" <<EOF
{
  "database_count_m05": ${local_database_count},
  "database_count_m06": ${remote_database_count},
  "database_compound_sha256_m05": "${local_database_compound}",
  "database_compound_sha256_m06": "${remote_database_compound}",
  "cross_host_identical": true,
  "source_scope": "approved43_union_sealed8_only"
}
EOF

printf 'online_observability_cpu_gate\n' > "${RUN_HOST}/state"
docker exec "${TRAINER_CONTAINER}" env PYTHONPATH="${RUNTIME_CONTAINER}" python3 \
  "${RUNTIME_CONTAINER}/scripts/replay_qwen38_tiered_observability.py" \
  --rollout-dir "${FROZEN96_RUN_CONTAINER}/private/rollouts" \
  --dataset "${FROZEN96_RUN_CONTAINER}/private/canary20.sensitive.parquet" \
  --database-root "${RUN_CONTAINER}/private/pi_sandbox" \
  --output "${RUN_CONTAINER}/audit/offline96_observability_replay.safe.json" \
  > "${RUN_HOST}/audit/offline96_observability_replay.log"
docker exec -i "${TRAINER_CONTAINER}" python3 - "${RUN_CONTAINER}/audit/offline96_observability_replay.safe.json" <<'PY'
import json, sys

value = json.load(open(sys.argv[1], encoding="utf-8"))
finals = value["final_component_counts"]
if value["rows"] != 96 or value["task_join_missing"] != 0:
    raise SystemExit("offline96 identity/row gate failed")
if value["task_component_counts"] != {"PASS:task_and_database_available": 96}:
    raise SystemExit("offline96 database repair gate failed")
if not any(key.startswith("PASS:") for key in finals) or not any(key.startswith("FAIL:") for key in finals):
    raise SystemExit("offline96 final component lacks credible PASS/FAIL")
if value["after"]["judge_state_counts"].get("PASS", 0) != 0:
    raise SystemExit("offline96 missing tool evidence was promoted to PASS")
PY
for dataset_name in canary20 sealed8; do
  docker exec "${TRAINER_CONTAINER}" env PYTHONPATH="${RUNTIME_CONTAINER}" python3 \
    "${RUNTIME_CONTAINER}/scripts/validate_qwen38_tiered_online_observability.py" \
    --dataset "${RUN_CONTAINER}/private/${dataset_name}.sensitive.parquet" \
    --database-root "${RUN_CONTAINER}/private/pi_sandbox" \
    --tokenizer-path "${MODEL_PATH}" \
    --output "${RUN_CONTAINER}/audit/${dataset_name}.m05.observability.safe.json" \
    > "${RUN_HOST}/audit/${dataset_name}.m05.observability.log"
  ssh -o BatchMode=yes "root@${ROLLOUT_HOST}" \
    "docker exec '${ROLLOUT_CONTAINER}' env PYTHONPATH='${RUNTIME_CONTAINER}' python3 '${RUNTIME_CONTAINER}/scripts/validate_qwen38_tiered_online_observability.py' --dataset '${RUN_CONTAINER}/private/${dataset_name}.sensitive.parquet' --database-root '${RUN_CONTAINER}/private/pi_sandbox' --tokenizer-path '${MODEL_PATH}' --output '${RUN_CONTAINER}/audit/${dataset_name}.m06.observability.safe.json'" \
    > "${RUN_HOST}/audit/${dataset_name}.m06.observability.log"
  scp -p \
    "root@${ROLLOUT_HOST}:${RUN_HOST}/audit/${dataset_name}.m06.observability.safe.json" \
    "${RUN_HOST}/audit/${dataset_name}.m06.observability.safe.json"
done

printf 'starting_isolated_ray\n' > "${RUN_HOST}/state"
docker exec "${TRAINER_CONTAINER}" env \
  PROJECT_ROOT="${RUNTIME_CONTAINER}" RAY_HEAD_PORT=36379 \
  PI_AGENT_SANDBOX_LOWER="${RUN_CONTAINER}/private/pi_sandbox" \
  PI_AGENT_TOKENIZER_PATH="${MODEL_PATH}" \
  RAY_TEMP_DIR=/tmp/q38-tiered-canary-ray-m05 \
  bash "${RUNTIME_CONTAINER}/scripts/start_ray_qwen38_smoke_m05.sh" \
  > "${RUN_HOST}/ray_m05.log" 2>&1
ray_started=true
ssh -o BatchMode=yes "root@${ROLLOUT_HOST}" \
  "docker exec '${ROLLOUT_CONTAINER}' env PROJECT_ROOT='${RUNTIME_CONTAINER}' RAY_HEAD_ADDRESS='${RAY_ADDRESS}' PI_AGENT_SANDBOX_LOWER='${RUN_CONTAINER}/private/pi_sandbox' PI_AGENT_TOKENIZER_PATH='${MODEL_PATH}' bash '${RUNTIME_CONTAINER}/scripts/start_ray_qwen38_smoke_m06.sh'" \
  > "${RUN_HOST}/ray_m06.log" 2>&1
docker exec "${TRAINER_CONTAINER}" ray status --address="${RAY_ADDRESS}" > "${RUN_HOST}/audit/ray_status_before.safe.txt"

printf 'training_canary_active\n' > "${RUN_HOST}/state"
docker exec "${TRAINER_CONTAINER}" env \
  PROJECT_ROOT="${RUNTIME_CONTAINER}" \
  PACKAGE_ROOT="${PACKAGE_CONTAINER}" \
  TASKS_FILE="${TASKS_CONTAINER}" \
  RAW100_FILE="${RAW100_CONTAINER}" \
  RUN_NAME="${RUN_NAME}" \
  OUTPUT_DIR="${RUN_CONTAINER}" \
  MODEL_PATH="${MODEL_PATH}" \
  RAY_ADDRESS="${RAY_ADDRESS}" \
  MEGATRON_BRIDGE_ROOT="${CONTAINER_PROJECT_ROOT}/reference/Megatron-Bridge-de93536e/src" \
  LLIN_CANARY_AUDIT_DIR="${RUN_CONTAINER}/private/parameter_audit" \
  PI_AGENT_SANDBOX_LOWER="${RUN_CONTAINER}/private/pi_sandbox" \
  PI_AGENT_TOKENIZER_PATH="${MODEL_PATH}" \
  bash "${RUNTIME_CONTAINER}/scripts/run_pi_qwen38_approved43_tiered_canary_v1.sh" \
  > "${RUN_HOST}/training.log" 2>&1

printf 'post_canary_gates\n' > "${RUN_HOST}/state"
model_hash_local_after="$(compound_hash_local)"
model_hash_remote_after="$(compound_hash_remote)"
[[ "${model_hash_local_after}" == "${model_hash_local_before}" ]]
[[ "${model_hash_remote_after}" == "${model_hash_remote_before}" ]]
checkpoint_dir="${RUN_HOST}/private_recovery/checkpoints/global_step_5"
[[ -d "${checkpoint_dir}/actor" ]]
find "${checkpoint_dir}" -type f -printf '%P\n' | LC_ALL=C sort > "${RUN_HOST}/audit/checkpoint_files.safe.txt"
find "${checkpoint_dir}" -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > "${RUN_HOST}/private/checkpoint_files.sha256"
chmod 600 "${RUN_HOST}/private/checkpoint_files.sha256"
checkpoint_tree_sha256="$(sha256sum "${RUN_HOST}/private/checkpoint_files.sha256" | cut -d' ' -f1)"
cat > "${RUN_HOST}/audit/post_canary.safe.json" <<EOF
{
  "model_source_hash_m05_unchanged": true,
  "model_source_hash_m06_unchanged": true,
  "temporary_checkpoint_global_step": 5,
  "temporary_checkpoint_tree_sha256": "${checkpoint_tree_sha256}",
  "full_training_allowed": false,
  "next_action": "main_thread_reward_and_hacking_review"
}
EOF
printf 'canary_complete_pending_review\n' > "${RUN_HOST}/state"
