#!/usr/bin/env bash
set -Eeuo pipefail

HOST_PROJECT_ROOT="${HOST_PROJECT_ROOT:-/data3/llin/qwen3.6-27b-verl-grpo}"
CONTAINER_PROJECT_ROOT="${CONTAINER_PROJECT_ROOT:-/workspace/llin-verl-grpo}"
TRAINER_CONTAINER="${TRAINER_CONTAINER:-llin-verl-trainer-m05-20260730}"
PIPELINE_NAME="${PIPELINE_NAME:-llin-accuracy-unattended-$(date +%Y%m%d-%H%M%S)}"
PIPELINE_DIR="${HOST_PROJECT_ROOT}/runs/${PIPELINE_NAME}"
CONTAINER_PIPELINE_DIR="${CONTAINER_PROJECT_ROOT}/runs/${PIPELINE_NAME}"
START_STAGE="${START_STAGE:-stage1}"
DIAGNOSTIC_PIPELINE_NAME="${DIAGNOSTIC_PIPELINE_NAME:-${PIPELINE_NAME}}"
DIAGNOSTIC_PIPELINE_DIR="${HOST_PROJECT_ROOT}/runs/${DIAGNOSTIC_PIPELINE_NAME}"
BOSS_ROOT="${BOSS_ROOT:-/data/renjunxiang/coding/huawei_train}"
BOSS_TASK_MANIFEST="${BOSS_TASK_MANIFEST:-${BOSS_ROOT}/datasets/sandboxes/raw/sft/20260628_v15/dwh_tasks.jsonl}"
BOSS_DB="${BOSS_DB:-${BOSS_ROOT}/datasets/sandboxes/raw/sft/20260628_v15/logistics.sqlite}"
SOURCE_VAL="${CONTAINER_PROJECT_ROOT}/data/boss_v15_dwh_full276_20260806/dataset/boss_pi_val.parquet"
STEP120_CHECKPOINT="${CONTAINER_PROJECT_ROOT}/runs/llin-pi-dense-correctness-step100-to-step120-20260810-01/checkpoints/global_step_120"
FIRST100_ROLLOUTS="${CONTAINER_PROJECT_ROOT}/runs/llin-v15-dwh-bossreward-12groups-100step-20260805-03/rollouts"
SECOND100_ROLLOUTS="${CONTAINER_PROJECT_ROOT}/runs/llin-v15-dwh-bossreward-step100-to-step200-12groups-20260806-01/rollouts"

CURRENT_STAGE="initializing"
FINAL_EXIT=1

write_stage() {
  CURRENT_STAGE="$1"
  printf '%s\n' "${CURRENT_STAGE}" > "${PIPELINE_DIR}/current_stage"
  date --iso-8601=seconds > "${PIPELINE_DIR}/stage_updated_at"
}

cleanup() {
  set +e
  failed_stage="${CURRENT_STAGE}"
  printf 'cleanup\n' > "${PIPELINE_DIR}/current_stage"
  date --iso-8601=seconds > "${PIPELINE_DIR}/stage_updated_at"
  docker exec "${TRAINER_CONTAINER}" bash -lc 'ray stop --force' \
    >> "${PIPELINE_DIR}/cleanup.log" 2>&1
  date --iso-8601=seconds > "${PIPELINE_DIR}/cleanup_finished_at"
  if [[ "${FINAL_EXIT}" == "0" ]]; then
    write_stage "done"
    printf 'complete\n' > "${PIPELINE_DIR}/DONE"
  else
    printf '%s\n' "${failed_stage}" > "${PIPELINE_DIR}/FAILED"
  fi
  printf '%s\n' "${FINAL_EXIT}" > "${PIPELINE_DIR}/exit_code"
  date --iso-8601=seconds > "${PIPELINE_DIR}/finished_at"
}
trap cleanup EXIT

mkdir -p "${PIPELINE_DIR}"
exec 9>"${HOST_PROJECT_ROOT}/runs/.accuracy_unattended.lock"
if ! flock -n 9; then
  printf 'another unattended accuracy pipeline holds the lock\n' >&2
  exit 9
fi
printf '%s\n' "$$" > "${PIPELINE_DIR}/supervisor.pid"
date --iso-8601=seconds > "${PIPELINE_DIR}/started_at"

for path in "${BOSS_TASK_MANIFEST}" "${BOSS_DB}"; do
  if [[ ! -f "${path}" ]]; then
    printf 'required boss asset missing: %s\n' "${path}" >&2
    exit 2
  fi
done
docker inspect "${TRAINER_CONTAINER}" >/dev/null
install -m 0644 "${BOSS_TASK_MANIFEST}" "${PIPELINE_DIR}/boss_v15_dwh_tasks.jsonl"

score_boss_exact() {
  local label="$1"
  local run_name="$2"
  local policy_step="$3"
  local parquet="$4"
  local run_host="${HOST_PROJECT_ROOT}/runs/${run_name}"
  local run_container="${CONTAINER_PROJECT_ROOT}/runs/${run_name}"
  mkdir -p "${run_host}/boss_exact"
  docker exec "${TRAINER_CONTAINER}" bash -lc \
    "python3 '${CONTAINER_PROJECT_ROOT}/scripts/prepare_boss_exact_evaluation.py' \
      --validation '${run_container}/validation/${policy_step}.jsonl' \
      --parquet '${parquet}' \
      --task-manifest '${CONTAINER_PIPELINE_DIR}/boss_v15_dwh_tasks.jsonl' \
      --trajectory-output '${run_container}/boss_exact/${label}.openai.jsonl' \
      --manifest-output '${run_container}/boss_exact/${label}.manifest.jsonl' \
      --summary-output '${run_container}/boss_exact/${label}.adapter.json'"
  python3 "${BOSS_ROOT}/scripts/data/reward_judge.py" \
    --manifest "${run_host}/boss_exact/${label}.manifest.jsonl" \
    --db-path "${BOSS_DB}" \
    --input "${run_host}/boss_exact/${label}.openai.jsonl" \
    --output "${run_host}/boss_exact/${label}.reward.jsonl" \
    --judge-out "${run_host}/boss_exact/${label}.judge.jsonl"
}

case "${START_STAGE}" in
  stage1)
    write_stage "stage1_prepare_oracle"
    docker exec "${TRAINER_CONTAINER}" bash -lc \
      "python3 '${CONTAINER_PROJECT_ROOT}/scripts/prepare_oracle_ladder.py' \
        --input '${SOURCE_VAL}' \
        --output-dir '${CONTAINER_PROJECT_ROOT}/data/oracle_ladder_step120_20260810' \
        --manifest '${CONTAINER_PIPELINE_DIR}/oracle_dataset_manifest.json'"

    for arm in control contract oracle; do
      write_stage "stage1_oracle_${arm}"
      run_name="${PIPELINE_NAME}-oracle-${arm}"
      timeout --signal=TERM --kill-after=120 7200 \
        docker exec "${TRAINER_CONTAINER}" bash -lc \
          "ORACLE_ARM='${arm}' RUN_NAME='${run_name}' SOURCE_CHECKPOINT='${STEP120_CHECKPOINT}' \
           bash '${CONTAINER_PROJECT_ROOT}/scripts/run_pi_oracle_ladder_step120.sh'" \
        > "${PIPELINE_DIR}/oracle_${arm}.log" 2>&1
      printf '0\n' > "${HOST_PROJECT_ROOT}/runs/${run_name}/exit_code"
      score_boss_exact \
        "oracle_${arm}" "${run_name}" 120 \
        "${CONTAINER_PROJECT_ROOT}/data/oracle_ladder_step120_20260810/oracle_${arm}.parquet"
    done

    write_stage "stage1_oracle_analysis"
    docker exec "${TRAINER_CONTAINER}" bash -lc \
      "cd '${CONTAINER_PROJECT_ROOT}' && python3 scripts/analyze_oracle_ladder.py \
        --control '${CONTAINER_PROJECT_ROOT}/runs/${PIPELINE_NAME}-oracle-control/boss_exact/oracle_control.reward.jsonl' \
        --contract '${CONTAINER_PROJECT_ROOT}/runs/${PIPELINE_NAME}-oracle-contract/boss_exact/oracle_contract.reward.jsonl' \
        --oracle '${CONTAINER_PROJECT_ROOT}/runs/${PIPELINE_NAME}-oracle-oracle/boss_exact/oracle_oracle.reward.jsonl' \
        --output '${CONTAINER_PIPELINE_DIR}/oracle_summary.json'"

    write_stage "stage3_banded_reward_replay"
    docker exec "${TRAINER_CONTAINER}" bash -lc \
      "cd '${CONTAINER_PROJECT_ROOT}' && python3 scripts/replay_banded_reward_gate.py \
        --rollout-dir '${FIRST100_ROLLOUTS}' \
        --rollout-dir '${SECOND100_ROLLOUTS}' \
        --expected-group-size 4 \
        --output '${CONTAINER_PIPELINE_DIR}/banded_reward_replay_gate.json'"
    ;;
  stage4)
    for artifact in oracle_summary.json banded_reward_replay_gate.json; do
      if [[ ! -f "${DIAGNOSTIC_PIPELINE_DIR}/${artifact}" ]]; then
        printf 'required diagnostic artifact missing: %s\n' \
          "${DIAGNOSTIC_PIPELINE_DIR}/${artifact}" >&2
        exit 2
      fi
      install -m 0644 \
        "${DIAGNOSTIC_PIPELINE_DIR}/${artifact}" "${PIPELINE_DIR}/${artifact}"
    done
    python3 - "${PIPELINE_DIR}/banded_reward_replay_gate.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    gate = json.load(handle)
if gate.get("gate_passed") is not True:
    raise SystemExit("refusing stage4 resume because replay gate did not pass")
PY
    printf '%s\n' "${DIAGNOSTIC_PIPELINE_NAME}" > \
      "${PIPELINE_DIR}/resumed_from_pipeline"
    ;;
  *)
    printf 'START_STAGE must be stage1 or stage4, got: %s\n' \
      "${START_STAGE}" >&2
    exit 2
    ;;
esac

CANARY_RUN="${PIPELINE_NAME}-banded-2x8-step120-to125"
write_stage "stage4_train_5step"
timeout --signal=TERM --kill-after=180 28800 \
  docker exec "${TRAINER_CONTAINER}" bash -lc \
    "SOURCE_CHECKPOINT='${STEP120_CHECKPOINT}' START_POLICY_STEP=120 NEW_TRAINING_STEPS=5 \
     LOAD_OPTIMIZER_STATE=false RUN_NAME='${CANARY_RUN}' \
     bash '${CONTAINER_PROJECT_ROOT}/scripts/launch_pi_banded_2x8_resume.sh'"
score_boss_exact \
  "step125" "${CANARY_RUN}" 125 \
  "${CONTAINER_PROJECT_ROOT}/data/boss_v15_dwh_full276_20260806/dataset/boss_pi_val.parquet"

write_stage "stage4_gate_5step"
docker exec "${TRAINER_CONTAINER}" bash -lc \
  "cd '${CONTAINER_PROJECT_ROOT}' && python3 scripts/analyze_accuracy_gate.py \
    --rollout-dir '${CONTAINER_PROJECT_ROOT}/runs/${CANARY_RUN}/rollouts' \
    --boss-reward '${CONTAINER_PROJECT_ROOT}/runs/${CANARY_RUN}/boss_exact/step125.reward.jsonl' \
    --expected-group-size 8 \
    --output '${CONTAINER_PIPELINE_DIR}/step125_accuracy_gate.json'"

FINAL_RUN="${PIPELINE_NAME}-banded-2x8-step125-to145"
write_stage "stage5_train_20step"
timeout --signal=TERM --kill-after=180 64800 \
  docker exec "${TRAINER_CONTAINER}" bash -lc \
    "SOURCE_CHECKPOINT='${CONTAINER_PROJECT_ROOT}/runs/${CANARY_RUN}/checkpoints/global_step_125' \
     START_POLICY_STEP=125 NEW_TRAINING_STEPS=20 LOAD_OPTIMIZER_STATE=false \
     RUN_NAME='${FINAL_RUN}' \
     bash '${CONTAINER_PROJECT_ROOT}/scripts/launch_pi_banded_2x8_resume.sh'"
score_boss_exact \
  "step145" "${FINAL_RUN}" 145 \
  "${CONTAINER_PROJECT_ROOT}/data/boss_v15_dwh_full276_20260806/dataset/boss_pi_val.parquet"

write_stage "stage5_final_analysis"
docker exec "${TRAINER_CONTAINER}" bash -lc \
  "cd '${CONTAINER_PROJECT_ROOT}' && python3 scripts/analyze_accuracy_gate.py \
    --rollout-dir '${CONTAINER_PROJECT_ROOT}/runs/${FINAL_RUN}/rollouts' \
    --boss-reward '${CONTAINER_PROJECT_ROOT}/runs/${FINAL_RUN}/boss_exact/step145.reward.jsonl' \
    --expected-group-size 8 \
    --output '${CONTAINER_PIPELINE_DIR}/step145_final_summary.json' \
    --report-only"

FINAL_EXIT=0
