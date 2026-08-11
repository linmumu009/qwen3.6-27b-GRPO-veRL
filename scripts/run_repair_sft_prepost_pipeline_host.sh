#!/usr/bin/env bash
set -Eeuo pipefail

HOST_PROJECT_ROOT="${HOST_PROJECT_ROOT:-/data3/llin/qwen3.6-27b-verl-grpo}"
CONTAINER_PROJECT_ROOT="${CONTAINER_PROJECT_ROOT:-/workspace/llin-verl-grpo}"
TRAINER_CONTAINER="${TRAINER_CONTAINER:-llin-verl-trainer-m05-20260730}"
PIPELINE_NAME="${PIPELINE_NAME:-llin-repair-sft-prepost-20260811-01}"
PIPELINE_DIR="${HOST_PROJECT_ROOT}/runs/${PIPELINE_NAME}"
CONTAINER_PIPELINE_DIR="${CONTAINER_PROJECT_ROOT}/runs/${PIPELINE_NAME}"
BASELINE_RUN_NAME="${BASELINE_RUN_NAME:-llin-repair-sft-replay-step120-20260811-01}"
POST_RUN_NAME="${POST_RUN_NAME:-llin-repair-sft-replay-post-sft-20260811-01}"
POST_MODEL_DIST_CKPT="${POST_MODEL_DIST_CKPT:-${CONTAINER_PROJECT_ROOT}/runs/llin-repair-sft-train236-overfit-step120-20260811-01/checkpoints/global_step_5/model/dist_ckpt}"
REPLAY_PARQUET="${CONTAINER_PROJECT_ROOT}/data/repair_sft_train236_20260811/repair_sft_replay.parquet"
BOSS_ROOT="${BOSS_ROOT:-/data/renjunxiang/coding/huawei_train}"
BOSS_TASK_MANIFEST="${BOSS_TASK_MANIFEST:-${BOSS_ROOT}/datasets/sandboxes/raw/sft/20260628_v15/dwh_tasks.jsonl}"
BOSS_DB="${BOSS_DB:-${BOSS_ROOT}/datasets/sandboxes/raw/sft/20260628_v15/logistics.sqlite}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-7200}"

CURRENT_STAGE=initializing
FINAL_EXIT=1

write_stage() {
  CURRENT_STAGE="$1"
  printf '%s\n' "${CURRENT_STAGE}" > "${PIPELINE_DIR}/current_stage"
  date --iso-8601=seconds > "${PIPELINE_DIR}/stage_updated_at"
}

finish() {
  set +e
  printf '%s\n' "${FINAL_EXIT}" > "${PIPELINE_DIR}/exit_code"
  date --iso-8601=seconds > "${PIPELINE_DIR}/finished_at"
  if [[ "${FINAL_EXIT}" == "0" ]]; then
    write_stage done
    printf 'complete\n' > "${PIPELINE_DIR}/DONE"
  else
    printf '%s\n' "${CURRENT_STAGE}" > "${PIPELINE_DIR}/FAILED"
  fi
}
trap finish EXIT

wait_run() {
  local run_name="$1"
  local elapsed=0
  local exit_file="${HOST_PROJECT_ROOT}/runs/${run_name}/exit_code"
  while [[ ! -f "${exit_file}" ]]; do
    if (( elapsed >= WAIT_TIMEOUT_SECONDS )); then
      printf 'timed out waiting for %s\n' "${run_name}" >&2
      return 124
    fi
    sleep 30
    elapsed=$((elapsed + 30))
  done
  local status
  status="$(tr -d '[:space:]' < "${exit_file}")"
  if [[ "${status}" != "0" ]]; then
    printf '%s exited with %s\n' "${run_name}" "${status}" >&2
    return 1
  fi
}

score_boss_exact() {
  local label="$1"
  local run_name="$2"
  local run_host="${HOST_PROJECT_ROOT}/runs/${run_name}"
  local run_container="${CONTAINER_PROJECT_ROOT}/runs/${run_name}"
  local validation="${run_container}/validation/0.jsonl"
  mkdir -p "${run_host}/boss_exact"
  docker exec "${TRAINER_CONTAINER}" bash -lc \
    "python3 '${CONTAINER_PROJECT_ROOT}/scripts/prepare_boss_exact_evaluation.py' \
      --validation '${validation}' \
      --parquet '${REPLAY_PARQUET}' \
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

mkdir -p "${PIPELINE_DIR}"
date --iso-8601=seconds > "${PIPELINE_DIR}/started_at"
for path in "${BOSS_TASK_MANIFEST}" "${BOSS_DB}"; do
  if [[ ! -f "${path}" ]]; then
    printf 'required boss asset missing: %s\n' "${path}" >&2
    exit 2
  fi
done
docker inspect "${TRAINER_CONTAINER}" >/dev/null
install -m 0644 "${BOSS_TASK_MANIFEST}" "${PIPELINE_DIR}/boss_v15_dwh_tasks.jsonl"

write_stage wait_step120_baseline
wait_run "${BASELINE_RUN_NAME}"

write_stage launch_post_sft_replay
docker exec "${TRAINER_CONTAINER}" bash -lc \
  "MODEL_DIST_CKPT='${POST_MODEL_DIST_CKPT}' LABEL='post_sft' RUN_NAME='${POST_RUN_NAME}' \
   bash '${CONTAINER_PROJECT_ROOT}/scripts/launch_repair_sft_replay.sh'"
wait_run "${POST_RUN_NAME}"

write_stage boss_exact_step120
score_boss_exact step120 "${BASELINE_RUN_NAME}"
write_stage boss_exact_post_sft
score_boss_exact post_sft "${POST_RUN_NAME}"

write_stage compare
docker exec "${TRAINER_CONTAINER}" bash -lc \
  "cd '${CONTAINER_PROJECT_ROOT}' && python3 scripts/compare_boss_exact_evaluations.py \
    --left '${CONTAINER_PROJECT_ROOT}/runs/${BASELINE_RUN_NAME}/boss_exact/step120.reward.jsonl' \
    --right '${CONTAINER_PROJECT_ROOT}/runs/${POST_RUN_NAME}/boss_exact/post_sft.reward.jsonl' \
    --left-label step120 \
    --right-label post_sft \
    --left-trajectories '${CONTAINER_PROJECT_ROOT}/runs/${BASELINE_RUN_NAME}/boss_exact/step120.openai.jsonl' \
    --right-trajectories '${CONTAINER_PROJECT_ROOT}/runs/${POST_RUN_NAME}/boss_exact/post_sft.openai.jsonl' \
    --output '${CONTAINER_PIPELINE_DIR}/comparison.json'"

docker exec -i "${TRAINER_CONTAINER}" python3 - "${CONTAINER_PIPELINE_DIR}/comparison.json" "${CONTAINER_PIPELINE_DIR}/gate.json" <<'PY'
import json
import sys

comparison = json.load(open(sys.argv[1], encoding="utf-8"))
pre = comparison["step120"]
post = comparison["post_sft"]
correct = int(post["correct_numeric_count"])
gate = {
    "contract": "train236-repair-sft-replay-gate-v1",
    "same_task_count": int(post["n"]),
    "prompt_identity": comparison.get("prompt_identity"),
    "step120_correct": int(pre["correct_numeric_count"]),
    "post_sft_correct": correct,
    "correct_delta": correct - int(pre["correct_numeric_count"]),
    "step120_complete": int(pre["complete_count"]),
    "post_sft_complete": int(post["complete_count"]),
    "step120_reward_mean": pre["reward_total_mean"],
    "post_sft_reward_mean": post["reward_total_mean"],
    "minimum_exact_successes": 14,
    "gate_passed": correct >= 14,
    "heldout_claim_allowed": False,
}
with open(sys.argv[2], "w", encoding="utf-8") as handle:
    json.dump(gate, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
print(json.dumps(gate, ensure_ascii=False, indent=2))
PY

FINAL_EXIT=0
