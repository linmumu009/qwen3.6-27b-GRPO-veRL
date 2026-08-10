#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/llin-verl-grpo}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data/boss_v15_dwh_full276_20260806/dataset}"
SOURCE_VAL="${SOURCE_VAL:-${DATA_DIR}/boss_pi_val.parquet}"
SENTINEL_DIR="${SENTINEL_DIR:-${PROJECT_ROOT}/data/force_final_sentinel_20260810}"
SENTINEL_FILE="${SENTINEL_FILE:-${SENTINEL_DIR}/sentinel6.parquet}"
SENTINEL_MANIFEST="${SENTINEL_MANIFEST:-${SENTINEL_DIR}/sentinel6.manifest.json}"
SENTINEL_TASK_IDS="${SENTINEL_TASK_IDS:-task_000070,task_000080,task_000133,task_000196,task_000048,task_000269}"
FORCE_FINAL_AFTER_ASSISTANT_TURNS="${FORCE_FINAL_AFTER_ASSISTANT_TURNS:-22}"
FORCE_FINAL_RESERVE_RESPONSE_TOKENS="${FORCE_FINAL_RESERVE_RESPONSE_TOKENS:-4096}"
FORCE_FINAL_MAX_RESPONSE_TOKENS="${FORCE_FINAL_MAX_RESPONSE_TOKENS:-4096}"
FORCE_FINAL_MAX_RETRIES="${FORCE_FINAL_MAX_RETRIES:-1}"
GATE_DESCRIPTION="${GATE_DESCRIPTION:-rescue_at_least_2_of_4_and_no_guardrail_completion_regression}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-${PROJECT_ROOT}/runs/llin-pi-dense-correctness-step100-to-step120-20260810-01/checkpoints/global_step_120}"
RUN_NAME="${RUN_NAME:-llin-step120-force-final-48k-sentinel6-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/runs/${RUN_NAME}}"

if [[ ! -f "${SOURCE_CHECKPOINT}/actor/ckpt_contents.json" ]]; then
  printf 'Step 120 checkpoint not found: %s\n' "${SOURCE_CHECKPOINT}" >&2
  exit 2
fi

mkdir -p "${SENTINEL_DIR}" "${OUTPUT_DIR}"
IFS=',' read -r -a sentinel_task_ids <<< "${SENTINEL_TASK_IDS}"
prepare_task_args=()
for task_id in "${sentinel_task_ids[@]}"; do
  prepare_task_args+=(--task-id "${task_id}")
done
python3 "${PROJECT_ROOT}/scripts/prepare_force_final_sentinel.py" \
  --input "${SOURCE_VAL}" \
  --output "${SENTINEL_FILE}" \
  --manifest "${SENTINEL_MANIFEST}" \
  "${prepare_task_args[@]}"

cat > "${OUTPUT_DIR}/experiment_contract.txt" <<EOF
mode=validation_only
checkpoint=${SOURCE_CHECKPOINT}
context_tokens=49152
force_final_after_assistant_turns=${FORCE_FINAL_AFTER_ASSISTANT_TURNS}
force_final_reserve_response_tokens=${FORCE_FINAL_RESERVE_RESPONSE_TOKENS}
force_final_max_response_tokens=${FORCE_FINAL_MAX_RESPONSE_TOKENS}
force_final_max_retries=${FORCE_FINAL_MAX_RETRIES}
sentinel_tasks=${SENTINEL_TASK_IDS}
gate=${GATE_DESCRIPTION}
EOF

RUN_NAME="${RUN_NAME}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
EVAL_FILE="${SENTINEL_FILE}" \
MAX_ASSISTANT_TURNS=26 \
MAX_USER_TURNS=25 \
bash "${PROJECT_ROOT}/scripts/run_pi_frozen_baseline.sh" \
  +actor_rollout_ref.rollout.multi_turn.force_final_after_assistant_turns="${FORCE_FINAL_AFTER_ASSISTANT_TURNS}" \
  +actor_rollout_ref.rollout.multi_turn.force_final_reserve_response_tokens="${FORCE_FINAL_RESERVE_RESPONSE_TOKENS}" \
  +actor_rollout_ref.rollout.multi_turn.force_final_max_response_tokens="${FORCE_FINAL_MAX_RESPONSE_TOKENS}" \
  +actor_rollout_ref.rollout.multi_turn.force_final_max_retries="${FORCE_FINAL_MAX_RETRIES}" \
  trainer.resume_mode=resume_path \
  trainer.resume_from_path="${SOURCE_CHECKPOINT}" \
  trainer.del_local_ckpt_after_load=False \
  actor_rollout_ref.actor.megatron.forward_only=False \
  actor_rollout_ref.actor.megatron.optimizer_offload=True \
  actor_rollout_ref.actor.megatron.grad_offload=True \
  actor_rollout_ref.actor.megatron.use_dist_checkpointing=True \
  'actor_rollout_ref.actor.checkpoint.load_contents=[model,extra]' \
  "$@"
