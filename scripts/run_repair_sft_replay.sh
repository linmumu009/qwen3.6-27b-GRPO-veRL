#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/llin-verl-grpo}"
MODEL_DIST_CKPT="${MODEL_DIST_CKPT:?MODEL_DIST_CKPT is required}"
LABEL="${LABEL:?LABEL is required}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data/repair_sft_train236_20260811}"
EVAL_FILE="${EVAL_FILE:-${DATA_DIR}/repair_sft_replay.parquet}"
RUN_NAME="${RUN_NAME:-llin-repair-sft-replay-${LABEL}-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/runs/${RUN_NAME}}"
MAX_ASSISTANT_TURNS="${MAX_ASSISTANT_TURNS:-26}"
MAX_USER_TURNS="${MAX_USER_TURNS:-25}"

if [[ ! -f "${MODEL_DIST_CKPT}/.metadata" ]]; then
  printf 'distributed model checkpoint missing: %s\n' "${MODEL_DIST_CKPT}" >&2
  exit 2
fi
if [[ ! -f "${EVAL_FILE}" ]]; then
  printf 'repair replay parquet missing: %s\n' "${EVAL_FILE}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"
cat > "${OUTPUT_DIR}/experiment_contract.txt" <<EOF
purpose=train236_repair_sft_same_task_replay
label=${LABEL}
model_dist_checkpoint=${MODEL_DIST_CKPT}
evaluation_rows=16
evaluation_split=train236_same_task_not_heldout
sampling=greedy_n1
system_tools=boss_exact
context_tokens=49152
max_tool_result_turns=${MAX_USER_TURNS}
max_assistant_turns=${MAX_ASSISTANT_TURNS}
weight_sync=forced_actor_dist_checkpoint_to_vllm
EOF

export LLIN_VAL_ONLY_FORCE_DIST_SYNC=1
RUN_NAME="${RUN_NAME}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
TRAIN_FILE="${EVAL_FILE}" \
EVAL_FILE="${EVAL_FILE}" \
MAX_ASSISTANT_TURNS="${MAX_ASSISTANT_TURNS}" \
MAX_USER_TURNS="${MAX_USER_TURNS}" \
ROLLOUT_GPU_MEMORY_UTILIZATION=0.80 \
ROLLOUT_MAX_BATCHED_TOKENS=16384 \
ROLLOUT_MAX_SEQS=12 \
bash "${PROJECT_ROOT}/scripts/run_pi_frozen_baseline.sh" \
  actor_rollout_ref.actor.megatron.use_dist_checkpointing=True \
  actor_rollout_ref.actor.megatron.dist_checkpointing_path="${MODEL_DIST_CKPT}" \
  'actor_rollout_ref.actor.checkpoint.load_contents=[]' \
  trainer.resume_mode=disable \
  trainer.val_only=True \
  trainer.validation_data_dir="${OUTPUT_DIR}/validation" \
  "$@"
