#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/llin-verl-grpo}"
MODEL_DIST_CKPT="${MODEL_DIST_CKPT:?MODEL_DIST_CKPT is required}"
EVAL_FILE="${EVAL_FILE:?EVAL_FILE is required}"
RUN_NAME="${RUN_NAME:-llin-runtime-parity-verl-step120-20260813-01}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/runs/${RUN_NAME}}"
EXPECTED_EVAL_ROWS="${EXPECTED_EVAL_ROWS:-80}"

if [[ ! -f "${MODEL_DIST_CKPT}/.metadata" ]]; then
  printf 'distributed model checkpoint missing: %s\n' "${MODEL_DIST_CKPT}" >&2
  exit 2
fi
if [[ ! -f "${EVAL_FILE}" ]]; then
  printf 'diagnostic parquet missing: %s\n' "${EVAL_FILE}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"
cat > "${OUTPUT_DIR}/experiment_contract.txt" <<EOF
purpose=pi_agent_vs_verl_runtime_parity
model=step120_identical_weights
evaluation_rows=${EXPECTED_EVAL_ROWS}
tasks=10
samples_per_task=8
sampling_temperature=1.0
sampling_top_p=0.95
sampling_top_k=20
system_tools=boss_exact
context_tokens=49152
max_tool_result_turns=25
max_assistant_turns=26
forward_only=true
optimizer_initialized=false
checkpoint_saved=false
fastest_k=false
all_samples_retained=true
EOF

RUN_NAME="${RUN_NAME}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
TRAIN_FILE="${EVAL_FILE}" \
EVAL_FILE="${EVAL_FILE}" \
MAX_ASSISTANT_TURNS=26 \
MAX_USER_TURNS=25 \
ROLLOUT_GPU_MEMORY_UTILIZATION=0.80 \
ROLLOUT_MAX_BATCHED_TOKENS=16384 \
ROLLOUT_MAX_SEQS=24 \
bash "${PROJECT_ROOT}/scripts/run_pi_frozen_baseline.sh" \
  actor_rollout_ref.actor.megatron.use_dist_checkpointing=True \
  actor_rollout_ref.actor.megatron.dist_checkpointing_path="${MODEL_DIST_CKPT}" \
  'actor_rollout_ref.actor.checkpoint.load_contents=[]' \
  ++trainer.val_only_force_dist_sync=True \
  actor_rollout_ref.rollout.val_kwargs.n=8 \
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  actor_rollout_ref.rollout.val_kwargs.top_k=20 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  trainer.resume_mode=disable \
  trainer.val_only=True \
  trainer.log_val_generations=80 \
  trainer.validation_data_dir="${OUTPUT_DIR}/validation"
