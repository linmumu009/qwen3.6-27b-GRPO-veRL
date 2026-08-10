#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/llin-verl-grpo}"
ORACLE_ARM="${ORACLE_ARM:?ORACLE_ARM must be control, contract, or oracle}"
ORACLE_DIR="${ORACLE_DIR:-${PROJECT_ROOT}/data/oracle_ladder_step120_20260810}"
EVAL_FILE="${ORACLE_DIR}/oracle_${ORACLE_ARM}.parquet"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-${PROJECT_ROOT}/runs/llin-pi-dense-correctness-step100-to-step120-20260810-01/checkpoints/global_step_120}"
RUN_NAME="${RUN_NAME:-llin-step120-oracle-${ORACLE_ARM}-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/runs/${RUN_NAME}}"

case "${ORACLE_ARM}" in
  control|contract|oracle) ;;
  *) printf 'Unknown oracle arm: %s\n' "${ORACLE_ARM}" >&2; exit 2 ;;
esac
if [[ ! -f "${EVAL_FILE}" ]]; then
  printf 'Oracle evaluation file not found: %s\n' "${EVAL_FILE}" >&2
  exit 2
fi
if [[ ! -f "${SOURCE_CHECKPOINT}/actor/ckpt_contents.json" ]]; then
  printf 'Step 120 checkpoint not found: %s\n' "${SOURCE_CHECKPOINT}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"
cat > "${OUTPUT_DIR}/experiment_contract.txt" <<EOF
mode=validation_only
oracle_arm=${ORACLE_ARM}
checkpoint=${SOURCE_CHECKPOINT}
context_tokens=49152
tasks=12
sampling=greedy_n1
EOF

RUN_NAME="${RUN_NAME}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
EVAL_FILE="${EVAL_FILE}" \
MAX_ASSISTANT_TURNS=26 \
MAX_USER_TURNS=25 \
bash "${PROJECT_ROOT}/scripts/run_pi_frozen_baseline.sh" \
  trainer.resume_mode=resume_path \
  trainer.resume_from_path="${SOURCE_CHECKPOINT}" \
  trainer.del_local_ckpt_after_load=False \
  actor_rollout_ref.actor.megatron.forward_only=False \
  actor_rollout_ref.actor.megatron.optimizer_offload=True \
  actor_rollout_ref.actor.megatron.grad_offload=True \
  actor_rollout_ref.actor.megatron.use_dist_checkpointing=True \
  'actor_rollout_ref.actor.checkpoint.load_contents=[model,extra]'
