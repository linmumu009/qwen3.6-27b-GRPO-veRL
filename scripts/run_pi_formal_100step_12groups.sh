#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/llin-verl-grpo}"
export PYTHONPATH="${PROJECT_ROOT}/runtime:${PROJECT_ROOT}:${PYTHONPATH:-}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data/boss_v15_dwh_full276_20260806/dataset}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/boss_pi_train.parquet}"
VAL_FILE="${VAL_FILE:-${DATA_DIR}/boss_pi_val.parquet}"
RUN_NAME="${RUN_NAME:-llin-pi-formal-grpo-12groups-100step-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/runs/${RUN_NAME}}"

# This formal entry is deliberately fixed to the approved experiment contract.
TOTAL_TRAINING_STEPS=100
GROUPS_PER_STEP=4
FINAL_EVAL_STEP="${TOTAL_TRAINING_STEPS}"
FINAL_SAVE_STEP="${TOTAL_TRAINING_STEPS}"
LEARNING_RATE="${LEARNING_RATE:-1e-7}"
PREWARM_GROUPS=8
MAX_QUEUE_GROUPS=8
STALENESS_THRESHOLD=2.0
TARGET_CONCURRENT_GROUPS=12
MAX_CONTEXT_TOKENS=49152
MAX_QUEUE_TOKENS="$((MAX_QUEUE_GROUPS * 4 * MAX_CONTEXT_TOKENS))"
ROLLOUT_GPU_MEMORY_UTILIZATION=0.80
ROLLOUT_MAX_BATCHED_TOKENS=16384
ROLLOUT_MAX_SEQS=24
AGENT_WORKERS=12
# The largest indivisible parameter is the 248320 x 5120 BF16 embedding
# (2425 MiB), so the HCCL bucket must exceed it. HCCL keeps send+receive
# buffers and Ascend PyHCCL creates another same-sized broadcast output;
# 2560 MiB is the smallest practical aligned bucket and trims the theoretical
# sync transient from ~9 GiB at 3072 MiB to ~7.5 GiB.
WEIGHT_BUCKET_MB=2560
ROLLOUT_DP=2
RESPONSES_PER_GROUP=4

# Fail closed if a later edit makes the physical sequence capacity smaller than
# the approved 12 complete groups. The update batch remains 4 groups; 12 is the
# bounded number of groups that may be in flight, not the optimizer batch size.
if (( ROLLOUT_DP * ROLLOUT_MAX_SEQS < TARGET_CONCURRENT_GROUPS * RESPONSES_PER_GROUP )); then
  printf 'Rollout capacity cannot hold %s groups x %s responses\n' \
    "${TARGET_CONCURRENT_GROUPS}" "${RESPONSES_PER_GROUP}" >&2
  exit 2
fi

# A formal run may only consume the source-joined, reviewed boss contract.
python3 "${PROJECT_ROOT}/scripts/check_boss_alignment_contract.py" \
  --data-dir "${DATA_DIR}"

for path in "${TRAIN_FILE}" "${VAL_FILE}"; do
  if [[ ! -f "${path}" ]]; then
    printf 'Formal PI data file not found: %s\n' "${path}" >&2
    exit 2
  fi
done

python3 "${PROJECT_ROOT}/scripts/check_formal_data_on_ray.py" \
  --train-file "${TRAIN_FILE}" \
  --val-file "${VAL_FILE}" \
  --ray-address "${RAY_ADDRESS:-192.168.202.5:26379}"

# Exact 4->4 sampling keeps every generated trajectory and introduces no
# fastest-K selection. The trainer validates only when policy step 100 is
# reached and saves only global_step_100; max_actor_ckpt_to_keep=1 is an
# additional retention guard. Eight completed groups are enough to begin, while
# staleness=2 allows up to 12 groups to remain in flight across two TP8 replicas.
RUN_NAME="${RUN_NAME}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
DATA_FILE="${TRAIN_FILE}" \
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS}" \
TOTAL_ROLLOUT_GROUPS="$((TOTAL_TRAINING_STEPS * GROUPS_PER_STEP))" \
GROUPS_PER_STEP="${GROUPS_PER_STEP}" \
SAVE_FREQ="${FINAL_SAVE_STEP}" \
FASTEST_K=4 \
OVERSAMPLE_CANDIDATES=4 \
PREWARM_GROUPS="${PREWARM_GROUPS}" \
STALENESS_THRESHOLD="${STALENESS_THRESHOLD}" \
MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS}" \
MAX_PROMPT_TOKENS=4096 \
MAX_RESPONSE_TOKENS=45056 \
MAX_ASSISTANT_TURNS=26 \
MAX_USER_TURNS=25 \
MAX_QUEUE_TOKENS="${MAX_QUEUE_TOKENS}" \
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
ROLLOUT_MAX_BATCHED_TOKENS="${ROLLOUT_MAX_BATCHED_TOKENS}" \
ROLLOUT_MAX_SEQS="${ROLLOUT_MAX_SEQS}" \
AGENT_WORKERS="${AGENT_WORKERS}" \
WEIGHT_BUCKET_MB="${WEIGHT_BUCKET_MB}" \
bash "${PROJECT_ROOT}/scripts/run_pi_grpo_fully_async_tp4_pp2_cp2.sh" \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="${PROJECT_ROOT}/configs/pi_workspace_tools.yaml" \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="${PROJECT_ROOT}/configs/pi_agent_loops.yaml" \
  actor_rollout_ref.actor.optim.lr="${LEARNING_RATE}" \
  actor_rollout_ref.actor.megatron.optimizer_offload=False \
  actor_rollout_ref.actor.megatron.use_dist_checkpointing=True \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=False \
  trainer.val_before_train=False \
  trainer.test_freq="${FINAL_EVAL_STEP}" \
  trainer.log_val_generations=20 \
  trainer.validation_data_dir="${OUTPUT_DIR}/validation" \
  trainer.save_freq="${FINAL_SAVE_STEP}" \
  trainer.max_actor_ckpt_to_keep=1 \
  async_training.use_trainer_do_validate=False \
  "$@"
