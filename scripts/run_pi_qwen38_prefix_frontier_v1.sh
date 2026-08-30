#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/llin-verl-grpo}"
MODEL_PATH="${MODEL_PATH:-/models/Qwen3.8-27B}"
DATA_FILE="${DATA_FILE:?DATA_FILE must be one frozen prefix-state runtime Parquet}"
RUN_NAME="${RUN_NAME:?RUN_NAME is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
VALIDATION_DIR="${VALIDATION_DIR:-${OUTPUT_DIR}/private/frontier_validation}"
SAMPLES_PER_STATE="${SAMPLES_PER_STATE:-4}"

EXPECTED_CONFIG_SHA256="191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab"
EXPECTED_MODEL_COMPOUND_SHA256="e2c3b44e4e198e94fcd74903983fc8997f8e504a21575e397f9d59db1cc2fc8f"

[[ -f "${DATA_FILE}" ]] || { printf 'frontier dataset missing\n' >&2; exit 2; }
[[ "$(sha256sum "${MODEL_PATH}/config.json" | awk '{print $1}')" == "${EXPECTED_CONFIG_SHA256}" ]]
mapfile -t shards < <(printf '%s\n' "${MODEL_PATH}"/model-*-of-00018.safetensors | LC_ALL=C sort)
[[ "${#shards[@]}" == 18 ]]
[[ "$(LC_ALL=C sha256sum "${shards[@]}" | sha256sum | awk '{print $1}')" == "${EXPECTED_MODEL_COMPOUND_SHA256}" ]]
[[ "${SAMPLES_PER_STATE}" == 4 || "${SAMPLES_PER_STATE}" == 8 ]]

mkdir -p "${VALIDATION_DIR}" "${OUTPUT_DIR}/audit"
chmod 700 "${VALIDATION_DIR}" "${OUTPUT_DIR}/audit"
cat > "${OUTPUT_DIR}/audit/frontier_rollout_contract.safe.txt" <<EOF
execution=rollout_only
actor_initialization=${MODEL_PATH}
optimizer_initialized=false
optimizer_steps=0
actor_parameter_updates=0
checkpoints=0
api_requests=0
samples_per_prefix_state=${SAMPLES_PER_STATE}
reward=tiered-query-cost-trajectory-shadow-v1
response_mask=generated_suffix_assistant_tokens_only
EOF

export LLIN_PREFIX_FRONTIER_ROLLOUT_ONLY=1
export PI_AGENT_TOKENIZER_PATH="${MODEL_PATH}"

MODEL_PATH="${MODEL_PATH}" \
DATA_FILE="${DATA_FILE}" \
RUN_NAME="${RUN_NAME}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
RAY_ADDRESS="${RAY_ADDRESS:-192.168.202.5:36379}" \
TRAIN_TP=4 TRAIN_PP=2 TRAIN_CP=2 TRAIN_NPUS=16 \
ROLLOUT_TP=4 ROLLOUT_NPUS=16 ROLLOUT_NNODES=1 \
TOTAL_TRAINING_STEPS=1 TOTAL_ROLLOUT_GROUPS=1 GROUPS_PER_STEP=1 \
RESPONSES_PER_GROUP="${SAMPLES_PER_STATE}" FASTEST_K="${SAMPLES_PER_STATE}" \
OVERSAMPLE_CANDIDATES="${SAMPLES_PER_STATE}" PREWARM_GROUPS=0 STALENESS_THRESHOLD=0 \
MAX_PROMPT_TOKENS=32768 MAX_RESPONSE_TOKENS=90112 MAX_CONTEXT_TOKENS=122880 \
MAX_QUEUE_TOKENS="$((SAMPLES_PER_STATE * 122880))" \
MAX_ASSISTANT_TURNS=26 MAX_USER_TURNS=25 AGENT_TIMEOUT_SECONDS=1800 \
MAX_PARALLEL_TOOL_CALLS=4 MAX_TOOL_RESPONSE_CHARS=32768 \
ROLLOUT_GPU_MEMORY_UTILIZATION=0.80 ROLLOUT_MAX_BATCHED_TOKENS=16384 ROLLOUT_MAX_SEQS=16 \
AGENT_WORKERS=12 CONCURRENT_SAMPLES_PER_REPLICA=6 \
OPTIMIZER_CPU_OFFLOAD=false ENGINE_OPTIMIZER_OFFLOAD=false SAVE_FREQ=-1 \
bash "${PROJECT_ROOT}/scripts/run_pi_grpo_fully_async_tp4_pp2_cp2.sh" \
  data.train_files="${DATA_FILE}" data.val_files="${DATA_FILE}" \
  data.shuffle=false data.seed=20260830 \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="${PROJECT_ROOT}/configs/pi_workspace_tools_relaxed1800.yaml" \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="${PROJECT_ROOT}/configs/pi_agent_loops.yaml" \
  actor_rollout_ref.rollout.agent.default_agent_loop=pi_agent \
  actor_rollout_ref.rollout.temperature=1.0 actor_rollout_ref.rollout.top_p=0.95 actor_rollout_ref.rollout.top_k=20 \
  actor_rollout_ref.rollout.val_kwargs.n="${SAMPLES_PER_STATE}" \
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.actor.megatron.forward_only=True \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0 \
  algorithm.use_kl_in_reward=False \
  reward.custom_reward_function.path="${PROJECT_ROOT}/llin_verl/pi_reward.py" \
  reward.custom_reward_function.name=compute_score_tiered_query_cost_v1 \
  trainer.val_before_train=True trainer.val_only=True \
  trainer.validation_data_dir="${VALIDATION_DIR}" \
  trainer.rollout_data_dir=null trainer.default_local_dir="${OUTPUT_DIR}/private/no_checkpoint" \
  trainer.save_freq=-1 trainer.test_freq=-1 trainer.resume_mode=disable \
  trainer.project_name=llin-qwen38-prefix-frontier \
  'actor_rollout_ref.actor.checkpoint.save_contents=[extra]'

if find "${OUTPUT_DIR}" -path '*/global_step_*' -print -quit | grep -q .; then
  printf 'rollout-only frontier unexpectedly created a checkpoint\n' >&2
  exit 3
fi
printf 'optimizer_steps=0\nactor_parameter_updates=0\ncheckpoints=0\n' \
  > "${OUTPUT_DIR}/audit/frontier_postconditions.safe.txt"
