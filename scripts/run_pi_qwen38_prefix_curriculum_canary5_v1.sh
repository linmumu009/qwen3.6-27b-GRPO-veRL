#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/llin-verl-grpo}"
VERL_ROOT="${VERL_ROOT:-/verl}"
MODEL_PATH="${MODEL_PATH:-/models/Qwen3.8-27B}"
TRAIN_FILE="${TRAIN_FILE:?TRAIN_FILE must be the gated 20-group prefix schedule}"
SEALED_FILE="${SEALED_FILE:?SEALED_FILE must be the frozen heldout prefix endpoints}"
RUN_NAME="${RUN_NAME:?RUN_NAME is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"

EXPECTED_CONFIG_SHA256="191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab"
EXPECTED_MODEL_COMPOUND_SHA256="e2c3b44e4e198e94fcd74903983fc8997f8e504a21575e397f9d59db1cc2fc8f"
TARGET_ACTUAL_OPTIMIZER_STEPS=5
MAX_NOMINAL_GROUPS=20

[[ -f "${TRAIN_FILE}" && -f "${SEALED_FILE}" ]]
[[ "$(sha256sum "${MODEL_PATH}/config.json" | awk '{print $1}')" == "${EXPECTED_CONFIG_SHA256}" ]]
mapfile -t shards < <(printf '%s\n' "${MODEL_PATH}"/model-*-of-00018.safetensors | LC_ALL=C sort)
[[ "${#shards[@]}" == 18 ]]
[[ "$(LC_ALL=C sha256sum "${shards[@]}" | sha256sum | awk '{print $1}')" == "${EXPECTED_MODEL_COMPOUND_SHA256}" ]]

mkdir -p "${OUTPUT_DIR}/audit" "${OUTPUT_DIR}/private/rollouts" "${OUTPUT_DIR}/private_recovery/checkpoints"
chmod 700 "${OUTPUT_DIR}/audit" "${OUTPUT_DIR}/private" "${OUTPUT_DIR}/private/rollouts" "${OUTPUT_DIR}/private_recovery" "${OUTPUT_DIR}/private_recovery/checkpoints"

python3 "${PROJECT_ROOT}/scripts/patch_verl_fastest_k_oversampling.py" \
  --rollouter "${VERL_ROOT}/verl/experimental/fully_async_policy/fully_async_rollouter.py" \
  --agent-loop "${VERL_ROOT}/verl/experimental/agent_loop/agent_loop.py" \
  --tool-agent-loop "${VERL_ROOT}/verl/experimental/agent_loop/tool_agent_loop.py" \
  --llm-server "${VERL_ROOT}/verl/workers/rollout/llm_server.py"
python3 "${PROJECT_ROOT}/scripts/patch_verl_hard_gate_resampling.py" \
  --agent-loop "${VERL_ROOT}/verl/experimental/agent_loop/agent_loop.py"
python3 "${PROJECT_ROOT}/scripts/patch_verl_grpo_strict_variance_gate.py" \
  --trainer "${VERL_ROOT}/verl/experimental/separation/ray_trainer.py" \
  --fully-async-trainer "${VERL_ROOT}/verl/experimental/fully_async_policy/fully_async_trainer.py"
python3 "${PROJECT_ROOT}/scripts/patch_verl_canary_rollout_audit.py" \
  --trainer "${VERL_ROOT}/verl/experimental/separation/ray_trainer.py"

cat > "${OUTPUT_DIR}/audit/canary_contract.safe.txt" <<EOF
contract=prefix-state-curriculum-grpo-canary5-v1
actor_initialization=${MODEL_PATH}
reference_initialization=${MODEL_PATH}
qwen36_or_historical_checkpoint_reused=false
reward=tiered-query-cost-trajectory-shadow-v1
reward_scope=generated_suffix_only
final_correctness_scope=combined_prefix_plus_suffix
algorithm_use_kl_in_reward=false
actor_use_kl_loss=true
actor_kl_loss_coef=0.001
actor_kl_loss_type=low_var_kl
learning_rate=5e-8_constant
entropy=0
hard_staleness=0
group_size=8
groups_per_nominal_batch=1
target_actual_optimizer_steps=${TARGET_ACTUAL_OPTIMIZER_STEPS}
maximum_nominal_groups=${MAX_NOMINAL_GROUPS}
checkpoint_policy=only_global_step_5_on_success
full_training_allowed=false
EOF

export LLIN_CANARY_TARGET_OPTIMIZER_STEPS="${TARGET_ACTUAL_OPTIMIZER_STEPS}"
export PI_AGENT_TOKENIZER_PATH="${MODEL_PATH}"

MODEL_PATH="${MODEL_PATH}" DATA_FILE="${TRAIN_FILE}" RUN_NAME="${RUN_NAME}" OUTPUT_DIR="${OUTPUT_DIR}" \
RAY_ADDRESS="${RAY_ADDRESS:-192.168.202.5:36379}" \
TRAIN_TP=4 TRAIN_PP=2 TRAIN_CP=2 TRAIN_NPUS=16 \
ROLLOUT_TP=4 ROLLOUT_NPUS=16 ROLLOUT_NNODES=1 \
TOTAL_TRAINING_STEPS="${MAX_NOMINAL_GROUPS}" TOTAL_ROLLOUT_GROUPS="${MAX_NOMINAL_GROUPS}" \
GROUPS_PER_STEP=1 RESPONSES_PER_GROUP=8 FASTEST_K=8 OVERSAMPLE_CANDIDATES=16 \
PREWARM_GROUPS=0 STALENESS_THRESHOLD=0 \
MAX_PROMPT_TOKENS=32768 MAX_RESPONSE_TOKENS=90112 MAX_CONTEXT_TOKENS=122880 \
MAX_QUEUE_TOKENS="$((8 * 122880))" \
MAX_ASSISTANT_TURNS=26 MAX_USER_TURNS=25 AGENT_TIMEOUT_SECONDS=1800 \
MAX_PARALLEL_TOOL_CALLS=4 MAX_TOOL_RESPONSE_CHARS=32768 \
ROLLOUT_GPU_MEMORY_UTILIZATION=0.80 ROLLOUT_MAX_BATCHED_TOKENS=16384 ROLLOUT_MAX_SEQS=16 \
AGENT_WORKERS=12 CONCURRENT_SAMPLES_PER_REPLICA=6 WEIGHT_BUCKET_MB=2560 \
OPTIMIZER_CPU_OFFLOAD=false ENGINE_OPTIMIZER_OFFLOAD=false SAVE_FREQ=5 \
bash "${PROJECT_ROOT}/scripts/run_pi_grpo_fully_async_tp4_pp2_cp2.sh" \
  data.train_files="${TRAIN_FILE}" data.val_files="${SEALED_FILE}" \
  data.shuffle=false data.seed=20260830 \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="${PROJECT_ROOT}/configs/pi_workspace_tools_relaxed1800.yaml" \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="${PROJECT_ROOT}/configs/pi_agent_loops.yaml" \
  actor_rollout_ref.rollout.agent.default_agent_loop=pi_agent \
  actor_rollout_ref.rollout.temperature=1.0 actor_rollout_ref.rollout.top_p=0.95 actor_rollout_ref.rollout.top_k=20 \
  actor_rollout_ref.actor.optim.lr=5e-8 actor_rollout_ref.actor.optim.lr_decay_style=constant \
  actor_rollout_ref.actor.entropy_coeff=0 actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  +actor_rollout_ref.ref.megatron.override_transformer_config.use_flash_attn=True \
  algorithm.use_kl_in_reward=False algorithm.rollout_correction.bypass_mode=True \
  reward.custom_reward_function.path="${PROJECT_ROOT}/llin_verl/pi_reward.py" \
  reward.custom_reward_function.name=compute_score_tiered_query_cost_v1 \
  trainer.val_before_train=true trainer.test_freq=5 \
  actor_rollout_ref.rollout.val_kwargs.n=4 \
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0 actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  trainer.validation_data_dir="${OUTPUT_DIR}/private/sealed_validation" \
  trainer.rollout_data_dir="${OUTPUT_DIR}/private/rollouts" \
  trainer.default_local_dir="${OUTPUT_DIR}/private_recovery/checkpoints" \
  trainer.save_freq=5 trainer.max_actor_ckpt_to_keep=1 trainer.resume_mode=disable \
  'actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra]'

mapfile -t checkpoints < <(find "${OUTPUT_DIR}/private_recovery/checkpoints" -maxdepth 1 -type d -name 'global_step_*' -printf '%f\n' | LC_ALL=C sort)
[[ "${#checkpoints[@]}" == 1 && "${checkpoints[0]}" == "global_step_5" ]] || {
  printf 'canary did not leave exactly global_step_5\n' >&2
  exit 4
}
