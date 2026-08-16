#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/llin-verl-grpo}"
VERL_ROOT="${VERL_ROOT:-/verl}"
VLLM_ROOT="${VLLM_ROOT:-/vllm}"
MODEL_PATH="${MODEL_PATH:-/models/Qwen3.6-27B}"
DATA_FILE="${DATA_FILE:-${PROJECT_ROOT}/data/pi_verified_smoke.parquet}"
RUN_NAME="${RUN_NAME:-pi-grpo-fully-async-tp4-pp2-cp2-tp8-dp2-20260730}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/runs/${RUN_NAME}}"
MEGATRON_BRIDGE_ROOT="${MEGATRON_BRIDGE_ROOT:-${PROJECT_ROOT}/reference/Megatron-Bridge-de93536e/src}"

TRAIN_TP="${TRAIN_TP:-4}"
TRAIN_PP="${TRAIN_PP:-2}"
TRAIN_CP="${TRAIN_CP:-2}"
TRAIN_NPUS="${TRAIN_NPUS:-16}"
ROLLOUT_TP="${ROLLOUT_TP:-8}"
ROLLOUT_NPUS="${ROLLOUT_NPUS:-16}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-20}"
GROUPS_PER_STEP="${GROUPS_PER_STEP:-4}"
RESPONSES_PER_GROUP="${RESPONSES_PER_GROUP:-4}"
TOTAL_ROLLOUT_GROUPS="${TOTAL_ROLLOUT_GROUPS:-$((TOTAL_TRAINING_STEPS * GROUPS_PER_STEP))}"
SAVE_FREQ="${SAVE_FREQ:-20}"
WEIGHT_BUCKET_MB="${WEIGHT_BUCKET_MB:-3072}"
MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-49152}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-4096}"
MAX_RESPONSE_TOKENS="${MAX_RESPONSE_TOKENS:-45056}"
MAX_ASSISTANT_TURNS="${MAX_ASSISTANT_TURNS:-25}"
MAX_USER_TURNS="${MAX_USER_TURNS:-24}"
MAX_PARALLEL_TOOL_CALLS="${MAX_PARALLEL_TOOL_CALLS:-4}"
MAX_TOOL_RESPONSE_CHARS="${MAX_TOOL_RESPONSE_CHARS:-32768}"
ROLLOUT_MAX_BATCHED_TOKENS="${ROLLOUT_MAX_BATCHED_TOKENS:-8192}"
ROLLOUT_MAX_SEQS="${ROLLOUT_MAX_SEQS:-16}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.60}"
AGENT_WORKERS="${AGENT_WORKERS:-8}"
CONCURRENT_SAMPLES_PER_REPLICA="${CONCURRENT_SAMPLES_PER_REPLICA:-16}"
FASTEST_K="${FASTEST_K:-4}"
OVERSAMPLE_CANDIDATES="${OVERSAMPLE_CANDIDATES:-6}"
PREWARM_GROUPS="${PREWARM_GROUPS:-0}"
STALENESS_THRESHOLD="${STALENESS_THRESHOLD:-0.5}"
PI_DENSE_CORRECTNESS_WEIGHT="${PI_DENSE_CORRECTNESS_WEIGHT:-0}"
# One complete training batch is GROUPS_PER_STEP groups × rollout.n responses.
# Keeping this many worst-case tokens prevents an oversized-group
# producer from blocking before the trainer can collect its first full batch.
MAX_QUEUE_TOKENS="${MAX_QUEUE_TOKENS:-$((GROUPS_PER_STEP * RESPONSES_PER_GROUP * MAX_CONTEXT_TOKENS))}"

if (( TRAIN_TP * TRAIN_PP * TRAIN_CP != TRAIN_NPUS )); then
  printf 'Invalid training topology: TP(%s) * PP(%s) * CP(%s) != NPUs(%s)\n' \
    "${TRAIN_TP}" "${TRAIN_PP}" "${TRAIN_CP}" "${TRAIN_NPUS}" >&2
  exit 2
fi
if (( ROLLOUT_NPUS % ROLLOUT_TP != 0 )); then
  printf 'Invalid rollout topology: NPUs(%s) is not divisible by TP(%s)\n' \
    "${ROLLOUT_NPUS}" "${ROLLOUT_TP}" >&2
  exit 2
fi
if (( MAX_PROMPT_TOKENS + MAX_RESPONSE_TOKENS != MAX_CONTEXT_TOKENS )); then
  printf 'Invalid context budget: prompt(%s) + response(%s) != context(%s)\n' \
    "${MAX_PROMPT_TOKENS}" "${MAX_RESPONSE_TOKENS}" "${MAX_CONTEXT_TOKENS}" >&2
  exit 2
fi
if (( RESPONSES_PER_GROUP <= 1 || GROUPS_PER_STEP <= 0 )); then
  printf 'Invalid GRPO shape: groups_per_step=%s responses_per_group=%s\n' \
    "${GROUPS_PER_STEP}" "${RESPONSES_PER_GROUP}" >&2
  exit 2
fi
if (( FASTEST_K != RESPONSES_PER_GROUP )); then
  printf 'Invalid fastest-K group size: fastest_k(%s) must equal rollout.n(%s)\n' \
    "${FASTEST_K}" "${RESPONSES_PER_GROUP}" >&2
  exit 2
fi
if (( OVERSAMPLE_CANDIDATES < FASTEST_K )); then
  printf 'Invalid oversampling: candidates(%s) must be >= fastest_k(%s)\n' \
    "${OVERSAMPLE_CANDIDATES}" "${FASTEST_K}" >&2
  exit 2
fi
if (( PREWARM_GROUPS < 0 )); then
  printf 'Invalid prewarm group count: %s\n' "${PREWARM_GROUPS}" >&2
  exit 2
fi
if [[ ! -d "${MEGATRON_BRIDGE_ROOT}/megatron/bridge" ]]; then
  printf 'Megatron-Bridge source not found: %s\n' "${MEGATRON_BRIDGE_ROOT}" >&2
  exit 2
fi

export PYTHONPATH="${VLLM_ROOT}:${MEGATRON_BRIDGE_ROOT}:${PROJECT_ROOT}/runtime:${PROJECT_ROOT}:${PYTHONPATH:-}"
export RAY_ADDRESS="${RAY_ADDRESS:-192.168.202.5:26379}"
export LLIN_PIN_RAY_ROLES=1
export LLIN_TRAINER_RESOURCE=llin_trainer
export LLIN_ROLLOUT_RESOURCE=llin_rollout
export HCCL_EXEC_TIMEOUT=60000
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_ALGO="${HCCL_ALGO:-broadcast=level0:NA;level1:NHR}"
export TOKENIZERS_PARALLELISM=true
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PI_DENSE_CORRECTNESS_WEIGHT

python3 "${PROJECT_ROOT}/scripts/patch_verl_megatron_bridge_compat.py" \
  --target "${VERL_ROOT}/verl/models/mcore/bridge.py"
python3 "${PROJECT_ROOT}/scripts/patch_verl_megatron_dist_checkpoint_init.py" \
  --target "${VERL_ROOT}/verl/workers/engine/megatron/transformer_impl.py"
python3 "${PROJECT_ROOT}/scripts/patch_verl_vllm_dp_weight_sync.py" \
  --target "${VERL_ROOT}/verl/workers/rollout/vllm_rollout/utils.py"
python3 "${PROJECT_ROOT}/scripts/patch_verl_fully_async_continuous_token.py" \
  --target "${VERL_ROOT}/verl/experimental/fully_async_policy/fully_async_main.py"
python3 "${PROJECT_ROOT}/scripts/patch_verl_agent_loop_continuous_token.py" \
  --target "${VERL_ROOT}/verl/experimental/agent_loop/agent_loop.py"
python3 "${PROJECT_ROOT}/scripts/patch_verl_force_final_config.py" \
  --target "${VERL_ROOT}/verl/workers/config/rollout.py"
python3 "${PROJECT_ROOT}/scripts/patch_verl_fully_async_group_token_queue.py" \
  --message-queue "${VERL_ROOT}/verl/experimental/fully_async_policy/message_queue.py" \
  --rollouter "${VERL_ROOT}/verl/experimental/fully_async_policy/fully_async_rollouter.py"
python3 "${PROJECT_ROOT}/scripts/patch_verl_fastest_k_oversampling.py" \
  --rollouter "${VERL_ROOT}/verl/experimental/fully_async_policy/fully_async_rollouter.py" \
  --agent-loop "${VERL_ROOT}/verl/experimental/agent_loop/agent_loop.py" \
  --tool-agent-loop "${VERL_ROOT}/verl/experimental/agent_loop/tool_agent_loop.py" \
  --llm-server "${VERL_ROOT}/verl/workers/rollout/llm_server.py"
python3 "${PROJECT_ROOT}/scripts/patch_verl_fastest_k_abort_observability.py" \
  --agent-loop "${VERL_ROOT}/verl/experimental/agent_loop/agent_loop.py" \
  --llm-server "${VERL_ROOT}/verl/workers/rollout/llm_server.py"
python3 "${PROJECT_ROOT}/scripts/patch_verl_fastest_k_abort_retry.py" \
  --agent-loop "${VERL_ROOT}/verl/experimental/agent_loop/agent_loop.py" \
  --llm-server "${VERL_ROOT}/verl/workers/rollout/llm_server.py"
python3 "${PROJECT_ROOT}/scripts/patch_verl_vllm_abort_api.py" \
  --target "${VERL_ROOT}/verl/workers/rollout/vllm_rollout/vllm_async_server.py"
python3 "${PROJECT_ROOT}/scripts/patch_verl_abort_partial_tokens.py" \
  --llm-server "${VERL_ROOT}/verl/workers/rollout/llm_server.py"
python3 "${PROJECT_ROOT}/scripts/patch_verl_fully_async_observability.py" \
  --trainer "${VERL_ROOT}/verl/experimental/fully_async_policy/fully_async_trainer.py" \
  --main "${VERL_ROOT}/verl/experimental/fully_async_policy/fully_async_main.py"

cd "${VERL_ROOT}"

python3 -m verl.experimental.fully_async_policy.fully_async_main \
  --config-path=config \
  --config-name=fully_async_ppo_megatron_trainer.yaml \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  algorithm.rollout_correction.bypass_mode=True \
  data.train_files="${DATA_FILE}" \
  data.val_files="${DATA_FILE}" \
  data.train_batch_size=0 \
  data.gen_batch_size=1 \
  data.max_prompt_length="${MAX_PROMPT_TOKENS}" \
  data.max_response_length="${MAX_RESPONSE_TOKENS}" \
  data.filter_overlong_prompts=True \
  data.filter_overlong_prompts_workers=4 \
  data.dataloader_num_workers=4 \
  data.return_raw_chat=True \
  data.return_multi_modal_inputs=False \
  data.truncation=error \
  data.continuous_token.enable=True \
  data.continuous_token.model_family=qwen35 \
  actor_rollout_ref.actor.strategy=megatron \
  critic.strategy=megatron \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.lora_rank=0 \
  actor_rollout_ref.hybrid_engine=False \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.lr_decay_style=constant \
  actor_rollout_ref.actor.optim.lr_decay_steps="${TOTAL_TRAINING_STEPS}" \
  +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
  +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1 \
  +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
  actor_rollout_ref.actor.ppo_mini_batch_size="${GROUPS_PER_STEP}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=False \
  actor_rollout_ref.actor.use_rollout_log_probs=True \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.megatron.use_mbridge=True \
  actor_rollout_ref.actor.megatron.vanilla_mbridge=False \
  actor_rollout_ref.actor.megatron.use_remove_padding=False \
  actor_rollout_ref.actor.megatron.tensor_model_parallel_size="${TRAIN_TP}" \
  actor_rollout_ref.actor.megatron.pipeline_model_parallel_size="${TRAIN_PP}" \
  actor_rollout_ref.actor.megatron.context_parallel_size="${TRAIN_CP}" \
  actor_rollout_ref.actor.megatron.param_offload=False \
  actor_rollout_ref.actor.megatron.optimizer_offload=True \
  actor_rollout_ref.actor.megatron.grad_offload=True \
  actor_rollout_ref.actor.megatron.dtype=bfloat16 \
  actor_rollout_ref.actor.megatron.use_distributed_optimizer=True \
  ++actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=auto \
  +actor_rollout_ref.actor.megatron.override_transformer_config.context_parallel_algo=kvallgather_cp_algo \
  +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
  +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
  +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
  +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True \
  +actor_rollout_ref.actor.megatron.override_transformer_config.sequence_parallel=True \
  actor_rollout_ref.actor.checkpoint.strict=False \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
  actor_rollout_ref.ref.megatron.use_mbridge=True \
  actor_rollout_ref.ref.megatron.vanilla_mbridge=False \
  actor_rollout_ref.ref.megatron.use_remove_padding=False \
  actor_rollout_ref.ref.megatron.tensor_model_parallel_size="${TRAIN_TP}" \
  actor_rollout_ref.ref.megatron.pipeline_model_parallel_size="${TRAIN_PP}" \
  actor_rollout_ref.ref.megatron.context_parallel_size="${TRAIN_CP}" \
  actor_rollout_ref.ref.megatron.param_offload=True \
  actor_rollout_ref.ref.megatron.dtype=bfloat16 \
  ++actor_rollout_ref.ref.megatron.override_transformer_config.attention_backend=auto \
  +actor_rollout_ref.ref.megatron.override_transformer_config.context_parallel_algo=kvallgather_cp_algo \
  +actor_rollout_ref.ref.megatron.override_transformer_config.sequence_parallel=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
  actor_rollout_ref.rollout.data_parallel_size="$((ROLLOUT_NPUS / ROLLOUT_TP))" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_BATCHED_TOKENS}" \
  actor_rollout_ref.rollout.max_model_len="${MAX_CONTEXT_TOKENS}" \
  actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_SEQS}" \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.n="${RESPONSES_PER_GROUP}" \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.enable_prefix_caching=True \
  actor_rollout_ref.rollout.disable_log_stats=False \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="${PROJECT_ROOT}/configs/pi_sqlite_tool.yaml" \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns="${MAX_ASSISTANT_TURNS}" \
  actor_rollout_ref.rollout.multi_turn.max_user_turns="${MAX_USER_TURNS}" \
  actor_rollout_ref.rollout.multi_turn.max_parallel_calls="${MAX_PARALLEL_TOOL_CALLS}" \
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length="${MAX_TOOL_RESPONSE_CHARS}" \
  actor_rollout_ref.rollout.multi_turn.format=qwen3_coder \
  actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=disable \
  actor_rollout_ref.rollout.agent.num_workers="${AGENT_WORKERS}" \
  actor_rollout_ref.rollout.checkpoint_engine.backend=nccl \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes="${WEIGHT_BUCKET_MB}" \
  reward.custom_reward_function.path="${PROJECT_ROOT}/llin_verl/pi_reward.py" \
  reward.custom_reward_function.name=compute_score \
  reward.reward_manager.name=naive \
  trainer.critic_warmup=0 \
  trainer.val_before_train=False \
  trainer.logger='["console"]' \
  trainer.project_name=llin-qwen36-verl-grpo \
  trainer.experiment_name="${RUN_NAME}" \
  trainer.default_local_dir="${OUTPUT_DIR}/checkpoints" \
  trainer.rollout_data_dir="${OUTPUT_DIR}/rollouts" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.max_actor_ckpt_to_keep=1 \
  trainer.test_freq=-1 \
  trainer.total_epochs="${TOTAL_TRAINING_STEPS}" \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
  trainer.resume_mode=disable \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node="${TRAIN_NPUS}" \
  rollout.nnodes=1 \
  rollout.n_gpus_per_node="${ROLLOUT_NPUS}" \
  rollout.n="${RESPONSES_PER_GROUP}" \
  rollout.total_rollout_steps="${TOTAL_ROLLOUT_GROUPS}" \
  async_training.staleness_threshold="${STALENESS_THRESHOLD}" \
  async_training.trigger_parameter_sync_step=1 \
  async_training.require_batches=1 \
  async_training.partial_rollout=True \
  +async_training.max_queue_tokens="${MAX_QUEUE_TOKENS}" \
  +async_training.fastest_k="${FASTEST_K}" \
  +async_training.oversample_candidates="${OVERSAMPLE_CANDIDATES}" \
  +async_training.prewarm_groups="${PREWARM_GROUPS}" \
  async_training.concurrent_samples_per_replica="${CONCURRENT_SAMPLES_PER_REPLICA}" \
  'actor_rollout_ref.actor.checkpoint.save_contents=[model,extra]' \
  "$@"
