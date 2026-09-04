#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/llin-verl-grpo}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/runs/llin-step120-opensource-20260825-02/hf_export_step120_opensource}"
MEGATRON_BRIDGE_ROOT="${MEGATRON_BRIDGE_ROOT:-${PROJECT_ROOT}/reference/Megatron-Bridge-de93536e/src}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-${PROJECT_ROOT}/runs/llin-step120-opensource-20260825-02/checkpoints/global_step_120}"
SOURCE_MODEL_DIST_CKPT="${SOURCE_MODEL_DIST_CKPT:-${SOURCE_CHECKPOINT}/actor/model/dist_ckpt}"
TRAIN_FILE="${TRAIN_FILE:-${PROJECT_ROOT}/runs/logistics-cpt-20260903/private/handbook8e_cpt_4096.parquet}"
RUN_NAME="${RUN_NAME:-logistics-cpt-book-exposure-curve-2x4x-20260904-01}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/runs/${RUN_NAME}}"
NPROC="${NPROC:-16}"
TP="${TP:-4}"
PP="${PP:-2}"
CP="${CP:-2}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
STEPS_PER_EXPOSURE="${STEPS_PER_EXPOSURE:-29}"
TOTAL_EXPOSURES="${TOTAL_EXPOSURES:-4}"
TOTAL_STEPS="${TOTAL_STEPS:-116}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-58}"
LEARNING_RATE="${LEARNING_RATE:-5e-7}"
MIN_LEARNING_RATE="${MIN_LEARNING_RATE:-1e-7}"
# 0.025 * 116 is truncated by Megatron to two warmup steps, matching the
# effective two-step warmup observed in the original 29-step run.
WARMUP_RATIO="${WARMUP_RATIO:-0.025}"
TOKENIZER_SHA256="${TOKENIZER_SHA256:-06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523}"

for path in "${MODEL_PATH}/config.json" "${MODEL_PATH}/tokenizer.json" "${SOURCE_MODEL_DIST_CKPT}/.metadata" "${TRAIN_FILE}"; do
  if [[ ! -e "${path}" ]]; then
    printf 'required logistics CPT exposure-curve input missing: %s\n' "${path}" >&2
    exit 2
  fi
done
if [[ ! -d "${MEGATRON_BRIDGE_ROOT}/megatron/bridge" ]]; then
  printf 'pinned Megatron-Bridge source missing: %s\n' "${MEGATRON_BRIDGE_ROOT}" >&2
  exit 2
fi
if [[ "${TRAIN_BATCH_SIZE}" -ne 4 || "${STEPS_PER_EXPOSURE}" -ne 29 || \
      "${TOTAL_EXPOSURES}" -ne 4 || "${TOTAL_STEPS}" -ne 116 || "${CHECKPOINT_STEPS}" -ne 58 ]]; then
  printf '2x/4x exposure contract requires batch=4, steps_per_exposure=29, exposures=4, total_steps=116, checkpoint_steps=58\n' >&2
  exit 2
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
  printf 'refusing to overwrite existing exposure-curve output: %s\n' "${OUTPUT_DIR}" >&2
  exit 2
fi

exec 9>"${PROJECT_ROOT}/runs/.logistics-cpt-book-exposure-curve.lock"
if ! flock -n 9; then
  printf 'another logistics CPT exposure-curve run holds the lock\n' >&2
  exit 3
fi

export CUDA_DEVICE_MAX_CONNECTIONS=1
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=true
export TRANSFORMERS_VERBOSITY=error
export PYTHONPATH="${PROJECT_ROOT}:${MEGATRON_BRIDGE_ROOT}:${PROJECT_ROOT}/runtime:/verl:${PYTHONPATH:-}"
umask 077

mkdir -p "${OUTPUT_DIR}/torchrun_logs"
python3 "${PROJECT_ROOT}/scripts/check_logistics_cpt_dataset.py" \
  --train-file "${TRAIN_FILE}" \
  --model-path "${MODEL_PATH}" \
  --max-length "${MAX_LENGTH}" \
  --expected-records 116 \
  --expected-content-tokens 336586 \
  --expected-tokenizer-sha256 "${TOKENIZER_SHA256}" \
  --output "${OUTPUT_DIR}/tokenization_gate.safe.json"

cat > "${OUTPUT_DIR}/experiment_contract.txt" <<EOF
purpose=single_book_cpt_exposure_curve_2x_4x
promotion_allowed=false
promotion_gate=fixed_nll_memorization_independent_knowledge_public_and_general_agent_regression
source_checkpoint=${SOURCE_CHECKPOINT}
source_model_dist_ckpt=${SOURCE_MODEL_DIST_CKPT}
checkpoint_resume=false
checkpoint_initialization=model_only_dist_ckpt
optimizer_state=fresh_continuous_through_step_116
optimizer_reset_at_step_58=false
optimizer_placement=device_side_matching_step120
dataloader_state=fresh
objective=raw_text_all_real_next_token_targets_including_eos
chat_template_applied=false
rights_basis=user_attested_written_permission
source_sha256=b2b4ed0156dea4076f86894478c8db2643ee05f25fb8af6bdf4622ee70b93bd1
private_parquet_sha256=975dc8719da2ff6ba0f2eb5db4faa03cb72f2f1ee36c14ed5e35460d4ca4c1c2
records_per_exposure=116
sequence_tokens_with_eos_per_exposure=336702
steps_per_exposure=${STEPS_PER_EXPOSURE}
planned_exposures=${TOTAL_EXPOSURES}
expected_total_sequence_tokens_with_eos=1346808
train_batch_size=${TRAIN_BATCH_SIZE}
total_training_steps=${TOTAL_STEPS}
evaluation_checkpoints=58,116
evaluation_exposures=2,4
learning_rate=${LEARNING_RATE}
minimum_learning_rate=${MIN_LEARNING_RATE}
warmup_ratio=${WARMUP_RATIO}
expected_effective_warmup_steps=2
lr_decay_style=cosine_across_all_116_steps
sampler=distributed_shuffle_deterministic_per_epoch
mtp_enable=false
mtp_enable_train=false
mtp_policy=keep_step120_frozen_mtp_tensors_outside_training_checkpoint
save_contents=model_and_extra_only
optimizer_checkpoint_saved=false
topology=tp${TP}_pp${PP}_cp${CP}
EOF

torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC}" \
  --log-dir="${OUTPUT_DIR}/torchrun_logs" --redirects=3 --tee=0 \
  -m verl.trainer.sft_trainer \
  "data.train_files=${TRAIN_FILE}" \
  data.val_files=null \
  +data.text_key=text \
  data.train_max_samples=-1 \
  "data.train_batch_size=${TRAIN_BATCH_SIZE}" \
  data.micro_batch_size_per_gpu=1 \
  data.use_dynamic_bsz=false \
  "data.max_token_len_per_gpu=${MAX_LENGTH}" \
  "data.max_length=${MAX_LENGTH}" \
  data.pad_mode=no_padding \
  data.truncation=error \
  data.num_workers=0 \
  "data.custom_cls.path=${PROJECT_ROOT}/scripts/qwen36_causal_lm_dataset.py" \
  data.custom_cls.name=Qwen36CausalLMDataset \
  model=hf_model \
  "model.path=${MODEL_PATH}" \
  model.trust_remote_code=true \
  model.use_remove_padding=false \
  model.lora_rank=0 \
  model.mtp.enable=false \
  model.mtp.enable_train=false \
  model.mtp.enable_rollout=false \
  optim=megatron \
  "optim.lr=${LEARNING_RATE}" \
  "optim.min_lr=${MIN_LEARNING_RATE}" \
  "optim.lr_warmup_steps_ratio=${WARMUP_RATIO}" \
  optim.weight_decay=0 \
  'optim.betas=[0.9,0.95]' \
  optim.clip_grad=1.0 \
  optim.lr_decay_style=cosine \
  +optim.override_optimizer_config.optimizer_cpu_offload=false \
  engine=megatron \
  "engine.tensor_model_parallel_size=${TP}" \
  "engine.pipeline_model_parallel_size=${PP}" \
  "engine.context_parallel_size=${CP}" \
  engine.use_mbridge=true \
  engine.vanilla_mbridge=false \
  engine.use_megatron_fsdp=false \
  engine.use_remove_padding=false \
  engine.param_offload=false \
  engine.optimizer_offload=false \
  engine.grad_offload=true \
  engine.dtype=bfloat16 \
  engine.use_distributed_optimizer=true \
  engine.use_dist_checkpointing=true \
  "engine.dist_checkpointing_path=${SOURCE_MODEL_DIST_CKPT}" \
  ++engine.override_transformer_config.attention_backend=auto \
  ++engine.override_transformer_config.context_parallel_algo=kvallgather_cp_algo \
  ++engine.override_transformer_config.recompute_method=uniform \
  ++engine.override_transformer_config.recompute_granularity=full \
  ++engine.override_transformer_config.recompute_num_layers=1 \
  ++engine.override_transformer_config.use_flash_attn=true \
  ++engine.override_transformer_config.sequence_parallel=true \
  'checkpoint.load_contents=[]' \
  'checkpoint.save_contents=[model,extra]' \
  "trainer.default_local_dir=${OUTPUT_DIR}/checkpoints" \
  trainer.project_name=llin-logistics-cpt \
  "trainer.experiment_name=${RUN_NAME}" \
  'trainer.logger=["console"]' \
  "trainer.total_epochs=${TOTAL_EXPOSURES}" \
  "trainer.total_training_steps=${TOTAL_STEPS}" \
  "trainer.save_freq=${CHECKPOINT_STEPS}" \
  trainer.test_freq=-1 \
  trainer.resume_mode=disable \
  trainer.max_ckpt_to_keep=2 \
  trainer.nnodes=1 \
  "trainer.n_gpus_per_node=${NPROC}"
