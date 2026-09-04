#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/llin-verl-grpo}"
ARM="${ARM:?ARM must be direct or rewritten}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/runs/llin-step120-opensource-20260825-02/hf_export_step120_opensource}"
MEGATRON_BRIDGE_ROOT="${MEGATRON_BRIDGE_ROOT:-${PROJECT_ROOT}/reference/Megatron-Bridge-de93536e/src}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-${PROJECT_ROOT}/runs/llin-step120-opensource-20260825-02/checkpoints/global_step_120}"
SOURCE_MODEL_DIST_CKPT="${SOURCE_MODEL_DIST_CKPT:-${SOURCE_CHECKPOINT}/actor/model/dist_ckpt}"
TRAIN_FILE="${TRAIN_FILE:?TRAIN_FILE is required}"
EXPECTED_TRAIN_FILE_SHA256="${EXPECTED_TRAIN_FILE_SHA256:?EXPECTED_TRAIN_FILE_SHA256 is required}"
EXPECTED_CONTENT_TOKENS="${EXPECTED_CONTENT_TOKENS:?EXPECTED_CONTENT_TOKENS is required}"
RUN_NAME="${RUN_NAME:-logistics-exam-cpt-${ARM}-4x-20260904-01}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/runs/${RUN_NAME}}"
NPROC="${NPROC:-16}"
TP="${TP:-4}"
PP="${PP:-2}"
CP="${CP:-2}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
RECORDS_PER_EXPOSURE="${RECORDS_PER_EXPOSURE:-64}"
EXPECTED_SEQUENCE_TOKENS="${EXPECTED_SEQUENCE_TOKENS:-$((EXPECTED_CONTENT_TOKENS + RECORDS_PER_EXPOSURE))}"
STEPS_PER_EXPOSURE="${STEPS_PER_EXPOSURE:-16}"
TOTAL_EXPOSURES="${TOTAL_EXPOSURES:-4}"
TOTAL_STEPS="${TOTAL_STEPS:-64}"
LEARNING_RATE="${LEARNING_RATE:-5e-7}"
MIN_LEARNING_RATE="${MIN_LEARNING_RATE:-1e-7}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03125}"
TOKENIZER_SHA256="${TOKENIZER_SHA256:-06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523}"

if [[ "${ARM}" != direct && "${ARM}" != rewritten ]]; then
  printf 'unsupported experiment arm: %s\n' "${ARM}" >&2
  exit 2
fi
for path in "${MODEL_PATH}/config.json" "${MODEL_PATH}/tokenizer.json" "${SOURCE_MODEL_DIST_CKPT}/.metadata" "${TRAIN_FILE}"; do
  if [[ ! -e "${path}" ]]; then
    printf 'required logistics exam CPT input missing: %s\n' "${path}" >&2
    exit 2
  fi
done
if [[ ! -d "${MEGATRON_BRIDGE_ROOT}/megatron/bridge" ]]; then
  printf 'pinned Megatron-Bridge source missing: %s\n' "${MEGATRON_BRIDGE_ROOT}" >&2
  exit 2
fi
if [[ "${TRAIN_BATCH_SIZE}" -ne 4 || "${RECORDS_PER_EXPOSURE}" -ne 64 || \
      "${STEPS_PER_EXPOSURE}" -ne 16 || "${TOTAL_EXPOSURES}" -ne 4 || "${TOTAL_STEPS}" -ne 64 ]]; then
  printf 'exam CPT contract requires 64 blocks, batch 4, 16 steps/exposure, and 4 exposures/64 steps\n' >&2
  exit 2
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
  printf 'refusing to overwrite existing exam CPT output: %s\n' "${OUTPUT_DIR}" >&2
  exit 2
fi

actual_train_hash="$(sha256sum "${TRAIN_FILE}" | awk '{print $1}')"
if [[ "${actual_train_hash}" != "${EXPECTED_TRAIN_FILE_SHA256}" ]]; then
  printf 'training Parquet hash mismatch\n' >&2
  exit 2
fi

exec 9>"${PROJECT_ROOT}/runs/.logistics-exam-cpt.lock"
if ! flock -n 9; then
  printf 'another logistics exam CPT run holds the lock\n' >&2
  exit 3
fi

source /usr/local/Ascend/ascend-toolkit/set_env.sh
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
  --expected-records "${RECORDS_PER_EXPOSURE}" \
  --expected-content-tokens "${EXPECTED_CONTENT_TOKENS}" \
  --expected-tokenizer-sha256 "${TOKENIZER_SHA256}" \
  --output "${OUTPUT_DIR}/tokenization_gate.safe.json"

cat > "${OUTPUT_DIR}/experiment_contract.txt" <<EOF
purpose=controlled_benchmark_contamination_${ARM}_gold_text_cpt
promotion_allowed=false
delivery_metric_valid=false
interpretation=same_item_association_or_semantic_transfer_only
source_checkpoint=${SOURCE_CHECKPOINT}
source_model_dist_ckpt=${SOURCE_MODEL_DIST_CKPT}
checkpoint_resume=false
checkpoint_initialization=model_only_dist_ckpt
optimizer_state=fresh
dataloader_state=fresh
objective=raw_text_all_real_next_token_targets_including_eos
chat_template_applied=false
question_arm=${ARM}
distractors_rendered=0
option_indices_or_letters_rendered_by_template=false
records_per_exposure=${RECORDS_PER_EXPOSURE}
steps_per_exposure=${STEPS_PER_EXPOSURE}
planned_exposures=${TOTAL_EXPOSURES}
content_tokens_per_exposure=${EXPECTED_CONTENT_TOKENS}
sequence_tokens_with_eos_per_exposure=${EXPECTED_SEQUENCE_TOKENS}
train_batch_size=${TRAIN_BATCH_SIZE}
total_training_steps=${TOTAL_STEPS}
learning_rate=${LEARNING_RATE}
minimum_learning_rate=${MIN_LEARNING_RATE}
warmup_ratio=${WARMUP_RATIO}
lr_decay_style=cosine
mtp_enable=false
mtp_enable_train=false
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
  trainer.project_name=llin-logistics-exam-cpt \
  "trainer.experiment_name=${RUN_NAME}" \
  'trainer.logger=["console"]' \
  "trainer.total_epochs=${TOTAL_EXPOSURES}" \
  "trainer.total_training_steps=${TOTAL_STEPS}" \
  "trainer.save_freq=${TOTAL_STEPS}" \
  trainer.test_freq=-1 \
  trainer.resume_mode=disable \
  trainer.max_ckpt_to_keep=1 \
  trainer.nnodes=1 \
  "trainer.n_gpus_per_node=${NPROC}"

python3 "${PROJECT_ROOT}/scripts/summarize_logistics_cpt_exposure_curve.py" \
  --run-dir "${OUTPUT_DIR}" \
  --experiment "logistics_exam_gold_text_cpt_${ARM}_4x" \
  --steps-per-exposure "${STEPS_PER_EXPOSURE}" \
  --total-exposures "${TOTAL_EXPOSURES}" \
  --sequence-tokens-per-exposure "${EXPECTED_SEQUENCE_TOKENS}" \
  --checkpoint-exposure 4 \
  --output "${OUTPUT_DIR}/training_summary.safe.json"

python3 "${PROJECT_ROOT}/scripts/export_megatron_dist_to_hf.py" \
  --actor-checkpoint "${OUTPUT_DIR}/checkpoints/global_step_${TOTAL_STEPS}" \
  --base-model "${MODEL_PATH}" \
  --output-dir "${OUTPUT_DIR}/hf_export_step_${TOTAL_STEPS}"

printf 'complete\n' > "${OUTPUT_DIR}/pipeline_status.txt"
