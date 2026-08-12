#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/llin-verl-grpo}"
MODEL_PATH="${MODEL_PATH:-/models/Qwen3.6-27B}"
MEGATRON_BRIDGE_ROOT="${MEGATRON_BRIDGE_ROOT:-${PROJECT_ROOT}/reference/Megatron-Bridge-de93536e/src}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-${PROJECT_ROOT}/runs/llin-pi-dense-correctness-step100-to-step120-20260810-01/checkpoints/global_step_120}"
SOURCE_MODEL_DIST_CKPT="${SOURCE_MODEL_DIST_CKPT:-${SOURCE_CHECKPOINT}/actor/model/dist_ckpt}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data/chosen_only_schema_action_20260813}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/chosen_only_schema_action_train48.parquet}"
DATASET_CONTRACT="${DATASET_CONTRACT:-${DATA_DIR}/contract.json}"
CANARY_DECISION="${CANARY_DECISION:-${PROJECT_ROOT}/runs/llin-chosen-schema-action-tf-step120-cal16-20260813-01/canary_decision.json}"
RUN_NAME="${RUN_NAME:?RUN_NAME is required}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/runs/${RUN_NAME}}"
TOKENIZATION_GATE="${TOKENIZATION_GATE:-${OUTPUT_DIR}/train48_tokenization_gate.json}"
NPROC="${NPROC:-16}"
TP="${TP:-4}"
PP="${PP:-2}"
CP="${CP:-2}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-48}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
TOOL_STRUCTURE_WEIGHT="${TOOL_STRUCTURE_WEIGHT:-0.25}"
SQL_PAYLOAD_WEIGHT="${SQL_PAYLOAD_WEIGHT:-8.0}"

if (( TP * PP * CP != NPROC )); then
  printf 'invalid SFT topology: TP(%s) * PP(%s) * CP(%s) != NPROC(%s)\n' "${TP}" "${PP}" "${CP}" "${NPROC}" >&2
  exit 2
fi
for path in "${MODEL_PATH}/config.json" "${SOURCE_MODEL_DIST_CKPT}/.metadata" "${TRAIN_FILE}" "${DATASET_CONTRACT}" "${CANARY_DECISION}"; do
  if [[ ! -e "${path}" ]]; then
    printf 'required chosen-only canary input missing: %s\n' "${path}" >&2
    exit 2
  fi
done
if [[ ! -d "${MEGATRON_BRIDGE_ROOT}/megatron/bridge" ]]; then
  printf 'pinned Megatron-Bridge source missing: %s\n' "${MEGATRON_BRIDGE_ROOT}" >&2
  exit 2
fi

# Keep the project root ahead of /verl.  The container ships a regular
# /verl/scripts package, which would otherwise shadow this repository's
# namespace-style scripts directory during the fail-closed CPU gate.
export PYTHONPATH="${MEGATRON_BRIDGE_ROOT}:${PROJECT_ROOT}/runtime:${PROJECT_ROOT}:/verl:${PYTHONPATH:-}"
mkdir -p "${OUTPUT_DIR}"
python3 "${PROJECT_ROOT}/scripts/check_chosen_only_schema_action_sft.py" \
  --data-file "${TRAIN_FILE}" \
  --dataset-contract "${DATASET_CONTRACT}" \
  --model-path "${MODEL_PATH}" \
  --max-length "${MAX_LENGTH}" \
  --output "${TOKENIZATION_GATE}"

python3 - "${DATASET_CONTRACT}" "${CANARY_DECISION}" "${TOKENIZATION_GATE}" "${TRAIN_FILE}" "${TRAIN_BATCH_SIZE}" "${TOOL_STRUCTURE_WEIGHT}" "${SQL_PAYLOAD_WEIGHT}" <<'PY'
import hashlib
import json
import sys

dataset = json.load(open(sys.argv[1], encoding="utf-8"))
decision = json.load(open(sys.argv[2], encoding="utf-8"))
token = json.load(open(sys.argv[3], encoding="utf-8"))
train_path = sys.argv[4]
batch = int(sys.argv[5])
structure_weight = float(sys.argv[6])
sql_weight = float(sys.argv[7])
sha256 = lambda path: hashlib.sha256(open(path, "rb").read()).hexdigest()
errors = []
if dataset.get("contract") != "chosen-only-schema-conditioned-first-action-sft-v1":
    errors.append("unexpected chosen-only dataset contract")
if dataset.get("train_rows") != batch or batch != 48:
    errors.append("canary batch must contain train48 exactly once")
if dataset.get("calibration_rows") != 16:
    errors.append("calibration16 split drifted")
if dataset.get("training_allowed") is not False:
    errors.append("source dataset unexpectedly authorizes training")
if ((dataset.get("outputs") or {}).get("train") or {}).get("sha256") != sha256(train_path):
    errors.append("train48 hash differs from dataset contract")
if decision.get("contract") != "chosen-only-first-action-baseline-decision-v1":
    errors.append("unexpected canary decision contract")
canary = decision.get("one_step_canary") or {}
if canary.get("allowed") is not True or canary.get("training_steps") != 1:
    errors.append("one-step canary is not authorized")
if (decision.get("decision") or {}).get("training_scope") != "one_step_train48_only":
    errors.append("training scope is not train48-only")
weights = canary.get("loss_weights") or {}
if weights != {"tool_structure": structure_weight, "sql_payload": sql_weight}:
    errors.append("runtime loss weights differ from sealed decision")
if token.get("rows") != 48:
    errors.append("train48 tokenization row count drifted")
for flag in (
    "all_rows_tokenize_without_truncation",
    "all_rows_loss_exactly_one_assistant_tool_action",
    "all_nonassistant_context_loss_zero",
    "all_tool_structure_and_sql_masks_nonempty_disjoint_and_complete",
):
    if token.get(flag) is not True:
        errors.append(f"tokenization gate failed: {flag}")
if errors:
    raise SystemExit("; ".join(errors))
PY

cat > "${OUTPUT_DIR}/experiment_contract.txt" <<EOF
purpose=chosen_only_schema_conditioned_first_action_one_step_canary
source_checkpoint=step120
source_model_dist_ckpt=${SOURCE_MODEL_DIST_CKPT}
checkpoint_initialization=model_only_dist_ckpt
optimizer_state=fresh_cpu_offloaded_adam
dataloader_state=fresh
train_rows=48
calibration_rows_excluded=16
heldout_frozen16_val20_test20_overlap=0
train_batch_size=48
micro_batch_size_per_gpu=1
total_training_steps=1
total_epochs=1
learning_rate=${LEARNING_RATE}
max_length=${MAX_LENGTH}
loss_weights=${TOOL_STRUCTURE_WEIGHT}_${SQL_PAYLOAD_WEIGHT}
save_policy=final_model_and_extra_only
optimizer_checkpoint_saved=false
intermediate_validation=false
topology=tp${TP}_pp${PP}_cp${CP}
oracle_relevant_table_selection=true
deployment_ready=false
promotion_allowed=false
next_gate=forward_only_calibration16_pre_registered_thresholds
EOF

export CUDA_DEVICE_MAX_CONNECTIONS=1
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=true

torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC}" \
  -m verl.trainer.sft_trainer \
  "data.train_files=${TRAIN_FILE}" \
  data.val_files=null \
  data.messages_key=messages \
  data.tools_key=tools \
  data.enable_thinking_key=enable_thinking \
  data.enable_thinking_default=false \
  "data.train_batch_size=${TRAIN_BATCH_SIZE}" \
  data.micro_batch_size_per_gpu=1 \
  data.use_dynamic_bsz=false \
  "data.max_token_len_per_gpu=${MAX_LENGTH}" \
  "data.max_length=${MAX_LENGTH}" \
  data.pad_mode=no_padding \
  data.truncation=error \
  data.ignore_input_ids_mismatch=false \
  "data.custom_cls.path=${PROJECT_ROOT}/scripts/qwen36_first_action_weighted_sft_dataset.py" \
  data.custom_cls.name=Qwen36FirstActionWeightedSFTDataset \
  "+data.tool_structure_weight=${TOOL_STRUCTURE_WEIGHT}" \
  "+data.sql_payload_weight=${SQL_PAYLOAD_WEIGHT}" \
  model=hf_model \
  "model.path=${MODEL_PATH}" \
  model.trust_remote_code=true \
  model.use_remove_padding=false \
  model.lora_rank=0 \
  optim=megatron \
  "optim.lr=${LEARNING_RATE}" \
  optim.lr_warmup_steps_ratio=0 \
  optim.weight_decay=0 \
  'optim.betas=[0.9,0.95]' \
  optim.clip_grad=1.0 \
  optim.lr_decay_style=constant \
  +optim.override_optimizer_config.optimizer_cpu_offload=true \
  +optim.override_optimizer_config.optimizer_offload_fraction=1 \
  +optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=true \
  engine=megatron \
  "engine.tensor_model_parallel_size=${TP}" \
  "engine.pipeline_model_parallel_size=${PP}" \
  "engine.context_parallel_size=${CP}" \
  engine.use_mbridge=true \
  engine.vanilla_mbridge=false \
  engine.use_megatron_fsdp=false \
  engine.use_remove_padding=false \
  engine.param_offload=false \
  engine.optimizer_offload=true \
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
  trainer.project_name=llin-chosen-only-first-action \
  "trainer.experiment_name=${RUN_NAME}" \
  'trainer.logger=["console"]' \
  trainer.total_epochs=1 \
  trainer.total_training_steps=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.resume_mode=disable \
  trainer.max_ckpt_to_keep=1 \
  trainer.nnodes=1 \
  "trainer.n_gpus_per_node=${NPROC}" \
  "$@"
