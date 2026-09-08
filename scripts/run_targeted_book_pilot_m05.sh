#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/workspace/llin-verl-grpo
OUT=/opt/llin-targeted-book-pilot-20260908
BASE=${ROOT}/runs/logistics-cpt-book-exposure-curve-2x4x-20260904-01/hf_export_step_116
SOURCE=${ROOT}/runs/logistics-cpt-book-exposure-curve-2x4x-20260904-01/checkpoints/global_step_116/model/dist_ckpt
DATA=${ROOT}/runs/targeted-book-pilot-prepared-20260908
[[ ! -e "$OUT" && -e "$SOURCE/.metadata" && -e "$DATA/manifest.safe.json" ]] || exit 2
available=$(df -B1 --output=avail /opt | tail -n 1 | tr -d ' ')
[[ "$available" -ge 400000000000 ]] || exit 2
exec 9>"${ROOT}/runs/.logistics-exam-cpt.lock"
flock -n 9 || exit 3
umask 077
mkdir -p "$OUT"
exec >"${OUT}/pipeline.log" 2>&1
trap 'printf "failed_at_line_%s\n" "$LINENO" > "$OUT/status.txt"' ERR
python3 - "$DATA" <<'PY'
import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1]);m=json.loads((p/'manifest.safe.json').read_text())
assert m['train_count']==80 and m['steps']==10 and m['dropped_tail']==0
assert hashlib.sha256((p/'train.parquet').read_bytes()).hexdigest()==m['train_parquet_sha256']
PY
cp "$DATA/manifest.safe.json" "$OUT/data_manifest.safe.json"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH="${ROOT}:${ROOT}/reference/Megatron-Bridge-de93536e/src:${ROOT}/runtime:/verl:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=1 HYDRA_FULL_ERROR=1 TOKENIZERS_PARALLELISM=true
unset ASCEND_RT_VISIBLE_DEVICES
printf 'training\n' > "$OUT/status.txt"
torchrun --standalone --nnodes=1 --nproc_per_node=16 --log-dir="$OUT/torchrun_logs" --redirects=3 --tee=0 \
 -m verl.trainer.sft_trainer \
 "data.train_files=$DATA/train.parquet" data.val_files=null \
 data.train_max_samples=-1 data.train_batch_size=8 data.micro_batch_size_per_gpu=1 \
 data.use_dynamic_bsz=false data.max_token_len_per_gpu=4096 data.max_length=4096 \
 data.pad_mode=no_padding data.truncation=error data.num_workers=0 \
 "data.custom_cls.path=$ROOT/scripts/qwen36_mcq_answer_dataset.py" data.custom_cls.name=Qwen36MCQAnswerDataset \
 model=hf_model "model.path=$BASE" model.trust_remote_code=true model.use_remove_padding=false model.lora_rank=0 \
 model.mtp.enable=false model.mtp.enable_train=false model.mtp.enable_rollout=false \
 optim=megatron optim.lr=5e-7 optim.min_lr=5e-7 optim.lr_warmup_steps_ratio=0 \
 optim.weight_decay=0 'optim.betas=[0.9,0.999]' optim.clip_grad=1.0 optim.lr_decay_style=constant \
 +optim.override_optimizer_config.adam_beta1=0.9 +optim.override_optimizer_config.adam_beta2=0.999 \
 +optim.override_optimizer_config.optimizer_cpu_offload=false \
 engine=megatron engine.tensor_model_parallel_size=4 engine.pipeline_model_parallel_size=2 engine.context_parallel_size=2 \
 engine.use_mbridge=true engine.vanilla_mbridge=false engine.use_megatron_fsdp=false engine.use_remove_padding=false \
 engine.param_offload=false engine.optimizer_offload=false engine.grad_offload=true engine.dtype=bfloat16 \
 engine.use_distributed_optimizer=true engine.use_dist_checkpointing=true "engine.dist_checkpointing_path=$SOURCE" \
 ++engine.override_transformer_config.attention_backend=auto ++engine.override_transformer_config.context_parallel_algo=kvallgather_cp_algo \
 ++engine.override_transformer_config.recompute_method=uniform ++engine.override_transformer_config.recompute_granularity=full \
 ++engine.override_transformer_config.recompute_num_layers=1 ++engine.override_transformer_config.use_flash_attn=true \
 ++engine.override_transformer_config.sequence_parallel=true 'checkpoint.load_contents=[]' 'checkpoint.save_contents=[model,extra]' \
 "trainer.default_local_dir=$OUT/checkpoints" trainer.project_name=llin-targeted-book \
 trainer.experiment_name=targeted-80-short-pilot 'trainer.logger=["console"]' \
 trainer.total_epochs=1 trainer.total_training_steps=10 trainer.save_freq=5 trainer.test_freq=-1 \
 trainer.resume_mode=disable trainer.max_ckpt_to_keep=2 trainer.nnodes=1 trainer.n_gpus_per_node=16
for step in 5 10; do
 printf 'exporting_step_%s\n' "$step" > "$OUT/status.txt"
 python3 "$ROOT/scripts/export_megatron_dist_to_hf.py" --actor-checkpoint "$OUT/checkpoints/global_step_$step" \
  --base-model "$BASE" --output-dir "$OUT/hf_step_$step" > "$OUT/export_$step.log" 2>&1
 printf 'book_probe_step_%s\n' "$step" > "$OUT/status.txt"
 export PYTHONPATH="/vllm:${ROOT}/runtime:${ROOT}:/verl:${PYTHONPATH:-}"
 export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
 python3 "$ROOT/scripts/probe_targeted_book_groups.py" --root "$ROOT" --data "$DATA/evaluation" \
  --model "$OUT/hf_step_$step" --model-label "targeted_book_pilot_step_$step" \
  --output "$OUT/book_probe_step_$step" > "$OUT/book_probe_$step.log" 2>&1
done
# Fixed final checkpoint only: benchmark scores do not choose between checkpoints.
printf 'official_evaluation_step_10\n' > "$OUT/status.txt"
MODEL_PATH="$OUT/hf_step_10" MODEL_LABEL=targeted_book_pilot_step10 \
 CASES_PATH="$ROOT/runs/logistics-cpt-diagnostics-20260904/private/public_eval/frozen_cases_source.jsonl" \
 REPEATS=3 MAX_MODEL_LEN=4096 MAX_OUTPUT_TOKENS=96 PRIVATE_OUTPUT="$OUT/evaluation.jsonl" \
 SAFE_OUTPUT="$OUT/evaluation.safe.json" bash "$ROOT/scripts/run_logistics_mcq_on_m05.sh" > "$OUT/evaluation.log" 2>&1
python3 "$ROOT/scripts/summarize_mcq_repeats.py" --repeat "$OUT/evaluation.jsonl" --repeat "$OUT/evaluation.repeat2.jsonl" \
 --repeat "$OUT/evaluation.repeat3.jsonl" --cases "$ROOT/runs/logistics-cpt-diagnostics-20260904/private/public_eval/frozen_cases_source.jsonl" \
 --model-label targeted_book_pilot_step10 --private-output "$OUT/majority.private.jsonl" --safe-output "$OUT/majority.safe.json"
printf 'inference_complete_grading_pending\n' > "$OUT/status.txt"
