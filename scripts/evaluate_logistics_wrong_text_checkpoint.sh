#!/usr/bin/env bash
set -Eeuo pipefail
RUN=${WRONG_TEXT_RUN:?}
ROOT=/workspace/llin-verl-grpo
CODE=$(cd -- "$(dirname -- "$0")" && pwd)
BASE=${WRONG_TEXT_BASE_MODEL:-$ROOT/runs/llin-step120-opensource-20260825-02/hf_export_step120_opensource}
CASES=$ROOT/runs/logistics-cpt-diagnostics-20260904/private/public_eval/frozen_cases_source.jsonl
STEP=${1:?}
[[ $STEP =~ ^[0-9]+$ && -d $RUN ]] || exit 2
MODEL=$BASE
export PYTHONPATH="/vllm:$(dirname -- "$CODE"):$ROOT:$ROOT/reference/Megatron-Bridge-de93536e/src:$ROOT/runtime:/verl:${PYTHONPATH:-}"
if [[ $STEP != 0 ]]; then
  MODEL=$RUN/hf/step_$STEP
  python3 "$CODE/export_megatron_dist_to_hf.py" --actor-checkpoint "$RUN/checkpoints/global_step_$STEP" \
    --base-model "$BASE" --output-dir "$MODEL"
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ASCEND_ENABLE_MLAPO=0
HCCL_HOST_SOCKET_PORT_RANGE=43000-43199 HCCL_NPU_SOCKET_PORT_RANGE=44000-44199 \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 "$CODE/run_vllm_logistics_mcq.py" \
  --model "$MODEL" --model-label "wrong_text_step_$STEP" --cases "$CASES" \
  --shard-count 2 --shard-index 0 --tensor-parallel-size 8 --max-model-len 4096 \
  --max-num-seqs 64 --max-output-tokens 128 --gpu-memory-utilization 0.65 \
  --private-output "$RUN/private/step_$STEP.shard0.jsonl" --safe-output "$RUN/safe/step_$STEP.shard0.json" \
  > "$RUN/eval_step_$STEP.shard0.log" 2>&1 &
P0=$!
HCCL_HOST_SOCKET_PORT_RANGE=45000-45199 HCCL_NPU_SOCKET_PORT_RANGE=46000-46199 \
ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 python3 "$CODE/run_vllm_logistics_mcq.py" \
  --model "$MODEL" --model-label "wrong_text_step_$STEP" --cases "$CASES" \
  --shard-count 2 --shard-index 1 --tensor-parallel-size 8 --max-model-len 4096 \
  --max-num-seqs 64 --max-output-tokens 128 --gpu-memory-utilization 0.65 \
  --private-output "$RUN/private/step_$STEP.shard1.jsonl" --safe-output "$RUN/safe/step_$STEP.shard1.json" \
  > "$RUN/eval_step_$STEP.shard1.log" 2>&1 &
P1=$!
trap 'kill "$P0" "$P1" 2>/dev/null || true; wait || true' EXIT
STATUS=0
wait "$P0" || STATUS=1
wait "$P1" || STATUS=1
trap - EXIT
[[ $STATUS == 0 ]] || exit 3
python3 "$CODE/logistics_wrong_text_selection.py" --run "$RUN" --cases "$CASES" --step "$STEP" --model "$MODEL"
