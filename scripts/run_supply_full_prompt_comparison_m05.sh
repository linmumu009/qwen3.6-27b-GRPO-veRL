#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/workspace/llin-verl-grpo
CODE=$(cd -- "$(dirname -- "$0")" && pwd)
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH="/vllm:$ROOT/runtime:$ROOT:/verl:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
umask 077
exec python3 "$CODE/run_supply_full_prompt_comparison.py" > "$ROOT/runs/supply-chain-full-prompt-comparison-20260909-01.launch.log" 2>&1
