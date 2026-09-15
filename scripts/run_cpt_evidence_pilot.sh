#!/usr/bin/env bash
set -Eeuo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
CODE=$(cd -- "$(dirname -- "$0")" && pwd)
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="/vllm:$(dirname -- "$CODE"):/workspace/llin-verl-grpo:${PYTHONPATH:-}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
# Import gate catches toolkit-path loss before loading the model.
python3 -c 'import acl; from acl.rt import memcpy'
exec python3 "$CODE/run_cpt_evidence_pilot.py" "$@"
