#!/usr/bin/env bash
set -Eeuo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
CODE=$(cd -- "$(dirname -- "$0")" && pwd)
export PYTHONPATH="/vllm:$CODE/..:/workspace/llin-verl-grpo:/workspace/llin-verl-grpo/runtime:/verl:${PYTHONPATH:-}"
python3 -c 'import acl; from acl.rt import memcpy'
exec python3 "$CODE/run_cpt_p1_stability.py" "$@"
