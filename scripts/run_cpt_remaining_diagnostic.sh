#!/usr/bin/env bash
set -Eeuo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
CODE=$(cd -- "$(dirname -- "$0")" && pwd)
export PYTHONPATH="/vllm:/workspace/llin-verl-grpo:${PYTHONPATH:-}"
python3 -c 'import acl; from acl.rt import memcpy'
exec python3 "$CODE/run_cpt_remaining_diagnostic.py" "$@"
