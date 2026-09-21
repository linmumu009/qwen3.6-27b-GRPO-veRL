#!/usr/bin/env bash
set -Eeuo pipefail
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
unset ASCEND_RT_VISIBLE_DEVICES ASCEND_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export PYTHONPATH="/delivery:/delivery/frameworks/verl:/delivery/frameworks/Megatron-Bridge/src:${PYTHONPATH:-}"
case "$1" in GRPO|agentic-GRPO) python3 /delivery/environment/apply_vllm_pageable.py ;; esac
exec python3 /delivery/tools/launch.py "$@"
