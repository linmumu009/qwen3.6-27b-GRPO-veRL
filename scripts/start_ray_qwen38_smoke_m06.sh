#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/llin-verl-grpo}"
VLLM_ROOT="${VLLM_ROOT:-/vllm}"
RAY_HEAD_ADDRESS="${RAY_HEAD_ADDRESS:-192.168.202.5:36379}"
RAY_MIN_WORKER_PORT="${RAY_MIN_WORKER_PORT:-38000}"
RAY_MAX_WORKER_PORT="${RAY_MAX_WORKER_PORT:-38999}"

export PYTHONPATH="${VLLM_ROOT}:${PROJECT_ROOT}/runtime:${PROJECT_ROOT}:${PYTHONPATH:-}"
export LLIN_PIN_RAY_ROLES=1
export HCCL_IF_IP=192.168.202.4
export HCCL_SOCKET_IFNAME=eno0
export HCCL_IF_BASE_PORT=62000
export HCCL_HOST_SOCKET_PORT_RANGE=62100-62163
export HCCL_NPU_SOCKET_PORT_RANGE=62200-62263
export HCCL_ALGO="broadcast=level0:NA;level1:NHR"

python3 "${PROJECT_ROOT}/scripts/patch_verl_force_final_config.py" --target /verl/verl/workers/config/rollout.py
python3 "${PROJECT_ROOT}/scripts/patch_verl_vllm_dp_weight_sync.py" --target /verl/verl/workers/rollout/vllm_rollout/utils.py
python3 "${PROJECT_ROOT}/scripts/patch_verl_agent_loop_continuous_token.py" --target /verl/verl/experimental/agent_loop/agent_loop.py
python3 "${PROJECT_ROOT}/scripts/patch_verl_fully_async_group_token_queue.py" --message-queue /verl/verl/experimental/fully_async_policy/message_queue.py --rollouter /verl/verl/experimental/fully_async_policy/fully_async_rollouter.py
python3 "${PROJECT_ROOT}/scripts/patch_verl_fastest_k_oversampling.py" --rollouter /verl/verl/experimental/fully_async_policy/fully_async_rollouter.py --agent-loop /verl/verl/experimental/agent_loop/agent_loop.py --tool-agent-loop /verl/verl/experimental/agent_loop/tool_agent_loop.py --llm-server /verl/verl/workers/rollout/llm_server.py
python3 "${PROJECT_ROOT}/scripts/patch_verl_fastest_k_abort_observability.py" --agent-loop /verl/verl/experimental/agent_loop/agent_loop.py --llm-server /verl/verl/workers/rollout/llm_server.py
python3 "${PROJECT_ROOT}/scripts/patch_verl_fastest_k_abort_retry.py" --agent-loop /verl/verl/experimental/agent_loop/agent_loop.py --llm-server /verl/verl/workers/rollout/llm_server.py
python3 "${PROJECT_ROOT}/scripts/patch_verl_vllm_abort_api.py" --target /verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py
python3 "${PROJECT_ROOT}/scripts/patch_verl_abort_partial_tokens.py" --llm-server /verl/verl/workers/rollout/llm_server.py
python3 "${PROJECT_ROOT}/scripts/patch_verl_none_rollout_logprobs.py" --agent-loop /verl/verl/experimental/agent_loop/agent_loop.py --detach-utils /verl/verl/experimental/fully_async_policy/detach_utils.py
python3 "${PROJECT_ROOT}/scripts/patch_verl_fully_async_observability.py" --trainer /verl/verl/experimental/fully_async_policy/fully_async_trainer.py --main /verl/verl/experimental/fully_async_policy/fully_async_main.py
python3 "${PROJECT_ROOT}/scripts/patch_verl_fully_async_validation_step.py" --trainer /verl/verl/experimental/fully_async_policy/fully_async_trainer.py --rollouter /verl/verl/experimental/fully_async_policy/fully_async_rollouter.py

ray start \
  --address="${RAY_HEAD_ADDRESS}" \
  --node-ip-address=192.168.202.4 \
  --num-cpus=64 \
  --resources='{"llin_rollout": 1}' \
  --disable-usage-stats \
  --min-worker-port="${RAY_MIN_WORKER_PORT}" \
  --max-worker-port="${RAY_MAX_WORKER_PORT}"
