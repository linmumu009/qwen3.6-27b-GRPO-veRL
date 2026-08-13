#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/llin-verl-grpo}"
VLLM_ROOT="${VLLM_ROOT:-/vllm}"
RAY_ROLE_RESOURCE="${RAY_ROLE_RESOURCE:-llin_trainer}"
if [[ ! "${RAY_ROLE_RESOURCE}" =~ ^[A-Za-z0-9_]+$ ]]; then
  printf 'invalid RAY_ROLE_RESOURCE: %s\n' "${RAY_ROLE_RESOURCE}" >&2
  exit 2
fi
mkdir -p "${PROJECT_ROOT}/runs/ray/m05"

export PYTHONPATH="${VLLM_ROOT}:${PROJECT_ROOT}/runtime:${PROJECT_ROOT}:${PYTHONPATH:-}"
export LLIN_PIN_RAY_ROLES=1
export HCCL_IF_IP=192.168.202.5
export HCCL_SOCKET_IFNAME=eno0
export HCCL_IF_BASE_PORT=60000
export HCCL_HOST_SOCKET_PORT_RANGE=60100-60163
export HCCL_NPU_SOCKET_PORT_RANGE=60200-60263
# The checkpoint engine forms an asymmetric 1-trainer + 16-rollout
# communicator. Force the NHR broadcast path, which supports Broadcast and
# nonuniform cross-server rank placement on Atlas A3.
export HCCL_ALGO="broadcast=level0:NA;level1:NHR"

python3 "${PROJECT_ROOT}/scripts/patch_verl_force_final_config.py" \
  --target "/verl/verl/workers/config/rollout.py"

python3 "${PROJECT_ROOT}/scripts/patch_verl_fully_async_group_token_queue.py" \
  --message-queue "/verl/verl/experimental/fully_async_policy/message_queue.py" \
  --rollouter "/verl/verl/experimental/fully_async_policy/fully_async_rollouter.py"
python3 "${PROJECT_ROOT}/scripts/patch_verl_fastest_k_oversampling.py" \
  --rollouter "/verl/verl/experimental/fully_async_policy/fully_async_rollouter.py" \
  --agent-loop "/verl/verl/experimental/agent_loop/agent_loop.py" \
  --tool-agent-loop "/verl/verl/experimental/agent_loop/tool_agent_loop.py" \
  --llm-server "/verl/verl/workers/rollout/llm_server.py"
python3 "${PROJECT_ROOT}/scripts/patch_verl_fastest_k_abort_observability.py" \
  --agent-loop "/verl/verl/experimental/agent_loop/agent_loop.py" \
  --llm-server "/verl/verl/workers/rollout/llm_server.py"
python3 "${PROJECT_ROOT}/scripts/patch_verl_fastest_k_abort_retry.py" \
  --agent-loop "/verl/verl/experimental/agent_loop/agent_loop.py" \
  --llm-server "/verl/verl/workers/rollout/llm_server.py"
python3 "${PROJECT_ROOT}/scripts/patch_verl_vllm_abort_api.py" \
  --target "/verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py"
python3 "${PROJECT_ROOT}/scripts/patch_verl_fully_async_observability.py" \
  --trainer "/verl/verl/experimental/fully_async_policy/fully_async_trainer.py" \
  --main "/verl/verl/experimental/fully_async_policy/fully_async_main.py"
python3 "${PROJECT_ROOT}/scripts/patch_verl_fully_async_validation_step.py" \
  --trainer "/verl/verl/experimental/fully_async_policy/fully_async_trainer.py" \
  --rollouter "/verl/verl/experimental/fully_async_policy/fully_async_rollouter.py"

ray start \
  --head \
  --node-ip-address=192.168.202.5 \
  --port=26379 \
  --num-cpus=64 \
  --resources="{\"${RAY_ROLE_RESOURCE}\": 1}" \
  --include-dashboard=false \
  --disable-usage-stats \
  --min-worker-port=27000 \
  --max-worker-port=27999 \
  --temp-dir="${PROJECT_ROOT}/runs/ray/m05"
