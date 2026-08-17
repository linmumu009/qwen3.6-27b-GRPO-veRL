#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/llin-verl-grpo}"
VLLM_ROOT="${VLLM_ROOT:-/vllm}"
NODE_IP="${NODE_IP:?NODE_IP is required}"
RAY_PORT="${RAY_PORT:?RAY_PORT is required}"
RAY_RESOURCE="${RAY_RESOURCE:?RAY_RESOURCE is required}"
EXPECTED_NPUS="${EXPECTED_NPUS:?EXPECTED_NPUS is required}"
RAY_MIN_WORKER_PORT="${RAY_MIN_WORKER_PORT:?RAY_MIN_WORKER_PORT is required}"
RAY_MAX_WORKER_PORT="${RAY_MAX_WORKER_PORT:?RAY_MAX_WORKER_PORT is required}"
RAY_TEMP_DIR="${RAY_TEMP_DIR:?RAY_TEMP_DIR is required}"

export PYTHONPATH="${VLLM_ROOT}:${PROJECT_ROOT}/runtime:${PROJECT_ROOT}:${PYTHONPATH:-}"
export LLIN_PIN_RAY_ROLES=1
export HCCL_IF_IP="${HCCL_IF_IP:?HCCL_IF_IP is required}"
export HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:?HCCL_SOCKET_IFNAME is required}"
export HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:?HCCL_IF_BASE_PORT is required}"
export HCCL_HOST_SOCKET_PORT_RANGE="${HCCL_HOST_SOCKET_PORT_RANGE:?HCCL_HOST_SOCKET_PORT_RANGE is required}"
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:?HCCL_NPU_SOCKET_PORT_RANGE is required}"
export HCCL_ALGO="broadcast=level0:NA;level1:NHR"

python3 "${PROJECT_ROOT}/scripts/patch_verl_force_final_config.py" --target /verl/verl/workers/config/rollout.py
python3 "${PROJECT_ROOT}/scripts/patch_verl_vllm_dp_weight_sync.py" --target /verl/verl/workers/rollout/vllm_rollout/utils.py
python3 "${PROJECT_ROOT}/scripts/patch_verl_agent_loop_continuous_token.py" --target /verl/verl/experimental/agent_loop/agent_loop.py
python3 "${PROJECT_ROOT}/scripts/patch_verl_fastest_k_oversampling.py" --rollouter /verl/verl/experimental/fully_async_policy/fully_async_rollouter.py --agent-loop /verl/verl/experimental/agent_loop/agent_loop.py --tool-agent-loop /verl/verl/experimental/agent_loop/tool_agent_loop.py --llm-server /verl/verl/workers/rollout/llm_server.py
python3 "${PROJECT_ROOT}/scripts/patch_verl_fastest_k_abort_observability.py" --agent-loop /verl/verl/experimental/agent_loop/agent_loop.py --llm-server /verl/verl/workers/rollout/llm_server.py
python3 "${PROJECT_ROOT}/scripts/patch_verl_fastest_k_abort_retry.py" --agent-loop /verl/verl/experimental/agent_loop/agent_loop.py --llm-server /verl/verl/workers/rollout/llm_server.py
python3 "${PROJECT_ROOT}/scripts/patch_verl_vllm_abort_api.py" --target /verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py
python3 "${PROJECT_ROOT}/scripts/patch_verl_abort_partial_tokens.py" --llm-server /verl/verl/workers/rollout/llm_server.py

observed_npus="$(python3 -c 'import torch_npu; print(torch_npu.npu.device_count())')"
if [[ "${observed_npus}" != "${EXPECTED_NPUS}" ]]; then
  printf 'expected %s visible NPUs, observed %s\n' "${EXPECTED_NPUS}" "${observed_npus}" >&2
  exit 2
fi

mkdir -p "${RAY_TEMP_DIR}"
ray start --head \
  --node-ip-address="${NODE_IP}" \
  --port="${RAY_PORT}" \
  --num-cpus=64 \
  --resources="{\"${RAY_RESOURCE}\": 1}" \
  --include-dashboard=false \
  --disable-usage-stats \
  --min-worker-port="${RAY_MIN_WORKER_PORT}" \
  --max-worker-port="${RAY_MAX_WORKER_PORT}" \
  --temp-dir="${RAY_TEMP_DIR}"
