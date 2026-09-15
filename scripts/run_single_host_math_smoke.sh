#!/usr/bin/env bash
# Run inside the existing m05 training container with its idle single-node Ray head.
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/workspace/llin-verl-grpo}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/runs/math-single-host-$(date +%Y%m%d-%H%M%S)}"
python3 - <<'PY'
import subprocess
import ray
status = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True, check=True).stdout
if status.count('No running processes found') != 8:
    raise SystemExit('Expected all 16 m05 chips idle; another NPU process is present.')
ray.init(address='192.168.202.5:26379', logging_level='ERROR')
nodes = [n for n in ray.nodes() if n['Alive']]
free = ray.available_resources()
if (len(nodes) != 1 or nodes[0]['NodeManagerAddress'] != '192.168.202.5'
        or free.get('NPU', 0) != 16
        or not all(free.get(k, 0) >= 1 for k in ['llin_trainer', 'llin_rollout'])):
    raise SystemExit('Expected one idle m05 Ray node with 16 NPU and both role labels.')
ray.shutdown()
PY
python3 "${PROJECT_ROOT}/scripts/prepare_single_host_math_smoke.py" \
  --project-root "${PROJECT_ROOT}" --output "${OUTPUT_DIR}"
set +e
bash "${OUTPUT_DIR}/run.sh" 2>&1 | tee "${OUTPUT_DIR}/driver.log"
result=${PIPESTATUS[0]}
printf '%s\n' "${result}" > "${OUTPUT_DIR}/exit_code"
exit "${result}"
