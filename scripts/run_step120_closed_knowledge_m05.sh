#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/workspace/llin-verl-grpo
BASE=${ROOT}/runs/step120-closed-knowledge-20260907
[[ ! -e "${BASE}/inference" ]] || { printf 'refusing overwrite\n' >&2; exit 2; }
exec 9>"${ROOT}/runs/.logistics-exam-cpt.lock"
flock -n 9 || { printf 'another logistics job holds the lock\n' >&2; exit 3; }
available_bytes=$(df -B1 --output=avail "${ROOT}/runs" | tail -n 1 | tr -d ' ')
[[ "${available_bytes}" -ge 10000000000 ]] || { printf 'need 10GB free\n' >&2; exit 2; }
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH="/vllm:${ROOT}/runtime:${ROOT}:/verl:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
unset ASCEND_RT_VISIBLE_DEVICES
umask 077
exec >"${BASE}/pipeline.log" 2>&1
trap 'printf "failed at line %s\n" "${LINENO}" > "${BASE}/failure.txt"' ERR
python3 "${ROOT}/scripts/run_step120_closed_knowledge.py" --root "${ROOT}" --protocol "${BASE}/protocol.private.json" --output "${BASE}/inference"
