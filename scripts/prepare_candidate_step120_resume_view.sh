#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/llin-verl-grpo}"
ROLE="${1:-}"
SOURCE_CHECKPOINT="${SOURCE_MODEL_CHECKPOINT:-${PROJECT_ROOT}/runs/llin-pi-dense-correctness-step100-to-step120-20260810-01/checkpoints/global_step_120}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-${PROJECT_ROOT}/runs/resume-views/llin-candidate128-curriculum-step120/global_step_120}"

if [[ "${ROLE}" != "trainer" && "${ROLE}" != "rollout" ]]; then
  printf 'Usage: %s trainer|rollout\n' "$0" >&2
  exit 2
fi

mkdir -p "${RESUME_CHECKPOINT}"
if [[ -e "${RESUME_CHECKPOINT}/data.pt" || -L "${RESUME_CHECKPOINT}/data.pt" ]]; then
  printf 'Refusing stale dataloader state: %s/data.pt\n' "${RESUME_CHECKPOINT}" >&2
  exit 2
fi

if [[ "${ROLE}" == "trainer" ]]; then
  if [[ ! -f "${SOURCE_CHECKPOINT}/actor/ckpt_contents.json" ]]; then
    printf 'Source actor checkpoint not found: %s\n' "${SOURCE_CHECKPOINT}" >&2
    exit 2
  fi
  if [[ -e "${RESUME_CHECKPOINT}/actor" || -L "${RESUME_CHECKPOINT}/actor" ]]; then
    actual="$(readlink -f "${RESUME_CHECKPOINT}/actor")"
    expected="$(readlink -f "${SOURCE_CHECKPOINT}/actor")"
    if [[ "${actual}" != "${expected}" ]]; then
      printf 'Existing resume actor points to %s, expected %s\n' \
        "${actual}" "${expected}" >&2
      exit 2
    fi
  else
    ln -s "${SOURCE_CHECKPOINT}/actor" "${RESUME_CHECKPOINT}/actor"
  fi
  printf 'model_and_rng_from=%s\noptimizer_state=reset\ndataloader_state=reset_for_candidate_curriculum\n' \
    "${SOURCE_CHECKPOINT}" > "${RESUME_CHECKPOINT}/resume_contract.txt"
else
  # The rollout node only needs the cumulative step encoded in the directory
  # name.  Omitting data.pt makes the new 640-row curriculum start at row zero.
  printf 'dataloader_state=reset_for_candidate_curriculum\nsource_cursor_not_loaded=%s/data.pt\n' \
    "${SOURCE_CHECKPOINT}" > "${RESUME_CHECKPOINT}/resume_contract.txt"
fi

printf 'Prepared %s candidate resume view at %s\n' "${ROLE}" "${RESUME_CHECKPOINT}"
