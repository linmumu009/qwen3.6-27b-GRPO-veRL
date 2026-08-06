#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/llin-verl-grpo}"
ROLE="${1:-}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-${PROJECT_ROOT}/runs/llin-v15-dwh-bossreward-12groups-100step-20260805-03/checkpoints/global_step_100}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-${PROJECT_ROOT}/runs/resume-views/llin-v15-step100-train236/global_step_100}"

if [[ "${ROLE}" != "trainer" && "${ROLE}" != "rollout" ]]; then
  printf 'Usage: %s trainer|rollout\n' "$0" >&2
  exit 2
fi

mkdir -p "${RESUME_CHECKPOINT}"

if [[ "${ROLE}" == "trainer" ]]; then
  if [[ ! -f "${SOURCE_CHECKPOINT}/actor/ckpt_contents.json" ]]; then
    printf 'Source actor checkpoint not found: %s\n' "${SOURCE_CHECKPOINT}" >&2
    exit 2
  fi
  if [[ -e "${RESUME_CHECKPOINT}/actor" || -L "${RESUME_CHECKPOINT}/actor" ]]; then
    actual="$(readlink -f "${RESUME_CHECKPOINT}/actor")"
    expected="$(readlink -f "${SOURCE_CHECKPOINT}/actor")"
    if [[ "${actual}" != "${expected}" ]]; then
      printf 'Existing resume actor points to %s, expected %s\n' "${actual}" "${expected}" >&2
      exit 2
    fi
  else
    ln -s "${SOURCE_CHECKPOINT}/actor" "${RESUME_CHECKPOINT}/actor"
  fi
  printf 'model_and_rng_from=%s\noptimizer_state=reset_missing_from_source\n' \
    "${SOURCE_CHECKPOINT}" > "${RESUME_CHECKPOINT}/resume_contract.txt"
else
  # The source cursor was saved for train237. The corrected formal dataset has
  # train236, so the rollout role intentionally exposes no data.pt and starts a
  # fresh deterministic dataloader while retaining the cumulative group count.
  if [[ -e "${RESUME_CHECKPOINT}/data.pt" ]]; then
    printf 'Refusing stale dataloader state: %s/data.pt\n' "${RESUME_CHECKPOINT}" >&2
    exit 2
  fi
  printf 'dataloader_state=reset_for_train236\nsource_cursor_not_loaded=%s/data.pt\n' \
    "${SOURCE_CHECKPOINT}" > "${RESUME_CHECKPOINT}/resume_contract.txt"
fi

printf 'Prepared %s resume view at %s\n' "${ROLE}" "${RESUME_CHECKPOINT}"
