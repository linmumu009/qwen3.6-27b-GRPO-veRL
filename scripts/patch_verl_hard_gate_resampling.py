#!/usr/bin/env python3
"""Make fastest-K select H=1 trajectories and fail closed at the attempt cap."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


MARKER = "LLIN_TRISTATE_UNKNOWN_RESAMPLE_V2"
OLD_MARKER = "LLIN_HARD_GATE_RESAMPLE_QUORUM"
CONCAT_MARKER = "LLIN_NON_TENSOR_CONCAT_SCHEMA_V1"


def normalize_non_tensor_chunks(outputs: list) -> list:
    """Fill absent optional columns without changing rows, values, or order."""
    if not outputs:
        return outputs
    all_non_tensor_keys = set().union(*(item.non_tensor_batch for item in outputs))
    for item in outputs:
        for key in all_non_tensor_keys - set(item.non_tensor_batch):
            missing = np.empty(len(item), dtype=object)
            missing[:] = None
            item.non_tensor_batch[key] = missing
    return outputs


def patch(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    changed = False
    if MARKER in text:
        pass
    elif OLD_MARKER in text:
        text = text.replace(OLD_MARKER, MARKER)
        text = text.replace('reward_info.get("online_eligible", 0)', 'reward_info.get("train_mask", 0)')
        text = text.replace("hard_gate_eligible", "tristate_trainable")
        text = text.replace("hard_gate_rejected", "tristate_unknown")
        text = text.replace("hard_gate_cap_exhausted", "tristate_cap_exhausted")
        text = text.replace("hard-gate attempt cap", "tristate UNKNOWN attempt cap")
        changed = True
    else:
        old = """\
                    for task in done:
                        if len(outputs) < fastest_k:
                            outputs.append(await task)
                            selected_indices.append(task_to_index[task])
                        else:
                            await task
                            completed_but_discarded += 1

                pending_indices = [task_to_index[task] for task in pending]
"""
        new = """\
                    # LLIN_TRISTATE_UNKNOWN_RESAMPLE_V2: PASS and FAIL have
                    # train_mask=1; UNKNOWN has train_mask=0 and is resampled.
                    # UNKNOWN candidates are retained only as fail-closed
                    # placeholders if the attempt cap is exhausted.
                    if 'tristate_unknown' not in locals():
                        tristate_unknown = []
                    for task in done:
                        candidate = await task
                        reward_info = candidate.extra_fields.get("reward_extra_info", {})
                        eligible = bool(reward_info.get("train_mask", 0))
                        if eligible and len(outputs) < fastest_k:
                            outputs.append(candidate)
                            selected_indices.append(task_to_index[task])
                        else:
                            tristate_unknown.append((task_to_index[task], candidate))
                            completed_but_discarded += 1

                pending_indices = [task_to_index[task] for task in pending]
"""
        if old not in text:
            raise RuntimeError("fastest-K selection anchor not found; apply fastest-K patch first")
        text = text.replace(old, new, 1)
        old = '''\
            print(
                "[LLIN_FASTEST_K] "
'''
        new = '''\
            eligible_selected = len(outputs)
            tristate_cap_exhausted = eligible_selected < fastest_k
            if tristate_cap_exhausted:
                for rejected_index, rejected_output in tristate_unknown:
                    if len(outputs) >= fastest_k:
                        break
                    outputs.append(rejected_output)
                    selected_indices.append(rejected_index)
            if len(outputs) != fastest_k:
                raise RuntimeError(
                    f"tristate UNKNOWN attempt cap returned {len(outputs)} of {fastest_k} required placeholders"
                )
            print(
                "[LLIN_FASTEST_K] "
'''
        if old not in text:
            raise RuntimeError("fastest-K observability anchor not found")
        text = text.replace(old, new, 1)
        text = text.replace(
            'f"physical_aborts={physically_aborted} "',
            'f"physical_aborts={physically_aborted} "\n                f"tristate_trainable={eligible_selected} "\n                f"tristate_unknown={len(tristate_unknown)} "\n                f"tristate_cap_exhausted={tristate_cap_exhausted} "',
            1,
        )
        changed = True

    if CONCAT_MARKER not in text:
        old = """\
        output = DataProto.concat(outputs)
"""
        new = """\
        # LLIN_NON_TENSOR_CONCAT_SCHEMA_V1: different agent chunks can
        # legitimately observe different optional evidence fields (for
        # example, a no-tool guess beside a tool-using trajectory).  Preserve
        # those samples and fill only the missing column with None; the
        # tri-state judge will treat absent evidence as FAIL or UNKNOWN rather
        # than crashing the whole atomic shard.
        all_non_tensor_keys = set().union(*(item.non_tensor_batch for item in outputs))
        for item in outputs:
            for key in all_non_tensor_keys - set(item.non_tensor_batch):
                missing = np.empty(len(item), dtype=object)
                missing[:] = None
                item.non_tensor_batch[key] = missing
        output = DataProto.concat(outputs)
"""
        if old not in text:
            raise RuntimeError("AgentLoopManager concat anchor not found")
        text = text.replace(old, new, 1)
        changed = True

    if changed:
        path.write_text(text, encoding="utf-8")
        return "patched"
    return "already-patched"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-loop", default="/verl/verl/experimental/agent_loop/agent_loop.py")
    args = parser.parse_args()
    print(f"{patch(Path(args.agent_loop))}: {args.agent_loop}")


if __name__ == "__main__":
    main()
