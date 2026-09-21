"""Use pageable host buffers in this disposable RL container only."""
import hashlib
import json
from pathlib import Path


def main():
    replacements = {
        'platform.py': (
            '    def is_pin_memory_available(cls):\n        return True\n',
            '    def is_pin_memory_available(cls):\n        return False  # DELIVERY_PAGEABLE_INPUTS\n'),
        'device_allocator/camem.py': (
            'torch.empty(size_in_bytes, dtype=torch.uint8, device="cpu", pin_memory=True)',
            'torch.empty(size_in_bytes, dtype=torch.uint8, device="cpu", pin_memory=False)'),
        '/MindSpeed/mindspeed/core/context_parallel/rotary_pos_embedding_utils.py': (
            '[cp_rank, (2 * cp_size - cp_rank - 1)], device="cpu", pin_memory=True\n    ).cuda(non_blocking=True)',
            '[cp_rank, (2 * cp_size - cp_rank - 1)], device="cpu", pin_memory=False\n    ).cuda(non_blocking=False)'),
        '/MindSpeed/mindspeed/core/context_parallel/get_batch_utils.py': (
            "torch.tensor(rearrange_index, device='cpu', pin_memory=True).to(device='npu', non_blocking=True)",
            "torch.tensor(rearrange_index, device='cpu', pin_memory=False).to(device='npu', non_blocking=False)"),
        '/Megatron-LM/megatron/core/models/common/embeddings/rope_utils.py': (
            '[cp_rank, (2 * cp_size - cp_rank - 1)], device="cpu", pin_memory=True\n    ).cuda(non_blocking=True)',
            '[cp_rank, (2 * cp_size - cp_rank - 1)], device="cpu", pin_memory=False\n    ).cuda(non_blocking=False)'),
    }
    edits = []
    for relative, (old, new) in replacements.items():
        path = Path('/vllm-ascend/vllm_ascend') / relative
        before = path.read_text()
        if before.count(old) != 1:
            raise RuntimeError(f'Unexpected Ascend source: {relative}; refusing an ambiguous patch')
        if relative.endswith('camem.py') and 'memcpy(cpu_ptr, dest_max, ptr, size_in_bytes, ACL_MEMCPY_DEVICE_TO_HOST)' not in before:
            raise RuntimeError('Pageable offload requires the existing synchronous ACL memcpy')
        edits.append((path, before, before.replace(old, new, 1)))
    for path, before, after in edits:
        path.write_text(after)
        print(json.dumps({'delivery_vllm_pageable': str(path),
                          'original_sha256': hashlib.sha256(before.encode()).hexdigest(),
                          'patched_sha256': hashlib.sha256(after.encode()).hexdigest()}), flush=True)


if __name__ == '__main__':
    main()
