"""Execute the patched position-index function for both CP2 ranks on NPU."""
import ast
from pathlib import Path
from types import SimpleNamespace
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu  # noqa: F401

torch_npu.npu.set_device(0)
path = Path('/MindSpeed/mindspeed/core/context_parallel/rotary_pos_embedding_utils.py')
source = ast.parse(path.read_text())
function = next(n for n in source.body if isinstance(n, ast.FunctionDef)
                and n.name == '_get_pos_emb_on_this_cp_rank_in_megatron_cp')
compiled = compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec')
for rank in range(2):
    namespace = {'torch': torch, 'parallel_state': SimpleNamespace(
        get_context_parallel_world_size=lambda: 2, get_context_parallel_rank=lambda: rank)}
    exec(compiled, namespace)
    value = torch.arange(16, dtype=torch.float32, device='npu:0').requires_grad_()
    output = namespace[function.name](value, 0)
    indices = [*range(rank * 4, rank * 4 + 4), *range((3-rank)*4, (4-rank)*4)]
    assert output.cpu().tolist() == indices
    output.sum().backward()
    assert value.grad.cpu().tolist() == [float(i in indices) for i in range(16)]
print('Both CP2 position slices and gradients verified with pageable indices', flush=True)
