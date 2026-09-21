"""Check synchronous ACL copies with pageable CPU memory on the target NPU."""
import acl
import torch
import torch_npu

torch_npu.npu.set_device(0)
expected = torch.arange(4096, dtype=torch.int32)
source = expected.to('npu:0')
target = torch.empty_like(expected, device='cpu', pin_memory=False)
restored = torch.empty_like(source)
torch_npu.npu.synchronize()
size = expected.numel() * expected.element_size()
assert acl.rt.memcpy(target.data_ptr(), size, source.data_ptr(), size, 2) == 0
assert torch.equal(target, expected)
assert acl.rt.memcpy(restored.data_ptr(), size, target.data_ptr(), size, 1) == 0
torch_npu.npu.synchronize()
assert torch.equal(restored.cpu(), expected)
print('Pageable CPU synchronous ACL D2H/H2D roundtrip passed', flush=True)
