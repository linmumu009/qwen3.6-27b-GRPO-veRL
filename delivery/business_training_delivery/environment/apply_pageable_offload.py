"""Delivery-only A3 fix: synchronous pageable model offload buffers.

The original snapshot stays unchanged in verl.tar.gz. This explicit extra patch
avoids large concurrent aclrtMallocHost allocations; the existing synchronous
D2H copy and persistent buffer lifetime are retained.
"""
import difflib
import hashlib
import json
from pathlib import Path
import tarfile

OLD = '                                device="cpu",\n                                pin_memory=True,\n'
NEW = ('                                device="cpu",\n'
       '                                # DELIVERY_PAGEABLE_OFFLOAD: D2H copy below is synchronous.\n'
       '                                pin_memory=False,\n')


def patch_text(text):
    if NEW not in text and text.count(OLD) != 1:
        raise ValueError("Unexpected upstream offload allocation; refusing an ambiguous patch")
    if "buffer.param_data.cpu_data.copy_(buffer.param_data.data, non_blocking=False)" not in text:
        raise ValueError("Pageable buffer requires synchronous D2H completion before freeing device storage")
    text = text.replace(OLD, NEW, 1)
    # On Ascend, non_blocking D2H .to() implicitly allocates pinned host
    # buffers as well (including FP32 master weights and optimizer states).
    text = text.replace('.to("cpu", non_blocking=True)', '.to("cpu", non_blocking=False)')
    text = text.replace('copy_(buffer.param_data.cpu_data, non_blocking=True)',
                        'copy_(buffer.param_data.cpu_data, non_blocking=False)')
    return text


def main():
    root = Path(__file__).resolve().parents[1]
    path = root / "frameworks/verl/verl/utils/megatron_utils.py"
    before = path.read_text()
    with tarfile.open(root / "frameworks/verl.tar.gz") as archive:
        original = archive.extractfile("verl/utils/megatron_utils.py").read().decode()
    after = patch_text(original)
    if before not in {original, original.replace(OLD, NEW, 1), after}:
        raise ValueError("Unrecognized working-tree edits; refusing to overwrite")
    if after == before:
        print("Pageable offload patch already applied")
        return
    patch = "".join(difflib.unified_diff(original.splitlines(True), after.splitlines(True),
                                       fromfile="a/verl/utils/megatron_utils.py", tofile="b/verl/utils/megatron_utils.py"))
    (root / "frameworks/verl-delivery-pageable-offload.patch").write_text(patch)
    path.write_text(after)
    manifest = root / "environment/source_manifest.json"
    content = json.loads(manifest.read_text())
    content["verl"]["delivery_extra_patch"] = {
        "file": "frameworks/verl-delivery-pageable-offload.patch",
        "sha256": hashlib.sha256(patch.encode()).hexdigest(),
        "patched_file_sha256": hashlib.sha256(after.encode()).hexdigest(),
    }
    manifest.write_text(json.dumps(content, indent=2))
    print("Applied synchronous pageable model-offload patch")


if __name__ == "__main__":
    main()
