"""Fix the archived historical DP rank adapter for colocated TP-only replicas."""
import difflib
import hashlib
import json
from pathlib import Path
import tarfile

OLD = '        local_rank = _resolve_vllm_weight_sync_local_rank(self.local_rank, parallel_config)\n'
NEW = ('        from tools.weight_sync import colocated_local_rank\n'
       '        local_rank = colocated_local_rank(self.local_rank, parallel_config)\n')


def main():
    root = Path(__file__).resolve().parents[1]
    member = 'verl/workers/rollout/vllm_rollout/utils.py'
    path = root / 'frameworks/verl' / member
    with tarfile.open(root / 'frameworks/verl.tar.gz') as archive:
        original = archive.extractfile(member).read().decode()
    if original.count(OLD) != 1:
        raise ValueError('Unexpected historical sync adapter')
    after = original.replace(OLD, NEW, 1)
    if path.read_text() not in {original, after}:
        raise ValueError('Unrecognized source modifications; refusing to overwrite')
    patch = ''.join(difflib.unified_diff(original.splitlines(True), after.splitlines(True),
                                       fromfile='a/' + member, tofile='b/' + member))
    patch_file = root / 'frameworks/verl-delivery-colocated-sync.patch'
    patch_file.write_text(patch)
    path.write_text(after)
    manifest = root / 'environment/source_manifest.json'
    value = json.loads(manifest.read_text())
    value['verl']['delivery_colocated_sync_patch'] = {
        'file': str(patch_file.relative_to(root)),
        'sha256': hashlib.sha256(patch.encode()).hexdigest(),
        'patched_file_sha256': hashlib.sha256(after.encode()).hexdigest(),
    }
    manifest.write_text(json.dumps(value, indent=2))
    print('Colocated replica-local weight synchronization patch applied')


if __name__ == '__main__':
    main()
