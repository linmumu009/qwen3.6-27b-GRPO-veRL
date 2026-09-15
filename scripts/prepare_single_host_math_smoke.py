"""Prepare an isolated copy of the diagnosed 9B single-host experiment (inside container)."""
import argparse
import hashlib
import json
import shutil
import math
import struct
import shlex
from pathlib import Path
from patch_verl_vllm_device_namespace import patch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', type=Path, default=Path('/workspace/llin-verl-grpo'))
    parser.add_argument('--verl-root', type=Path, default=Path('/verl'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, output = args.project_root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = root / 'scripts/run_math_grpo_smoke.sh'
    original = source.read_text()
    # FSDP exports full FP32 parameters, regardless of on-disk BF16 weights.
    largest = 0
    for weights in (root / 'models/Qwen3.5-9B').glob('*.safetensors'):
        with weights.open('rb') as stream:
            header_len = struct.unpack('<Q', stream.read(8))[0]
            if header_len > 100_000_000:
                raise ValueError('Invalid safetensors header size')
            header = json.loads(stream.read(header_len))
        largest = max(largest, *(math.prod(meta['shape']) * 4 for name, meta in header.items() if name != '__metadata__'))
    if not 0 < largest <= 4096 * 1024**2:
        raise ValueError(f'Largest FP32 tensor does not fit 4096 MiB bucket: {largest}')
    runtime = output / 'runtime'
    shutil.copytree(args.verl_root / 'verl', runtime / 'verl', ignore=shutil.ignore_patterns('__pycache__'))
    patch(runtime / 'verl')
    shutil.copyfile(root / 'llin_verl/math_smoke_reward.py', output / 'reward.py')
    import pyarrow as pa
    import pyarrow.parquet as pq
    rows = pq.read_table(root / 'data/math_smoke.parquet').to_pylist()
    for row in rows:
        gold = row.get('ground_truth')
        if not isinstance(gold, dict) or 'golden_answer' not in gold:
            raise ValueError('Missing golden answer')
        row['reward_model'] = {'style': 'rule', 'ground_truth': gold}
    pq.write_table(pa.Table.from_pylist(rows), output / 'data.parquet')
    replacements = {
        '${PROJECT_ROOT}/data/math_smoke.parquet': str(output / 'data.parquet'),
        '${PROJECT_ROOT}/llin_verl/math_reward.py': str(output / 'reward.py'),
        'update_weights_bucket_megabytes=2560': 'update_weights_bucket_megabytes=4096',
        'export PYTHONPATH="/vllm-ascend:': f'export PYTHONPATH="{runtime}:/vllm-ascend:',
        'VERL_ROOT="${VERL_ROOT:-/verl}"': f'VERL_ROOT="{runtime}"',
        'trainer.save_freq=1': 'trainer.save_freq=3',
        'actor_rollout_ref.actor.fsdp_config.param_offload=True': 'actor_rollout_ref.actor.fsdp_config.param_offload=False',
        'actor_rollout_ref.actor.fsdp_config.optimizer_offload=True': 'actor_rollout_ref.actor.fsdp_config.optimizer_offload=False',
    }
    script = original
    for old, new in replacements.items():
        if script.count(old) != 1:
            raise ValueError(f'Unexpected original script anchor: {old}')
        script = script.replace(old, new, 1)
    # Explicit Ray runtime settings reach workers even with an existing Ray head.
    settings = [
        '+ray_kwargs.ray_init.runtime_env.env_vars.HCCL_EXEC_TIMEOUT=\'"600"\'',
        '+ray_kwargs.ray_init.runtime_env.env_vars.HCCL_CONNECT_TIMEOUT=\'"300"\'',
        '+ray_kwargs.ray_init.runtime_env.env_vars.VERL_LOGGING_LEVEL=\'"INFO"\'',
    ]
    script = script.replace('  "$@"', '  ' + ' \\\n  '.join(settings) + ' \\\n  "$@"')
    environment = {'RUN_NAME': output.name, 'OUTPUT_DIR': str(output),
                   'PROJECT_ROOT': str(root), 'MODEL_PATH': str(root / 'models/Qwen3.5-9B'),
                   'DATA_FILE': str(output / 'data.parquet')}
    exports = '\n'.join(f'export {key}={shlex.quote(value)}' for key, value in environment.items())
    script = script.replace('set -euo pipefail', 'set -euo pipefail\n' + exports)
    (output / 'run.sh').write_text(script)
    (output / 'manifest.json').write_text(json.dumps({'original_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'rows': len(rows), 'runtime': str(runtime), 'bucket_mib': 4096, 'largest_fp32_tensor_bytes': largest}, indent=2))
    print(output / 'run.sh')


if __name__ == '__main__':
    main()
