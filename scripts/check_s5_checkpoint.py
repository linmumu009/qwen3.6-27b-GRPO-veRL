"""Validate S5's explicit model-and-RNG checkpoint; optimizer resume is unavailable."""
import json
from pathlib import Path


def checkpoint_gate(path):
    path = Path(path).resolve()
    manifest = json.loads((path/'ckpt_contents.json').read_text())
    if set(manifest['save_contents']) != {'model', 'extra'}:
        raise ValueError('S5 requires exactly model and extra save contents')
    if manifest['global_step'] != 77 or manifest['world_size'] != 16:
        raise ValueError('S5 checkpoint step/topology mismatch')
    if any(k in manifest['contents'] for k in ('optimizer', 'lr_scheduler')):
        raise ValueError('Unexpected optimizer state in S5 model-only contract')
    for key in ('model', 'rng_state'):
        relative = Path(manifest['contents'][key]['path'])
        directory = (path/relative).resolve()
        if relative.is_absolute() or '..' in relative.parts or not directory.is_relative_to(path):
            raise ValueError('Unsafe checkpoint component path')
        if not (directory/'.metadata').is_file() or not any(
                f.stat().st_size > 0 for f in directory.glob('*.distcp')):
            raise ValueError('Missing checkpoint component: '+key)
    if not list(path.glob('data_*.pt')):
        raise ValueError('Missing dataloader state')
    return dict(manifest=manifest, optimizer_resume_available=False)
