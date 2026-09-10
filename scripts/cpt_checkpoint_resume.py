"""Normalize a saved end-of-epoch iterator in memory; never edit checkpoint files."""
from copy import deepcopy


def normalize_epoch_boundary(state, *, global_step, steps_per_epoch, records):
    if global_step <= 0 or global_step % steps_per_epoch:
        return deepcopy(state)
    if state.get('_iterator_finished') is True:
        return deepcopy(state)
    if (state.get('_num_yielded') != steps_per_epoch or
            state.get('_sampler_iter_yielded') != steps_per_epoch or
            state.get('_sampler_iter_state', {}).get('samples_yielded') != records):
        raise ValueError('Saved state does not match the claimed complete epoch')
    result = deepcopy(state)
    result['_iterator_finished'] = True
    return result


def install_epoch_boundary_restore():
    """Patch only this process's loader restoration, retaining model/Adam/RNG restore."""
    import torch
    from pathlib import Path
    from verl.utils.checkpoint.checkpoint_handler import CheckpointHandler, extract_step
    original = CheckpointHandler._load_dataloader_state
    def restore(self, checkpoint_path):
        path = Path(checkpoint_path)/f'data_{self.dp_rank}.pt'
        if not path.exists():
            raise FileNotFoundError('Resume probe requires saved loader state')
        original(self, checkpoint_path)
        state = torch.load(path, map_location='cpu', weights_only=False)
        normalized = normalize_epoch_boundary(state, global_step=extract_step(checkpoint_path),
            steps_per_epoch=len(self.train_dataloader), records=len(self.train_dataloader.dataset))
        self.train_dataloader.load_state_dict(normalized)
    CheckpointHandler._load_dataloader_state = restore
