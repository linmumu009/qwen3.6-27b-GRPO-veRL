"""SFT trainer entrypoint with a frozen paired sampler, scoped to this process only."""
import hashlib
import json
import os
from pathlib import Path


def validate_plan(plan, arm):
    if arm not in ('original', 'related'):
        raise ValueError('Unknown paired arm')
    assert plan['records'] == 116 and plan['batch_size'] == 4 and len(plan['epochs']) == 4
    for epoch in plan['epochs']:
        assert sorted(epoch[arm]) == list(range(116))
        a, b = epoch['original_supervised_tokens'], epoch['related_supervised_tokens']
        assert len(a) == len(b) == 29
        assert max(abs(y-x)/x for x, y in zip(a, b)) <= .01
        for key in ('original', 'related'):
            indices = epoch[key]
            actual = [sum(plan['lengths'][key][i] for i in indices[j:j+4]) for j in range(0, 116, 4)]
            assert actual == epoch[key+'_supervised_tokens'], 'Budget metadata differs from sample permutation'


def main():
    plan_path = Path(os.environ['CPT_BATCH_PLAN'])
    arm = os.environ['CPT_ARM']
    if hashlib.sha256(plan_path.read_bytes()).hexdigest() != os.environ['CPT_BATCH_PLAN_SHA']:
        raise ValueError('Frozen batch plan changed')
    plan = json.loads(plan_path.read_text())
    validate_plan(plan, arm)
    if hashlib.sha256(Path(os.environ['TRAIN_FILE']).read_bytes()).hexdigest() != plan['dataset_sha256'][arm]:
        raise ValueError('Training source differs from frozen paired plan')
    from torch.utils.data import DistributedSampler
    from verl.trainer import sft_trainer
    class PairedSampler(DistributedSampler):
        def __init__(self, dataset, **kwargs):
            super().__init__(dataset, **kwargs)
            assert self.num_replicas == 1 and self.rank == 0 and len(dataset) == 116
            actual = dataset.dataframe['token_count'].astype(int).tolist()
            assert actual == plan['lengths'][arm], 'Dataset order changed after budget matching'

        def __iter__(self):
            if not 0 <= self.epoch < len(plan['epochs']):
                raise ValueError('Epoch outside frozen training budget')
            return iter(plan['epochs'][self.epoch][arm])
    sft_trainer.DistributedSampler = PairedSampler
    sft_trainer.main()


if __name__ == '__main__':
    main()
