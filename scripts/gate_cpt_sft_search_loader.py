"""Exercise the installed trainer's loader with DP=1 before allocating the model."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    a = p.parse_args()
    import torch_npu  # register the same device backend as training
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from verl.trainer.sft_trainer import SFTTrainer
    from qwen36_mcq_answer_dataset import Qwen36MCQAnswerDataset
    from audit_cpt_sft_search_data import EXPECTED
    config = OmegaConf.create(dict(data=dict(train_batch_size=3, num_workers=0,
                    pad_mode='no_padding', truncation='error', max_length=4096)))
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True, trust_remote_code=True)
    datasets = {s: Qwen36MCQAnswerDataset(str(a.data/(s+'.parquet')), tok, config.data)
                for s in ('train', 'dev')}
    holder = SimpleNamespace(config=config, train_dataset=datasets['train'], val_dataset=datasets['dev'],
              engine=SimpleNamespace(get_data_parallel_rank=lambda: 0, get_data_parallel_size=lambda: 1))
    SFTTrainer._build_dataloader(holder)
    report = {}
    for split in ('train', 'dev'):
        sampler = getattr(holder, split+'_sampler' if split == 'train' else 'val_sampler')
        loader = getattr(holder, split+'_dataloader' if split == 'train' else 'val_dataloader')
        order = list(sampler)
        count, tokens, loss, _ = EXPECTED[split]
        if sorted(order) != list(range(count)):
            raise ValueError('sampler does not cover every row exactly once')
        seen = seq_total = loss_total = batches = 0
        for batch in loader:
            ids = list(batch['input_ids'].unbind())
            masks = list(batch['loss_mask'].unbind())
            positions = list(batch['position_ids'].unbind())
            if len(ids) != 3:
                raise ValueError('unexpected batch size')
            for actual, mask, pos in zip(ids, masks, positions):
                expected = datasets[split][order[seen]]
                for value, key in ((actual, 'input_ids'), (mask, 'loss_mask'), (pos, 'position_ids')):
                    if not value.equal(expected[key]):
                        raise ValueError('loader content/mask/order mismatch')
                seen += 1
                seq_total += actual.numel()
                loss_total += int(mask.sum())
            batches += 1
        if (seen, seq_total, loss_total) != (count, tokens, loss):
            raise ValueError('incomplete loader budget')
        report[split] = dict(records=seen, batches=batches, sequence_tokens=seq_total,
                             loss_tokens=loss_total, order_sha256=hashlib.sha256(json.dumps(order).encode()).hexdigest())
    report.update(installed_loader_passed=True, data_parallel_size=1,
                  full_distributed_training_verified=False)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
