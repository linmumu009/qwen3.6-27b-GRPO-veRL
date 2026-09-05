"""Exact evaluation-prefix token IDs, with only gold JSON and EOS supervised."""


def validate_record(ids, start, eos_token_id, max_length):
    if not 0 < start < len(ids)-1 or len(ids) > max_length:
        raise ValueError('invalid answer boundary or sequence length')
    if ids[-1] != eos_token_id or any(not isinstance(x, int) or x < 0 for x in ids):
        raise ValueError('invalid token IDs or EOS')
    return [0]*start + [1]*(len(ids)-start)


class Qwen36MCQAnswerDataset:
    def __init__(self, parquet_files, tokenizer, config, processor=None, max_samples=-1):
        import pandas as pd
        from omegaconf import ListConfig
        if config.get('pad_mode') != 'no_padding' or config.get('truncation') != 'error':
            raise ValueError('MCQ fit requires no padding and no truncation')
        paths = list(parquet_files) if isinstance(parquet_files, (list,ListConfig)) else [parquet_files]
        self.rows = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True).to_dict('records')
        if max_samples > 0:
            self.rows = self.rows[:max_samples]
        self.maximum = int(config.get('max_length',4096))
        self.eos = tokenizer.eos_token_id
        for row in self.rows:
            row['input_ids'] = [int(x) for x in row['input_ids']]
            validate_record(row['input_ids'], int(row['answer_start']), self.eos, self.maximum)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import torch
        row = self.rows[index]
        ids = row['input_ids']
        mask = validate_record(ids, int(row['answer_start']), self.eos, self.maximum)
        return {'input_ids': torch.tensor(ids,dtype=torch.long),
                'position_ids': torch.arange(len(ids),dtype=torch.long),
                'loss_mask': torch.tensor(mask,dtype=torch.long)}
