"""Read-only identity and token audit. Does not certify the distributed loader."""
import argparse
import hashlib
import json
from pathlib import Path

from qwen36_mcq_answer_dataset import validate_record

EXPECTED = {
    'train': (591, 88819, 18579, '2f10b42c9a8ad0bde49b2c1887a6216727352c4871d1afdc5e84145e70dd9051'),
    'dev': (105, 16345, 3295, '54410c6d33db3cbbdcaa542a6a3a978f10d5acd4470dd8d132493629127c0e3a'),
}
TOKENIZER_SHA = '06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523'

S3_EXPECTED = {
    'train': (399, 72014, 6896, '349a19d47aee422bac6e6d52c77325d82e4fcca4800cc77e0e6d0cca64dff4dd'),
    'dev': (33, 5622, 212, '3a35e5307cd46027b5e91f598917c70c0bc470175ba1de4ec534120edddc41f0'),
}


S4_EXPECTED = {
    'train': (231, 39887, 4276, 'b0f083fc3f15b128bcce6295d162f25fbfe02d191272944eb587110c9dea82f8'),
    'dev': S3_EXPECTED['dev'],
}


def expected_for(profile):
    return {'S1S2': EXPECTED, 'S3': S3_EXPECTED, 'S4': S4_EXPECTED}[profile]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_records(rows, messages, tokenizer, batch_size):
    if batch_size <= 0 or len(rows) % batch_size:
        raise ValueError('batch would drop records or is invalid')
    ids = [r['id'] for r in rows]
    mids = [r['id'] for r in messages]
    if len(set(ids)) != len(ids) or len(set(mids)) != len(mids) or set(ids) != set(mids):
        raise ValueError('duplicate or mismatched record IDs')
    by_id = {r['id']: r['messages'] for r in messages}
    total = loss = maximum = 0
    for row in rows:
        values = list(row['input_ids'])
        start = row['answer_start']
        # pyarrow supplies native integers; reject coercion of floats/bools.
        if type(start) is not int or any(type(v) is not int for v in values):
            raise ValueError('noninteger token or boundary')
        mask = validate_record(values, start, tokenizer.eos_token_id, 4096)
        msg = by_id[row['id']]
        if len(msg) != 2 or [m['role'] for m in msg] != ['user', 'assistant']:
            raise ValueError('unexpected conversation structure')
        prompt = tokenizer.apply_chat_template(msg[:-1], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
        prefix = tokenizer.encode(prompt, add_special_tokens=False)
        answer = tokenizer.encode(msg[-1]['content'], add_special_tokens=False)
        if len(prefix) != start or prefix + answer + [tokenizer.eos_token_id] != values:
            raise ValueError('stored tokens differ from actual model tokenizer')
        if sum(mask) != row['loss_tokens']:
            raise ValueError('loss token count mismatch')
        total += len(values)
        loss += sum(mask)
        maximum = max(maximum, len(values))
    return dict(count=len(rows), sequence_tokens=total, loss_tokens=loss,
                max_length=maximum, batches=len(rows)//batch_size, dropped_records=0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--profile', choices=['S1S2', 'S3', 'S4'], default='S1S2')
    a = p.parse_args()
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    if digest(a.model/'tokenizer.json') != TOKENIZER_SHA:
        raise ValueError('unexpected tokenizer identity')
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True, local_files_only=True)
    if tok.eos_token_id != 248046:
        raise ValueError('unexpected EOS')
    report = {}
    split_ids = {}
    for split, (count, sequence, loss, sha) in expected_for(a.profile).items():
        path = a.data/(split+'.parquet')
        if digest(path) != sha:
            raise ValueError('unexpected '+split+' parquet identity')
        rows = pq.read_table(path).to_pylist()
        messages = [json.loads(s) for s in (a.data/(split+'.messages.private.jsonl')).read_text(encoding='utf-8').splitlines()]
        result = audit_records(rows, messages, tok, 3)
        if (result['count'], result['sequence_tokens'], result['loss_tokens']) != (count, sequence, loss):
            raise ValueError('unexpected '+split+' budget')
        report[split] = dict(result, parquet_sha256=sha)
        split_ids[split] = {r['id'] for r in rows}
    if split_ids['train'] & split_ids['dev']:
        raise ValueError('train/dev ID overlap')
    report.update(tokenizer_sha256=TOKENIZER_SHA, offline_data_audit_passed=True,
                  distributed_loader_verified=False, training_authorized_by_this_audit=False)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
