"""Read-only training and answer-shape audit; save aggregate evidence only."""
import json,re,statistics,hashlib,math
from pathlib import Path
ROOT=Path('/workspace/llin-verl-grpo')
PILOT=Path('/opt/llin-targeted-book-pilot-20260908')

def main():
    steps={}
    for f in (PILOT/'torchrun_logs').rglob('*.log'):
        for line in f.read_text(errors='replace').splitlines():
            if not re.match(r'^step:\d+ - ',line):continue
            fields=dict(re.findall(r'(step|train/loss|train/grad_norm|train/lr|train/global_tokens):([-+\deE.]+)',line))
            if len(fields)!=5:continue
            row={k:float(v) for k,v in fields.items()};step=int(row['step'])
            if step in steps and steps[step]!=row:raise ValueError('Conflicting step logs')
            steps[step]=row
    assert set(steps)==set(range(1,11))
    assert all(math.isfinite(v) for r in steps.values() for v in r.values())
    data=ROOT/'runs/targeted-book-pilot-prepared-20260908'
    meta=json.loads((data/'manifest.safe.json').read_text())
    import pandas as pd
    from transformers import AutoTokenizer
    from probe_targeted_book_groups import prompt_messages
    from qwen36_mcq_answer_dataset import validate_record
    assert hashlib.sha256((data/'train.parquet').read_bytes()).hexdigest()==meta['train_parquet_sha256']
    rr={r['id']:r for r in map(json.loads,(data/'train.private.jsonl').read_text().splitlines())}
    tt=pd.read_parquet(data/'train.parquet').to_dict('records')
    tokenizer=AutoTokenizer.from_pretrained(meta['source_model'],trust_remote_code=True)
    loss=0;seq=0
    for r in tt:
        q=rr[r['id']];prefix=tokenizer.encode(tokenizer.apply_chat_template(prompt_messages(q,0),tokenize=False,add_generation_prompt=True,enable_thinking=False),add_special_tokens=False)
        expected=prefix+tokenizer.encode(q['answer'],add_special_tokens=False)+[tokenizer.eos_token_id]
        actual=[int(x) for x in r['input_ids']]
        assert actual==expected and r['answer_start']==len(prefix)
        mask=validate_record(actual,len(prefix),tokenizer.eos_token_id,4096)
        assert sum(mask)==r['loss_tokens'];loss+=sum(mask);seq+=len(actual)
    assert len(tt)==len(rr)==80 and seq==15491 and loss==7183
    assert sum(r['train/global_tokens'] for r in steps.values())==seq
    ids={r['id'] for r in map(json.loads,(data/'evaluation/candidates.private.jsonl').read_text().splitlines())}
    base=[]
    for name in ('targeted-book-probe-20260908','targeted-book-supplement-probe-20260908'):
        base+=json.loads((ROOT/'runs'/name/'answers.private.json').read_text())
    answer_sets={'baseline':base,**{f'step{x}':json.loads((PILOT/f'book_probe_step_{x}/answers.private.json').read_text()) for x in (5,10)}}
    length={}
    for model,aa in answer_sets.items():
        values=[r['tokens'] for r in aa if r['id'] in ids and r['condition'].startswith('closed')]
        assert len(values)==110
        length[model]={'count':len(values),'mean':statistics.mean(values),'median':statistics.median(values)}
    safe={'steps':[steps[k] for k in sorted(steps)],'all_logged_metrics_finite':True,
          'sequence_tokens_reconciled':seq,'supervised_tokens_revalidated':loss,'exact_retokenized_records':len(tt),
          'closed_answer_lengths':length,'training_answer_mean_tokens_including_eos':loss/80,
          'limitations':['Different batches per step; loss trend is not a fixed-set before/after comparison.',
                         'Finite logged gradients and correct supervision do not prove full numerical correctness.',
                         'Answer shortening is observed; its causal role is not established.']}
    out=ROOT/'runs/targeted-book-pilot-record-audit-20260908';out.mkdir(exist_ok=False)
    (out/'summary.safe.json').write_text(json.dumps(safe,indent=2));print(json.dumps(safe,indent=2))

if __name__=='__main__':main()
