"""Freeze actual trained opportunities for evaluation only."""
import json,hashlib,os
from pathlib import Path
ROOT=Path('/workspace/llin-verl-grpo')

def main():
    os.umask(0o077)
    p=ROOT/'runs/targeted-book-pilot-prepared-20260908'
    rows=list(map(json.loads,(p/'train.private.jsonl').read_text().splitlines()))
    import pandas as pd
    meta=json.loads((p/'manifest.safe.json').read_text())
    assert hashlib.sha256((p/'train.parquet').read_bytes()).hexdigest()==meta['train_parquet_sha256']
    assert {r['id'] for r in rows}==set(pd.read_parquet(p/'train.parquet')['id'])
    selected=[r for r in rows if r['decision']!='maintenance_pool']
    assert len(rows)==80 and len(selected)==64 and all(r['split']=='train' for r in selected)
    out=ROOT/'runs/targeted-book-train64-20260908';out.mkdir(exist_ok=False)
    f=out/'candidates.private.jsonl';f.write_text(''.join(json.dumps(r)+'\n' for r in selected))
    (out/'summary.safe.json').write_text(json.dumps({'items':64,'candidates_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),
        'actual_training_ids_verified':True,'evaluation_only':True,'not_generalization':True}))

if __name__=='__main__':main()
