"""Freeze DB/documents only and export DB schema into a new evaluation sandbox."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    sys.path.insert(0,str(a.root))
    import pyarrow.parquet as pq
    from scripts.pi_runtime_preflight import validate_dataset_runtime_environments
    data=a.root/'data/boss_v15_dwh_full276_20260806/dataset'
    rows={s:pq.read_table(data/f'boss_pi_{s}.parquet').to_pylist() for s in ('train','val','test')}
    ids={s:{x['extra_info']['instruction_sha256'] for x in rs} for s,rs in rows.items()}
    assert [len(rows[s]) for s in ('train','val','test')]==[236,20,20]
    assert all(len(ids[s])==len(rows[s]) and '' not in ids[s] for s in rows)
    assert not (ids['train']&ids['val'] or ids['train']&ids['test'] or ids['val']&ids['test'])
    envs={x['extra_info']['environment_id'] for x in rows['test']}
    assert envs=={'sft/20260628_v15'}
    source=Path('/pi_sandbox/sft/20260628_v15').resolve(strict=True)
    files=[source/'logistics.sqlite']+sorted(x for x in (source/'documents').rglob('*') if x.is_file())
    assert all(not x.is_symlink() and x.resolve().is_relative_to(source) for x in files)
    assert not (source/'logistics.sqlite-wal').exists()
    assert sum(x.stat().st_size for x in files)<100_000_000
    before={str(x.relative_to(source)):digest(x) for x in files}
    a.out.mkdir(parents=True,exist_ok=False)
    target=a.out/'sandbox/sft/20260628_v15';target.mkdir(parents=True)
    for x in files:
        dest=target/x.relative_to(source);dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(x,dest)
        assert digest(dest)==before[str(x.relative_to(source))]
    assert all(digest(x)==before[str(x.relative_to(source))] for x in files)
    db=sqlite3.connect(f'file:{(target/"logistics.sqlite").as_posix()}?mode=ro',uri=True)
    try:
        schema=[x[0] for x in db.execute("SELECT sql FROM sqlite_master WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%' ORDER BY type,name")]
    finally:db.close()
    assert schema
    (target/'schema_dictionary.md').write_text('# Database schema\n\nDeterministic SQLite DDL export; no rows or task answers.\n\n```sql\n'+';\n'.join(schema)+';\n```\n')
    # Private source manifest stays on server. Safe report includes only aggregate digests.
    (a.out/'source_manifest.private.json').write_text(json.dumps(before,indent=2))
    preflight=validate_dataset_runtime_environments(data/'boss_pi_test.parquet',a.out/'sandbox')
    safe=dict(status='isolated_sandbox_preflight_passed',preflight=preflight,
              source_files=len(files),source_bytes=sum(x.stat().st_size for x in files),
              source_manifest_sha256=digest(a.out/'source_manifest.private.json'),
              database_sha256=digest(target/'logistics.sqlite'),schema_sha256=digest(target/'schema_dictionary.md'),
              dataset_sha256={s:digest(data/f'boss_pi_{s}.parquet') for s in rows},
              split_sizes={s:len(rs) for s,rs in rows.items()},instruction_overlap=0,
              shared_sandbox_modified=False,generated_reports_excluded=True,
              historical_protocol_exact_reproduction=False,model_inference_run=False,
              required_runtime_env={'PI_AGENT_SANDBOX_LOWER':str(a.out/'sandbox')})
    (a.out/'summary.safe.json').write_text(json.dumps(safe,indent=2));print(json.dumps(safe,indent=2))


if __name__=='__main__':main()
