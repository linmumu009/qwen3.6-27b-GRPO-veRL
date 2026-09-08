"""Partition reviewed candidate rows; explicitly NOT training-ready export."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

OPPORTUNITIES={'knowledge_access_opportunity','mixed_opportunity','application_opportunity'}


def pool(row):
    d=row['decision']
    if d in OPPORTUNITIES and row['split']=='train':return 'opportunity'
    if d=='maintenance_pool' and row['split']=='train':return 'maintenance'
    if d=='development_only' and row['split']=='dev':return 'development'
    return 'quarantine'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();os.umask(0o077)
    rows=list(map(json.loads,a.input.read_text().splitlines()))
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate IDs')
    groups={}
    for r in rows:groups.setdefault(r['concept_group'],set()).add(r['split'])
    if any(len(s)!=1 for s in groups.values()):raise ValueError('Cross-split concept group')
    a.output.mkdir(parents=True,exist_ok=False)
    counts=Counter(pool(r) for r in rows);hashes={}
    for name in ('opportunity','maintenance','development','quarantine'):
        path=a.output/f'{name}.private.jsonl'
        with path.open('x') as f:
            for r in rows:
                if pool(r)==name:f.write(json.dumps(r)+'\n')
        hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    safe={'items':len(rows),'pools':dict(counts),'hashes':hashes,'input_sha256':hashlib.sha256(a.input.read_bytes()).hexdigest(),
      'training_ready':False,'training_started':False,'messages_training_export_created':False,
      'limitations':['Automatic learning-opportunity proxy, not measured training gain.',
         'Post-grading exclusions can leave incomplete concept groups; no training export until reviewed.',
         'This partitioner does not establish semantic deduplication or coverage; consult the upstream screening and merged-selection summaries.']}
    (a.output/'summary.safe.json').write_text(json.dumps(safe,indent=2));print(json.dumps(safe,indent=2))


if __name__=='__main__':main()
