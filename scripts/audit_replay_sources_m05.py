"""Host-only source provenance audit; does not export training messages."""
import json,hashlib,sys,os
from pathlib import Path
from collections import Counter
ROOT=Path('/data3/llin/qwen3.6-27b-verl-grpo')
sys.path.insert(0,str(ROOT))
from llin_verl.opensource_reward import extract_explicit_answer

def main():
    os.umask(0o077)
    p=ROOT/'data/step120_opensource_20260824';f=p/'opensource_step120_train.jsonl'
    quality=json.loads((p/'opensource_step120_quality_report.json').read_text())
    assert hashlib.sha256(f.read_bytes()).hexdigest()==quality['train_jsonl_sha256']
    rows=list(map(json.loads,f.read_text().splitlines()));assert len(rows)==80
    sources={};status=Counter();datasets=Counter()
    for r in rows:
        e=r['extra_info'];path=Path(e['source_path']);ds=e['dataset'];datasets[ds]+=1
        if not path.exists():status['missing_source']+=1;continue
        if str(path) not in sources:
            raw=path.read_bytes();sources[str(path)]={'sha256':hashlib.sha256(raw).hexdigest(),'rows':raw.decode().splitlines()}
        source=json.loads(sources[str(path)]['rows'][e['source_row']]);gt=r['reward_model']['ground_truth']['answer']
        if ds=='MATH':answer,found=extract_explicit_answer(source.get('solution',''));solution=source.get('solution','')
        elif ds=='GSM8K':solution=source.get('answer','');answer=solution.rsplit('####',1)[-1].strip();found='####' in solution
        else:answer=str(source.get('answer',''));found=bool(answer);solution=source.get('solution',source.get('explanation',''))
        status['source_answer_matches' if found and str(answer).strip()==str(gt).strip() else 'answer_needs_review']+=1
        status['has_solution_text']+=bool(str(solution).strip())
    agent=json.loads((ROOT/'data/repair_sft_train236_20260811/contract.json').read_text())
    safe={'opensource_rows':80,'datasets':dict(datasets),'checks':dict(status),
        'source_file_hashes':{k:v['sha256'] for k,v in sources.items()},
        'historical_train_jsonl_sha256':quality['train_jsonl_sha256'],
        'agent_candidate_rows':agent['rows'],'agent_semantic_warning_rows':agent['semantic_warning_rows'],
        'agent_contract_promotion_allowed':agent['promotion_allowed'],
        'replay_training_ready':False,'training_exported':False,
        'limits':['Source final-answer consistency is not solution reasoning validation.',
          'Historical train-only provenance is not a newly verified full benchmark decontamination audit.',
          'Agent warning rows and separate held-out regression require validation; textbook maintenance is not Agent replay.']}
    out=ROOT/'runs/sft-next-replay-audit-20260908';out.mkdir(exist_ok=False)
    (out/'summary.safe.json').write_text(json.dumps(safe,indent=2));print(json.dumps(safe,indent=2))

if __name__=='__main__':main()
