"""Rebuild all model results and paired contrasts from retained raw outputs."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
from cpt_list_probe import sha,read,save,digest,specs,prompt,summarize,SEED
from run_cpt_list_probe import TOKENIZER_SHA,MODELS

def independent_index(raw,n):
    try:
        if raw['finish_reason']!='stop':return None
        value=json.loads(raw['text'])
        assert type(value) is dict and list(value)==['answers']
        items=value['answers'];assert type(items) is list and len(items)==1
        index=items[0];assert type(index) is int and index in range(n)
        return index
    except (ValueError,TypeError,KeyError,AssertionError):return None

def verify(packet_dir,package,run,tokenizer):
    from transformers import AutoTokenizer
    manifest=json.loads((packet_dir/'manifest.safe.json').read_text());packet_file=packet_dir/'packet.private.json'
    assert sha(packet_file)==manifest['packet_sha256'];packet=json.loads(packet_file.read_text(encoding='utf-8'))
    package_manifest=json.loads((package/'package.safe.json').read_text())
    for name,h in package_manifest['files'].items():assert sha(package/name)==h
    for name in ('cpt_list_probe.py','run_cpt_list_probe.py'):assert sha(Path(__file__).with_name(name))==sha(package/'scripts'/name)
    assert sha(package/'packet/packet.private.json')==sha(packet_file)
    assert sha(tokenizer/'tokenizer.json')==TOKENIZER_SHA
    tok=AutoTokenizer.from_pretrained(tokenizer,local_files_only=True,trust_remote_code=True)
    plan=specs(packet);tokens=[]
    for s in plan[:192]:
        text=prompt(packet,s)
        ids=tok.encode(tok.apply_chat_template([dict(role='user',content=text)],tokenize=False,add_generation_prompt=True,enable_thinking=False),add_special_tokens=False)
        tokens.append((digest(ids),len(ids)))
    summaries={};raw_hashes={};raws={}
    for label in MODELS:
        folder=run/label;reg=json.loads((folder/'registration.safe.json').read_text())
        assert json.loads((folder/'status.safe.json').read_text())['status']=='completed_pending_local_verification'
        checks=dict(model_label=label,model=MODELS[label].as_posix(),model_manifest_sha256=sha(package/(label+'.model.safe.json')),
            packet_sha256=sha(packet_file),runner_sha256=sha(package/'scripts/run_cpt_list_probe.py'),protocol_sha256=sha(package/'scripts/cpt_list_probe.py'),
            seed=SEED,temperature=0,thinking=False,tp=8,max_model_len=8192,max_tokens=96,max_num_seqs=16,chunk=8,calls=576,repeats=3,identical_batches_repeated=True,training_allowed=False)
        assert reg==checks
        raw=read(folder/'predictions.private.jsonl');summary=summarize(packet,raw)
        assert summary==json.loads((folder/'summary.safe.json').read_text())
        for i,(s,r,d) in enumerate(zip(plan,raw,summary['per_call'])):
            assert (r['prompt_token_sha256'],r['prompt_tokens'])==tokens[i%192]
            pred=independent_index(r,s['size']);assert pred==d['prediction'] and (pred==s['expected'])==d['correct']
        summaries[label]=summary;raws[label]=raw
        for name in ('predictions.private.jsonl','registration.safe.json','summary.safe.json','status.safe.json'):
            raw_hashes[label+'/'+name]=sha(folder/name)
    for label in ('p1','p4'):
        for a,b in zip(raws['step120_current'],raws[label]):
            assert all(a[k]==b[k] for k in ('id','size','position','mode','repeat','order','expected','prompt_text_sha256','prompt_token_sha256','prompt_tokens'))
    contrasts={};between={}
    for label,summary in summaries.items():
        grouped={(d['id'],d['size'],d['position'],d['mode'],d['repeat']):d for d in summary['per_call']}
        counts=defaultdict(Counter)
        for d in summary['per_call']:
            if d['size']==4:continue
            short=grouped[(d['id'],4,d['position'],d['mode'],d['repeat'])]
            counts[d['mode']+'/4_to_'+str(d['size'])].update(pairs=1,short_correct=short['correct'],long_correct=d['correct'],
                losses=short['correct'] and not d['correct'],gains=d['correct'] and not short['correct'])
        contrasts[label]={k:dict(v) for k,v in counts.items()}
    for before,after in (('step120_current','p1'),('p1','p4')):
        counts=defaultdict(Counter)
        for a,b in zip(summaries[before]['per_call'],summaries[after]['per_call']):
            counts[a['mode']+'/'+str(a['size'])].update(pairs=1,before_correct=a['correct'],after_correct=b['correct'],
                gains=b['correct'] and not a['correct'],losses=a['correct'] and not b['correct'])
        between[before+'_to_'+after]={k:dict(v) for k,v in counts.items()}
    for s in summaries.values():del s['per_call']
    return dict(status='completed_verified_local',new_calls=1728,packet_sha256=sha(packet_file),models=summaries,
        within_model_short_long=contrasts,between_models=between,raw_sha256=raw_hashes,
        all_token_fingerprints_verified=True,all_paired_prompts_identical=True,independent_prediction_reparse=True,
        verifier_sha256=sha(Path(__file__)),training_allowed=False,limitations=packet['limitations'])

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('packet','package','run','tokenizer','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();result=verify(a.packet,a.package,a.run,a.tokenizer);save(a.out,result)
    print(json.dumps(dict(status=result['status'],groups={k:v['groups'] for k,v in result['models'].items()}),indent=2))
