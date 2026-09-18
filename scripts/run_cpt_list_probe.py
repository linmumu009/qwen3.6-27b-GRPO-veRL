"""Fixed, repeated nested-list measurement for three frozen models."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from cpt_list_probe import sha,save,digest,specs,prompt,summarize,SEED

ROOT=Path('/workspace/llin-verl-grpo')
MODELS={
 'step120_current':ROOT/'runs/llin-step120-opensource-20260825-02/hf_export_step120_opensource',
 'p1':ROOT/'runs/llin-transfer-p1-run-20260916-02/training/llin-step120-p1-hf',
 'p4':ROOT/'runs/llin-transfer-p4-run-20260918-01/training/llin-step120-p4-hf'}
TOKENIZER_SHA='06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523'

def identity(model):
    index=json.loads((model/'model.safetensors.index.json').read_text());shards=sorted(set(index['weight_map'].values()))
    assert len(index['weight_map'])==1199 and len(shards)==15
    out=dict(path=str(model),metadata_sha256={},shards_stat={},weight_content_hashed=False)
    for name in ('config.json','generation_config.json','tokenizer.json','tokenizer_config.json','model.safetensors.index.json'):
        out['metadata_sha256'][name]=sha(model/name)
    assert out['metadata_sha256']['tokenizer.json']==TOKENIZER_SHA
    for name in shards:
        assert Path(name).name==name
        stat=(model/name).stat();out['shards_stat'][name]=dict(bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)
    return out

def run(packet_dir,model_manifest,out,label):
    import fcntl
    manifest=json.loads((packet_dir/'manifest.safe.json').read_text());path=packet_dir/'packet.private.json'
    assert sha(path)==manifest['packet_sha256']
    packet=json.loads(path.read_text());assert packet['training_allowed'] is False
    frozen=json.loads(model_manifest.read_text());model=MODELS[label];assert identity(model)==frozen
    os.umask(0o077);out.mkdir(exist_ok=False)
    def status(stage,**kwargs):
        tmp=out/'status.tmp';save(tmp,dict(status=stage,training_allowed=False,**kwargs));tmp.replace(out/'status.safe.json')
    status('waiting_for_lock')
    try:
        with (ROOT/'runs/.logistics-exam-cpt.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            from transformers import AutoTokenizer
            from vllm import LLM,SamplingParams
            tok=AutoTokenizer.from_pretrained(model,trust_remote_code=True,local_files_only=True)
            plan=specs(packet);texts=[prompt(packet,s) for s in plan];prompts=[]
            for text in texts:
                ids=tok.encode(tok.apply_chat_template([dict(role='user',content=text)],tokenize=False,add_generation_prompt=True,enable_thinking=False),add_special_tokens=False)
                assert len(ids)+96<=8192;prompts.append(dict(prompt_token_ids=ids))
            assert prompts[:192]==prompts[192:384]==prompts[384:]
            save(out/'registration.safe.json',dict(model_label=label,model=str(model),model_manifest_sha256=sha(model_manifest),
                packet_sha256=sha(path),runner_sha256=sha(Path(__file__)),protocol_sha256=sha(Path(__file__).with_name('cpt_list_probe.py')),
                seed=SEED,temperature=0,thinking=False,tp=8,max_model_len=8192,max_tokens=96,max_num_seqs=16,chunk=8,
                calls=576,repeats=3,identical_batches_repeated=True,training_allowed=False))
            status('loading_model')
            llm=LLM(model=str(model),tensor_parallel_size=8,dtype='bfloat16',trust_remote_code=True,max_model_len=8192,
                max_num_seqs=16,gpu_memory_utilization=.8,seed=SEED,enforce_eager=True)
            raw=[]
            with (out/'predictions.private.jsonl').open('x',encoding='utf-8') as f:
                for first in range(0,len(plan),8):
                    batch=prompts[first:first+8]
                    outputs=sorted(llm.generate(batch,SamplingParams(temperature=0,max_tokens=96,seed=SEED)),key=lambda x:int(x.request_id))
                    assert len(outputs)==len(batch)
                    for i,o in enumerate(outputs,first):
                        ids=prompts[i]['prompt_token_ids'];assert list(o.prompt_token_ids)==ids;v=o.outputs[0]
                        r=dict(plan[i],text=v.text,finish_reason=v.finish_reason,output_tokens=len(v.token_ids),prompt_tokens=len(ids),
                            prompt_text_sha256=hashlib.sha256(texts[i].encode()).hexdigest(),prompt_token_sha256=digest(ids))
                        raw.append(r);f.write(json.dumps(r,ensure_ascii=False)+'\n')
                    f.flush();status('running',completed=len(raw),total=576)
            assert identity(model)==frozen
            save(out/'summary.safe.json',summarize(packet,raw));status('completed_pending_local_verification')
    except BaseException as e:status('failed',error=type(e).__name__+': '+str(e));raise

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('packet','model-manifest','out'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--model-label',choices=list(MODELS),required=True);a=p.parse_args();run(a.packet,a.model_manifest,a.out,a.model_label)
