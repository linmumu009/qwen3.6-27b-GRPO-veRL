"""Build only released source-authored training records; no evaluation inputs."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
from prepare_cpt_p4_cumulative import OLD_MESSAGES_SHA,OLD_PARQUET_SHA,TOKENIZER_SHA
from prepare_cpt_mechanism import sha,read,budget,save
from qwen36_mcq_answer_dataset import validate_record
from cpt_two_family import variant

DEV_SHA='3a35e5307cd46027b5e91f598917c70c0bc470175ba1de4ec534120edddc41f0'
DEV_MESSAGES_SHA='dea806503ed0a13a0e8218bf0036b7074ea10af26df134ed481e3a8ca821da25'


def released_tasks(tasks_path):
    packet=json.loads(tasks_path.read_text())
    quality_path=tasks_path.parent/'quality.safe.json';author_path=tasks_path.parent/'train_author.v2.private.json'
    quality=json.loads(quality_path.read_text());author=json.loads(author_path.read_text())
    assert quality['passed'] is True and packet['quality_sha256']==sha(quality_path)
    hashes=[value for name,value in quality['inputs'].items() if Path(name.replace('\\','/')).name=='train_author.v2.private.json']
    assert hashes==[sha(author_path)] and packet['tasks']==author['candidates'],'reviewed training content changed'
    assert packet['quality_released'] is True and packet['split']=='train'
    return packet['tasks']


def validate_budgets(d,t,unused=None):
    for b in (d,t):
        assert b['records']==351 and b['sequence_tokens']<=160000 and b['supervised_tokens']<=35000
    for key,limit in [('supervised_tokens',.05),('sequence_tokens',.10)]:
        assert abs(d[key]-t[key])/min(d[key],t[key])<=limit,(key,d[key],t[key])


def prepare(original,tasks_path,model,out):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    assert sha(original/'train.messages.private.jsonl')==OLD_MESSAGES_SHA
    assert sha(original/'train.parquet')==OLD_PARQUET_SHA
    assert sha(original/'dev.parquet')==DEV_SHA and sha(original/'dev.messages.private.jsonl')==DEV_MESSAGES_SHA
    assert sha(model/'tokenizer.json')==TOKENIZER_SHA
    tasks=released_tasks(tasks_path);assert len(tasks)==24 and len({t['id'] for t in tasks})==24
    assert Counter(t['arm'] for t in tasks)=={'D':12,'T':12}
    assert len({t['family'] for t in tasks})==2
    for family in {t['family'] for t in tasks}:
        for pair in range(1,7):
            selected=[t for t in tasks if t['family']==family and t['pair']==pair]
            assert len(selected)==2 and {t['arm'] for t in selected}=={'D','T'}
            assert set(selected[0]['fact_ids'])==set(selected[1]['fact_ids'])
    tok=AutoTokenizer.from_pretrained(model,local_files_only=True)
    messages=read(original/'train.messages.private.jsonl');rows=pq.read_table(original/'train.parquet').to_pylist()
    assert len(messages)==len(rows)==315
    def encode(x):
        m=x['messages'];prefix=tok.encode(tok.apply_chat_template(m[:-1],tokenize=False,add_generation_prompt=True,enable_thinking=False),add_special_tokens=False)
        ids=prefix+tok.encode(m[-1]['content'],add_special_tokens=False)+[tok.eos_token_id]
        mask=validate_record(ids,len(prefix),tok.eos_token_id,4096)
        return dict(id=x['id'],input_ids=ids,answer_start=len(prefix),loss_tokens=sum(mask))
    assert [encode(x) for x in messages]==rows
    built={};budgets={}
    for arm in ('D','T'):
        additions=[]
        for t in sorted((t for t in tasks if t['arm']==arm),key=lambda t:(t['family'],t['pair'])):
            for exposure in range(3):
                v=variant(t,exposure);msg=v['messages'];answer=json.dumps({'answers':v['expected']},separators=(',',':'))
                if exposure:
                    msg[0]['content']='Briefly explain the governing rule and its application, then return one JSON object with key "answers" containing all correct zero-based option indices.'
                    answer='\n'.join(f'Option {i}: '+t['option_reasons'][j] for i,j in enumerate(v['order']))+'\n'+answer
                msg.append(dict(role='assistant',content=answer))
                additions.append(dict(id=f"two-family-{t['id']}-{exposure}",messages=msg))
        extra_rows=[encode(x) for x in additions]
        built[arm]=(messages+additions,rows+extra_rows);budgets[arm]=budget(rows+extra_rows)
    validate_budgets(budgets['D'],budgets['T'])
    out.mkdir(exist_ok=False)
    for arm,(msgs,data) in built.items():
        folder=out/arm;folder.mkdir()
        pq.write_table(pa.Table.from_pylist(data),folder/'train.parquet')
        (folder/'train.messages.private.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in msgs))
        for name in ('dev.parquet','dev.messages.private.jsonl'):shutil.copyfile(original/name,folder/name)
    result=dict(records_per_arm=351,steps_per_arm=117,lr=5e-7,batch=3,epochs=1,seed=1,budgets=budgets,
        training_packet_sha256=sha(tasks_path),original_315_messages_and_tokens_unchanged=True,
        evaluation_inputs_read=False,training_allowed=False,pending='screening and installed-loader gates',
        files={arm:{n:sha(out/arm/n) for n in ['train.parquet','train.messages.private.jsonl','dev.parquet','dev.messages.private.jsonl']} for arm in ('D','T')})
    save(out/'budget.safe.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('original','tasks','model','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();print(json.dumps(prepare(a.original,a.tasks,a.model,a.out),indent=2))
