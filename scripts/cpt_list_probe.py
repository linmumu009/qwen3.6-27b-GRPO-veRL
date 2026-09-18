"""Source-authored nested-list diagnostic. No benchmark input or training output."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

SOURCE_SHA = '5df7dc2b209938986cd7081aecdc9105fb989c9a872d572ce0a0c5a16dbbf8a4'
SEED = 20923
SIZES = (4, 32, 128)
REPEATS = 3
# source position, authored definition/application, three short-list distractors
TASKS = [
    (2, 'A product engineer needs a structured list of the raw materials, components and subassemblies needed to manufacture one product. Which specific concept names this list?', [22,7,16]),
    (11, 'A parcel carrier derives a pricing weight from a package\'s length, width and height instead of using only its scale reading. Which specific concept names this pricing technique?', [119,126,37]),
    (17, 'A process begins at one recorded instant and is completed at another. Which specific concept names the elapsed interval between those two instants?', [7,42,89]),
    (29, 'A tracking installation automatically identifies tags attached to goods by means of electromagnetic fields. Which specific identification technology is described?', [38,16,45]),
    (37, 'Several individual items are combined into one assemblage for easier storage and handling. The concept applies to a pallet load as well as a container load and is not restricted to aircraft equipment. Which specific concept names this assemblage?', [93,4,5]),
    (38, 'Warehouse workers receive spoken directions and interact through speech-recognition software. Which specific warehouse technology names this interaction method?', [29,45,40]),
    (44, 'An inventory classification assigns very tight controls and accurate records to class A, moderate controls to B, and simple controls with minimal records to C. Which specific classification method is described?', [28,20,78]),
    (45, 'Computer-controlled equipment automatically places loads into defined storage locations and retrieves them from those locations. Which specific type of system is described?', [40,16,38]),
]

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(x): return hashlib.sha256(json.dumps(x,separators=(',',':')).encode()).hexdigest()
def read(p): return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]
def save(p,obj): p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def build(source):
    assert sha(source)==SOURCE_SHA
    sources=read(source);bank=sources[:128]
    assert len(bank)==128 and len({s['title'].casefold() for s in bank})==128
    rows=[]
    for target,question,core in TASKS:
        assert len(set([target,*core]))==4
        remaining=[i for i in range(128) if i not in [target,*core]]
        random.Random(SEED+target).shuffle(remaining)
        rows.append(dict(id='list-probe-'+str(target),question=question,target=target,
            target_label=bank[target]['title'],distractors=core+remaining,
            source={k:sources[target][k] for k in ('id','title','scope','source_text','evidence','proposed_split')},
            training_allowed=False))
    return dict(version='list-probe-v1',rows=rows,bank=[dict(index=i,id=s['id'],label=s['title']) for i,s in enumerate(bank)],
        source_sha256=SOURCE_SHA,sizes=list(SIZES),repeats=REPEATS,seed=SEED,training_allowed=False,
        limitations=['Eight familiar concepts and one heterogeneous bilingual vocabulary, not population-level evidence.',
            'Source-authored questions; sources may have appeared in earlier training. No knowledge novelty claim.',
            'Nested distractor sets change length and interference together; this is not pure token-length isolation.',
            'Relative answer positions are matched, absolute indices differ by size except position zero.',
            '128 options do not reproduce the formal 269-option material-handling vocabulary.',
            'Exact-label mode deliberately supplies the answer term; diagnostic control only, never training.',
            'All cases are evaluation-only; no formal question or label is used for authoring.'])

def specs(packet):
    base=[]
    for row in packet['rows']:
        for n in SIZES:
            for position,index in enumerate((0,n//3,2*n//3,n-1)):
                order=row['distractors'][:n-1].copy();order.insert(index,row['target'])
                for mode in ('concept','exact_label'):
                    base.append(dict(id=row['id'],size=n,position=position,expected=index,order=order,mode=mode))
    random.Random(SEED).shuffle(base)
    return [dict(s,repeat=repeat) for repeat in range(REPEATS) for s in base]

def prompt(packet,s):
    row=next(r for r in packet['rows'] if r['id']==s['id'])
    question=row['question'] if s['mode']=='concept' else 'Find the option whose complete text is exactly '+json.dumps(row['target_label'],ensure_ascii=False)+'.'
    return ('Return only a JSON object with key "answers" containing exactly one zero-based option index. Select the single best matching specific entry.\nQuestion:\n'
        +question+'\nOptions:\n'+'\n'.join(f'[{i}] {packet["bank"][j]["label"]}' for i,j in enumerate(s['order'])))

def answer(text,n):
    try:
        value=json.loads(text)
        if not isinstance(value,dict) or set(value)!={'answers'}:return None
        a=value['answers']
        if not isinstance(a,list) or len(a)!=1 or type(a[0]) is not int or not 0<=a[0]<n:return None
        return a[0]
    except (ValueError,TypeError):return None

def summarize(packet,raw):
    plan=specs(packet);assert len(plan)==len(raw)==576
    totals=defaultdict(Counter);by_prompt=defaultdict(list);details=[]
    for s,r in zip(plan,raw):
        assert all(s[k]==r[k] for k in s)
        assert r['prompt_text_sha256']==hashlib.sha256(prompt(packet,s).encode()).hexdigest()
        assert 0<r['output_tokens']<=96 and r['prompt_tokens']+96<=8192
        pred=answer(r['text'],s['size']) if r['finish_reason']=='stop' else None
        d={k:s[k] for k in ('id','size','position','mode','repeat')}
        d.update(correct=pred==s['expected'],prediction=pred,expected=s['expected'],invalid=pred is None,truncated=r['finish_reason']!='stop')
        details.append(d);key=(s['id'],s['size'],s['position'],s['mode']);by_prompt[key].append(d)
        for group in (s['mode']+'/'+str(s['size']),s['mode']+'/'+str(s['size'])+'/position'+str(s['position'])):
            totals[group].update(calls=1,correct=d['correct'],invalid=d['invalid'],truncated=d['truncated'])
    stability=[]
    for key,ds in by_prompt.items():
        assert len(ds)==3
        stability.append(dict(zip(('id','size','position','mode'),key),correct_repeats=sum(d['correct'] for d in ds),
            invariant=len({d['prediction'] for d in ds})==1,all_valid=all(not d['invalid'] for d in ds)))
    return dict(calls=576,groups={k:dict(v) for k,v in totals.items()},identical_prompt_groups=192,
        unstable_identical_prompts=sum(not d['invariant'] for d in stability),per_prompt=stability,per_call=details,training_allowed=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    packet=build(a.source);a.out.mkdir(exist_ok=False);save(a.out/'packet.private.json',packet)
    save(a.out/'manifest.safe.json',dict(packet_sha256=sha(a.out/'packet.private.json'),source_sha256=SOURCE_SHA,
        concepts=8,unique_prompts_per_model=192,calls_per_model=576,total_calls=1728,sizes=list(SIZES),repeats=3,
        authoring_script_sha256=sha(Path(__file__)),training_allowed=False,limitations=packet['limitations']))
