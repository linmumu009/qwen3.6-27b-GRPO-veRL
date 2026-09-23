"""Release source-authored tasks only after independent labels and separation pass."""
import argparse
from collections import Counter
import json
import re
from pathlib import Path
from cpt_two_family import BASE,read,sha,digest,freeze,validate,screen_packet,FORMS


def texts(value):
    if isinstance(value,dict):
        if isinstance(value.get('question'),str) and isinstance(value.get('options'),list):yield value
        for v in value.values():yield from texts(v)
    elif isinstance(value,list):
        for v in value:yield from texts(v)


def grams(s):
    words=re.findall(r'\w+',s.casefold())
    return {tuple(words[i:i+5]) for i in range(max(0,len(words)-4))}


def audit():
    sources=read(BASE/'source_only.private.json');families={f['family_id']:f for f in sources['families']}
    train=read(BASE/'train_author.v2.private.json');dev=read(BASE/'transfer_author.private.json')
    tasks=train['candidates']+dev['items'];review=read(BASE/'blind_review.v2.private.json')
    judgments={r['id']:r for r in review['items']}
    assert len(tasks)==len(judgments)==32 and {t['id'] for t in tasks}==set(judgments)
    faults=[]
    for t in tasks:
        validate(t,families[t['family']]['evidence_text'])
        r=judgments[t['id']]
        checks=['supported','unambiguous','self_contained','scope_preserved','not_answer_leaking','design_satisfied','pass']
        if not all(r[k] is True for k in checks) or sorted(r['derived_indices'])!=t['correct_indices']:faults.append(t['id'])
    for f in families:
        assert Counter(t['form'] for t in dev['items'] if t['family']==f)=={form:1 for form in FORMS}
        for pair in range(1,7):
            group=[t for t in train['candidates'] if t['family']==f and t['pair']==pair]
            assert len(group)==2 and {t['arm'] for t in group}=={'D','T'}
            assert set(group[0]['fact_ids'])==set(group[1]['fact_ids'])
    old={t['id']:t for t in read(BASE/'train_author.private.json')['candidates']}
    changed={t['id'] for t in train['candidates'] if t!=old[t['id']]}
    assert changed=={'TRAIN-U-T1','TRAIN-U-T3','TRAIN-U-T6','TRAIN-R-T2'}
    assert all(t['fact_ids']==old[t['id']]['fact_ids'] for t in train['candidates'])
    # A fixed historical scan, not a correctness-based selection step.
    history_paths=[Path('CPT_resources/llin-transfer-audit-20260915-01/frozen_cases.private.jsonl')]
    for folder in ['llin-logistika-independent-20260922-01','llin-remaining-operations-20260923-01']:
        base=Path('CPT_resources')/folder
        history_paths.extend(sorted(base.glob('author*.private.json')))
        history_paths.extend(sorted(base.glob('diagnostic.private.json')))
    corpus=[]
    for path in history_paths:
        value=[json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()] if path.suffix=='.jsonl' else read(path)
        corpus.extend((str(path),r['question'],grams(r['question'])) for r in texts(value))
    hits=[]
    for t in tasks:
        g=grams(t['question'])
        for path,text,h in corpus:
            score=len(g&h)/len(g|h) if g|h else 0
            if score>=.6:hits.append(dict(id=t['id'],history=path,jaccard=score,history_question_sha256=digest(text)))
    conflicts=review['cross_split_conflicts']
    passed=not faults and not conflicts and not hits
    report=dict(passed=passed,tasks=32,training_tasks=24,transfer_tasks=8,content_failures=len(faults),
        cross_split_conflicts=len(conflicts),historical_near_duplicate_hits=len(hits),history_candidates=len(corpus),
        v1_cross_split_conflicts=len(read(BASE/'blind_review.private.json')['cross_split_conflicts']),
        pre_generation_quality_amendments=1,student_calls_at_release=0,training_allowed=False,
        inputs={str(p):sha(p) for p in [BASE/'source_only.private.json',BASE/'train_author.private.json',BASE/'train_author.v2.private.json',
            BASE/'transfer_author.private.json',BASE/'blind_review.v2.private.json',Path('docs/cpt_two_family_registration_20260923.md')]},
        historical_inputs={str(p):sha(p) for p in history_paths},
        limitation='Five-word overlap scan is not semantic proof; independent cross-split review remains required.')
    freeze(BASE/'quality.safe.json',report)
    freeze(BASE/'quality_details.private.json',dict(faults=faults,conflicts=conflicts,history_hits=hits))
    freeze(Path('docs/cpt_two_family_quality_20260923.safe.json'),report)
    if passed:
        freeze(BASE/'train_released.private.json',dict(quality_released=True,split='train',quality_sha256=sha(BASE/'quality.safe.json'),tasks=train['candidates']))
        packet=screen_packet(dev['items'],families);freeze(BASE/'packet.private.json',packet)
        previous=read(Path('CPT_resources/llin-presentation-20260923-01/execution.safe.json'))
        reg=dict(id='llin-two-family-screen-20260923-01',max_calls=80,training_allowed=False,quality_released=True,
            quality_sha256=sha(BASE/'quality.safe.json'),packet_sha256=sha(BASE/'packet.private.json'),
            runner_sha256=sha(Path('scripts/run_cpt_two_family_screen.py')),protocol_code_sha256=previous['protocol_code_sha256'],
            model_path=previous['model_path'],repeats='4 rotations plus two repeated first-order closed prompts',
            temperature=0,seed=1024,max_tokens=96,max_model_len=8192,tp=8,max_num_seqs=32,
            registered_plan_sha256=sha(Path('docs/cpt_two_family_registration_20260923.md')))
        freeze(BASE/'execution.safe.json',reg);freeze(Path('docs/cpt_two_family_screen_registration_20260923.safe.json'),reg)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':audit()
