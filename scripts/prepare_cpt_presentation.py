"""Freeze a prefix x existing candidate-pool diagnostic, without generating data."""
import hashlib
import json
from pathlib import Path
from audit_cpt_transfer import CASE_SHA, read_rows, verify_predictions, verify_protocols
from build_cpt_p1_error_map import matching_pairs
from evaluate_logistics_knowledge import EvalItem, build_messages

BASE = Path('CPT_resources/llin-presentation-20260923-01')
MAP = Path('CPT_resources/llin-p1-error-map-20260923-01')
HISTORY = Path('CPT_resources/llin-transfer-formal-20260917-01/p1')
CASES = Path('CPT_resources/llin-transfer-audit-20260915-01/frozen_cases.private.jsonl')
MARKER = 'Which of the following best matches:'


def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(x): return hashlib.sha256(json.dumps(x, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
def item(c): return EvalItem(**{k:c[k] for k in EvalItem.__dataclass_fields__})
def freeze(p, x):
    text = json.dumps(x, ensure_ascii=False, indent=2) + '\n'
    if p.exists():
        if p.read_text(encoding='utf-8') != text: raise ValueError('Frozen file differs: ' + str(p))
    else: p.write_text(text, encoding='utf-8')


def crossed(large, small):
    lp, ld = large['question'].split(MARKER, 1)
    sp, sd = small['question'].split(MARKER, 1)
    if ld != sd: raise ValueError('Definition bytes differ')
    if sorted(large['options'][i] for i in large['expected']) != sorted(small['options'][i] for i in small['expected']):
        raise ValueError('Answer text differs')
    result = {}
    for prefix, value in [('L', lp), ('S', sp)]:
        for pool, case in [('L', large), ('S', small)]:
            item = dict(case, question=value + MARKER + ld)
            messages = build_messages(EvalItem(**{k:item[k] for k in EvalItem.__dataclass_fields__}))
            result[prefix+pool] = dict(messages=messages, expected=case['expected'], options=case['options'])
    return result


def main():
    BASE.mkdir(parents=True, exist_ok=True)
    assert sha(CASES) == CASE_SHA
    cases_list = read_rows(CASES); cases = {c['item_hash']: c for c in cases_list}
    raw = read_rows(HISTORY/'predictions.private.jsonl')
    scores = verify_predictions(cases, raw, read(HISTORY/'scores.safe.json'))
    protocol = read(HISTORY/'protocol.safe.json'); verify_protocols({'p1': protocol}, CASE_SHA, 1672)
    queue = read(MAP/'decision_queues.private.json')['queues']['presentation_control_first']
    ledger = {r['item_hash']: r for r in read(MAP/'ledger.private.json')}
    assert len(queue) == len(set(queue)) == 25 and all(ledger[k]['source_status']=='A' for k in queue)
    pairs = sorted(matching_pairs(cases)); chosen = []
    for a in sorted(queue):
        eligible = [b for aa,b in pairs if aa==a and scores[b]['correct']]
        assert eligible and not scores[a]['correct']
        chosen.append(('target', a, min(eligible)))
    seen = set()
    for a,b in pairs:
        if a not in seen and scores[a]['correct'] and scores[b]['correct']:
            chosen.append(('retention', a, b)); seen.add(a)
            if len(seen)==25: break
    assert len(chosen)==50
    token_hashes = {c['item_hash']:h for c,h in zip(cases_list,protocol['prompt_hashes'])}
    raw_by = {k:[r for r in raw if r['item_hash']==k] for k in cases}
    groups=[]; requests=[]; reuse=[]
    for cohort,a,b in chosen:
        conditions=crossed(cases[a],cases[b]); group_id=a
        groups.append(dict(id=group_id,cohort=cohort,large=a,small=b,conditions=conditions,
            source_review=ledger[a]['source_review'] if cohort=='target' else None,
            retention_labels_independently_revalidated=False if cohort=='retention' else None))
        for condition,c in conditions.items():
            if condition in ('LL','SS'):
                key=a if condition=='LL' else b
                assert c['messages']==build_messages(item(cases[key]))
                reuse.append(dict(group=group_id,condition=condition,messages=c['messages'],
                    historical_prompt_token_sha256=token_hashes[key],rows=raw_by[key]))
            else:
                for repeat in range(3):
                    requests.append(dict(id=f'{group_id}:{condition}:{repeat}',group=group_id,cohort=cohort,
                        condition=condition,repeat=repeat,messages=c['messages']))
    assert len(requests)==300 and len(reuse)==100 and sum(len(r['rows']) for r in reuse)==300
    freeze(BASE/'packet.private.json',dict(groups=groups,requests=requests,reuse=reuse,training_allowed=False))
    old=read(Path('CPT_resources/llin-remaining-operations-20260923-01/execution.safe.json'))
    reg=dict(id='llin-presentation-20260923-01',max_calls=300,reused_calls=300,groups=50,targets=25,retention=25,
        conditions={'LL':'original large prefix, full original pool; reuse','SS':'original small prefix, original small pool; reuse',
                    'LS':'original large prefix, original small pool; new','SL':'original small prefix, full original pool; new'},
        selection='All 25 source-A discordant pairs; 25 both-correct pairs sorted by large hash then small hash, unique large IDs.',
        packet_sha256=sha(BASE/'packet.private.json'),runner_sha256=sha(Path('scripts/run_cpt_presentation.py')),
        inputs={str(p):sha(p) for p in [CASES,MAP/'decision_queues.private.json',MAP/'ledger.private.json',
            HISTORY/'predictions.private.jsonl',HISTORY/'protocol.safe.json',HISTORY/'scores.safe.json']},
        protocol_code_sha256=old['protocol_code_sha256'],model_path=protocol['model'],training_allowed=False,
        repeats=3,temperature=0,seed=1024,max_tokens=96,max_model_len=8192,tp=8,max_num_seqs=32,
        primary='Strict-majority target repair and retention loss for both prefix effects and both pool effects; list all four cells.',
        limitations=['Development-selected pairs; no generalization claim','Pool size, distractors and order change together',
                    'Small prefix provides domain cues; this is replacement, not neutral removal',
                    'Historical baselines reused; batch/runtime timing differences remain','Retention uses existing labels, not a new source audit'],
        training_gate='Original Stage C unchanged; this diagnostic alone never opens training.')
    freeze(BASE/'execution.safe.json',reg)
    freeze(Path('docs/cpt_presentation_registration_20260923.safe.json'),reg)
    print(json.dumps({k:reg[k] for k in ['id','max_calls','reused_calls','targets','retention','packet_sha256']}))


if __name__=='__main__':main()
