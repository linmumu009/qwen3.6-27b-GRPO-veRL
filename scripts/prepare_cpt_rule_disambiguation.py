"""Freeze three bounded diagnostic contrasts; reuse four exact historical cells."""
import hashlib
import json
from pathlib import Path
from transformers import AutoTokenizer


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    private = Path('CPT_resources')
    out = private/'llin-rule-disambiguation-20260922-01'
    assert not (out/'cases.private.json').exists(), 'Frozen inputs already exist'
    instructions = read(out/'instructions.private.json')
    previous = private/'llin-context-bridge-20260922-01'
    old = read(previous/'cases.private.json')
    source = read(private/'llin-source-nll-20260922-01/cases.private.json')
    tokdir = private/'llin-transfer-p4-tokenizer-20260918-01'
    tok = AutoTokenizer.from_pretrained(tokdir, local_files_only=True)
    cases = []

    def add(module, condition, template, messages, reuse=False):
        ids = tok.encode(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                         enable_thinking=False), add_special_tokens=False)
        for repeat in range(3):
            cases.append(dict(id=f'{module}-{condition}-{repeat}', module=module, condition=condition,
                repeat=repeat, unit_id=template['unit_id'], fact_index=template['fact_index'],
                required=template['required'], target_sentence=template['target_sentence'],
                messages=messages, prompt_ids=ids, reused_case_id=template['id'] if reuse and repeat==0 else None))
        if reuse:
            assert ids == template['prompt_ids'] and messages == template['messages']

    rail = next(c for c in old if c['unit_id'].endswith('284682e6d05c9f063d99') and c['condition']=='source')
    original = next(c for c in source if c['unit_id']==rail['unit_id'])['messages']
    for cond in ('broad','focused'):
        msgs = json.loads(json.dumps(original))
        if cond == 'focused':
            msgs[-1]['content'] += ' '+instructions['rail_focus']
        msgs[-1]['content'] += ' Answer in one sentence.'
        add('rail', cond, rail, msgs)
    barge = next(c for c in old if c['unit_id'].endswith('031182c8e97a128c8c31') and c['condition']=='source')
    add('barge', 'ordinary', barge, barge['messages'], reuse=True)
    msgs = json.loads(json.dumps(barge['messages']))
    msgs[-1]['content'] += ' '+instructions['fine_classification']
    add('barge', 'per_class', barge, msgs)
    load = next(c for c in old if c['unit_id'].endswith('9adae425cbbabbd2bf0c') and c['condition']=='self')
    add('load', 'original_note', load, load['messages'], reuse=True)
    body, note = load['messages'][2]['content'].rsplit('\n\nNote: ',1)
    assert len(tok.encode('Note: '+note,add_special_tokens=False)) == len(tok.encode(instructions['neutral_note'],add_special_tokens=False)) == 25
    for condition, context in [('neutral_note',body+'\n\n'+instructions['neutral_note']),('removed_note',body)]:
        msgs = json.loads(json.dumps(load['messages']))
        msgs[2]['content'] = context
        add('load', condition, load, msgs)
    neutral = next(c for c in cases if c['module']=='load' and c['condition']=='neutral_note')
    assert len(neutral['prompt_ids']) == len(load['prompt_ids'])
    assert len(cases)==21 and sum(c['reused_case_id'] is None for c in cases)==19
    p = out/'cases.private.json'
    p.write_text(json.dumps(cases,indent=2)+'\n',encoding='utf-8')
    prior = read(Path('docs/cpt_context_bridge_results_20260922.safe.json'))
    reg = dict(id=out.name,date='2026-09-22',models=['K','R'],max_new_response_calls=38,
        reused_calls=4,total_observations=42,repeats=3,max_new_response_tokens=14592,
        probability_calls=0,training_steps=0,formal_reruns=0,promotion=False,
        cases_sha256=sha(p),instructions_sha256=sha(out/'instructions.private.json'),
        preparation_sha256=sha(Path(__file__)),runner_sha256=sha(Path('scripts/run_cpt_rule_disambiguation.py')),
        chat_template_sha256=sha(tokdir/'chat_template.jinja'),
        prior_cases_sha256=sha(previous/'cases.private.json'),prior_raw_sha256=prior['raw_sha256'],
        prior_ratings_sha256=prior['ratings_sha256'],
        contrasts=dict(rail='No assistant history; same original question and one-sentence instruction, add only target-topic focus.',
            barge='Same source context and ordinary question; add only per-class answer precision request.',
            load='Same K historical context and question; keep original note, replace it with a 25-token neutral glossary note, or remove it. Original and neutral entire prompt token counts identical.'),
        protocol=dict(tp=8,dtype='bfloat16',temperature=0,seed=1024,max_model_len=8192,
            max_num_seqs=32,max_tokens=384,enable_thinking=False,enable_prefix_caching=False),
        scoring='Frozen target fact complete, no contradiction or uncertainty. Review new responses with model and condition labels hidden; unchanged prior scores reused. Freeze before reveal. Report truncation separately.',
        decision_rules=dict(rail='Candidate cue sensitivity if focused>=2/3 and broad<=1/3; compare K and R; otherwise mixed or no such signal.',
            barge='Candidate output-granularity sensitivity if per_class>=2/3 and ordinary<=1/3; no claim of new knowledge.',
            load='If original<=1/3 and both neutral and removed>=2/3: candidate note-content effect. If only removed>=2/3: removal/length confounded. If original>=2/3: earlier failure not stable. Otherwise mixed.'),
        limitations='Engineering decision gates, not significance tests. Repeats probe inference stability, not training seeds. Four repeat-0 observations reused from older batch; report repeats 1/2 separately. Three historically selected units, no generalization or formal gains.',
        stop='No additional prompts, models, repetitions, probability scoring, training or tuning. No automatic resubmission with unresolved reservation. Deliver all branches even if negative.')
    Path('docs/cpt_rule_disambiguation_registration_20260922.safe.json').write_text(json.dumps(reg,indent=2)+'\n',encoding='utf-8')
    print('Frozen: 21 cells/model, 19 new/model; matched note and total prompt token counts')


if __name__ == '__main__':
    main()
