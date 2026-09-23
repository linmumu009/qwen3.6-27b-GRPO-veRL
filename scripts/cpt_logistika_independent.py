"""Source-only author input and answer-hidden review for the Logistika pilot.

No generation, remote inference, training, or automatic quality release.
"""
import argparse
import hashlib
import json
from pathlib import Path

BASE = Path('CPT_resources/llin-logistika-independent-20260922-01')
FORMS = ('scope', 'competing_rules', 'application', 'counterexample')


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def freeze(p, value):
    raw = (json.dumps(value, ensure_ascii=False, indent=2)+'\n').encode('utf-8')
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        if p.read_bytes() != raw:
            raise ValueError('Existing frozen artifact differs: '+p.name)
    else:
        with p.open('xb') as f:
            f.write(raw)


def prepare(base):
    source = Path('CPT_resources/llin-structured-candidate-20260922-01/writer_handoff.private.json')
    gate = Path('docs/cpt_structured_candidate_gate_20260922.safe.json')
    if sha(source) != read(gate)['writer_handoff_sha256']:
        raise ValueError('Source handoff fingerprint changed')
    groups = sorted(read(source)['groups'], key=lambda g:g['id'])
    for group in groups:
        if hashlib.sha256(group['source_text'].encode()).hexdigest() != group['source_text_sha256']:
            raise ValueError('Source text fingerprint changed')
    slots = [dict(id=g['id']+'__'+f, group=g['id'], form=f) for g in groups for f in FORMS]
    if len(groups)!=8 or len(slots)!=32:
        raise ValueError('Frozen scope changed')
    freeze(base/'writer_input.private.json', dict(
        schema=1, groups=groups, slots=slots,
        input_scope='Only this file; no task history, repository search, formal sets, prior drafts, predictions, or web',
        output_path=str(base/'author.private.json'),
        instructions=[
            'Write one English multi-select task per slot or an explicit skip; no replacements or revisions in this run.',
            'Exactly minimum_option_count distinct options, question at most 150 words, each option at most 40 words.',
            'Include an unambiguous select-all instruction. No option-position references, all/none-of-above, or answer hints.',
            'Preserve source conditions, exceptions, classification granularity, and genuine competing rules.',
            'Every option needs an entailed/contradicted judgment, exact source quote(s), and concise inference.',
            'A merely unsupported option is insufficient and must not be treated as contradicted. Skip if unresolved.',
            'Do not invent quantitative relationships or missing figures. Hypothetical scenario facts must be explicit.',
            'For counterexamples, the scenario must negate the tested claim, not merely describe a different topic.',
            'No claimed expert or external review. Declare all inputs accessed and historical exposure honestly.'
        ],
        output_schema=dict(inputs_accessed=['absolute input path'],historical_context_seen=False,
            records=[dict(id='slot id',question='text',options=['text'],
                option_evidence=[dict(option_index=0,relation='entailed or contradicted',
                    source_quotes=['exact source substring'],reason='short inference')],
                reasoning_structure='judgment operation; no chain-of-thought required')]),
        skip_schema=dict(id='slot id',skip_reason='specific missing source/design condition')))
    freeze(base/'registration.private.json', dict(
        source_handoff_sha256=sha(source), writer_input_sha256=sha(base/'writer_input.private.json'),
        slots=slots, maximum_draft_records=32, correction_rounds=0,
        maximum_diagnostic_calls=432, training_allowed=False,
        review_rule='Freeze independent answer-hidden judgments before comparing author evidence; root overlap/source audit still required'))
    print(json.dumps(dict(groups=8,slots=32,writer_input_sha256=sha(base/'writer_input.private.json'))))


def validate_author(base):
    registration=read(base/'registration.private.json')
    if sha(base/'writer_input.private.json')!=registration['writer_input_sha256']:
        raise ValueError('Writer input changed')
    author=read(base/'author.private.json')
    expected={s['id'] for s in registration['slots']}
    rows=author['records']
    if not expected or len(rows)!=len(expected) or {r['id'] for r in rows}!=expected:
        raise ValueError('Missing or duplicate slots')
    if author['historical_context_seen'] is not False or not author['inputs_accessed']:
        raise ValueError('Author isolation not declared')
    return registration,author


def blind(base):
    registration,author=validate_author(base)
    by_id={r['id']:r for r in author['records']}
    rows=[]
    for slot in registration['slots']:
        task=by_id[slot['id']]
        # Whitelist only: do not expose skip reasons, evidence, keys, or rationale.
        rows.append(dict(slot,question=task.get('question'),options=task.get('options'),
                         authored='skip_reason' not in task))
    freeze(base/'reviewer_input.private.json',dict(
        schema=1, groups=read(base/'writer_input.private.json')['groups'], records=rows,
        author_sha256=sha(base/'author.private.json'),
        instructions=[
            'Read only this input; no author file, conversation history, repository searches, model predictions, or web.',
            'Independently judge every option as entailed, contradicted, or insufficient; provide exact source quotes and short reasons.',
            'Assess scope, competing rules, application and counterexample quality, leakage, ambiguity and cross-form duplication.',
            'Reject weak naming/keyword matches used in place of the required judgment operation.',
            'Do not repair tasks. Freeze all judgments before any author key or evidence is revealed.'
        ],
        output_schema=dict(reviewer_input_sha256='hash of this file',inputs_accessed=['absolute input path'],
            historical_context_seen=False,author_answers_seen=False,
            records=[dict(id='slot id',quality_pass=False,reason='concise quality rationale',
                option_evidence=[dict(option_index=0,relation='entailed/contradicted/insufficient',
                    source_quotes=['exact substring'],reason='short inference')])])) )


def evidence_key(evidence, options, source):
    if not isinstance(options,list) or not options or len(set(options))!=len(options):
        raise ValueError('Invalid options')
    if len(evidence)!=len(options) or {r['option_index'] for r in evidence}!=set(range(len(options))):
        raise ValueError('Incomplete option judgments')
    for row in evidence:
        if type(row['option_index']) is not int or row['relation'] not in ('entailed','contradicted'):
            raise ValueError('Unresolved option')
        if not row['reason'].strip() or not row['source_quotes']:
            raise ValueError('Missing evidence')
        if any(not isinstance(q,str) or len(q)<12 or q not in source for q in row['source_quotes']):
            raise ValueError('Source quote mismatch')
    key=sorted(r['option_index'] for r in evidence if r['relation']=='entailed')
    if not 0<len(key)<len(options):
        raise ValueError('Degenerate key')
    return key


def compare(base):
    registration,author=validate_author(base)
    blind(base)  # Idempotent; rejects input replacement after blind export.
    review=read(base/'review.private.json')
    if review['reviewer_input_sha256']!=sha(base/'reviewer_input.private.json'):
        raise ValueError('Review input mismatch')
    if review['historical_context_seen'] is not False or review['author_answers_seen'] is not False or not review['inputs_accessed']:
        raise ValueError('Reviewer isolation not declared')
    rows=review['records']; ids={s['id'] for s in registration['slots']}
    if not ids or len(rows)!=len(ids) or {r['id'] for r in rows}!=ids:
        raise ValueError('Incomplete review')
    freeze(base/'review_freeze.private.json',dict(review_sha256=sha(base/'review.private.json'),
        reviewer_input_sha256=sha(base/'reviewer_input.private.json')))
    authors={r['id']:r for r in author['records']}; reviews={r['id']:r for r in rows}
    groups={g['id']:g for g in read(base/'writer_input.private.json')['groups']}
    result=[]
    for slot in registration['slots']:
        task=authors[slot['id']]; judgment=reviews[slot['id']]; g=groups[slot['group']]; errors=[]
        try:
            if 'skip_reason' in task:raise ValueError('Author skipped')
            if len(task['options'])!=g['minimum_option_count']:raise ValueError('Option count')
            if len(task['question'].split())>150 or any(len(o.split())>40 for o in task['options']):raise ValueError('Word budget')
            a=evidence_key(task['option_evidence'],task['options'],g['source_text'])
            b=evidence_key(judgment['option_evidence'],task['options'],g['source_text'])
            if a!=b:raise ValueError('Author/reviewer disagreement')
            if judgment['quality_pass'] is not True:raise ValueError('Reviewer rejected quality')
        except (KeyError,TypeError,ValueError) as exc:errors.append(str(exc))
        result.append(dict(slot,evidence_agreement_pass=not errors,errors=errors))
    freeze(base/'comparison.safe.json',dict(records=result,review_sha256=sha(base/'review.private.json'),
        author_sha256=sha(base/'author.private.json'),
        diagnostic_released=False,root_source_and_overlap_audit_required=True,training_allowed=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('prepare','blind','compare'))
    parser.add_argument('--base',type=Path,default=BASE)
    args=parser.parse_args()
    {'prepare':prepare,'blind':blind,'compare':compare}[args.mode](args.base)
