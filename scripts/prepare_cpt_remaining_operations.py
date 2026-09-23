"""Freeze a source-only author batch after the bounded ten-item pre-screen."""
import argparse
import hashlib
from pathlib import Path
from cpt_logistika_independent import FORMS, freeze, read, sha

BASE = Path('CPT_resources/llin-remaining-operations-20260923-01')
SOURCE_HASH = '26b92190dc3425bab810a5051db198e8037bceba2b309c9fb33cba68d4b2a51e'


def prepare(base):
    source_path = base / 'source_input.private.json'
    if sha(source_path) != SOURCE_HASH:
        raise ValueError('Frozen ten-item source input changed')
    selection = read(base / 'selection.private.json')
    all_groups = {g['id']: g for g in read(source_path)['groups']}
    ids = selection['selected_ids']
    if not 0 < len(ids) <= 10 or len(set(ids)) != len(ids) or not set(ids) <= set(all_groups):
        raise ValueError('Invalid source selection')
    groups = []
    for gid in sorted(ids):
        original = all_groups[gid]
        for s in original['sources']:
            if hashlib.sha256(s['text'].encode()).hexdigest() != s.get('sha256', s.get('text_sha256')):
                raise ValueError('Source text changed')
        text = '\n\n'.join(s['text'] for s in original['sources'])
        groups.append(dict(id=gid, source_text=text, source_text_sha256=hashlib.sha256(text.encode()).hexdigest(),
            minimum_option_count=4, scope=selection['author_scope'][gid],
            source_ids=[s['id'] for s in original['sources']]))
    slots = [dict(id=g['id']+'__'+f, group=g['id'], form=f) for g in groups for f in FORMS]
    # Reuse the source-only format requirements, never old sources or questions.
    template = read(Path('CPT_resources/llin-logistika-independent-20260922-01/writer_input.private.json'))
    instructions = template['instructions'] + [
        'Stay within the named group scope. Do not use unrelated terms from the same source page to fill four forms.',
        'Scope must require a boundary decision; application and competing_rules must require conditional reasoning, not naming or arithmetic.',
        'Four forms must differ in reasoning operation, not just reverse the same definition.',
        'Do not quote governing rules or give the answer in the question. State scenario facts needed for a unique answer.',
        'Skip an unsupportable slot rather than filling it with a weak substitute.'
    ]
    freeze(base / 'writer_input.private.json', dict(schema=1, groups=groups, slots=slots,
        input_scope=template['input_scope'], instructions=instructions,
        output_path=str(base / 'author.private.json'), output_schema=template['output_schema'], skip_schema=template['skip_schema']))
    n = len(groups)
    freeze(base / 'registration.private.json', dict(source_input_sha256=SOURCE_HASH,
        selection_sha256=sha(base / 'selection.private.json'), writer_input_sha256=sha(base / 'writer_input.private.json'),
        slots=slots, maximum_draft_records=len(slots), correction_rounds=0,
        maximum_diagnostic_calls=48*n+2*min(24,16*n), training_allowed=False,
        review_rule='Freeze hidden-answer review, then root evidence and historical-operation audit before diagnostic release'))
    print(f'Frozen {n} groups / {len(slots)} slots; no diagnostic release')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--base', type=Path, default=BASE)
    prepare(p.parse_args().base)
