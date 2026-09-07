"""Bind eight source-inspection decisions to immutable private inputs; no API."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

DECISIONS = {
    'revision40-005': ('minor_revision', 'ambiguous_timeframe_wording'),
    'revision40-037': ('keep', 'source_supported_definition_in_scenario'),
    'revision40-010': ('keep', 'source_supported_comparison'),
    'revision40-034': ('type_revision', 'comparison_label_without_explicit_comparison'),
    'revision40-007': ('scope_revision', 'engine_compatibility_condition_omitted'),
    'revision40-031': ('keep', 'source_supported_factors_and_visibility'),
    'revision40-004': ('scope_revision', 'illustration_generalized_without_network_conditions'),
    'revision40-020': ('minor_revision', 'import_clearance_scope_needs_explicit_wording'),
}


def selected(rows):
    rng = random.Random(20260909)
    result = []
    for kind in ('definition', 'comparison', 'conditions', 'application'):
        result.extend(rng.sample(sorted([r for r in rows if r['kind'] == kind], key=lambda r:r['id']), 2))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pilot', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    raw = (a.pilot/'candidates.private.jsonl').read_bytes()
    source_raw = a.source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    source_digest = hashlib.sha256(source_raw).hexdigest()
    if digest != '2277f4ac340a7467f660184c5e0568179f54f493aab53a9a0cb6d441971d893d' or source_digest != '21071ddfc147f1b2c3f1ffc4f6df66363927d4232260b41914b162a296a5816c':
        raise ValueError('Unexpected frozen inputs')
    rows = list(map(json.loads, raw.decode().splitlines()))
    sources = {r['record_id']:r for r in map(json.loads,source_raw.decode().splitlines())}
    sample = selected(rows)
    if set(r['id'] for r in sample) != set(DECISIONS):
        raise ValueError('Sample differs from reviewed set')
    records = []
    for r in sample:
        text = sources[r['source_id']]['text']
        if hashlib.sha256(text.encode()).hexdigest() != r['source_text_sha256']:
            raise ValueError('Lineage mismatch')
        qa = r['qa']
        if any(q not in text for q in qa['support_quotes']):
            raise ValueError('Evidence mismatch')
        decision, reason = DECISIONS[r['id']]
        records.append({'id':r['id'], 'kind':r['kind'], 'source_id':r['source_id'],
            'qa_sha256':hashlib.sha256(json.dumps(qa,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
            'decision':decision,'reason_code':reason})
    result = {'candidate_sha256':digest,'source_sha256':source_digest,'seed':20260909,
        'sample_n':8,'records':records,'counts':dict(Counter(r['decision'] for r in records)),
        'reviewer':'Codex agent reading candidate QA and original supporting excerpts; not a human expert',
        'scope':'source fidelity, question completeness and type; not current external factual/legal verification',
        'selection_before_reading':True,'decision_record_created_after_reading':True,
        'new_api_calls':0,'new_qa_generated':0,'unreviewed_candidates':32,
        'human_review_complete':False,'training_started':False,'expanded':False,'ready_for_training':False}
    with a.output.open('x',encoding='utf-8') as f:
        json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
