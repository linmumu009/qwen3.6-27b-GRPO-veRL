"""Reconstruct screening lineage; separately parse raw baseline responses.

This verifies mechanics, not the factual quality of model-generated labels.
An optional agent-authored semantic review can remove false knowledge gaps.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path


def rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unique(records, key='id'):
    result = {r[key]: r for r in records}
    if len(result) != len(records):
        raise ValueError('duplicate record identifiers')
    return result


def independent_answer(text, count):
    decoder = json.JSONDecoder()
    matches = []
    for offset, char in enumerate(text):
        if char != '{':
            continue
        try:
            value, _ = decoder.raw_decode(text[offset:])
        except ValueError:
            continue
        if isinstance(value, dict) and set(value) == {'answers'}:
            matches.append(value['answers'])
    if len(matches) != 1:
        return None
    a = matches[0]
    if not isinstance(a, list) or not a or len(a) >= count or any(type(x) is not int or x < 0 or x >= count for x in a) or len(a) != len(set(a)):
        return None
    return sorted(a)


def verify(source_path, result_dir, semantic_path=None):
    spec = importlib.util.spec_from_file_location('runner', Path(__file__).with_name('run_cpt_transfer_screen.py'))
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    base = Path(result_dir)
    source = unique(rows(source_path))
    registration = json.loads((base/'registration.safe.json').read_text())
    assert sha(source_path) == registration['source_sha256']
    assert registration['baseline'] == dict(max_tokens=96, seed=1024, temperature=0)
    assert registration['thinking'] is False and registration['repeats'] == 1
    generation = unique(rows(base/'generation.private.jsonl'), 'source_id')
    candidates = unique(rows(base/'candidates.private.jsonl'))
    review = unique(rows(base/'review.private.jsonl'))
    accepted = unique(rows(base/'filtered.private.jsonl'))
    prediction = unique(rows(base/'baseline.private.jsonl'))
    assert set(generation) == set(source) and set(review) == set(candidates) and set(prediction) == set(accepted)
    def check_prompt(raw, text):
        assert raw['prompt_text_sha256'] == hashlib.sha256(text.encode()).hexdigest()
    for key, raw in generation.items():
        check_prompt(raw, runner.generation_prompt(source[key]))
    for key, c in candidates.items():
        r = source[c['unit_id']]
        assert c['category'] == r['category'] and c['group'] == r['group']
        assert c['training_allowed'] is False
        assert c['task'] == runner.validate_task(c['task'], r)
        original = runner.parse_object(generation[c['unit_id']]['text'])['tasks']
        assert any(runner.validate_task(t, r) == c['task'] for t in original if t.get('form') == c['task']['form'])
        check_prompt(review[key], runner.review_prompt(c, r))
    provisional, invalid, truncated = 0, 0, 0
    gap_units = set()
    semantic = unique(rows(semantic_path)) if semantic_path else {}
    assert set(semantic) <= set(accepted)
    validated_gaps, rejected_gaps = set(), set()
    by_category = {}
    for key, c in accepted.items():
        assert {k: c[k] for k in candidates[key]} == candidates[key]
        assert c['review'] == runner.parse_object(review[key]['text'])
        assert review[key]['finish_reason'] == 'stop'
        assert all(c['review'].get(k) is True for k in ('supported','unambiguous','self_contained','scope_preserved','not_answer_leaking'))
        assert sorted(c['review']['correct_indices']) == c['task']['correct_indices']
        raw = prediction[key]
        check_prompt(raw, runner.baseline_prompt(c['task']))
        parsed = independent_answer(raw['text'], len(c['task']['options']))
        correct = parsed == c['task']['correct_indices'] and raw['finish_reason'] == 'stop'
        assert parsed == raw['parsed'] and correct == raw['provisional_correct']
        assert raw['output_tokens'] <= 96 and raw['prompt_tokens'] + 96 <= 8192
        provisional += correct
        invalid += parsed is None
        truncated += raw['finish_reason'] != 'stop'
        stats = by_category.setdefault(c['category'], Counter())
        stats.update(screened=1, provisional_correct=int(correct))
        if not correct:
            gap_units.add(c['unit_id'])
        if key in semantic:
            decision = semantic[key]
            assert decision['task_sha256'] == runner.digest(c['task'])
            assert decision['verdict'] in ('accept', 'reject', 'hold')
            assert isinstance(decision['reason'], str) and decision['reason'].strip()
            if decision['verdict'] == 'accept':
                assert decision['independent_correct_indices'] == c['task']['correct_indices']
                assert len(decision['option_audit']) == len(c['task']['options'])
                if not correct:
                    validated_gaps.add(c['unit_id'])
            elif not correct:
                rejected_gaps.add(key)
    return dict(
        source_units=len(source), historical_source_groups=len({r['group'] for r in source.values()}),
        candidate_tasks=len(candidates), model_filtered_tasks=len(accepted), baseline_requests=len(prediction),
        provisional_correct=provisional, invalid=invalid, truncated=truncated,
        provisional_gap_source_units=len(gap_units), category_results=by_category,
        mechanically_reparsed=True, raw_lineage_and_text_prompt_hashes_verified=True,
        token_prompt_reconstruction='requires archived tokenizer check on server',
        semantic_reviewed_tasks=len(semantic), semantically_validated_gap_source_units=len(validated_gaps),
        rejected_or_held_error_tasks=len(rejected_gaps), validated_gap_unit_ids=sorted(validated_gaps),
        training_ready=False, official_evaluation_changed=False,
        limitations='Source-unit counts require further semantic consolidation into independent operational rules. Same-model source review is not independent certification. No new training or generalization claim.',
        artifact_sha256={p.name: sha(p) for p in [Path(source_path), *sorted(base.glob('*.private.jsonl'))]},
    )


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--sources', type=Path, required=True)
    p.add_argument('--result', type=Path, required=True)
    p.add_argument('--semantic', type=Path)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    result = verify(a.sources, a.result, a.semantic)
    a.out.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
