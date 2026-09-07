import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('pilot', Path(__file__).parents[1] / 'scripts/build_book_sft_api_pilot.py')
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def test_grounding_requires_exact_quote():
    qa = {'question': 'What is inventory?', 'answer': 'Goods held for future use.',
          'support_quotes': ['Inventory is goods held for future use.']}
    assert pilot.validate(qa, qa['support_quotes'][0]) == []
    assert 'invalid_verbatim_support' in pilot.validate(qa, 'Unrelated source')


def test_missing_and_dependent_questions_rejected():
    assert 'missing_answer' in pilot.validate({}, '')
    assert 'context_dependency' in pilot.validate({'question': 'What does the above show?'}, '')
    assert 'context_dependency' in pilot.validate({'question': 'Based on the excerpt, what should the retailer do?'}, '')
    assert 'context_dependency' not in pilot.validate({'question': 'How do warehouse costs affect service levels?'}, '')


def test_chapter_balancing_and_repeatability():
    records = [{'record_id': str(i), 'chapter': i, 'text': 'x' * 1500} for i in range(1, 45)]
    tasks = pilot.plan(records, 200)
    assert tasks == pilot.plan(records, 200)
    assert len({t['id'] for t in tasks}) == 200
    assert len({t['source']['chapter'] for t in tasks}) == 44
    assert set(t['kind'] for t in tasks) == set(pilot.KINDS)
    assert max(sum(t['source']['chapter'] == i for t in tasks) for i in range(1, 45)) == 5
