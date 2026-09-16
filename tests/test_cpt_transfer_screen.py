import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('screen', ROOT / 'scripts/run_cpt_transfer_screen.py')
screen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(screen)
vspec = importlib.util.spec_from_file_location('verify_screen', ROOT / 'scripts/verify_cpt_transfer_screen.py')
verify_screen = importlib.util.module_from_spec(vspec)
vspec.loader.exec_module(verify_screen)


class ScreenContracts(unittest.TestCase):
    def setUp(self):
        self.source = dict(title='Rule', scope='Historical statistical snapshot', source_text='Only operating facilities are counted during the reference period.')
        self.task = dict(form='boundary_counterexample', question='Which listed facility satisfies the statistical reporting condition?', options=['Active', 'Dormant', 'Planned', 'Never commissioned'], correct_indices=[0], source_quote=self.source['source_text'], explanation='Only operating units count.', option_reasons=['Operating', 'Inactive', 'Inactive', 'Inactive'])

    def test_quote_must_be_exact(self):
        bad = dict(self.task, source_quote='Facilities are all counted during the reference period.')
        with self.assertRaises(ValueError):
            screen.validate_task(bad, self.source)

    def test_duplicate_options_rejected(self):
        bad = copy.deepcopy(self.task)
        bad['options'][1] = ' ACTIVE '
        with self.assertRaises(ValueError):
            screen.validate_task(bad, self.source)

    def test_bool_and_out_of_range_answers_rejected(self):
        for a in [[True], [4], [0, 0], [], [0, 1, 2, 3]]:
            with self.assertRaises(ValueError):
                screen.validate_task(dict(self.task, correct_indices=a), self.source)

    def test_baseline_excludes_source_and_labels(self):
        prompt = screen.baseline_prompt(self.task)
        self.assertNotIn(self.source['source_text'], prompt)
        self.assertNotIn(self.task['explanation'], prompt)
        self.assertNotIn('correct_indices', prompt)

    def test_reviewer_blind_to_proposed_answer(self):
        prompt = screen.review_prompt(dict(task=self.task), self.source)
        self.assertNotIn(self.task['explanation'], prompt)
        self.assertNotIn('Only operating units count.', prompt)
        self.assertIn(self.source['source_text'], prompt)

    def test_multiple_json_answers_invalid(self):
        self.assertIsNone(screen.parse_answer('{"answers":[0]} {"answers":[1]}', 4))
        self.assertIsNone(screen.parse_answer('{"answers":[true]}', 4))
        self.assertEqual(screen.parse_answer('{"answers":[2,0]}', 4), [0, 2])

    def test_generation_input_allowlist(self):
        row = dict(self.source, official_question='FORBIDDEN', expected=[3], private_prediction='FORBIDDEN')
        self.assertNotIn('FORBIDDEN', screen.generation_prompt(row))

    def test_valid_task(self):
        self.assertEqual(screen.validate_task(self.task, self.source)['correct_indices'], [0])

    def fixture(self, root, finish='stop'):
        source = dict(self.source, id='unit', group='group', category='category')
        candidate = dict(id='case', unit_id='unit', group='group', category='category', task=self.task, training_allowed=False)
        review = dict(correct_indices=[0], supported=True, unambiguous=True, self_contained=True, scope_preserved=True, not_answer_leaking=True)
        def raw(text, prompt):
            return dict(text=text, finish_reason='stop', prompt_text_sha256=hashlib.sha256(prompt.encode()).hexdigest(), prompt_tokens=100, output_tokens=8)
        generation = dict(source_id='unit', **raw(json.dumps(dict(tasks=[self.task])), screen.generation_prompt(source)))
        review_raw = dict(id='case', **raw(json.dumps(review), screen.review_prompt(candidate, source)))
        accepted = dict(candidate, review=review)
        baseline = dict(id='case', **raw('{"answers":[0]}', screen.baseline_prompt(self.task)), parsed=[0], provisional_correct=finish=='stop')
        baseline['finish_reason'] = finish
        files = {'sources.private.jsonl':[source], 'generation.private.jsonl':[generation], 'candidates.private.jsonl':[candidate], 'review.private.jsonl':[review_raw], 'filtered.private.jsonl':[accepted], 'baseline.private.jsonl':[baseline]}
        for name, records in files.items():
            (root/name).write_text(''.join(json.dumps(r)+'\n' for r in records), encoding='utf-8')
        (root/'registration.safe.json').write_text(json.dumps(dict(source_sha256=verify_screen.sha(root/'sources.private.jsonl'), baseline=dict(max_tokens=96,seed=1024,temperature=0), thinking=False, repeats=1)))
        return root/'sources.private.jsonl'

    def test_independent_raw_reparse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=self.fixture(root)
            result=verify_screen.verify(source,root)
            self.assertEqual(result['provisional_correct'],1)

    def test_truncation_is_wrong_even_with_correct_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=self.fixture(root,'length')
            result=verify_screen.verify(source,root)
            self.assertEqual(result['provisional_correct'],0)
            self.assertEqual(result['truncated'],1)

    def test_prompt_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=self.fixture(root)
            p=root/'baseline.private.jsonl';r=json.loads(p.read_text());r['prompt_text_sha256']='tampered';p.write_text(json.dumps(r)+'\n')
            with self.assertRaises(AssertionError):verify_screen.verify(source,root)

    def test_raw_prediction_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=self.fixture(root)
            p=root/'baseline.private.jsonl';r=json.loads(p.read_text());r['text']='{"answers":[1]}';p.write_text(json.dumps(r)+'\n')
            with self.assertRaises(AssertionError):verify_screen.verify(source,root)


if __name__ == '__main__':
    unittest.main()
