import copy
import unittest

from scripts.audit_cpt_transfer import audit_answer_content, paired, select_review, verify_predictions, verify_protocols
from scripts.run_cpt_evidence_pilot import messages_for, strip_section_ids
from scripts.evaluate_logistics_knowledge import EvalItem, build_messages


class TransferAuditTests(unittest.TestCase):
    def fixture(self, answers=(0, 0, 1)):
        cases = {'x': dict(dataset='SC-bench-knowledge', category='warehouse',
            question_type='single_choice', options=['right', 'wrong', 'third'], expected=[0])}
        rows = [dict(item_hash='x', dataset='SC-bench-knowledge', repeat=i,
            prediction='{"answers":['+str(a)+']}', parsed=[a], valid=True,
            correct=a == 0, truncated=False, output_tokens=8) for i, a in enumerate(answers)]
        saved = {'x': dict(dataset='SC-bench-knowledge', correct=answers.count(0) >= 2)}
        return cases, rows, saved

    def test_raw_reparse_and_majority(self):
        result = verify_predictions(*self.fixture())['x']
        self.assertTrue(result['correct'])
        self.assertTrue(result['repeat_unstable'])

    def test_no_majority_is_wrong(self):
        result = verify_predictions(*self.fixture((0, 1, 2)))['x']
        self.assertIsNone(result['answer'])
        self.assertFalse(result['correct'])

    def test_duplicate_repeat_rejected(self):
        c, r, s = self.fixture()
        r[1]['repeat'] = 0
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            verify_predictions(c, r, s)

    def test_stale_cached_score_rejected(self):
        c, r, s = self.fixture()
        r[0]['prediction'] = '{"answers":[1]}'
        with self.assertRaisesRegex(ValueError, 'raw answer'):
            verify_predictions(c, r, s)

    def test_truncated_correct_text_cannot_vote(self):
        c, r, s = self.fixture()
        for row in r[:2]:
            row.update(truncated=True, valid=False, correct=False)
        s['x']['correct'] = False
        result = verify_predictions(c, r, s)['x']
        self.assertFalse(result['correct'])
        self.assertEqual(result['truncated'], 2)
        self.assertEqual(result['invalid'], 2)

    def test_saved_majority_must_match(self):
        c, r, s = self.fixture()
        s['x']['correct'] = False
        with self.assertRaisesRegex(ValueError, 'saved majority'):
            verify_predictions(c, r, s)

    def test_output_budget_enforced(self):
        c, r, s = self.fixture()
        r[0]['output_tokens'] = 97
        with self.assertRaisesRegex(ValueError, 'budget'):
            verify_predictions(c, r, s)

    def test_protocol_and_prompt_drift_rejected(self):
        p = dict(model='a', cases_sha256='sha', cases=1, repeats=3,
            temperature=0, seed=1024, max_tokens=96, max_model_len=8192,
            tp=8, max_num_seqs=32, prompt='original', prompt_hashes=['abc'])
        q = dict(p, model='b')
        verify_protocols({'a': p, 'b': q}, 'sha', 1)
        for field, value in [('prompt_hashes', ['def']), ('max_tokens', 128)]:
            changed = copy.deepcopy(q)
            changed[field] = value
            with self.assertRaises(ValueError):
                verify_protocols({'a': p, 'b': changed}, 'sha', 1)

    def test_sampling_covers_strata_deterministically(self):
        cases = {str(i): dict(dataset='LogistikaBench' if i < 30 else 'SC-bench-knowledge',
            category=str(i % 3), question_type='single_choice', options=['a', 'b'], expected=[0])
            for i in range(35)}
        baseline = {k: dict(correct=False) for k in cases}
        selected, allocations = select_review(cases, baseline, quota=12)
        self.assertEqual(len(selected), 17)
        self.assertEqual(len(set(selected)), 17)
        self.assertTrue(all(str(i) in selected for i in range(30, 35)))
        self.assertTrue(all(a['selected'] > 0 for a in allocations))
        self.assertEqual(select_review(dict(reversed(list(cases.items()))), baseline, 12)[0], selected)

    def test_pairing_counts_regressions_and_repairs_separately(self):
        before = {str(i): dict(correct=i < 2) for i in range(4)}
        after = {str(i): dict(correct=i % 2 == 0) for i in range(4)}
        result = paired(sorted(before), before, after, uncertainty=True)
        self.assertEqual((result['gains'], result['losses'], result['delta_pp']), (1, 1, 0))
        self.assertEqual(result['mcnemar_exact_two_sided_p_unadjusted'], 1)

    def test_evidence_changes_only_user_message_and_closed_is_identical(self):
        c = dict(item_hash='x', dataset='test', source_id='1', category='math',
                 question_type='single_choice', question='Which?', options=['a', 'b'], expected=[0])
        expected = build_messages(EvalItem('x', 'test', '1', 'math', 'single_choice', 'Which?', ('a', 'b'), (0,)))
        self.assertEqual(messages_for(c, None), expected)
        evidence = messages_for(c, 'Rule text')
        self.assertEqual(evidence[0], expected[0])
        self.assertTrue(evidence[-1]['content'].startswith(expected[-1]['content']))
        self.assertTrue(evidence[-1]['content'].endswith('Rule text'))

    def test_content_collision_is_diagnostic_and_does_not_regrade(self):
        cases = {'x': dict(dataset='test', options=['TRAILER', 'Trailer'], expected=[0],
                           question_type='single_choice')}
        models = {'step120': {'x': dict(correct=False, answer=[1])}}
        result = audit_answer_content(cases, models)
        self.assertFalse(result['official_scores_changed'])
        self.assertEqual(result['datasets'][0]['items_with_gold_option_text_collision'], 1)
        self.assertEqual(result['datasets'][0]['models']['step120']['index_wrong_but_same_normalized_option_text'], 1)
        self.assertFalse(models['step120']['x']['correct'])

    def test_section_control_preserves_body_facts_and_numbers(self):
        text = 'E.II-05 SHIP (VESSEL)\nMore than 12 passengers.\nD.I/II-03 PIPELINE NETWORK\nAll pipelines.\n'
        self.assertEqual(strip_section_ids(text), 'SHIP (VESSEL)\nMore than 12 passengers.\nPIPELINE NETWORK\nAll pipelines.\n')


if __name__ == '__main__':
    unittest.main()
