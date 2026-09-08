import unittest
from scripts.run_logistics_strategy_diagnostic import choose, majority, messages_for, permutation, position_sensitive


def row(key='a', dataset='LogistikaBench', count=3):
    return dict(item_hash=key,dataset=dataset,source_id=key,category='synthetic',question_type='multiple_choice',question='Which entry?',options=[f'value {i}' for i in range(count)],expected=[1])


class DiagnosticTests(unittest.TestCase):
    def test_permutation_bijection_and_reproducibility(self):
        for count in (1,3,269):
            p=permutation('x',count)
            self.assertEqual(sorted(p),list(range(count)))
            self.assertEqual(p,permutation('x',count))
            for original in range(count): self.assertEqual(p[p.index(original)],original)

    def test_gold_independent_messages(self):
        a=row();b=dict(a,expected=[0,2])
        for condition in ('original','permuted','deliberate'):
            self.assertEqual(messages_for(a,condition),messages_for(b,condition))

    def test_selection_shortage_not_filled(self):
        rows=[row('a'),row('b',count=269),row('c',dataset='SC-bench-knowledge')]
        base={r['item_hash']:{'correct':False} for r in rows}
        picked,allocation=choose(rows,base)
        self.assertEqual(len(picked),3)
        self.assertEqual(sum(a['selected'] for a in allocation),3)
        self.assertEqual(sum(a['requested'] for a in allocation),96)

    def test_invalid_not_vote_and_not_stable_wrong(self):
        values=[dict(repeat=i,mapped=[1],parse_ok=i<1,correct=False,finish_reason='stop') for i in range(3)]
        result=majority(values)
        self.assertIsNone(result['answer'])
        self.assertFalse(result['stable_wrong'])
        self.assertEqual(result['parse_failures'],2)

    def test_correct_majority(self):
        values=[dict(repeat=i,mapped=[1],parse_ok=True,correct=True,finish_reason='stop') for i in range(3)]
        self.assertTrue(majority(values)['correct'])

    def test_position_reference_flag(self):
        r=row();r['options']=['All of the above','Other']
        self.assertTrue(position_sensitive(r))


if __name__=='__main__':unittest.main()
