from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from probe_book_task_aligned_value import messages,permutation,parse_selection
from grade_book_task_aligned_value import valid_grade,order_for,consensus,classify_open,classify_selection
from audit_book_value_context import context_flags


class ValueTests(unittest.TestCase):
    def test_no_gold_or_rubric_in_model_prompt(self):
        r=dict(id='synthetic',kind='definition',question='Question only',answer='SECRET_ANSWER',rubric=['SECRET_RUBRIC'])
        closed,_=messages(r,'closed','PRIVATE_SOURCE')
        self.assertNotIn('PRIVATE_SOURCE',str(closed));self.assertNotIn('SECRET',str(closed))
        opened,_=messages(r,'evidence','PRIVATE_SOURCE');self.assertIn('PRIVATE_SOURCE',str(opened));self.assertNotIn('SECRET',str(opened))

    def test_choice_permutation_is_gold_independent(self):
        r=dict(id='x',kind='evidence_selection',question='Choose',options=list('abcdef'),correct_indices=[1,4],answer='SECRET',rubric=['SECRET'])
        m,p=messages(r,'permuted','REF');self.assertEqual(sorted(p),list(range(6)));self.assertNotIn('SECRET',str(m))
        r['correct_indices']=[0];self.assertEqual(permutation(r),p)
        parsed,valid=parse_selection('{"indices":[0,2]}',6,p);self.assertTrue(valid);self.assertEqual(parsed,sorted([p[0],p[2]]))

    def test_strict_parser(self):
        for txt in ('{"indices":[true]}','{"indices":[1,1]}','{"indices":[6]}','not json'):
            self.assertFalse(parse_selection(txt,6,list(range(6)))[1])
        self.assertTrue(parse_selection('```json\n{"indices":[1]}\n```',6,list(range(6)))[1])

    def test_anonymous_order_reversal(self):
        self.assertEqual(order_for('synthetic',0),list(reversed(order_for('synthetic',1))))

    def test_grade_schema_and_exact_consensus(self):
        v=dict(question_valid=True,rubric_valid=True,answers=[dict(id=i,score=2,source_ids=['S001'],reason='supported') for i in 'ABCD'])
        self.assertTrue(valid_grade(v,{'S001':'source'}))
        v['answers'][0]['score']=True;self.assertFalse(valid_grade(v,{'S001':'source'}))
        a=dict(status='scored',repeat=0,scores={'a':2});b=dict(status='scored',repeat=1,scores={'a':1})
        self.assertEqual(consensus([a,dict(a,repeat=1)])[0],'judge_agreed')
        self.assertEqual(consensus([a,a])[0],'judge_invalid')
        self.assertEqual(consensus([a,b])[0],'judge_disagreement')
        self.assertEqual(consensus([a])[0],'judge_invalid')

    def test_value_is_candidate_not_training_permission(self):
        self.assertEqual(classify_open(1,2),'evidence_rescued_candidate')
        self.assertEqual(classify_open(2,1),'reference_interference')
        self.assertEqual(classify_open(0,1),'unresolved_both_conditions')
        self.assertEqual(classify_open(2,2),'maintenance_candidate')
        self.assertEqual(classify_selection(dict(parse_ok=True,correct=True),dict(parse_ok=True,correct=False)),'order_sensitive_candidate')
        self.assertEqual(classify_selection(dict(parse_ok=False,correct=False),dict(parse_ok=True,correct=True)),'invalid_response')

    def test_missing_reference_cues(self):
        self.assertTrue(context_flags('According to the source, what is the rule?'))
        self.assertTrue(context_flags('How does the approach in the provided text work?'))
        self.assertFalse(context_flags('How does activity-based costing allocate picking cost?'))

if __name__=='__main__':unittest.main()
