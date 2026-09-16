import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('composed',Path(__file__).resolve().parents[1]/'scripts/cpt_composed_probe.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import verify_cpt_composed_probe as verifier
import run_cpt_atomic_probe as atomic
import verify_cpt_atomic_probe as atomic_verifier


class ProbeContracts(unittest.TestCase):
    def setUp(self):
        self.c=dict(id='one',category='test',question='Choose the valid statements. Select all correct statements.',
            options=['A','B','C','D','E','F'],expected=[0,2,5],option_audit=['AUDIT_ONLY']*6,
            sources=[dict(id='source',scope='Archived scope',source_text='A.VI-04 DAMAGE\nThreshold 150000; preserve this quantity.')],
            arithmetic_checks=[['1-(1+1/2+1/3)/6','25/36']],training_allowed=False,purpose='screen_only')

    def test_exact_arithmetic(self):
        self.assertEqual(m.arithmetic('1-(1+1/2+1/3)/6'),m.arithmetic('25/36'))
        m.validate([self.c])

    def test_no_code_execution(self):
        for expr in ['open("x")','__import__("os")','x+1','2**100','[1][0]','True+1']:
            with self.assertRaises(ValueError):m.arithmetic(expr)

    def test_numerical_mismatch_fails(self):
        c=copy.deepcopy(self.c);c['arithmetic_checks']=[['2+2','5']]
        with self.assertRaises(ValueError):m.validate([c])

    def test_permutations_preserve_semantic_key(self):
        specs=m.specifications([self.c])
        self.assertEqual(len(specs),4)
        for s in specs:self.assertEqual(sorted(s['order'][i] for i in s['expected']),self.c['expected'])
        a,b=m.variants(self.c)
        self.assertTrue(all(x!=y for x,y in zip(a,b)))

    def test_closed_book_has_no_sources_or_rationale(self):
        text=m.prompt(self.c,m.variants(self.c)[0],'closed_book')
        self.assertNotIn('AUDIT_ONLY',text);self.assertNotIn('150000',text)

    def test_evidence_strips_only_chapter_identifiers(self):
        text=m.prompt(self.c,m.variants(self.c)[0],'source_evidence')
        self.assertNotIn('A.VI-04',text);self.assertIn('150000',text);self.assertNotIn('AUDIT_ONLY',text)

    def test_invalid_answer_sets(self):
        for text in ['{"answers":[true]}','{"answers":[0,0]}','{"answers":[6]}','{"answers":[]}','{"answers":[0]} {"answers":[1]}']:
            self.assertIsNone(m.parse(text,6))
        self.assertEqual(m.parse('{"answers":[4,1]}',6),[1,4])

    def test_duplicate_option_guard(self):
        c=copy.deepcopy(self.c);c['options'][1]=' a '
        with self.assertRaises(ValueError):m.validate([c])

    def test_training_release_rejected(self):
        c=dict(self.c,training_allowed=True)
        with self.assertRaises(ValueError):m.validate([c])

    def fixture(self,root,one_wrong=False,truncated=False):
        packet=root/'cases.private.jsonl';packet.write_text(json.dumps(self.c)+'\n')
        reg=dict(case_sha256=m.sha(packet),max_tokens=96,temperature=0,seed=1024,thinking=False,tp=8,max_num_seqs=16,max_model_len=8192,option_orders=2,repeats_per_order_condition=1,code_sha256='fixture')
        (root/'registration.safe.json').write_text(json.dumps(reg))
        records=[]
        for s in m.specifications([self.c]):
            wrong=one_wrong and s['variant']==1 and s['condition']=='closed_book'
            parsed=[i for i in range(6) if i not in s['expected']] if wrong else s['expected']
            finish='length' if truncated else 'stop'
            records.append(dict(s,text=json.dumps({'answers':parsed}),parsed=parsed,correct=not wrong and not truncated,finish_reason=finish,output_tokens=20,prompt_tokens=100,prompt_text_sha256=hashlib.sha256(m.prompt(self.c,s['order'],s['condition']).encode()).hexdigest()))
        (root/'predictions.private.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
        return packet

    def test_paired_result_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=self.fixture(root,one_wrong=True);result,details=verifier.verify(p,root)
            self.assertEqual(result['totals']['closed_one_correct'],1)
            self.assertTrue(details[0]['closed_content_changes_with_order'])
            self.assertEqual(result['totals']['evidence_both_correct'],1)

    def test_truncation_remains_wrong(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=self.fixture(root,truncated=True);result,_=verifier.verify(p,root)
            self.assertEqual(result['totals']['truncated'],4)
            self.assertEqual(result['totals']['closed_book_correct'],0)

    def test_permutation_key_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=self.fixture(root);f=root/'predictions.private.jsonl'
            rs=[json.loads(s) for s in f.read_text().splitlines()];rs[0]['expected']=[0];f.write_text(''.join(json.dumps(r)+'\n' for r in rs))
            with self.assertRaises(AssertionError):verifier.verify(p,root)

    def test_prompt_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=self.fixture(root);f=root/'predictions.private.jsonl'
            rs=[json.loads(s) for s in f.read_text().splitlines()];rs[0]['prompt_text_sha256']='altered';f.write_text(''.join(json.dumps(r)+'\n' for r in rs))
            with self.assertRaises(AssertionError):verifier.verify(p,root)

    def test_boolean_false_is_a_valid_answer(self):
        self.assertIs(atomic.parse_bool('{"correct":false}'),False)
        self.assertIs(atomic.parse_bool('{"correct":true}'),True)
        for text in ['{"correct":0}','{"correct":"false"}','{"correct":true,"other":1}','{"correct":true} more']:
            self.assertIsNone(atomic.parse_bool(text))

    def test_atomic_prompt_preserves_all_claims_and_no_key(self):
        text=atomic.prompt(self.c,'closed_book',2)
        for i,j in enumerate(m.variants(self.c)[0]):self.assertIn('['+str(i)+'] '+self.c['options'][j],text)
        self.assertNotIn('AUDIT_ONLY',text);self.assertNotIn('150000',text)
        self.assertNotIn('Select all correct statements.',text)

    def test_atomic_target_tracks_original_option_order(self):
        order=m.variants(self.c)[0]
        for target in range(6):
            self.assertIn('Designated statement: option ['+str(order.index(target))+']',atomic.prompt(self.c,'source_evidence',target))

    def test_atomic_independent_reconstruction_and_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cases=[dict(copy.deepcopy(self.c),id='case'+str(i)) for i in range(31)]
            packet=root/'cases.private.jsonl';packet.write_text(''.join(json.dumps(c)+'\n' for c in cases))
            audit=root/'audit.private.jsonl';audit.write_text(''.join(json.dumps(dict(id=c['id'],verdict='accept',case_sha256=m.sha(packet)))+'\n' for c in cases))
            reg=dict(case_sha256=m.sha(packet),semantic_review_sha256=m.sha(audit),max_tokens=16,per_case_condition_total_output_cap=96,repeated_input_compute=6,max_model_len=8192,tp=8,max_num_seqs=16,thinking=False,temperature=0,seed=1024,paired_joint_variant=0)
            (root/'registration.safe.json').write_text(json.dumps(reg))
            records=[];old=[]
            for c in cases:
                for condition in ('closed_book','source_evidence'):
                    order=m.variants(c)[0];answer=sorted(i for i,j in enumerate(order) if j in c['expected'])
                    old.append(dict(id=c['id'],condition=condition,variant=0,order=order,text=json.dumps(dict(answers=answer)),finish_reason='stop',correct=True))
                    for target in range(6):
                        truth=target in c['expected'];value=not truth if c['id']=='case0' and condition=='closed_book' and target==0 else truth
                        records.append(dict(id=c['id'],condition=condition,target=target,expected=truth,parsed=value,text=json.dumps(dict(correct=value)),correct=value==truth,finish_reason='stop',output_tokens=6,prompt_tokens=100,prompt_text_sha256=hashlib.sha256(atomic.prompt(c,condition,target).encode()).hexdigest()))
            predictions=root/'predictions.private.jsonl';predictions.write_text(''.join(json.dumps(r)+'\n' for r in records))
            jp=root/'joint.private.jsonl';jp.write_text(''.join(json.dumps(r)+'\n' for r in old))
            result,_=atomic_verifier.verify(packet,audit,root,jp)
            self.assertEqual(result['by_condition']['closed_book']['atomic_complete_correct'],30)
            self.assertEqual(result['by_condition']['closed_book']['joint_right_atomic_wrong'],1)
            records[0]['expected']=not records[0]['expected'];predictions.write_text(''.join(json.dumps(r)+'\n' for r in records))
            with self.assertRaises(AssertionError):atomic_verifier.verify(packet,audit,root,jp)


if __name__=='__main__':unittest.main()
