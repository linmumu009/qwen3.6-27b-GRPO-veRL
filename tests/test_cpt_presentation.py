import sys
import tempfile
import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from prepare_cpt_presentation import crossed
from run_cpt_presentation import batches
from summarize_cpt_presentation import score


class PresentationTests(unittest.TestCase):
    def test_preflight_cannot_select_worker(self):
        p=subprocess.run([sys.executable,str(Path(__file__).parents[1]/'scripts/run_cpt_presentation.py'),
            '--package','synthetic','--worker','--preflight-only'],capture_output=True,text=True)
        self.assertEqual(p.returncode,2)
        self.assertIn('not allowed',p.stderr)

    def test_strict_majority_and_truncation(self):
        rows=[dict(text='{"answers":[1]}',finish_reason='stop') for _ in range(3)]
        self.assertTrue(score(rows,[1],2)['stable_correct'])
        rows[0]['finish_reason']='length'
        result=score(rows,[1],2)
        self.assertTrue(result['correct']);self.assertFalse(result['stable_correct']);self.assertEqual(result['truncated'],1)
        rows[1]['text']='{"answers":[0]}'
        self.assertFalse(score(rows,[1],2)['correct'])

    def test_cross_preserves_answer_text_not_index(self):
        base=dict(item_hash='synthetic',dataset='synthetic',source_id='1',category='x',question_type='single',
            question='Large context Which of the following best matches: definition',options=['wrong','right','other'],expected=[1])
        small=dict(base,question='Specific context Which of the following best matches: definition',options=['right','wrong'],expected=[0])
        x=crossed(base,small)
        self.assertEqual(x['LS']['expected'],[0]);self.assertEqual(x['SL']['expected'],[1])
        self.assertIn('Large context',x['LS']['messages'][1]['content'])
        self.assertIn('[0] right',x['LS']['messages'][1]['content'])
        with self.assertRaisesRegex(ValueError,'Definition'):crossed(base,dict(small,question=small['question']+' '))
        with self.assertRaisesRegex(ValueError,'Answer'):crossed(base,dict(small,expected=[1]))

    def test_unresolved_reservation_blocks_all_new_generation(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);calls=[]
            class Fail:
                def generate(self,*args):calls.append(1);raise RuntimeError('failure after reserve')
            plan=[dict(id='x',messages=[])];prompts=[dict(prompt_token_ids=[1])]
            with self.assertRaisesRegex(RuntimeError,'after reserve'):batches(plan,prompts,folder,'hash',Fail,None)
            with self.assertRaisesRegex(AssertionError,'Unresolved'):batches(plan,prompts,folder,'hash',Fail,None)
            self.assertEqual(len(calls),1)

    def test_completed_batches_reused_without_loading_model(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);plan=[dict(id='x',messages=[])];prompts=[dict(prompt_token_ids=[1])]
            class Good:
                def generate(self,*args):return [SimpleNamespace(request_id='0',prompt_token_ids=[1],outputs=[
                    SimpleNamespace(token_ids=[2],text='{"answers":[0]}',finish_reason='stop')])]
            a=batches(plan,prompts,folder,'hash',Good,None)
            def forbidden():raise AssertionError('Must not load model')
            self.assertEqual(a,batches(plan,prompts,folder,'hash',forbidden,None))


if __name__=='__main__':unittest.main()
