import sys
import unittest
import tempfile
import json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from cpt_two_family import variant,screen_packet,screen_score,digest
from prepare_cpt_two_family_train import validate_budgets,released_tasks,sha


class TwoFamilyTests(unittest.TestCase):
    def task(self,i,family,form):
        return dict(id=str(i),family=family,form=form,question='Synthetic: Select all correct statements.',
            options=['first','second','third','fourth'],correct_indices=[0,2],option_reasons=['x']*4,
            source_quotes=['synthetic source'],scenario_signature=str(i),reasoning_structure='synthetic')

    def test_rotation_maps_multiple_answers(self):
        t=self.task(0,'a','scope');v=variant(t,1)
        self.assertEqual(v['order'],[1,2,3,0]);self.assertEqual(v['expected'],[1,3])
        self.assertIn('[1] third',v['messages'][1]['content'])

    def test_both_families_and_repeats_required(self):
        families={f:dict(evidence_text='synthetic source') for f in ('a','b')}
        tasks=[self.task(i,f,form) for i,(f,form) in enumerate((f,form) for f in families for form in ['scope','competing_rules','application','counterexample'])]
        packet=screen_packet(tasks,families);rows=[]
        for r in packet['requests']:
            t=tasks[int(r['group'])];shift=r.get('shift',0)
            key=variant(t,shift)['expected']
            correct=r['condition']=='source' or int(t['id'])%4<2
            import json
            rows.append(dict(id=r['id'],text=json.dumps({'answers':key if correct else []}),finish_reason='stop',messages_sha256=digest(r['messages'])))
        self.assertTrue(screen_score(packet,rows)[1]['training_value_passed'])
        row=next(r for r in rows if r['id']=='0:repeat:1');row['text']='{"answers":[]}'
        self.assertFalse(screen_score(packet,rows)[1]['training_value_passed'])
        row['text']='{"answers":[0,2]}'
        for r in rows:
            if r['id'].startswith('4:source:'):r['text']='{"answers":[]}'
            if r['id'].startswith('5:source:'):r['text']='{"answers":[]}'
        self.assertFalse(screen_score(packet,rows)[1]['training_value_passed'])

    def test_budget_cannot_be_silently_relaxed(self):
        b=dict(records=351,sequence_tokens=100000,supervised_tokens=20000)
        validate_budgets(b,dict(b,supervised_tokens=21000))
        for bad in [dict(b,records=354),dict(b,supervised_tokens=21001),dict(b,sequence_tokens=110001)]:
            with self.assertRaises(AssertionError):validate_budgets(b,bad)

    def test_release_is_bound_to_reviewed_tasks(self):
        with tempfile.TemporaryDirectory() as d:
            b=Path(d);author=b/'train_author.v2.private.json';quality=b/'quality.safe.json';packet=b/'train_released.private.json'
            def write(p,x):p.write_text(json.dumps(x))
            tasks=[{'id':'synthetic','correct_indices':[1]}]
            write(author,dict(candidates=tasks))
            q=dict(passed=True,inputs={'CPT_resources\\synthetic\\train_author.v2.private.json':sha(author)})
            write(quality,q)
            payload=dict(quality_released=True,split='train',quality_sha256=sha(quality),tasks=tasks)
            write(packet,payload);self.assertEqual(released_tasks(packet),tasks)
            payload['tasks']=[{'id':'synthetic','correct_indices':[0]}];write(packet,payload)
            with self.assertRaisesRegex(AssertionError,'content changed'):released_tasks(packet)
            q['passed']=False;write(quality,q);payload['quality_sha256']=sha(quality);write(packet,payload)
            with self.assertRaises(AssertionError):released_tasks(packet)


if __name__=='__main__':unittest.main()
