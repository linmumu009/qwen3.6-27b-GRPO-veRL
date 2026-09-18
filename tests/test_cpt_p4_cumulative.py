import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_cpt_p4_cumulative import append_released

class CumulativeTests(unittest.TestCase):
    def setUp(self):
        self.old=[dict(id=f'old-{i}',messages=[dict(role='assistant',content='unchanged')]) for i in range(315)]
        self.tasks=[dict(id=f'new-{i}',split='train',unit='unit',question='Choose the valid statements.',options=['A','B','C','D'],option_truth=[True,False,True,False],correct_indices=[0,2],option_reasons=['A is valid.','B is invalid.','C is valid.','D is invalid.']) for i in range(16)]
        self.release=dict(training_allowed=True,approved_train_ids=[t['id'] for t in self.tasks],evaluation_only_ids=['heldout'])

    def test_preserves_old_and_uses_only_author_reasons(self):
        before=copy.deepcopy(self.old);rows=append_released(self.old,self.tasks,self.release)
        self.assertEqual(rows[:315],before);self.assertEqual(self.old,before)
        self.assertEqual(len(rows),363)
        for i in range(16):
            for e in range(3):
                text=rows[315+3*i+e]['messages'][-1]['content']
                self.assertEqual(json.loads(text.splitlines()[-1]),{'answers':[0,2]})
                self.assertEqual('Option 0: A is valid.' in text,e!=0)

    def test_rejects_eval_id_even_if_added_to_approval(self):
        self.release['evaluation_only_ids'].append('new-0')
        with self.assertRaises(AssertionError):append_released(self.old,self.tasks,self.release)

    def test_rejects_wrong_split(self):
        self.tasks[0]['split']='dev'
        with self.assertRaises(AssertionError):append_released(self.old,self.tasks,self.release)

    def test_rejects_inconsistent_gold(self):
        self.tasks[0]['correct_indices']=[0]
        with self.assertRaises(AssertionError):append_released(self.old,self.tasks,self.release)

    def test_rejects_duplicate_or_unapproved_task(self):
        self.tasks[0]['id']='other'
        with self.assertRaises(AssertionError):append_released(self.old,self.tasks,self.release)

if __name__=='__main__':unittest.main()
