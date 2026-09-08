import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from probe_book_repair_value import conditions
from probe_book_task_aligned_value import messages

class RepairProbeTests(unittest.TestCase):
    def test_closed_and_evidence(self):
        self.assertEqual(conditions({'mode':'closed'}),('closed','evidence'))
        self.assertEqual(conditions({'mode':'evidence'}),('evidence',))
        with self.assertRaises(ValueError):conditions({'mode':'unknown'})

    def test_no_gold_in_model_prompt(self):
        q=dict(kind='definition',question='Question text',answer='SECRET_GOLD',rubric=['SECRET_RUBRIC'])
        closed,_=messages(q,'closed','SECRET_REFERENCE')
        evidence,_=messages(q,'evidence','SECRET_REFERENCE')
        self.assertNotIn('SECRET',str(closed))
        self.assertIn('SECRET_REFERENCE',str(evidence))
        self.assertNotIn('SECRET_GOLD',str(evidence))
        self.assertNotIn('SECRET_RUBRIC',str(evidence))

if __name__=='__main__':unittest.main()
