import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_cpt_knowledge_complete_pipeline import audit_training

class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.lengths=[381]*4911+[1872924-381*4911]
        self.metrics={i+1:{'train/global_tokens':sum(self.lengths[i*8:(i+1)*8]),'train/loss':2.,'train/grad_norm':1.,'train/lr':5e-7} for i in range(614)}
    def test_tokenizer_pin_unchanged(self):
        script=(Path(__file__).resolve().parents[1]/'scripts/run_cpt_knowledge_complete_one_epoch.sh').read_text()
        self.assertIn('06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523',script)
    def test_exact_epoch(self):
        self.assertEqual(audit_training(self.metrics,self.lengths)['steps'],614)
    def test_missing_tail_rejected(self):
        self.metrics.pop(614)
        with self.assertRaises(AssertionError):audit_training(self.metrics,self.lengths)
    def test_wrong_token_budget_rejected(self):
        self.metrics[1]['train/global_tokens']+=1
        with self.assertRaises(AssertionError):audit_training(self.metrics,self.lengths)
    def test_nonfinite_loss_rejected(self):
        self.metrics[1]['train/loss']=float('nan')
        with self.assertRaises(AssertionError):audit_training(self.metrics,self.lengths)
if __name__=='__main__':unittest.main()
