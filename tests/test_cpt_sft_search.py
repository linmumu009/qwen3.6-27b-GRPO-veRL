import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_cpt_sft_search import assess_target,target_correct

class SearchTargetTests(unittest.TestCase):
    def table(self,a=196,b=1224):
        return [dict(dataset='SC-bench-knowledge',n=226,before=189,after=a),
                dict(dataset='LogistikaBench',n=1446,before=1180,after=b)]
    def test_integer_thresholds(self):
        self.assertEqual(target_correct(189,226),196)
        self.assertEqual(target_correct(1180,1446),1224)
    def test_both_required(self):
        self.assertFalse(assess_target(self.table(a=195,b=1400))['numerical_target_met'])
        self.assertFalse(assess_target(self.table(a=220,b=1223))['numerical_target_met'])
    def test_threshold_pass_still_needs_confirmation(self):
        r=assess_target(self.table());self.assertTrue(r['numerical_target_met'])
        self.assertTrue(r['independent_confirmation_still_required'])
    def test_baseline_cannot_be_changed(self):
        rows=self.table();rows[0]['before']=188
        with self.assertRaises(ValueError):assess_target(rows)
    def test_duplicate_not_silently_selected(self):
        rows=self.table();rows.append(rows[0])
        with self.assertRaises(ValueError):assess_target(rows)
if __name__=='__main__':unittest.main()
