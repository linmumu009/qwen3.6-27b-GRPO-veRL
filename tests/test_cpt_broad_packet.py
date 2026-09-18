import json
from pathlib import Path
import sys
import unittest
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_cpt_broad_review import specs
from run_cpt_remediation_review import review_prompt
from cpt_broad_warehouse_tasks import add_warehouse
from cpt_broad_planning_tasks import add_planning
from cpt_broad_transport_tasks import add_transport
from cpt_broad_retention_tasks import add_retention

class BroadPacketTests(unittest.TestCase):
    def test_four_orders_cover_every_position_and_invert_gold(self):
        for count in range(1,5):
            row=dict(id='example-'+str(count),correct_indices=list(range(count)))
            plan=specs([row]);self.assertEqual(len(plan),4)
            for original in range(4):self.assertEqual({s['order'].index(original) for s in plan},set(range(4)))
            for s in plan:self.assertEqual(sorted(s['order'][i] for i in s['expected']),list(range(count)))

    def test_blind_review_payload_hides_author_labels_and_reasons(self):
        row=dict(question='Q',options=['A','B','C','D'],correct_indices=[1,3],option_truth=[False,True,False,True],
            option_reasons=['SECRET']*4,reasoning_requirement='scope',sources=[dict(title='title',scope='scope',source_text='source',extra_secret='SECRET')])
        payload=json.loads(review_prompt(row).split('INPUT:\n',1)[1])
        self.assertEqual(set(payload['task']),{'question','options'})
        self.assertNotIn('SECRET',json.dumps(payload))

    def test_authored_split_counts_and_cardinality_coverage(self):
        rows=[]
        def add(unit,split,design,q,options,checks=()):rows.append((unit,split,design,options))
        for f in (add_warehouse,add_planning,add_transport,add_retention):f(add)
        self.assertEqual(len(rows),88);self.assertEqual(len({(u,d) for u,_,d,_ in rows}),88)
        self.assertEqual(Counter(s for _,s,_,_ in rows),dict(train=48,dev=24,retention=16))
        self.assertEqual(Counter(sum(v[1] for v in opts) for _,s,_,opts in rows if s=='train'),{1:12,2:12,3:12,4:12})
        self.assertEqual(Counter(sum(v[1] for v in opts) for _,s,_,opts in rows if s=='dev'),{1:6,2:6,3:6,4:6})

if __name__=='__main__':unittest.main()
