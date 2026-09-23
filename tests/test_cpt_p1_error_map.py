import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from build_cpt_p1_error_map import matching_pairs


def case(key,n,definition,label,index=0):
    options=[f'Other {i}' for i in range(n)];options[index]=label
    return dict(item_hash=key,question='Header Which of the following best matches: '+definition,
                options=options,expected=[index])


class PairingTests(unittest.TestCase):
    def test_pairs_by_answer_text_not_position(self):
        a=case('large',269,'A  supplied definition','TARGET',198)
        b=case('small',4,'a supplied definition','target',1)
        self.assertEqual(matching_pairs({'large':a,'small':b}),[('large','small')])

    def test_different_definition_or_gold_never_paired(self):
        a=case('large',269,'One precise definition','TARGET')
        b=case('different_definition',4,'A related definition','TARGET')
        c=case('different_gold',4,'One precise definition','OTHER')
        self.assertEqual(matching_pairs({r['item_hash']:r for r in [a,b,c]}),[])

    def test_one_to_many_retained_for_explicit_deduplication(self):
        rows=[case('large',269,'Same definition','TARGET'),case('small1',3,'Same definition','TARGET'),case('small2',5,'Same definition','TARGET')]
        pairs=matching_pairs({r['item_hash']:r for r in rows})
        self.assertEqual(len(pairs),2)
        self.assertEqual(len({a for a,b in pairs}),1)


if __name__=='__main__':unittest.main()
