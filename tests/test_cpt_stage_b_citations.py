import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from cpt_stage_b_citations import locate,normalized_map,source_blocks,validate_option_evidence


class CitationsTest(unittest.TestCase):
    def test_source_blocks_cover_source_and_validate_provenance(self):
        g=dict(id='test',source_text=' First line\nsecond line.\n\nNext paragraph. ')
        b=source_blocks(g)
        self.assertEqual(len(b),2)
        for x in b:self.assertEqual(x['text'],g['source_text'][x['start']:x['end']])
        rows=[dict(option_index=0,relation='entailed',source_ids=[b[0]['id']],reason='reason'),dict(option_index=1,relation='insufficient',source_ids=[],reason='unknown')]
        self.assertTrue(validate_option_evidence(rows,2,b))
        for bad in [[],[dict(rows[0],source_ids=['unknown']),rows[1]],[rows[0],rows[0]],[dict(rows[0],source_ids=[]),rows[1]]]:
            with self.assertRaises(ValueError):validate_option_evidence(bad,2,b)

    def test_whitespace_with_original_offsets(self):
        s='  First\n rule.  Another\t rule. '
        row=locate(s,'First rule.')[0]
        self.assertTrue(row['located'])
        for m in row['matches']:
            self.assertEqual(s[m['start']:m['end']],m['source_span'])
            self.assertEqual(normalized_map(m['source_span'])[0],'First rule.')

    def test_disjoint_spans_are_not_invented_continuity(self):
        self.assertTrue(all(p['located'] for p in locate('Alpha. X. Beta.','Alpha. ... Beta.')))
        self.assertFalse(locate('Alpha. X. Beta.','Alpha. Beta.')[0]['located'])

    def test_no_fuzzy_word_or_punctuation_acceptance(self):
        for q in ['alpha.','Alpha!','Alpha is true.']:
            self.assertFalse(locate('Alpha.',q)[0]['located'])

    def test_repeated_matches_and_short_anchors(self):
        r=locate('Rail. Rail.','Rail.')[0]
        self.assertEqual(len(r['matches']),2);self.assertTrue(r['short_anchor'])
        self.assertEqual(locate('Rail.','...'),[])


if __name__=='__main__':unittest.main()
