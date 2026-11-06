"""Metadata/schema checks only; fixture strings are not reviewed training data."""
import unittest
from qwen_one_two.training.diagnostics import (BLOCKS,MASKS,block_for,choose_cases,first_difference,ranking,eligible)
from qwen_one_two.training.inputs import SLOTS,digest


class DiagnosticMetadataTests(unittest.TestCase):
    def rows(self):
        return [dict(id=f'SCHEMA-{r}-{i}',role=r,family=f'SCHEMA-FAMILY-{r}-{i}',
                     annotation_sha256=digest(f'SCHEMA-ONLY:{r}:{i}'.encode())) for r in SLOTS for i in range(5)]

    def test_all_component_combinations_once(self):
        self.assertEqual(len(MASKS),32);self.assertEqual(set(MASKS),set(range(32)))
        for branch in ('shared','target'):
            for i,block in enumerate(BLOCKS):self.assertEqual(block_for(f'pooler.{branch}.{block}.weight'),i)
        self.assertIsNone(block_for('T.0'))
        for name in ('pooler.foreign.queries','pooler.shared.unknown'):
            with self.assertRaises(ValueError):block_for(name)

    def test_splits_are_deterministic_distinct_and_family_disjoint(self):
        rows=self.rows();anchors=(rows[0]['id'],rows[1]['id'])
        result=choose_cases(rows,anchors);self.assertEqual(result,choose_cases(reversed(rows),anchors))
        self.assertTrue(set(anchors)<={r['id'] for r in result['search']})
        for stage in result.values():
            self.assertEqual(len(stage),20)
            for role in SLOTS:self.assertEqual(sum(r['role']==role for r in stage),2)
        for field in ('family','annotation_sha256'):
            self.assertFalse({r[field] for r in result['search']}&{r[field] for r in result['screen']})
        duplicate=dict(rows[0],id='ZZ-SCHEMA-duplicate')
        self.assertEqual(result,choose_cases(rows+[duplicate],anchors))

    def test_bad_splits_reject(self):
        rows=self.rows()
        with self.assertRaisesRegex(ValueError,'identities'):choose_cases(rows+rows[:1],())
        with self.assertRaisesRegex(ValueError,'insufficient'):choose_cases([r for r in rows if r['role']!='key'],())
        with self.assertRaisesRegex(ValueError,'overlap'):choose_cases(rows,(rows[0]['id'],rows[0]['id']))

    def test_difference_uses_only_shared_history(self):
        self.assertEqual(first_difference([1,2,3,248046],[1,2,4,248046]),2)
        for a,b in (([1,248046],[1,248046]),([1,248046],[1,2,248046]),([1],[2])):
            with self.assertRaises(ValueError):first_difference(a,b)

    def result(self,mask,ids,margin=0.):
        return dict(mask=mask,exact=len(ids),mean_content_margin=margin,mean_swap_effect=0.,
                    cases=[dict(id=i,recovery=dict(exact=True)) for i in ids])

    def test_ranking_and_no_regression_gate(self):
        base=self.result(0,['A']);gain=self.result(1,['A','B']);regression=self.result(2,['B','C'])
        self.assertTrue(eligible(gain,gain,base,base));self.assertFalse(eligible(regression,gain,base,base))
        self.assertFalse(eligible(gain,base,base,base));self.assertFalse(eligible(gain,regression,base,base))
        self.assertLess(ranking(gain),ranking(base))
        self.assertLess(ranking(self.result(1,['A'])),ranking(self.result(3,['A'])))
