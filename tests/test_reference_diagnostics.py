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

    def test_resume_reuses_complete_and_partial_results_and_rejects_corruption(self):
        import json,tempfile
        from pathlib import Path
        from qwen_one_two.training.inputs import canonical
        from qwen_one_two.training.diagnostics import load_resume,PROTOCOL
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows=choose_cases(self.rows(),anchors=())['search']
            measured=[dict(row,recovery=dict(exact=False,matched_tokens=0,target_tokens=2,
                        first_mismatch=dict(index=0,predicted_token=1,target_token=2))) for row in rows]
            artifacts={'report.json':dict(protocol=PROTOCOL,status='INCOMPLETE',completed_search_candidates=1,completed_screen_candidates=0),
                'cases.json':dict(splits=dict(search=rows)), 'feature-identities.json':dict(schema_fixture_only=True),
                'edits.json':{'0':dict(reference_id='SCHEMA-ONLY'), '20':dict(reference_id='SCHEMA-PARTIAL')},
                'search-00.json':dict(mask=0,stage='search',reference_id='SCHEMA-ONLY',cases=measured,executed=20,exact=0)}
            for name,value in artifacts.items():(root/name).write_bytes(canonical(value))
            (root/'events.jsonl').write_text(json.dumps(dict(event='recovery',stage='search',mask=20,**measured[0]))+'\n')
            receipt=root/'receipt.json';receipt.write_bytes(canonical(dict(format='qot-reference-block-search-evidence-v1',
                assets={str(p):digest(p.read_bytes()) for p in root.iterdir()})))
            result=load_resume(root,receipt)
            self.assertEqual(result['search'][0],artifacts['search-00.json'])
            self.assertEqual(result['partial'][20][rows[0]['id']]['recovery'],measured[0]['recovery'])
            (root/'search-00.json').write_bytes(b'{}')
            with self.assertRaisesRegex(ValueError,'artifact hash'):load_resume(root,receipt)

    def test_reused_recovery_requires_same_case_and_target(self):
        from qwen_one_two.training.diagnostics import reused_recovery
        row=self.rows()[0];previous=dict(row,recovery=dict(target_tokens=2,exact=True,matched_tokens=2,first_mismatch=None))
        self.assertEqual(reused_recovery(row,[2,248046],previous),previous['recovery'])
        with self.assertRaisesRegex(ValueError,'source identity'):reused_recovery(dict(row,family='FOREIGN-SCHEMA'),[2,248046],previous)
        with self.assertRaisesRegex(ValueError,'target identity'):reused_recovery(row,[1,2,248046],previous)
