"""Offline construction regressions. Synthetic schema fixtures, no model calls."""
import copy
from collections import Counter
import unittest
from scripts.produce_training_data import (scenario,make_view,permuted_choice,
                                          intervention,training_views,evaluation_views,DOMAINS)
from qwen_one_two.training.inputs import canonical


def fixture(i=0,split='train'):
 skin={'id':f'{split}-SCHEMA-{i}','subject':'SCHEMA target '+str(i),'question':'Assess the supplied fictional rule.','a':'first prerequisite','b':'second prerequisite','unit':'checkpoints'}
 return scenario(skin,split,DOMAINS[(i//3)%6],i)


class ProductionTests(unittest.TestCase):
 def test_decisive_false_unknown_and_conflict_stay_distinct(self):
  rows=fixture()
  self.assertEqual(rows[2]['labels']['status'],'Insufficient')
  self.assertIsNone(rows[2]['labels']['payload'])
  self.assertEqual(rows[3]['labels']['status'],'Conflicting')
  self.assertEqual(rows[5]['labels']['payload'],{'p_true':0})
  self.assertEqual(len(rows[5]['supplied_facts']),1)

 def test_choice_feasibility_is_not_ranking_or_missingness(self):
  rows=fixture(1)
  self.assertEqual(rows[0]['labels']['feasibility'],{'A':True,'B':True,'C':False})
  self.assertEqual(rows[0]['labels']['payload'],{'acceptable_selection_ids':['A']})
  self.assertEqual(rows[2]['labels']['feasibility']['A'],'unknown')
  self.assertEqual(rows[3]['labels']['status'],'Conflicting')
  self.assertEqual(rows[3]['labels']['feasibility']['B'],True)
  self.assertEqual(rows[5]['labels']['status'],'NoFeasibleChoice')
  self.assertIsNone(rows[5]['labels']['payload'])

 def test_bijection_changes_ids_and_preserves_unknowns(self):
  row=fixture(1)[3];changed=permuted_choice(row,1)
  self.assertEqual(changed['labels']['feasibility'],{'B':'unknown','C':True,'A':False})
  self.assertIn('B,C,A',changed['human_input']['prompt'])
  self.assertEqual(row['labels']['feasibility']['A'],'unknown')

 def test_reader_has_no_original_evidence_or_invented_cause(self):
  c=fixture()[1];r=make_view(c,'reader','reader-fixture')
  self.assertEqual(set(r['model_inputs']),{'bound_return','permitted_prefix'})
  self.assertNotIn('evidence',r['model_inputs'])
  self.assertIsNone(r['model_inputs']['bound_return']['reason'])
  self.assertNotIn('second prerequisite: no',str(r['supervision']))

 def test_hidden_world_twins_have_identical_complete_declared_inputs(self):
  c=fixture()[2];a=intervention(c,4,'a',0);b=intervention(c,4,'b',1)
  self.assertEqual(canonical(a['model_inputs']),canonical(b['model_inputs']))
  self.assertEqual(a['labels']['status'],b['labels']['status'])
  self.assertNotEqual(a['intervention'],b['intervention'])

 def test_changed_rubrics_update_value_meaning_and_native_target(self):
  c=fixture(8)[2];v=c['labels']['payload']['score_interval'][0]
  a=intervention(c,7,'a',1);b=intervention(c,8,'b',1)
  self.assertEqual(a['labels']['payload']['score_interval'],[1-v,1-v])
  self.assertEqual(b['labels']['payload']['score_interval'],[v/2,v/2])
  self.assertIn('remaining',a['semantics']['rubric']['dimension'])
  self.assertNotEqual(a['guidance']['acceptable_continuations'],c['guidance']['acceptable_continuations'])

 def test_clean_reader_follows_supplied_value_without_revealing_cause(self):
  c=fixture()[0];a=intervention(c,9,'a',0);b=intervention(c,9,'b',1)
  self.assertEqual(a['model_inputs']['bound_return']['payload'],{'p_true':1})
  self.assertEqual(b['model_inputs']['bound_return']['payload'],{'p_true':0})
  self.assertIn('No, according to',b['supervision']['continuations'][0])
  self.assertNotIn('facts',b['model_inputs'])

 def test_exact_training_counts_and_prescribed_corrections(self):
  cases=[c for i in range(18) for c in fixture(i)];rows=training_views(cases)
  self.assertEqual(Counter(r['category'] for r in rows),{'reference':500,'receiver':1500,'writer':750,'reader':750,'interleaved':2000,'direct':500,'language_replay':1000,'decision_replay':1000})
  corrections=[r['correction_specification'] for r in rows if 'correction_specification' in r]
  self.assertEqual(len(corrections),400)
  self.assertEqual(set(Counter(c['family'] for c in corrections).values()),{80})
  self.assertTrue(all(c['requires_runtime_materialization'] and not c['observed_model_failure'] for c in corrections))
  self.assertTrue(all(c['native_cancellation_target'] is None for c in corrections))

 def test_locked_allocations_and_mechanical_absence(self):
  cases=[c for i in range(18) for c in fixture(i,'locked')];rows=evaluation_views(cases,'locked')
  self.assertEqual(Counter(r['evaluation_slice'] for r in rows),{'RF':100,'W':100,'E':130,'R':130,'F':1000,'X':500,'M':40})
  for r in rows:
   if r['evaluation_slice']=='M':
    self.assertNotIn('status',r['labels']);self.assertNotIn('payload',r['labels'])

if __name__=='__main__':unittest.main()
