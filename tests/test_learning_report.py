"""CPU metadata/report checks; no model libraries or model execution."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from qwen_one_two.training.inputs import canonical,digest
from qwen_one_two.training.learning_report import probe_panel,summarize_learning
from qwen_one_two.training.behavior_qualification import fp32,grade


def row(op,status='Assessed',payload=None):
    return {'id':'fixture','operation':op,'supervision':{'result':{'operation':op,'status':status,'payload':payload}}}


class LearningReportTests(unittest.TestCase):
    def test_real_panel_is_training_only_reproducible_and_covers_all_cells(self):
        from collections import Counter
        from qwen_one_two.training.release import load_release
        from scripts.fit_reference import TRAIN_SHA
        release=load_release('data/training-collections-20261105/releases/train',expected_hash=TRAIN_SHA)
        panel=probe_panel(release)
        self.assertEqual(panel,probe_panel(release));self.assertEqual(len({p['family'] for p in panel}),32)
        self.assertEqual(Counter(p['operation'] for p in panel),{'Noul':11,'Choice':8,'Score':13})
        self.assertEqual(len({(p['operation'],p['status']) for p in panel}),13)
        noul=[release.record(p['id'])['supervision']['result']['payload']['p_true'] for p in panel
              if p['operation']=='Noul' and p['status']=='Assessed']
        self.assertEqual(sum(x>.5 for x in noul),4);self.assertEqual(sum(x<.5 for x in noul),4)
        self.assertTrue(all(release.record(p['id'])['split']=='train' for p in panel))

    def summarize(self,r,head,**extra):
        release=SimpleNamespace(record=lambda key:deepcopy(r))
        case={'record':r['id'],'record_sha256':digest(canonical(r)),'head':head,
              'summary':{'terminal':'Completed'},**extra}
        return summarize_learning(release,[case],[r['id']],kind='heads')['operations'][r['operation']]

    def test_correct_abstention_does_not_count_as_assessed_learning(self):
        r=row('Noul',payload={'p_true':1.})
        s=self.summarize(r,{'status':'Unresolved','value':.8,'raw':1.4})
        self.assertEqual(s['counts']['assessed_scheduled'],1)
        self.assertEqual(s['counts']['assessed_head_correct'],1)
        self.assertEqual(s['counts']['assessed_head_status_correct'],0)
        self.assertAlmostEqual(s['noul_brier']['mean'],.04)
        s=self.summarize(row('Noul','Unresolved'),{'status':'Unresolved'})
        self.assertEqual(s['counts']['assessed_scheduled'],0)
        self.assertEqual(s['counts']['status_correct'],1)

    def test_noul_tie_and_missing_or_failed_head_are_not_success(self):
        r=row('Noul',payload={'p_true':0.})
        s=self.summarize(r,{'status':'Assessed','value':.5})
        self.assertEqual(s['counts']['assessed_head_correct'],0);self.assertEqual(s['counts']['assessed_probability_ties'],1)
        s=self.summarize(r,{},failure='failed before head')
        self.assertEqual(s['counts']['scheduled'],1);self.assertEqual(s['counts']['assessed_scheduled'],1)
        self.assertEqual(s['counts']['failed'],1);self.assertEqual(s['noul_brier'],{'measured':0,'mean':None})

    def test_score_uses_existing_fp32_endpoints_without_widening(self):
        r=row('Score',payload={'score_interval':[.1,.3]})
        s=self.summarize(r,{'status':'Assessed','value':fp32(.3),'raw':fp32(.3)})
        self.assertEqual(s['counts']['assessed_head_status_correct'],1)
        s=self.summarize(r,{'status':'Assessed','value':fp32(.3)+1e-6,'raw':2.})
        self.assertEqual(s['counts']['assessed_head_correct'],0);self.assertEqual(s['counts']['assessed_clipped_readouts'],1)
        self.assertGreater(s['score_interval_distance']['mean'],0)

    def test_choice_matches_identity_and_coverage_is_exact(self):
        r=row('Choice',payload={'acceptable_selection_ids':['A']})
        self.assertEqual(self.summarize(r,{'status':'Assessed','selected':'A'})['counts']['assessed_head_status_correct'],1)
        self.assertEqual(self.summarize(r,{'status':'Assessed','selected':'B'})['counts']['assessed_head_correct'],0)
        with self.assertRaisesRegex(ValueError,'coverage'):
            summarize_learning(SimpleNamespace(record=lambda key:r),[],['fixture'],kind='heads')
        with self.assertRaisesRegex(ValueError,'training release'):probe_panel(SimpleNamespace(purpose='development'))

    def test_behavior_uses_actual_delivery_grade_and_separate_head_readout(self):
        r=row('Noul',payload={'p_true':1.})
        case={'id':'fixture','record_sha256':digest(canonical(r)),
              'actual':{'status':'Unresolved','value':None,'selected':None,'reason_present':True},
              'predicted_status':'Unresolved','predicted_value':.9,'transport':{},'failure':None}
        case['score']=grade(r,case)
        result=summarize_learning(SimpleNamespace(record=lambda key:r),[case],['fixture'],kind='behavior')
        counts=result['operations']['Noul']['counts']
        self.assertEqual(counts['assessed_head_correct'],1);self.assertEqual(counts['assessed_behavior_passed'],0)


if __name__=='__main__':unittest.main()
