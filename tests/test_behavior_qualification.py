"""Synthetic grading/schema tests ONLY; no model results or content approval."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from release_fixtures import row,write_release
from qwen_one_two.training.release import load_release
from qwen_one_two.training.inputs import digest
from qwen_one_two.training.behavior_qualification import grade,fp32,suite,summarize,CATEGORIES
from qwen_one_two.training.reference_qualification import load_reference_evidence
from qwen_one_two.training.component_producer import component_view


def annotation(op='Noul',status='Assessed',payload=None):
    if payload is None:payload={'p_true':1} if op=='Noul' else (
        {'acceptable_selection_ids':['A']} if op=='Choice' else {'score_interval':[.3,.3]})
    return {'operation':op,'status':status,'payload':payload}


def measured(op='Noul',value=1.,selected=None,status='Assessed'):
    # Explicitly invented schema fixtures: never written to real evidence paths.
    return {'actual':{'status':status,'value':value,'selected':selected,'reason_present':status!='Assessed'},
            'transport':dict.fromkeys(('latent_request','latent_return','bound_snapshot','candidate_binding',
                'consumed','single_prefill','cache_continuity','accounting'),True),'failure':None}


def record(op='Noul',**gold):
    return {'operation':op,'supervision':{'result':annotation(op,**gold)}}


class BehaviorQualificationTests(unittest.TestCase):
    def test_truth_meaning_not_canonical_text_or_probability_one(self):
        r=record();c=measured(value=.8)
        c['reconstruction']={'exact':False}
        self.assertTrue(grade(r,c)['passed'])
        for v in (.5,.2,float('nan'),1.01):
            self.assertFalse(grade(r,measured(value=v))['passed'])
        self.assertEqual(r['supervision']['result']['payload']['p_true'],1)

    def test_score_interval_has_no_widening(self):
        r=record('Score')
        self.assertTrue(grade(r,measured('Score',value=fp32(.3)))['passed'])
        self.assertFalse(grade(r,measured('Score',value=fp32(.300001)))['passed'])
        self.assertFalse(grade(r,measured('Score',value=.29))['passed'])
        self.assertEqual(r['supervision']['result']['payload']['score_interval'],[.3,.3])

    def test_choice_needs_actual_stable_identity(self):
        r=record('Choice',payload={'acceptable_selection_ids':['A','C']})
        for key,want in (('A',True),('C',True),('B',False),('0',False),(None,False)):
            self.assertEqual(grade(r,measured('Choice',value=None,selected=key))['passed'],want)

    def test_status_and_reason_presence_not_fabricated_values(self):
        r=record(status='Insufficient',payload={})
        self.assertTrue(grade(r,measured(status='Insufficient',value=None))['passed'])
        self.assertFalse(grade(r,measured(status='Insufficient',value=.2))['passed'])
        self.assertFalse(grade(r,measured(status='Conflicting',value=None))['passed'])
        c=measured(status='Insufficient',value=None);c['failure']='unbound Reason'
        self.assertFalse(grade(r,c)['passed'])

    def test_typed_result_presence_is_strict(self):
        c=measured('Choice',value=.3,selected='A')
        self.assertFalse(grade(record('Choice'),c)['passed'])
        c=measured();c['actual']['selected']='A'
        self.assertFalse(grade(record(),c)['passed'])
        c=measured();c['actual']['reason_present']=True
        self.assertFalse(grade(record(),c)['passed'])

    def test_existing_label_only_intervention_uses_reviewed_truth(self):
        r=record();r['labels']=r['supervision'].pop('result')
        r['supervision']['continuations']=['SCHEMA ONLY reconsider the supplied return']
        self.assertTrue(grade(r,measured())['passed'])
        self.assertFalse(grade(r,measured(value=.1))['passed'])

    def test_all_transport_properties_are_required(self):
        for key in measured()['transport']:
            c=measured();c['transport'][key]=False
            self.assertFalse(grade(record(),c)['passed'],key)

    def release(self):
        rows=[]
        for category,op in CATEGORIES.items():
            for side in (0,1):
                result=annotation(op,payload={'p_true':1-side} if op=='Noul' else None)
                rows.append(row(f'SCHEMA-{category}-{side}',operation=op,training_category=None,
                    evaluation_slice='X',family=f'SCHEMA-family-{category}',
                    construction={'template_version':'controlled-data-v1','scenario_spec':'SCHEMA ONLY',
                        'seed':17,'transformation':'SCHEMA ONLY'},
                    intervention={'category':category,'pair':f'SCHEMA-pair-{category}',
                        'side':side,'protocol':'clean_component' if category==4 else (None if category==6 else 'offline_semantic_intervention')},
                    semantics={'question':{'proposition':'SCHEMA ONLY'},'target':{'referent':'SCHEMA ONLY'}},
                    model_inputs={'evidence':{'facts':[],'rules':[]},
                        'prompt':'FORBIDDEN prompt shortcut','supplied_return':{'secret':'FORBIDDEN gold'}},
                    supervision={'result':result}))
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        write_release(temp.name,rows,purpose='development')
        return load_release(temp.name)

    def test_pair_failures_and_missing_cases_cannot_be_hidden(self):
        release=self.release();groups,_=suite(release);cases=[]
        for g in groups:
            for key in g['ids']:
                r=release.record(key);op=r['operation'];gold=r['supervision']['result']['payload']
                c=measured(op,value=gold.get('p_true',fp32(.3)) if op!='Choice' else None,
                           selected='A' if op=='Choice' else None);c['id']=key;cases.append(c)
        report=summarize(release,cases,unchanged=True)
        self.assertEqual(report['status'],'PASSED')  # schema arithmetic only
        cases[0]['actual']['value']=.1
        report=summarize(release,cases,unchanged=True)
        self.assertEqual(report['pairs_passed'],len(groups)-1)
        with self.assertRaisesRegex(ValueError,'missing/reordered'):
            summarize(release,cases[:-1],unchanged=True)
        self.assertIn('evaluated weights changed',summarize(release,cases,unchanged=False)['blockers'])

    def test_prompt_gold_and_reason_do_not_enter_receiver_view(self):
        release=self.release();r=release.record('SCHEMA-1-0')
        view,p,_=component_view(release,r,intervention=True)
        excerpts,body=view.materialize(p)
        self.assertNotIn(b'FORBIDDEN',body);self.assertFalse(excerpts)
        self.assertNotIn('reason',[r.role for r in view.references])
        changed=copy.deepcopy(r);changed['semantics']['target']['referent']='changed'
        with self.assertRaisesRegex(ValueError,'record/release drift'):
            component_view(release,changed,intervention=True)

    def test_old_exact_report_cannot_grant_semantic_acceptance(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'schema-only.json'
            raw=json.dumps({'protocol':'reference-recovery-v1','status':'PASSED',
                            'recovery':{'exact':100}}).encode();path.write_bytes(raw)
            with self.assertRaisesRegex(ValueError,'reconstruction is diagnostic only'):
                load_reference_evidence(path,self.release(),expected_hash=digest(raw))
            with self.assertRaisesRegex(ValueError,'report hash'):
                load_reference_evidence(path,self.release(),expected_hash='0'*64)


if __name__=='__main__':unittest.main()
