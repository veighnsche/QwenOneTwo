"""Joint reporting, loss dispatch and owner metadata; no neural computation."""
import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace,ModuleType
import importlib.util
import sys
import unittest
from unittest.mock import patch

from qwen_one_two.training.inputs import canonical,digest,Exposure
from qwen_one_two.training.eligibility import Eligibility
from qwen_one_two.training.learning_report import summarize_joint_learning
from qwen_one_two.training.behavior_qualification import grade
from qwen_one_two.training.receiver_profiles import receiver_parameter,direct_projection_owners
from qwen_one_two.g1_accounting import budget


def symbolic_episodes():
    # Replace only loss evaluation with named symbols; execute real dispatch and
    # masks without torch, tensors or a pretend CPU model.
    losses=ModuleType('qwen_one_two.training.losses');losses.Eligibility=Eligibility
    class Book:
        def __init__(self):self.fields={}
        def add(self,name,*args,**kwargs):self.fields[name]=name
        def finish(self):return dict(self.fields)
    losses.ExampleLosses=Book
    for name in ('ce','binary','permitted_set','regression','alignment'):setattr(losses,name,lambda *a,**kw:'symbolic-loss')
    spec=importlib.util.spec_from_file_location('qwen_one_two.training._rd05_symbolic_episodes','qwen_one_two/training/episodes.py')
    module=importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules,{'qwen_one_two.training.losses':losses,module.__name__:module}):spec.loader.exec_module(module)
    return module


class JointReceiverTests(unittest.TestCase):
    def test_real_dispatch_keeps_assessed_values_when_actual_status_abstains(self):
        from tests.test_direct_role_packets import fixture
        from qwen_one_two.codec.schema import Port,Status,Kind
        episodes=symbolic_episodes()
        for port,name in ((Port.NOUL,'noul'),(Port.CHOICE,'choice'),(Port.SCORE,'score')):
            rt,packet=fixture(port);snap=rt.registry.get(packet.snapshot)
            candidates=() if snap.bank is None else rt.registry.get(snap.bank).candidates
            labels=episodes.DecisionLabels(packet.snapshot,Status.ASSESSED,noul=.8,choice=candidates[:1],
                score_interval=(.3,.4),feasibility=tuple((r,0) for r in candidates))
            evaluated={'status_logits':'logits','raw':'raw','pointer_logits':'pointers',
                       'feasibility_logits':['f']*len(candidates)}
            losses,_=episodes.decision_losses(rt.registry,packet.snapshot,Status.UNRESOLVED,evaluated,labels)
            self.assertEqual(set(losses),{'status',name}|({'feasibility'} if candidates else set()))
            self.assertNotIn('alignment',losses)
            rejected,_=episodes.decision_losses(rt.registry,packet.snapshot,Status.ASSESSED,evaluated,
                                               episodes.DecisionLabels(None,Status.ASSESSED,noul=1.))
            self.assertEqual(rejected,{})

    def test_diagnostic_and_native_admission_guards_are_preserved(self):
        from tests.test_direct_role_packets import fixture
        from qwen_one_two.codec.schema import Port
        episodes=symbolic_episodes()
        with self.assertRaisesRegex(ValueError,'diagnostic reference'):
            episodes.collect_session(None,SimpleNamespace(episode_report={'diagnostic_reference':True}),{},Exposure('native_writer'))
        rt,_=fixture(Port.NOUL)
        with self.assertRaisesRegex(ValueError,'native semantics stay masked'):
            episodes.collect_session(None,SimpleNamespace(runtime=rt,episode_report={}),{},Exposure('native_writer'))

    def test_owner_list_and_masks_exclude_frozen_and_writer_only_parameters(self):
        for name in ('codec.direct_role_projection.weight','codec.direct_role_projection.bias','codec.status.weight','codec.score_row'):
            self.assertTrue(receiver_parameter('new_matrix',name))
        for name in ('codec.direct_role_length.weight','codec.budget_row','reference.pooler.weight','layers.0.R_A'):
            self.assertFalse(receiver_parameter('R',name))
        groups=direct_projection_owners()
        self.assertEqual([g['weight_decay'] for g in groups],[.01,0.])
        self.assertEqual([p['shape'] for g in groups for p in g['parameters']],[[1024,1024],[1024]])
        tree=ast.parse(Path('qwen_one_two/training/receiver_learning.py').read_text())
        function=ast.unparse(next(n for n in tree.body if getattr(n,'name',None)=='extend_profile_window'))
        self.assertIn('clone_state(state)',function);self.assertNotIn('.step(',function)
        self.assertNotIn('exp_avg',function)  # no fabricated moments or copied Adam algorithm

    def test_failed_empty_component_still_occupies_a_scheduled_exposure(self):
        from tests.test_direct_role_packets import fixture
        from qwen_one_two.codec.schema import Port
        episodes=symbolic_episodes();rt,_=fixture(Port.NOUL);rt.terminate('Failed',usable=False,emit_end=False)
        session=SimpleNamespace(runtime=rt,episode_report={'input_profile':'direct-role-states-v1'},
            state=SimpleNamespace(position=rt.position),summary=lambda:{**rt.summary(),'prefills':1,'lineage':1,'override':None})
        recorded=[];window=SimpleNamespace(phase='components',add_episode=lambda *args,**kw:recorded.append((args,kw)))
        result=episodes.collect_session(window,session,{},Exposure('isolated_receiver'))
        self.assertFalse(result['accepted']);self.assertEqual(recorded[0][0],({},))
        self.assertFalse(recorded[0][1]['complete']);self.assertEqual(recorded[0][1]['failure'],'Failed')
        with self.assertRaisesRegex(ValueError,'requires prepared receiver admission'):
            episodes.collect_session(window,session,{'status':'symbolic'},Exposure('isolated_receiver'))

    def report_fixture(self):
        rows={};heads=[];actual=[]
        for i,(op,status) in enumerate((('Noul','Assessed'),('Noul','Unresolved'),('Score','Assessed'))):
            payload={'p_true':1.} if op=='Noul' and status=='Assessed' else {'score_interval':[.3,.4]} if op=='Score' else None
            row={'id':str(i),'operation':op,'supervision':{'result':{'operation':op,'status':status,'payload':payload}}};rows[str(i)]=row
            # Correct abstention plus failed assessed status must stay separate.
            head={'record':str(i),'record_sha256':digest(canonical(row)),'input_sha256':digest(str(i).encode()),
                'head':{'status':'Unresolved','value':.35,'raw':.35},'losses':{'status':1.,**({'score':.1} if op=='Score' else {})},
                'reason_mask':'unsupported new-profile Reason','summary':{'terminal':'Completed'},
                'startup':{'raw':.5,'predicted_budget':budget(.5),'override':512,'granted_budget':512}}
            case={'id':str(i),'record_sha256':head['record_sha256'],'predicted_status':'Unresolved','predicted_value':.35,
                'actual':{'status':'Unresolved','value':None,'selected':None,'reason_present':True},'failure':None,
                'transport':{k:True for k in ('latent_request','latent_return','bound_snapshot','candidate_binding','consumed','single_prefill','cache_continuity','accounting')}}
            case['score']=grade(row,case);heads.append(head);actual.append(case)
        return SimpleNamespace(record=lambda key:deepcopy(rows[key])),heads,actual,list(rows)

    def test_joint_report_separates_abstentions_status_collapse_and_retention(self):
        release,heads,actual,ids=self.report_fixture();before=deepcopy(heads)
        heads[0]['startup'].update(raw=.6,predicted_budget=budget(.6))
        result=summarize_joint_learning(release,heads,actual,ids,startup_baseline=before)
        self.assertEqual(result['cells']['Noul/Assessed']['delivered_passed'],0)
        self.assertEqual(result['cells']['Noul/Unresolved']['delivered_passed'],1)
        self.assertEqual(result['cells']['Score/Assessed']['head_status_correct'],0)
        self.assertEqual(result['eligible_examples'],{'status':3,'score':1})
        self.assertEqual(result['startup_retention']['raw_changed'],1)
        self.assertEqual(result['startup_retention']['mapped_budget_changed'],1)
        self.assertEqual(result['startup_retention']['missing'],0)
        heads[0]['startup']['predicted_budget']=512
        with self.assertRaisesRegex(ValueError,'continuous mapping drift'):
            summarize_joint_learning(release,heads,actual,ids,startup_baseline=before)

    def test_missing_startup_and_failed_exposures_are_not_counted_as_measured_success(self):
        release,heads,actual,ids=self.report_fixture();heads[0]['failure']='model-free failed fixture'
        result=summarize_joint_learning(release,heads,actual,ids)
        self.assertEqual(result['eligible_examples']['status'],2)
        self.assertEqual(result['startup_retention']['paired'],0)
        self.assertEqual(result['startup_retention']['missing'],3)


if __name__=='__main__':unittest.main()
