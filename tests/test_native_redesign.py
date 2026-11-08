"""RD-06 metadata/provenance tests; literal symbols, no neural workloads."""
import ast
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import unittest
from unittest.mock import patch

from qwen_one_two.training.inputs import canonical,digest,Exposure
from qwen_one_two.training.release import load_release
from qwen_one_two.training.producer import native_plan,EpisodeProducer
from qwen_one_two.training.binding import request_targets,bind_labels,compare_request,compatible_return,request_fields,UnsupportedNativeTargets
from qwen_one_two.training.native import NativeEpisode,target_layout,GPU_CHECKS
from qwen_one_two.codec.layout import DIRECT
from qwen_one_two.codec.schema import Port
from tests.test_direct_role_encoding import MetadataTokenizer
from tests.test_direct_role_packets import fixture


class NativeRedesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train=load_release('data/training-collections-20261105/releases/train',expected_hash='c70d4f44135a34c00e7fa0348833b1ec56613def15f5753231589492ed35478f')
        cls.dev=load_release('data/training-collections-20261105/releases/development',expected_hash='5b8f4cffdc4f33e255d3a34306cac03a68c50dcd9dab905ae8d8adcf89c81e77')
        cls.writer=next(cls.train.record(k) for k in sorted(cls.train.selected_ids) if cls.train.record(k)['training_category']=='writer')

    def test_explicit_native_evaluation_has_no_gold_targets_or_guidance(self):
        for slice_ in ('W','F'):
            row=next(self.dev.record(k) for k in sorted(self.dev.selected_ids) if self.dev.record(k).get('evaluation_slice')==slice_)
            plan=native_plan(self.dev,row,'free_joint',1,evaluation=True)
            self.assertIsNone(plan.target);self.assertIsNone(plan.effort);self.assertIsNone(plan.override);self.assertEqual(plan.calls,())
            with self.assertRaisesRegex(ValueError,'held-out free-control'):
                native_plan(self.dev,row,'guided_joint',1,evaluation=True)

    def test_direct_training_targets_preflight_before_any_encoder_and_do_not_encode_Reason(self):
        calls=[];tokenizer=MetadataTokenizer();reference=SimpleNamespace(backbone=SimpleNamespace(tokenizer=tokenizer),frozen=True,
            artifact_id='reference-fixture',fingerprint=lambda:'reference-fixture')
        encoder=SimpleNamespace(source_policy='frozen-w-source-v1',role_policy=DIRECT,
            _body_states=lambda *args:(calls.append(args[0]) or ('symbolic-field',),None))
        with patch('qwen_one_two.training.binding.tensor_hash',lambda value:digest(canonical(value))), \
             patch('qwen_one_two.training.binding.encode_annotation',side_effect=AssertionError('pooled encoder forbidden')):
            target=request_targets(reference,self.train,self.writer,b'native prompt bytes',profile=DIRECT,
                                   source_encoder=encoder,source_tokens=8)
        self.assertEqual(len(calls),len(target.references));self.assertEqual(target.field_lengths,(1,)*len(calls))
        self.assertIsNone(target.reason);self.assertEqual(target.profile,DIRECT)
        with self.assertRaisesRegex(UnsupportedNativeTargets,'evidence count'):
            request_targets(reference,self.train,self.writer,b'bytes',profile=DIRECT,
                            source_encoder=lambda:(_ for _ in ()).throw(AssertionError('must reject before encoder')),source_tokens=257)

    def test_native_vectors_never_gain_labels_from_exact_targets_or_record_ID(self):
        target=SimpleNamespace(profile=DIRECT)
        self.assertIsNone(bind_labels(None,None,None,None,target)[0])
        self.assertFalse(compare_request(None,None,target)[0]);self.assertFalse(compatible_return(None,None,None,target))

    def test_rubric_fields_use_explicit_lengths_without_midpoint_padding(self):
        rt,packet=fixture(Port.SCORE)
        _,_,fields,_=request_fields(rt.registry,packet.snapshot)
        self.assertEqual([role for role,_ in fields],['question','target','dimension','conditions','zero','one'])
        self.assertEqual([len(values) for _,values in fields],[2,3,1,2,1,2])

    def test_native_receipt_masks_decision_post_call_continuation_and_dev_training(self):
        rt,_=fixture(Port.NOUL);source=rt.call.snapshot
        session=SimpleNamespace(runtime=rt,episode_report={'release_purpose':'training','evaluation':False})
        admission=NativeEpisode(rt,source,'symbolic-source','writer',False)
        session.episode_report['native_admission']=admission.receipt()
        with patch('qwen_one_two.training.binding.source_hash',return_value='symbolic-source'),patch.object(rt.registry,'get',return_value=None):
            admission.validate(session,{'gate':'symbolic','writer':'symbolic'})
            for name in ('native','status','noul','choice','score','feasibility','alignment'):
                with self.assertRaisesRegex(ValueError,'unsupported native'):admission.validate(session,{name:'symbolic'})
            session.episode_report['release_purpose']='development'
            with self.assertRaisesRegex(ValueError,'development native'):admission.validate(session,{})

    def test_components_and_supplied_readers_cannot_be_silently_converted(self):
        codec=SimpleNamespace(require_profile=lambda profile:None)
        producer=EpisodeProducer(None,codec,self.train,packet_profile=DIRECT)
        row=next(self.train.record(k) for k in sorted(self.train.selected_ids) if self.train.record(k)['training_category']=='reader')
        with self.assertRaisesRegex(ValueError,'cannot substitute'):producer(row,'components',1)
        with self.assertRaisesRegex(ValueError,'development needs explicit evaluation'):
            EpisodeProducer(None,codec,self.dev,packet_profile=DIRECT)

    def test_live_source_is_original_prefill_and_direct_targets_are_prepared_before_session(self):
        tree=ast.parse(Path('qwen_one_two/training/producer.py').read_text())
        cls=next(n for n in tree.body if getattr(n,'name',None)=='EpisodeProducer')
        code=ast.unparse(next(n for n in cls.body if getattr(n,'name',None)=='__call__'))
        self.assertLess(code.index('prepared[index] = request_targets'),code.index('session = Session'))
        self.assertIn('h[0, selected]',code);self.assertIn('h = session.prefill(ids)',code)
        method=ast.unparse(next(n for n in cls.body if getattr(n,'name',None)=='_guided_call'))
        self.assertIn('allow_attributes=False',method);self.assertNotIn('source_encoder',method)
        self.assertNotIn('reference_payload',method);self.assertIn('decode_result(self.codec, rt, evaluated)',method)
        self.assertIn('s.deliver(returned)',method)

    def test_model_free_modules_GPU_collection_and_guard_registration(self):
        subprocess.run([sys.executable,'-c',"""
import sys
class Block:
 def find_spec(self,name,*args):
  if name.split('.')[0] in {'torch','numpy','transformers','safetensors','peft'}:raise RuntimeError(name)
sys.meta_path.insert(0,Block())
import qwen_one_two.training.native
import qwen_one_two.training.native_coverage
import qwen_one_two.training.native_evaluation
import tests.test_native_redesign_gpu
"""],check=True)
        tree=ast.parse(Path('tests/test_native_redesign_gpu.py').read_text());cls=next(n for n in tree.body if getattr(n,'name',None)=='NativeRedesignGPU')
        methods={n.name:ast.unparse(n) for n in cls.body if isinstance(n,ast.FunctionDef)}
        self.assertEqual(set(GPU_CHECKS.values()),{n for n in methods if n.startswith('test_')})
        self.assertLess(methods['setUpClass'].index('CUDA_VISIBLE_DEVICES'),methods['setUpClass'].index('import torch'))
        self.assertLess(methods['setUpClass'].index('QOT_RD06_GPU_QUALIFY'),methods['setUpClass'].index('import torch'))
        for name in GPU_CHECKS.values():self.assertIn('assert',methods[name])

    def test_native_final_values_require_separate_native_observations(self):
        from qwen_one_two.training.native_evaluation import grade_final
        from qwen_one_two.training.behavior_qualification import fp32
        for op,payload,value,identity in [('Noul',{'p_true':1.},.8,None),
            ('Score',{'score_interval':[.2,.4]},fp32(.4),None),('Choice',{'acceptable_selection_ids':['A']},None,'A')]:
            row={'operation':op,'supervision':{'result':{'status':'Assessed','payload':payload}}}
            case={'record_sha256':digest(canonical(row)),'exposure':'actual_native','actual_return_consumed':True,
                  'one_prefill':True,'one_lineage':True,
                  'final_observation':{'origin':'generated_native_answer','status':'Assessed','value':value,'identity':identity}}
            self.assertTrue(grade_final(row,case)['passed'])
            if op=='Choice':case['final_observation']['value']=.5
            else:case['final_observation']['identity']='foreign'
            self.assertFalse(grade_final(row,case)['passed'])
            case['final_observation'].update(value=value,identity=identity)
            case['exposure']='supplied_return_reader';self.assertFalse(grade_final(row,case)['supported'])
            case['exposure']='actual_native';case['final_observation']['origin']='receiver_heads'
            self.assertFalse(grade_final(row,case)['supported'])

    def test_change_pairs_do_not_relabel_component_or_supplied_reader_evidence(self):
        from qwen_one_two.training.native_evaluation import grade_change_pairs
        manifest={'change_pairs':{'reader':{'exposure':'supplied_return_reader','cases':['A','B']},
                                  'native':{'exposure':'actual_native','cases':['C','D']}}}
        result=grade_change_pairs(None,[],manifest)
        self.assertEqual(len(result),2);self.assertFalse(any(r['supported'] or r['passed'] for r in result))


if __name__=='__main__':unittest.main()
