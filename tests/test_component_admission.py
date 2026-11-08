"""Symbolic binding/ABI tests only; no neural features, labels or CPU models."""
from copy import deepcopy
from dataclasses import replace
import ast
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace,ModuleType
import unittest
from unittest.mock import patch

from qwen_one_two.training.inputs import canonical,digest
from qwen_one_two.training.release import load_release
from qwen_one_two.training.component_producer import component_view
from qwen_one_two.training.component_admission import admit_prepared_component,GPU_CHECKS
from qwen_one_two.training.eligibility import Eligibility
from qwen_one_two.codec.records import Source,Span,Semantic
from qwen_one_two.codec.runtime import Runtime
from qwen_one_two.codec.schema import Kind,Port,Status
from qwen_one_two.codec.layout import DIRECT
from tests.test_direct_role_packets import SymbolicRegistry


def symbolic_hash(value):return digest(canonical(value))
def symbolic_source(value):
    return digest(canonical([value.text.hex(),value.token_ids,value.offsets,value.states]))


class ComponentAdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train=load_release('data/training-collections-20261105/releases/train',expected_hash='c70d4f44135a34c00e7fa0348833b1ec56613def15f5753231589492ed35478f')
        cls.dev=load_release('data/training-collections-20261105/releases/development',expected_hash='5b8f4cffdc4f33e255d3a34306cac03a68c50dcd9dab905ae8d8adcf89c81e77')

    def setUp(self):
        self.enterContext(patch('qwen_one_two.training.component_admission.tensor_hash',symbolic_hash))
        self.enterContext(patch('qwen_one_two.training.component_admission.source_hash',symbolic_source))

    def fixture(self,operation='Choice'):
        # Field extraction is replaced by literal symbols. This does not assert
        # that any real annotation fits RD-04's limits (none do).
        row=next(self.train.record(k) for k in sorted(self.train.selected_ids)
                 if self.train.record(k)['training_category']=='receiver' and self.train.record(k)['operation']==operation
                 and self.train.record(k)['supervision']['result']['status']=='Assessed')
        view,p,index=component_view(self.train,row);excerpts,body=view.materialize(p)
        rt=Runtime('RD05-SYMBOLIC',10,2048,16,profile=DIRECT);rt.registry=SymbolicRegistry(rt.registry.run)
        sources=[];audit=[]
        for i,text in enumerate(excerpts):
            record=Source(text,(1,),((0,len(text)),),('source-symbol-'+str(i),))
            sources.append(rt.registry.publish(Kind.SOURCE,record))
            audit.append({'source_policy':'frozen-w-source-v1','input_hash':digest(text),'offsets':record.offsets,
                          'tensor_hash':symbolic_hash(record.states)})
        original=rt.registry.fork();slots=tuple((f'field-{i}',) for i in range(len(view.references)))
        for ref,value in zip(view.references,slots):audit.append({'role':ref.role,'input_hash':ref.sha256,'tensor_hash':symbolic_hash(value)})
        encoded={'registry':original,'sources':tuple(sources),'slots':slots,'input_hash':digest(body),'audit':audit,'role_profile':DIRECT}
        rt.startup();rt.begin({'Noul':Port.NOUL,'Choice':Port.CHOICE,'Score':Port.SCORE}[operation])
        def publish(i,kind):return rt.publish(kind,Semantic(slots[i],profile=DIRECT,lengths=(1,)))
        q=publish(index['question'],Kind.QUESTION);t=publish(index['target'],Kind.TARGET)
        candidates=tuple(publish(i,Kind.ALTERNATIVE) for i in index['candidates']);rubric=None
        if index['rubric']:
            vals=sum((slots[i] for i in index['rubric']),());sizes=(1,)*len(index['rubric'])
            rubric=rt.publish(Kind.RUBRIC,Semantic(vals,direction=index['direction'],midpoint=len(sizes)==5,profile=DIRECT,
                lengths=sizes if len(sizes)==5 else sizes+(0,)))
        packet=rt.seal(q,t,tuple(Span(ref,0,1) for ref in sources),rubric,candidates)
        return row,view,p,encoded,index,rt,packet

    def admit(self,fixture):
        row,view,p,encoded,index,rt,packet=fixture
        return admit_prepared_component(self.train,row,view,p,encoded,index,rt,packet,input_profile=DIRECT)

    def test_all_operations_bind_actual_snapshot_without_qualification(self):
        for op in ('Noul','Choice','Score'):
            fixture=self.fixture(op);bound=self.admit(fixture);rt=fixture[-2]
            bound.validate(rt);receipt=bound.receipt()
            self.assertIsNone(receipt['reference_qualification']);self.assertFalse(receipt['receiver_behavior_accepted'])
            self.assertFalse(receipt['reason_supported']);self.assertEqual(bound.snapshot,rt.call.snapshot)
            with self.assertRaisesRegex(ValueError,'complete actual request forward'):
                bound.labels(rt,self.train,fixture[0])

    def test_same_record_id_does_not_admit_divergent_vectors_candidates_or_sources(self):
        for changed in ('role','source','evidence','rubric','candidates'):
            fixture=self.fixture('Score' if changed=='rubric' else 'Choice');rt=fixture[-2];snap=rt.registry.get(fixture[-1].snapshot)
            if changed=='role':rt.registry._records[snap.question]=replace(rt.registry.get(snap.question),slots=('foreign',))
            elif changed=='source':
                ref=fixture[3]['sources'][0];item=rt.registry.get(ref);rt.registry._records[ref]=replace(item,states=('foreign',))
            elif changed=='evidence':rt.registry._records[rt.call.snapshot]=replace(snap,evidence=())
            elif changed=='rubric':
                rt.registry._records[snap.rubric]=replace(rt.registry.get(snap.rubric),direction=3-rt.registry.get(snap.rubric).direction)
            else:
                bank=rt.registry.get(snap.bank);rt.registry._records[snap.bank]=replace(bank,candidates=tuple(reversed(bank.candidates)))
            with self.subTest(changed=changed),self.assertRaises(ValueError):self.admit(fixture)

    def test_certificate_rechecks_content_and_rejects_foreign_runtime(self):
        fixture=self.fixture();bound=self.admit(fixture);rt=fixture[-2]
        with self.assertRaisesRegex(ValueError,'foreign/divergent'):bound.validate(self.fixture()[-2])
        snap=rt.registry.get(bound.snapshot);rt.registry._records[snap.target]=replace(rt.registry.get(snap.target),slots=('mutated',))
        with self.assertRaisesRegex(ValueError,'contents diverged'):bound.validate(rt)

    def test_development_training_and_unreviewed_outcome_in_view_reject(self):
        fixture=self.fixture();row,view,p,encoded,index,rt,packet=fixture
        with self.assertRaisesRegex(ValueError,'never train on development'):
            admit_prepared_component(self.dev,row,view,p,encoded,index,rt,packet,input_profile=DIRECT)
        leaked=replace(view,purpose='reader')
        with self.assertRaisesRegex(ValueError,'unreviewed/divergent'):
            admit_prepared_component(self.train,row,leaked,p,encoded,index,rt,packet,input_profile=DIRECT)
        ref=view.references[0]
        leaked=replace(view,references=(replace(ref,content=canonical({'role':ref.role,'fields':{'expected_winner':'A'}})),)+view.references[1:])
        with self.assertRaisesRegex(ValueError,'unreviewed/divergent'):
            admit_prepared_component(self.train,row,leaked,p,encoded,index,rt,packet,input_profile=DIRECT)

    def test_positive_labels_use_actual_candidate_refs_and_keep_Reason_masked(self):
        from tests.test_joint_receiver import symbolic_episodes
        episodes=symbolic_episodes()
        fixture=self.fixture();bound=self.admit(fixture);row,_,_,_,index,rt,packet=fixture
        rt.semantic_positions(len(packet))
        with patch.dict(sys.modules,{'qwen_one_two.training.episodes':episodes}):labels=bound.labels(rt,self.train,row)
        self.assertEqual(labels.snapshot,packet.snapshot);self.assertIsNone(labels.reason_slots)
        actual=dict(bound.candidates);gold=row['supervision']['result']['payload']
        self.assertEqual(labels.choice,tuple(actual[k] for k in gold['acceptable_selection_ids']))
        self.assertTrue(all(r in actual.values() for r,_ in labels.feasibility))

    def test_unsupported_Reason_and_changed_labels_reject_post_forward(self):
        fixture=self.fixture();bound=self.admit(fixture);row,_,_,_,_,rt,packet=fixture
        rt.semantic_positions(len(packet))
        with self.assertRaisesRegex(ValueError,'unsupported new-profile Reason'):
            bound.labels(rt,self.train,row,reason_slots=('not-supported',))
        changed=deepcopy(row);changed['supervision']['result']['status']='Unresolved'
        with self.assertRaisesRegex(ValueError,'label record drift'):bound.labels(rt,self.train,changed)

    def test_status_value_and_reason_masks_remain_exact(self):
        cells=0
        for op in ('Noul','Choice','Score'):
            statuses=('Assessed','Insufficient','Conflicting','Unresolved')+(('NoFeasibleChoice',) if op=='Choice' else ())
            for gold in statuses:
                cells+=1
                for actual in statuses:
                    mask=Eligibility(True,gold,actual,op).mask()
                    self.assertTrue(mask['status']);self.assertFalse(mask['reason'])
                    self.assertEqual(mask[op.lower()],gold=='Assessed')
                    self.assertEqual(mask['feasibility'],op=='Choice')
                    supported=Eligibility(True,gold,actual,op,reason_supported=True).mask()
                    self.assertEqual(supported['reason'],gold==actual and gold!='Assessed')
                self.assertFalse(any(Eligibility(False,gold,gold,op,reason_supported=True).mask().values()))
                self.assertFalse(any(Eligibility(True,gold,gold,op,runtime_failure=True).mask().values()))
        self.assertEqual(cells,13)

    def test_model_free_collection_and_registered_GPU_methods(self):
        subprocess.run([sys.executable,'-c',"""
import sys
class Block:
 def find_spec(self,name,*args):
  if name.split('.')[0] in {'torch','numpy','safetensors','transformers','peft'}:raise RuntimeError(name)
sys.meta_path.insert(0,Block())
import qwen_one_two.training.component_admission
import qwen_one_two.training.eligibility
import tests.test_joint_receiver_gpu
"""],check=True)
        tree=ast.parse(Path('tests/test_joint_receiver_gpu.py').read_text())
        cls=next(n for n in tree.body if getattr(n,'name',None)=='JointReceiverGPU')
        methods={n.name:ast.unparse(n) for n in cls.body if isinstance(n,ast.FunctionDef)}
        self.assertEqual(set(GPU_CHECKS.values()),{name for name in methods if name.startswith('test_')})
        setup=methods['setUpClass'];self.assertLess(setup.index('QOT_RD05_GPU_QUALIFY'),setup.index('import torch'))
        self.assertLess(setup.index('CUDA_VISIBLE_DEVICES'),setup.index('import torch'))
        for name in GPU_CHECKS.values():self.assertIn('assert',methods[name])


if __name__=='__main__':unittest.main()
