"""Resume/export mechanisms on tiny CUDA fixtures; no capability fitting."""
import json
from pathlib import Path
import random
import tempfile
import unittest
import torch
from qwen_one_two.g1_proof import configure
from qwen_one_two.training.engine import Window
from qwen_one_two.training.checkpoint import save, resume, load_export, IDENTITIES


class CheckpointGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):configure()

    def make(self):
        p=torch.nn.Parameter(torch.tensor([.2,.4],device='cuda'))
        row=torch.nn.Parameter(torch.zeros(2,device='cuda'))
        other=torch.nn.Parameter(torch.ones(2,device='cuda'))
        weights={'p':p,'codec.budget_row':row,'codec.score_row':other}
        w=Window([{'name':'fixture','named_parameters':list(weights.items()),'lr':1e-4,'weight_decay':0.}])
        return w,weights

    def step(self,w,weights):
        for i in range(8):
            loss=(weights['p']*(i+1)).square().mean()
            w.add_episode({'native':loss},complete=True,positions=4)
        return w.step()

    def test_resume_next_update_rng_and_export_exact(self):
        w,weights=self.make();self.step(w,weights)
        identities={k:str(i+1)*64 for i,k in enumerate(sorted(IDENTITIES))}
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'checkpoint'
            save(path,w,weights,identities,{'cursor':8,'order':['a','b']},reference_frozen=True,reference_id=identities['reference'])
            expected_cuda=torch.rand(5,device='cuda');expected_python=random.random()
            self.step(w,weights)
            expected={n:p.detach().clone() for n,p in weights.items()}
            v,restored=self.make()
            manifest=resume(path,v,restored,identities)
            self.assertEqual(manifest['sampler']['cursor'],8)
            self.assertTrue(torch.equal(expected_cuda,torch.rand(5,device='cuda')))
            self.assertEqual(expected_python,random.random())
            self.step(v,restored)
            self.assertEqual(v.scheduled_windows,w.scheduled_windows)
            for n in expected:self.assertTrue(torch.equal(expected[n],restored[n]))
            for p,q in zip(w.parameters,v.parameters):
                for key,value in w.optimizer.state.get(p,{}).items():self.assertTrue(torch.equal(value,v.optimizer.state[q][key]))
            before={n:p.detach().clone() for n,p in restored.items()}
            with self.assertRaises(ValueError):resume(path,v,restored,dict(identities,protocol='0'*64))
            for n in before:self.assertTrue(torch.equal(before[n],restored[n]))
            load_export(path,restored,identities)
            self.assertFalse(torch.equal(restored['p'],before['p']))
            with self.assertRaises(ValueError):save(path,v,restored,identities,{},reference_frozen=True,reference_id=identities['reference'])

    def test_hash_tampering_and_midwindow_save_rejected(self):
        w,weights=self.make();identities={k:'0'*64 for k in IDENTITIES}
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'checkpoint'
            w.add_episode({'native':weights['p'].sum()},complete=True,positions=4)
            with self.assertRaises(ValueError):save(path,w,weights,identities,{},reference_frozen=True,reference_id=identities['reference'])
            w.abort()
            save(path,w,weights,identities,{},reference_frozen=True,reference_id=identities['reference'])
            with (path/'weights.safetensors').open('ab') as f:f.write(b'corrupt')
            before=weights['p'].detach().clone()
            with self.assertRaises(ValueError):load_export(path,weights,identities)
            self.assertTrue(torch.equal(before,weights['p']))
