"""Model-free plan/admission/resume tests. Binary fixtures are opaque bytes."""
import ast
from collections import Counter
from copy import deepcopy
import importlib.abc
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from qwen_one_two.training.inputs import canonical, digest


class BlockModels(importlib.abc.MetaPathFinder):
    def find_spec(self, name, *args):
        if name.split('.')[0] in {'torch', 'numpy', 'safetensors', 'transformers', 'peft'}:
            raise AssertionError('model library forbidden: '+name)


class ReceiverPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.block = BlockModels(); sys.meta_path.insert(0, cls.block)
        from qwen_one_two.training import receiver_plan
        from qwen_one_two.training.receiver_profiles import LEGACY
        cls.planning = receiver_plan
        saved = json.loads(Path('data/receiver-probe-20261107/run/plan.json').read_bytes())
        cls.snapshot = saved['snapshot']
        cls.plan = receiver_plan.prepare(cls.snapshot, LEGACY)

    @classmethod
    def tearDownClass(cls):
        sys.meta_path.remove(cls.block)

    def rebind(self, plan):
        from qwen_one_two.training.inputs import canonical, digest
        plan['run_sha256'] = digest(canonical(plan['contract']))
        plan['checkpoint_identities'] = dict(plan['contract']['initial_checkpoint']['identities'],
            protocol=plan['run_sha256'], sources=digest(canonical(plan['contract']['dependencies']['sources'])))
        return plan

    def bundle(self, path, plan, cursor=8, *, updates=1):
        receipts = {'exposures':dict(Counter(plan['contract']['training_order'][:cursor])),
                    'eligible':{'status':cursor-1}, 'failures':{'disclosed failed fixture':1}, 'measurements':[]}
        if cursor == 0: receipts.update(eligible={}, failures={})
        elif updates == 0: receipts.update(eligible={}, failures={'disclosed failed fixture':cursor})
        path.mkdir()
        for name in ('weights.safetensors', 'optimizer.pt', 'rng.pt'):
            (path/name).write_bytes(b'OPAQUE METADATA TEST FIXTURE; NEVER LOAD AS A TENSOR')
        manifest = {'format':'qot-training-bundle-v1', 'identities':plan['checkpoint_identities'],
            'files':{p.name:digest(p.read_bytes()) for p in path.iterdir()},
            'ownership':plan['contract']['owners'], 'phase':'components', 'schedule_profile':'a4-v1',
            'reference_frozen':True, 'reference_id':plan['contract']['dependencies']['reference'],
            'scheduled_windows':188+cursor//8, 'optimizer_updates':188+updates,
            'sampler':self.planning.sampler_state(plan,cursor,receipts)}
        self.write_bundle(path, manifest)
        return manifest

    def write_bundle(self, path, manifest):
        (path/'manifest.json').write_bytes(canonical(manifest))
        return digest((path/'manifest.json').read_bytes())

    def inspect_bundle(self, path, plan=None):
        return self.planning.inspect_resume(plan or self.plan,path,digest((path/'manifest.json').read_bytes()))

    def test_prepare_is_deterministic_with_models_blocked(self):
        again = self.planning.prepare(self.snapshot, self.plan['contract']['input_profile'])
        self.assertEqual(canonical(self.plan), canonical(again))
        self.assertEqual(len(self.plan['contract']['training_order']),256)
        self.assertEqual(self.plan['contract']['exposures']['windows'],32)
        self.assertEqual(len(self.plan['contract']['panels']['development_E']['ids']),130)
        self.assertEqual(len(self.plan['contract']['panels']['development_X']['ids']),180)

    def test_profiles_regimes_and_pending_owners_are_explicit(self):
        from qwen_one_two.training.receiver_profiles import profile
        with self.assertRaisesRegex(ValueError,'unknown input profile'):profile('typo','component')
        with self.assertRaisesRegex(ValueError,'component regime'):profile('direct-role-states-v1','native')
        p = deepcopy(self.plan)
        p['contract']['input_profile'] = 'direct-role-states-v1'
        p['contract']['profile'] = profile('direct-role-states-v1','component')
        self.rebind(p)
        self.assertNotEqual(p['checkpoint_identities'],self.plan['checkpoint_identities'])
        with self.assertRaisesRegex(ValueError,'pending RD-03..07'):
            self.planning.admit_execution(p,gpu=self.planning.PRIMARY,input_profile='direct-role-states-v1',regime='component')

    def test_v2_plan_retains_full_coverage_and_v1_audit_stays_historical(self):
        with self.assertRaisesRegex(ValueError,'stale direct layout coverage dependency'):
            self.planning.prepare(self.snapshot,'direct-role-states-v1')
        plan=self.planning.prepare(self.snapshot,'direct-role-states-v2')['contract']
        self.assertEqual(plan['exposures']['distinct_records'],1500)
        self.assertEqual(plan['exposures']['presentations'],6000)
        self.assertEqual(plan['exposures']['windows'],750)
        coverage=plan['layout_coverage']
        self.assertIn('all1500 training supported',coverage['admission'])
        self.assertEqual({name:(s['supported'],s['total']) for name,s in coverage['panels'].items()},
            {'train_receiver':(1500,1500),'development_E':(127,130),'development_X':(180,180)})
        owners={p['name']:p for group in plan['owners'] for p in group['parameters']}
        self.assertEqual(owners['codec.direct_role_projection.weight']['shape'],[1024,1024])
        self.assertEqual(owners['codec.direct_role_projection.bias']['shape'],[1024])
        self.assertFalse(any(name.startswith('codec.direct_role_length.') for name in owners))
        self.assertIn('codec.direct_role_length',plan['frozen_owners'])
        self.assertIsNone(plan['experimental_component_admission']['reference_qualification'])
        self.assertFalse(plan['experimental_component_admission']['receiver_behavior_accepted'])
        self.assertIn('unsupported/masked',plan['experimental_component_admission']['Reason'])
        self.assertTrue(plan['required_evidence']['qualification_required_before_fit'])
        self.assertIn('historical_RD07',plan['required_evidence'])
        self.assertIn('changed_v2',plan['required_evidence'])
        self.assertEqual(plan['objectives']['status'],1.)
        self.assertEqual(plan['objectives']['feasibility'],.5)

    def test_wrong_gpu_profile_and_regime_reject_before_models(self):
        profile = self.plan['contract']['input_profile']
        for gpu in (None,'cpu','0','GPU-744c1cf5-9578-6aad-95c1-7a6f1c9d11db'):
            with self.assertRaisesRegex(ValueError,'3090 UUID'):
                self.planning.admit_execution(self.plan,gpu=gpu,input_profile=profile,regime='component')
        for name,regime in [('direct-role-states-v1','component'),(profile,'native')]:
            with self.assertRaisesRegex(ValueError,'profile/regime'):
                self.planning.admit_execution(self.plan,gpu=self.planning.PRIMARY,input_profile=name,regime=regime)

    def test_performance_policy_binds_plan_and_blocks_unqualified_execution(self):
        from qwen_one_two.training.receiver_performance import RETAINED,policy,GPU_CHECKS
        faster=self.planning.prepare(self.snapshot,self.plan['contract']['input_profile'],execution=RETAINED)
        self.assertEqual(faster['contract']['execution_policy'],policy(RETAINED))
        for field in ('training_order','owners','objectives','initial_checkpoint','exposures'):
            self.assertEqual(faster['contract'][field],self.plan['contract'][field])
        self.assertNotEqual(faster['run_sha256'],self.plan['run_sha256'])
        self.assertEqual(set(faster['contract']['required_evidence']['performance_checks']),set(GPU_CHECKS))
        with self.assertRaisesRegex(ValueError,'performance GPU qualification pending'):
            self.planning.admit_execution(faster,gpu=self.planning.PRIMARY,
                input_profile=faster['contract']['input_profile'],regime='component')
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'plan.json';path.write_bytes(canonical(faster))
            with self.assertRaisesRegex(ValueError,'execution policy mismatch'):
                self.planning.load_plan(path,self.snapshot,faster['contract']['input_profile'])

    def test_plan_drift_and_stale_dependencies_reject(self):
        p = deepcopy(self.plan);p['contract']['releases']['training']='0'*64
        with self.assertRaisesRegex(ValueError,'identity drift'):self.planning.validate_plan(p)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'plan.json';path.write_bytes(canonical(self.plan))
            wrong=deepcopy(self.plan);wrong['contract']['dependencies']['precision']='0'*64;self.rebind(wrong)
            with patch.object(self.planning,'prepare',return_value=wrong):
                with self.assertRaisesRegex(ValueError,'stale profile/release/dependency/checkpoint'):
                    self.planning.load_plan(path,self.snapshot,self.plan['contract']['input_profile'])

    def test_initial_checkpoint_and_base_bytes_are_checked_without_loading(self):
        with patch.object(self.planning,'file_hash',return_value='0'*64):
            with self.assertRaisesRegex(ValueError,'initial checkpoint manifest drift'):
                self.planning.initial_state()
            with self.assertRaisesRegex(ValueError,'base snapshot identity mismatch'):
                self.planning.prepare(self.snapshot,self.plan['contract']['input_profile'])

    def test_preparation_receipt_has_no_executed_work_and_never_overwrites(self):
        with tempfile.TemporaryDirectory(dir=self.planning.ROOT/'data') as temp:
            output=Path(temp)/'prepared'
            receipt=self.planning.write_preparation(output,self.plan,{'fixture_host_clock_only':True})
            self.assertEqual(set(receipt['accounting'].values()),{0})
            self.assertEqual(receipt['plan_sha256'],digest((output/'plan.json').read_bytes()))
            with self.assertRaisesRegex(ValueError,'fresh local data output'):
                self.planning.write_preparation(output,self.plan,{})

    def test_cache_cannot_reuse_mutable_sources_or_cross_profile_or_mapping(self):
        with self.assertRaisesRegex(ValueError,'mutable D'):
            self.planning.cache_identity(self.plan,'source','1'*64,'2'*64)
        cached=self.planning.cache_identity(self.plan,'reference','1'*64,'2'*64)
        self.planning.validate_cache(self.plan,cached,kind='reference',input_sha256='1'*64,mapping_sha256='2'*64)
        for field,value in [('profile','foreign'),('mapping_sha256','3'*64),('input_sha256','4'*64),('regime','native')]:
            wrong=deepcopy(cached);wrong[field]=value
            with self.assertRaisesRegex(ValueError,'cache identity'):
                self.planning.validate_cache(self.plan,wrong,kind='reference',input_sha256='1'*64,mapping_sha256='2'*64)
        p=deepcopy(self.plan);p['contract']['dependencies']['reference']='5'*64;self.rebind(p)
        with self.assertRaisesRegex(ValueError,'cache identity'):
            self.planning.validate_cache(p,cached,kind='reference',input_sha256='1'*64,mapping_sha256='2'*64)

    def test_complete_window_resume_retains_failed_scheduled_exposure(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'checkpoint';self.bundle(path,self.plan,updates=0)
            saved=self.inspect_bundle(path)
            self.assertEqual(saved['sampler']['cursor'],8)
            self.assertEqual(sum(saved['sampler']['receipts']['exposures'].values()),8)
            self.assertEqual(saved['optimizer_updates'],188)
            self.assertEqual(saved['scheduled_windows'],189)
            self.assertEqual(saved['sampler']['receipts']['failures'],{'disclosed failed fixture':8})
            self.assertEqual(self.plan['contract']['training_order'][saved['sampler']['cursor']:][:8],
                             self.plan['contract']['training_order'][8:16])

    def test_resume_rejects_foreign_identity_ownership_cursor_and_counters(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'checkpoint';original=self.bundle(path,self.plan)
            changes=[lambda m:m['identities'].update(dataset='0'*64),
                     lambda m:m['identities'].update(protocol='0'*64),
                     lambda m:m.update(ownership=[]),lambda m:m.update(scheduled_windows=190),
                     lambda m:m.update(optimizer_updates=190),
                     lambda m:m.update(optimizer_updates=188.5),
                     lambda m:m['sampler'].update(cursor=7),
                     lambda m:m['sampler'].update(order_sha256='0'*64),
                     lambda m:m['sampler']['receipts'].update(exposures={}),
                     lambda m:m.update(reference_id='0'*64)]
            for change in changes:
                m=deepcopy(original);change(m);self.write_bundle(path,m)
                with self.subTest(change=change):
                    with self.assertRaises(ValueError):self.inspect_bundle(path)

    def test_corrupt_checkpoint_files_and_manifest_hash_reject(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'checkpoint';self.bundle(path,self.plan)
            with self.assertRaisesRegex(ValueError,'manifest drift'):
                self.planning.inspect_resume(self.plan,path,'0'*64)
            (path/'rng.pt').write_bytes(b'corrupted opaque fixture')
            with self.assertRaisesRegex(ValueError,'artifact hash mismatch'):self.inspect_bundle(path)

    def test_transactional_restore_delegates_to_existing_checkpoint_implementation(self):
        # No checkpoint/model import, fitting or substitute optimizer is executed.
        tree=ast.parse(Path('qwen_one_two/training/receiver_plan.py').read_text())
        restore=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='restore_prepared')
        self.assertIn('return restore_model',ast.unparse(restore))
        for path,names in [('qwen_one_two/training/engine.py',('Window',)),
                           ('qwen_one_two/training/checkpoint.py',('save','resume','restore_model','_load_weights','load_export'))]:
            old=ast.parse(subprocess.check_output(['git','show','47c58a3:'+path],text=True))
            current=ast.parse(Path(path).read_text())
            for name in names:
                before=next(n for n in old.body if getattr(n,'name',None)==name)
                after=next(n for n in current.body if getattr(n,'name',None)==name)
                self.assertEqual(ast.dump(before),ast.dump(after))

    def test_cli_rejects_ambiguous_execution_without_model_imports(self):
        import scripts.train_receiver_probe as runner
        base=['probe','--snapshot',self.snapshot,'--output','data/rd02-rejected-no-write']
        cases=[[],['--prepare','--execute-gpu'],['--prepare'],
               ['--prepare','--input-profile','direct-role-states-v1','--resume','x','--resume-sha256','0'*64],
               ['--execute-gpu','--input-profile','direct-role-states-v1'],
               ['--execute-gpu','--resume','x']]
        for flags in cases:
            with self.subTest(flags=flags),patch.object(sys,'argv',base+flags),patch.dict(os.environ,{},clear=True):
                with self.assertRaises((ValueError,SystemExit)):runner.main()
        self.assertFalse(Path('data/rd02-rejected-no-write').exists())


if __name__=='__main__':unittest.main()
