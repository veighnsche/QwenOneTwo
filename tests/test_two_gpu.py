"""CPU transport/device-policy fixtures only. No model or GPU results."""
import copy
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
from qwen_one_two.g1_device import (PRIMARY,SECONDARY,DEVICES,diagnostic_device,
                                   require_cuda,require_visibility,require_training_context)
from qwen_one_two.training.two_gpu import dispatch,compare_ready,validate_case


def fixture_worker(conn,uuid,mode='normal'):
    """Spawned CPU echo fixture. It never supplies actual model observations."""
    try:
        if mode=='exit' and uuid=='fixture-b':os._exit(7)
        conn.send({'kind':'ready','uuid':uuid,'fixture':True})
        if mode=='premature-final':conn.send({'kind':'final','uuid':uuid,'fixture':True});return
        while True:
            msg=conn.recv()
            if msg['kind']=='stop':break
            if mode=='error':conn.send({'kind':'error','error':'TEST_ONLY injected'});return
            index=msg['index']+(1 if mode=='wrong-index' else 0)
            conn.send({'kind':'result','index':index,'echo':msg['job'],'fixture':True})
            if mode=='duplicate':conn.send({'kind':'result','index':index,'echo':msg['job'],'fixture':True})
        conn.send({'kind':'final','uuid':uuid,'fixture':True})
    finally:conn.close()


def anchor():
    # Invented transport schema, not an annotation, review or GPU measurement.
    return {'id':'TEST_ONLY','fixture':True,'failure':None,'predicted_status':'Assessed',
        'predicted_value':.7,'actual':{'status':'Assessed','selected':None,'reason_present':False,'value':.7},
        'score':{'passed':True,'status_correct':True,'value_correct':True,'transport_valid':True},
        'transport':dict.fromkeys(('latent_request','latent_return','bound_snapshot','candidate_binding',
                                 'consumed','single_prefill','cache_continuity','accounting'),True),
        'runtime':{'P':1,'B':1,'A':0,'H':0,'total':2,'held_positions':0,'prefills':1,'lineage':1,
                   'boundaries':[['TEST_ONLY',2]],'predicted_budget':0}}


class DevicePolicyTests(unittest.TestCase):
    def test_diagnostic_admission_and_optimizer_prohibition_cover_engine_threads(self):
        with patch.dict(os.environ,CUDA_VISIBLE_DEVICES=SECONDARY):
            with diagnostic_device(SECONDARY,execute_gpu=True),ThreadPoolExecutor(max_workers=1) as pool:
                self.assertEqual(pool.submit(require_visibility).result(),SECONDARY)
                with self.assertRaises(ValueError):pool.submit(require_training_context).result()
                with self.assertRaises(RuntimeError):
                    pool.submit(lambda:diagnostic_device(PRIMARY,execute_gpu=True).__enter__()).result()
            with self.assertRaises(RuntimeError):require_visibility()
        require_training_context()
    def cuda(self,name):return Mock(is_available=Mock(return_value=True),get_device_name=Mock(return_value='NVIDIA GeForce '+name))

    def test_default_rejects_secondary_multiple_and_unknown_before_any_query(self):
        for value in (SECONDARY,PRIMARY+','+SECONDARY,'0','GPU-unknown',''):
            cuda=self.cuda('RTX 3060')
            with patch.dict(os.environ,CUDA_VISIBLE_DEVICES=value):
                with self.assertRaises(RuntimeError):require_cuda(cuda)
            cuda.is_available.assert_not_called();cuda.get_device_name.assert_not_called()

    def test_default_primary_and_readonly_scoped_secondary_restore(self):
        with patch.dict(os.environ,CUDA_VISIBLE_DEVICES=PRIMARY):
            require_cuda(self.cuda('RTX 3090'));require_training_context()
            with self.assertRaises(RuntimeError):
                with diagnostic_device(PRIMARY):pass
        for uuid in (PRIMARY,SECONDARY):
            with patch.dict(os.environ,CUDA_VISIBLE_DEVICES=uuid):
                with diagnostic_device(uuid,execute_gpu=True):
                    require_cuda(self.cuda(DEVICES[uuid]))
                    with self.assertRaises(ValueError):require_training_context()
                    with self.assertRaises(RuntimeError):
                        with diagnostic_device(uuid,execute_gpu=True):pass
                if uuid==SECONDARY:
                    with self.assertRaises(RuntimeError):require_visibility()
        require_training_context()

    def test_no_cpu_or_wrong_device_model_fallback(self):
        with patch.dict(os.environ,CUDA_VISIBLE_DEVICES=SECONDARY),diagnostic_device(SECONDARY,execute_gpu=True):
            for tensor in (SimpleNamespace(is_cuda=False),SimpleNamespace(is_cuda=True,device=SimpleNamespace(index=1))):
                cuda=self.cuda('RTX 3060')
                with self.assertRaises(RuntimeError):require_cuda(cuda,tensor)
                cuda.is_available.assert_not_called()
            with self.assertRaises(RuntimeError):require_cuda(self.cuda('RTX 3090'))
            unavailable=self.cuda('RTX 3060');unavailable.is_available.return_value=False
            with self.assertRaises(RuntimeError):require_cuda(unavailable)


class DispatchTests(unittest.TestCase):
    def test_single_resident_worker_covers_jobs_without_cross_device_claim(self):
        jobs=[{'ids':['TEST_ONLY-'+str(i)]} for i in range(3)];received=[];finals=[]
        result=dispatch(jobs,['fixture-a'],fixture_worker,(),on_ready=lambda m:self.assertEqual(set(m),{'fixture-a'}),
            on_result=lambda u,i,m:received.append((i,m['echo'])),on_final=lambda u,m:finals.append(u))
        self.assertEqual([j for i,j in sorted(received)],jobs)
        self.assertEqual(finals,['fixture-a']);self.assertEqual(result['ipc_sends'],4)
        self.assertEqual(result['ipc_receives'],5)

    def exercise(self,mode='normal',ready=None):
        jobs=[{'ids':['TEST_ONLY-'+str(i)]} for i in range(12)];results=[];finals={}
        result=dispatch(jobs,['fixture-a','fixture-b'],fixture_worker,(mode,),
            on_ready=ready or (lambda v:self.assertEqual(set(v),{'fixture-a','fixture-b'})),
            on_result=lambda u,i,m:results.append((u,i,m['echo'])),on_final=lambda u,m:finals.update({u:m}))
        return jobs,results,finals,result

    def test_spawned_workers_cover_jobs_once_and_restore_canonical_order(self):
        jobs,results,finals,r=self.exercise()
        self.assertEqual([v[2] for v in sorted(results,key=lambda v:v[1])],jobs)
        self.assertEqual(len({v[1] for v in results}),12);self.assertEqual(r['jobs_completed'],12)
        self.assertEqual(set(finals),{'fixture-a','fixture-b'})
        self.assertEqual(r['ipc_sends'],14)  # twelve jobs plus two stops
        self.assertEqual(r['ipc_receives'],16)  # twelve results, two ready/final pairs
        self.assertGreaterEqual(r['parent_dispatch_cpu_ns'],0)
        self.assertGreater(r['main_wall_seconds'],0)
        self.assertLessEqual(r['main_wall_seconds'],r['wall_seconds'])

    def test_failure_exit_or_foreign_result_never_returns_success(self):
        for mode in ('error','exit','wrong-index','premature-final','duplicate'):
            with self.subTest(mode=mode),self.assertRaises((RuntimeError,ValueError)):self.exercise(mode)

    def test_readiness_rejection_precedes_any_dispatch(self):
        def reject(_):raise ValueError('TEST_ONLY reject readiness')
        with self.assertRaisesRegex(ValueError,'reject readiness'):self.exercise(ready=reject)


class ReceiptTests(unittest.TestCase):
    def readiness(self):
        prepared={'receiver':'TEST_ONLY_RECEIVER','reference':'TEST_ONLY_REFERENCE','anchors':['TEST_ONLY']}
        messages={u:{'uuid':u,'receiver':prepared['receiver'],'reference':prepared['reference'],
            'base_frozen':True,'gradients_none':True,'optimizer_updates':0,'device_name':'TEST_ONLY '+DEVICES[u],
            'anchors':[anchor()]} for u in (PRIMARY,SECONDARY)}
        return prepared,messages

    def test_readout_comparison_uses_existing_tolerance_and_exact_controls(self):
        p,m=self.readiness();compare_ready(m,p)
        m[SECONDARY]['anchors'][0]['predicted_value']+=.001
        compare_ready(m,p)
        for change in (lambda c:c.update(predicted_value=.71),lambda c:c.update(predicted_value=float('nan')),
                       lambda c:c['runtime'].update(predicted_budget=1),lambda c:c.update(predicted_status='Insufficient'),
                       lambda c:c['actual'].update(selected='foreign'),lambda c:c['transport'].update(consumed=False)):
            p,m=self.readiness();change(m[SECONDARY]['anchors'][0])
            with self.assertRaises(ValueError):compare_ready(m,p)

    def test_readonly_or_identity_drift_rejects(self):
        for field,value in [('receiver','foreign'),('reference','foreign'),('base_frozen',False),('gradients_none',False),
                            ('device_name','wrong card'),('optimizer_updates',1)]:
            p,m=self.readiness();m[SECONDARY][field]=value
            with self.assertRaises(ValueError):compare_ready(m,p)

    def test_case_identity_grades_and_accounting_are_recomputed(self):
        from qwen_one_two.training.inputs import canonical,digest
        row={'id':'TEST_ONLY','operation':'Noul','supervision':{'result':{'operation':'Noul','status':'Assessed','payload':{'p_true':1}}}}
        release=SimpleNamespace(record=lambda key:row)
        expected={'id':'TEST_ONLY','record_sha256':digest(canonical(row)),'view_sha256':'TEST_ONLY_VIEW'}
        c=anchor();c.update(record_sha256=expected['record_sha256'],view_sha256=expected['view_sha256'])
        validate_case(c,expected,release)
        for change in (lambda v:v.update(id='foreign'),lambda v:v.update(view_sha256='changed'),
                       lambda v:v['score'].update(passed=False),lambda v:v['runtime'].update(total=3),
                       lambda v:v['runtime'].update(lineage=2),lambda v:v['runtime'].update(held_positions=1)):
            bad=copy.deepcopy(c);change(bad)
            with self.assertRaises(ValueError):validate_case(bad,expected,release)

    def test_execute_opt_in_rejected_before_any_model_import_or_path_read(self):
        from scripts.two_gpu_diagnostics import run,output_directory
        with self.assertRaisesRegex(ValueError,'--execute-gpu'):run('/no/such/plan.json')
        with self.assertRaisesRegex(ValueError,'repository data'):output_directory('/tmp/unrelated')
        from scripts.benchmark_recurrence_cache import run as cache_run
        with self.assertRaisesRegex(ValueError,'--execute-gpu'):cache_run({},None)

    def test_child_also_requires_explicit_execution_before_setting_visibility(self):
        from qwen_one_two.training.two_gpu_worker import worker
        conn=Mock();before=os.environ.get('CUDA_VISIBLE_DEVICES')
        with self.assertRaisesRegex(ValueError,'explicitly requested'):worker(conn,SECONDARY,{})
        self.assertEqual(os.environ.get('CUDA_VISIBLE_DEVICES'),before)
        self.assertEqual(conn.send.call_args.args[0]['kind'],'error');conn.close.assert_called_once()
