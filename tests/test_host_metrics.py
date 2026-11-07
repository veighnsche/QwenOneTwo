"""CPU-only measurement/call-order fixtures; never model or GPU execution."""
import ast
from collections import OrderedDict
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from qwen_one_two.training.host_metrics import Recorder,current,readback_bytes,aggregate
from qwen_one_two.training.inputs import canonical


class ByteFixture:
    """Method protocol over literal bytes, not a Tensor or a model."""
    def __init__(self,body=b'\x00abc',is_cuda=True):
        self.body=body;self.is_cuda=is_cuda;self.calls=[];self.shape=(len(body),);self.dtype='TEST_ONLY'
    def detach(self):self.calls.append('detach');return self
    def contiguous(self):self.calls.append('contiguous');return self
    def reshape(self,*shape):self.calls.append('reshape');return self
    def view(self,*dtype):self.calls.append('view');return self
    def cpu(self):self.calls.append('cpu');return self
    def numpy(self):self.calls.append('numpy');return self
    def tobytes(self):self.calls.append('tobytes');return self.body
    def numel(self):return len(self.body)
    def element_size(self):return 1


def definition(path,name):
    tree=ast.parse(Path(path).read_text())
    return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)


class HostMetricsTests(unittest.TestCase):
    def test_nested_intervals_exclusive_cost_and_scope_restore(self):
        ticks=iter([(0,0),(10,3),(30,9),(50,15)])
        r=Recorder(clock=lambda:next(ticks))
        with r.activate():
            self.assertIs(current(),r)
            with r.span('outer'):
                with r.span('inner'):pass
        self.assertIsNone(current())
        p=r.snapshot()['phases'];self.assertEqual(p['inner']['wall_ns'],20)
        self.assertEqual(p['outer']['cpu_ns'],15)
        self.assertEqual(p['outer']['exclusive_cpu_ns'],9)
        self.assertEqual(p['outer']['exclusive_wall_ns'],30)

    def test_patch_preserves_results_exceptions_and_inherited_attributes(self):
        sentinel=object();failure=ValueError('TEST_ONLY')
        class Base:
            def value(self):return sentinel
            def fail(self):raise failure
        class Child(Base):pass
        r=Recorder()
        with r.patch([(Child,'value','call'),(Child,'fail','failure')]):
            self.assertIs(Child().value(),sentinel)
            with self.assertRaises(ValueError) as e:Child().fail()
            self.assertIs(e.exception,failure)
        self.assertNotIn('value',vars(Child));self.assertNotIn('fail',vars(Child))
        self.assertEqual(r.snapshot()['phases']['failure']['failed_calls'],1)
        self.assertFalse(r.stack)

    def test_snapshot_delta_and_counters_are_not_nested_phase_sums(self):
        r=Recorder();before=r.snapshot()
        r.counts['TEST_ONLY']+=3;r.call('bookkeeping',lambda:None)
        delta=Recorder.difference(before,r.snapshot())
        self.assertEqual(delta['counts'],{'TEST_ONLY':3})
        self.assertEqual(delta['phases']['bookkeeping']['calls'],1)

    def test_readback_order_bytes_and_explicit_site_counts(self):
        for enabled in (False,True):
            value=ByteFixture();r=Recorder()
            if enabled:
                with r.activate():body=readback_bytes(value)
                self.assertEqual(r.snapshot()['counts'],{'audit_readback_calls':1,'audit_requested_bytes':4,'audit_d2h_calls':1})
            else:body=readback_bytes(value)
            self.assertEqual(body,b'\x00abc');self.assertEqual(value.calls,['cpu','numpy','tobytes'])
        cpu=ByteFixture(is_cuda=False)
        with r.activate():readback_bytes(cpu)
        self.assertEqual(r.counts['audit_d2h_calls'],1)

    def test_actual_fingerprint_body_preserves_header_and_byte_order(self):
        fn=definition('qwen_one_two/training/reference.py','state_fingerprint')
        scope={'__name__':'qwen_one_two.training._schema_fingerprint','__package__':'qwen_one_two.training',
               'hashlib':hashlib,'torch':SimpleNamespace(uint8='TEST_ONLY_BYTEVIEW'),'readback_bytes':readback_bytes}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'schema-fingerprint','exec'),scope)
        state={'b':ByteFixture(b'xyz'),'a':ByteFixture(b'\x00abc')};expected=hashlib.sha256()
        for name,value in sorted(state.items()):
            expected.update(canonical({'name':name,'shape':list(value.shape),'dtype':str(value.dtype)}));expected.update(value.body)
        r=Recorder()
        with r.activate():actual=scope['state_fingerprint'](state)
        self.assertEqual(actual,expected.hexdigest());self.assertEqual(r.counts['audit_requested_bytes'],7)
        self.assertEqual(state['a'].calls,['detach','contiguous','reshape','view','cpu','numpy','tobytes'])

    def test_actual_cache_metadata_function_counts_hit_miss_and_eviction(self):
        fn=definition('qwen_one_two/codec/recurrence.py','_graphs')
        scope={'torch':SimpleNamespace(cuda=SimpleNamespace(current_stream=lambda device:SimpleNamespace(cuda_stream=7))),
               '_CACHE':OrderedDict(),'CACHE_LIMIT':2,'current':current,'_Graphs':lambda values:object()}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'schema-cache','exec'),scope)
        def values(n):return (SimpleNamespace(shape=(1,n,2,3),dtype='TEST_ONLY',device=SimpleNamespace(index=0)),)
        r=Recorder()
        with r.activate():
            for n in (1,2,1,3):scope['_graphs'](values(n))
        self.assertEqual({k:v for k,v in r.counts.items() if not k.startswith('recurrence_key')},
                         {'recurrence_lookups':4,'recurrence_misses':3,'recurrence_hits':1,'recurrence_evictions':1})
        self.assertEqual(sum(v for k,v in r.counts.items() if k.startswith('recurrence_key')),4)

    def data(self):
        parent={'cpu_ns':20,'wall_ns':2500}
        workers={u:{'process_total':{'cpu_ns':100,'wall_ns':1000},'main_case_count':1} for u in ('TEST_A','TEST_B')}
        cases=[{'id':'TEST_ONLY_'+u,'uuid':u,'cpu_ns':v,'wall_ns':200} for u,v in [('TEST_A',20),('TEST_B',30)]]
        return parent,workers,cases

    def test_parent_and_worker_cpu_totals_and_gpu_unmeasured(self):
        p,w,c=self.data();r=aggregate(p,w,c,2500)
        self.assertEqual(r['parent_plus_worker_cpu_ns'],220)
        self.assertEqual(r['cpu_ns_per_main_case_including_setup'],110)
        self.assertEqual(r['main_worker_cpu_ns_per_case'],25)
        self.assertIsNone(r['gpu_compute_ns']);self.assertIsNone(r['gpu_idle_ns'])
        json.dumps(r,allow_nan=False)

    def test_aggregate_rejects_bad_coverage_and_intervals(self):
        for change in (lambda p,w,c:c.append(c[0]),lambda p,w,c:c[0].update(cpu_ns=-1),
                       lambda p,w,c:c[0].update(uuid='foreign'),lambda p,w,c:w['TEST_A'].update(main_case_count=2),
                       lambda p,w,c:c[0].update(cpu_ns=101)):
            p,w,c=self.data();change(p,w,c)
            with self.assertRaises(ValueError):aggregate(p,w,c,2500)


class ProfilingPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from qwen_one_two.training.two_gpu import plan
        cls.prepared=plan('/TEST_ONLY_NOT_OPENED',selection='cpu-profile',profile_host=True,trace_cuda=True)

    def test_existing_case_extrema_and_pair_are_pinned(self):
        p=self.prepared
        self.assertEqual(len(p['cases']),8);self.assertEqual(len(p['jobs']),7)
        self.assertEqual(sum(len(j['ids'])==2 for j in p['jobs']),1)
        self.assertEqual({c['operation'] for c in p['cases']},{'Noul','Choice','Score'})
        chosen={c['id'] for c in p['cases']}
        self.assertTrue(all(set(ids)<=chosen for ids in p['selection_evidence']['extrema'].values()))
        self.assertTrue(set(p['selection_evidence']['pair']['ids'])<=chosen)
        self.assertEqual(p['status'],'PREPARED_GPU_NOT_RUN')
        self.assertTrue(p['profiling']['host']);self.assertTrue(p['profiling']['cuda_trace_first_job_per_worker'])
        self.assertEqual(len(p['anchor_cases']),3)

    def test_invalid_options_reject_before_reading_dataset(self):
        from qwen_one_two.training.two_gpu import plan
        for kwargs in ({'selection':'unknown'},{'trace_cuda':True}):
            with self.assertRaisesRegex(ValueError,'profiling selection'):plan('/never-opened',**kwargs)
