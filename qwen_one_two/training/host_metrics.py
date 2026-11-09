"""Host interval/counter measurements. No model imports or GPU queries.

Process CPU time includes native threads. Wall intervals can include GPU waits;
neither is GPU compute time. Nested phase totals must not be added together.
"""
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import time

_active=ContextVar('qot_host_metrics',default=None)
_fingerprint_batch_bytes=ContextVar('qot_fingerprint_batch_bytes',default=0)


@contextmanager
def fingerprint_transfers(max_bytes):
    """Explicit process-local audit policy; never reuse a content hash."""
    if type(max_bytes) is not int or max_bytes < 0:
        raise ValueError('nonnegative fingerprint batch size required')
    token=_fingerprint_batch_bytes.set(max_bytes)
    try:yield
    finally:_fingerprint_batch_bytes.reset(token)


def fingerprint_batch_bytes():return _fingerprint_batch_bytes.get()


def packed_fingerprint(entries,read_group,max_bytes):
    """Hash the original interleaved metadata/bytes through grouped transfers.

    Entries are (canonical header bytes, opaque buffer, byte count), already in
    canonical name order. Groups hold at most max_bytes, except an individual
    larger buffer, which is transferred alone. No numeric operation or hash
    cache is involved. The reader must return exactly the requested bytes.
    """
    import hashlib
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError('positive fingerprint batch size required')
    h=hashlib.sha256();group=[];size=0
    def consume():
        body=memoryview(read_group([value for _,value,_ in group]))
        if len(body)!=size:raise ValueError('fingerprint readback byte count mismatch')
        offset=0
        for header,_,count in group:
            h.update(header);h.update(body[offset:offset+count]);offset+=count
    for header,value,count in entries:
        if not isinstance(header,bytes) or type(count) is not int or count<0:
            raise ValueError('invalid fingerprint entry')
        if group and size+count>max_bytes:
            consume();group=[];size=0
        group.append((header,value,count));size+=count
    if group:consume()
    return h.hexdigest()


def stamp():
    return time.perf_counter_ns(),time.process_time_ns()


def elapsed(start,end=None):
    end=stamp() if end is None else end
    return {'wall_ns':end[0]-start[0],'cpu_ns':end[1]-start[1]}


class Recorder:
    def __init__(self,clock=stamp):
        self.clock=clock;self.phases={};self.counts=Counter();self.stack=[]

    @contextmanager
    def span(self,name):
        start=self.clock();frame=[0,0];self.stack.append(frame);failed=False
        try:yield
        except BaseException:failed=True;raise
        finally:
            duration=elapsed(start,self.clock());wall=duration['wall_ns'];cpu=duration['cpu_ns'];self.stack.pop()
            item=self.phases.setdefault(name,dict(calls=0,failed_calls=0,wall_ns=0,cpu_ns=0,
                                                exclusive_wall_ns=0,exclusive_cpu_ns=0))
            item['calls']+=1;item['failed_calls']+=int(failed)
            item['wall_ns']+=wall;item['cpu_ns']+=cpu
            item['exclusive_wall_ns']+=wall-frame[0];item['exclusive_cpu_ns']+=cpu-frame[1]
            if self.stack:self.stack[-1][0]+=wall;self.stack[-1][1]+=cpu

    def snapshot(self):
        return {'phases':{k:dict(v) for k,v in self.phases.items()},'counts':dict(self.counts)}

    def call(self,name,fn,*args,**kwargs):
        with self.span(name):return fn(*args,**kwargs)

    @staticmethod
    def difference(before,after):
        return {'phases':{k:{n:v-before['phases'].get(k,{}).get(n,0) for n,v in values.items()}
                          for k,values in after['phases'].items()},
                'counts':dict(Counter(after['counts'])-Counter(before['counts']))}

    @contextmanager
    def activate(self):
        token=_active.set(self)
        try:yield self
        finally:_active.reset(token)

    @contextmanager
    def patch(self,targets):
        """Worker-local instrumentation, installed only for explicit profiling."""
        saved=[]
        try:
            for owner,attribute,label in targets:
                original=getattr(owner,attribute);saved.append((owner,attribute,original,attribute in vars(owner)))
                def wrapped(*args,_original=original,_label=label,**kwargs):
                    with self.span(_label):return _original(*args,**kwargs)
                setattr(owner,attribute,wraps(original)(wrapped))
            yield
        finally:
            for owner,attribute,original,owned in reversed(saved):
                if owned:setattr(owner,attribute,original)
                else:delattr(owner,attribute)


def current():return _active.get()


def readback_bytes(tensor):
    """The original cpu/numpy/tobytes sequence, with optional audit-site counts.

    Callers keep their original detach/contiguous/byte-view operations in place.
    Counts describe explicit requested copies, not all PCIe traffic or scalars.
    """
    meter=current()
    if meter is None:return tensor.cpu().numpy().tobytes()
    meter.counts['audit_readback_calls']+=1
    meter.counts['audit_requested_bytes']+=tensor.numel()*tensor.element_size()
    if tensor.is_cuda:meter.counts['audit_d2h_calls']+=1
    with meter.span('audit_readback'):host=tensor.cpu()
    with meter.span('audit_numpy_bytes'):return host.numpy().tobytes()


def aggregate(parent,workers,case_metrics,wall_ns):
    """CPU totals include setup/anchors/idle; main-only cost is separate."""
    if any(type(c.get('repetition',0)) is not int or c.get('repetition',0)<0 for c in case_metrics):
        raise ValueError('invalid measured repetition')
    ids=[(c.get('repetition',0),c['id']) for c in case_metrics]
    if len(ids)!=len(set(ids)):raise ValueError('duplicate measured main case')
    def interval(value):
        if any(type(value[k]) is not int or value[k]<0 for k in ('cpu_ns','wall_ns')):
            raise ValueError('invalid host interval')
    interval(parent)
    for value in workers.values():interval(value['process_total'])
    for value in case_metrics:interval(value)
    if sum(v['main_case_count'] for v in workers.values())!=len(ids):raise ValueError('worker measured case count')
    def owner(c):return c.get('worker_id',c.get('uuid'))
    if any(owner(c) not in workers for c in case_metrics):raise ValueError('unknown measured worker')
    for uuid,value in workers.items():
        own=[c for c in case_metrics if owner(c)==uuid]
        if len(own)!=value['main_case_count']:raise ValueError('per-worker case count')
        if any(sum(c[k] for c in own)>value['process_total'][k] for k in ('cpu_ns','wall_ns')):
            raise ValueError('case intervals exceed worker total')
    if type(wall_ns) is not int or wall_ns<0:raise ValueError('invalid total wall interval')
    cpu=parent['cpu_ns']+sum(v['process_total']['cpu_ns'] for v in workers.values())
    main_cpu=sum(v['cpu_ns'] for v in case_metrics);count=len(ids)
    return {'parent':parent,'workers':workers,'completed_main_cases':count,
        'parent_plus_worker_cpu_ns':cpu,'cpu_ns_per_main_case_including_setup':cpu/count if count else None,
        'main_worker_cpu_ns':main_cpu,'main_worker_cpu_ns_per_case':main_cpu/count if count else None,
        'end_to_end_wall_ns':wall_ns,'main_cases_per_second':count*1e9/wall_ns if wall_ns else None,
        'gpu_compute_ns':None,'gpu_idle_ns':None,
        'coverage':'process CPU intervals include native threads; parent cutoff precedes final report encoding; worker cutoff precedes final send; no GPU compute inferred'}
