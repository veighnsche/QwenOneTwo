"""Certification checks for the bounded G1 proof, all tensor math on CUDA."""
import gc
import math
import time
import torch
from .g1_proof import Episode, gradients, emit, bootstrap
from .g1_protocol import AdmissionError


def clear(b):
    for dual in b.duals: dual.evaluation_outputs.clear()
    gc.collect();torch.cuda.empty_cache()


def tensor_error(actual,reference,atol,rtol,rms_limit):
    assert actual.device.type==reference.device.type=='cuda'
    x,y=actual.double(),reference.double()
    if x.shape!=y.shape: return {'pass':False,'shape':True}
    delta=(x-y).abs()
    er=delta.square().mean().sqrt()/y.square().mean().sqrt().clamp_min(1e-6)
    return {'pass':bool(torch.isfinite(x).all() and torch.isfinite(y).all() and
                         (delta<=atol+rtol*y.abs()).all() and er<=rms_limit),
            'maximum_absolute':delta.max().item(),'normalized_rms':er.item()}


def grad_error(actual,reference,limit,cos_min):
    ss=rr=dot=xx=None;count=0;structure=len(actual)==len(reference)
    for x,y in zip(actual,reference):
        if (x is None)!=(y is None): structure=False;continue
        if x is None: continue
        xd,yd=x.double(),y.double()
        terms=((xd-yd).square().sum(),yd.square().sum(),(xd*yd).sum(),xd.square().sum())
        if ss is None: ss,rr,dot,xx=terms
        else: ss+=terms[0];rr+=terms[1];dot+=terms[2];xx+=terms[3]
        count+=x.numel()
    if not count: return {'pass':structure,'active':0}
    ry=(rr/count).sqrt();rx=(xx/count).sqrt()
    error=(ss/count).sqrt()/ry.clamp_min(1e-10)
    cosine=(dot/(rr*xx).sqrt()).item() if ry>1e-10 and rx>1e-10 else None
    return {'pass':structure and bool(torch.isfinite(error)) and error.item()<=limit and
                    (cosine is None or cosine>=cos_min),
            'normalized_rms':error.item(),'cosine':cosine,'active':count}


def freeze_episode(ep,grad):
    return {'summary':ep.summary(),'snapshots':ep.snapshots,'logits':ep.logit_snapshots,
            'gradients':grad.pop('_vectors',{})}


def compare(test,base,fp32=False):
    errors=[];details={}
    if len(test['snapshots'])!=len(base['snapshots']): errors.append('snapshot count')
    if len(test['logits'])!=len(base['logits']): errors.append('logit count')
    a,z=test['summary'],base['summary']
    for key in ('P','m','B','A','H','total','prefills','lineage','switches','records','attempts',
                'dispatched','consumed','held_positions','released_positions','predicted_budget','events'):
        if a[key]!=z[key]: errors.append('exact:'+key)
    for (label,pos,h,state),(rl,rp,rh,rs) in zip(test['snapshots'],base['snapshots']):
        if (label,pos)!=(rl,rp): errors.append('boundary alignment');continue
        if len(state)!=len(rs): errors.append('state tensor count')
        for index,(x,y) in enumerate([(h,rh),*zip(state,rs)]):
            e=tensor_error(y,x,.02,.02,.01) if fp32 else tensor_error(x,y,.02,.02,.01)
            if not e['pass']: errors.append(f'state:{label}:{index}')
            if e['normalized_rms']>details.get('worst_state_rms',0):
                details['worst_state_rms']=e['normalized_rms'];details['worst_state']=[label,index]
    for i,(x,y) in enumerate(zip(test['logits'],base['logits'])):
        e=tensor_error(y,x,.05,.02,.01) if fp32 else tensor_error(x,y,.05,.02,.01)
        if not e['pass']: errors.append('logits:'+str(i))
    details['value_absolute_error']=abs(a['value']-z['value'])
    if details['value_absolute_error']>.002: errors.append('value')
    for key,x in test['gradients'].items():
        e=grad_error(base['gradients'][key],x,.05,.995) if fp32 else grad_error(x,base['gradients'][key],1e-4,.99999)
        details['gradient_'+key]=e
        if not e['pass']: errors.append('gradient:'+key)
    return {'pass':not errors,'errors':errors[:30],**details}


def assert_accounting(ep):
    f=ep.fixture;s=ep.summary();port=f['port']
    assert s['prefills']==1 and s['lineage']==1
    assert [x['to'] for x in s['switches']]==['D','R','D','R']
    assert s['B']==(35 if port==1 else 58)+f['m']
    assert s['A']==8 and s['H']==6
    assert s['total']==f['P']+s['B']+14
    assert s['released_positions']==36 and s['held_positions']==0
    assert s['records']==(4 if port==1 else 5)
    assert [e['position'] for e in s['events']]==list(range(f['P']+1,s['total']+1))
    for state,layer in zip(ep.state.layers,ep.backbone.model.model.layers):
        if layer.block_type=='full_attention':
            assert state[0].shape[-2]==state[1].shape[-2]==s['total']
        else:
            assert state[0].shape[-1]==4 and state[1].dtype==torch.float32


def positive(g):
    if not g.get('scalar_path'): return False
    return g['budget_grad_none'] and all(g[k]['finite'] and g[k]['rms']>1e-10
        for k in ('slots','writer_R_via_slots','decision_D_via_value','during_evaluation','reader_R_operations','native_R'))


def mechanical(b,c,f):
    results=[]
    for repeat in range(3):
        with torch.no_grad():
            ep=Episode(b,c,f,override=0,capture=False).run()
            assert ep.ledger.dispatched==0 and ep.ledger.b==0 and ep.ledger.a==8
            results.append({'name':'zero_B','repeat':repeat,'pass':True,'summary':ep.summary()})
            del ep;clear(b)
            ep=Episode(b,c,f,override=71+f['m'],capture=False).run();assert_accounting(ep)
            assert ep.ledger.budget-ep.ledger.b==36
            results.append({'name':'exact_fit','repeat':repeat,'pass':True,'summary':ep.summary()})
            before=ep.summary();state=ep.backbone.snapshot_state(ep.state)
            # A completed binding cannot consume the same return a second time.
            rejected=False
            try: ep.ledger.finish_return(14+f['m'],14)
            except AssertionError: rejected=True
            assert rejected and ep.summary()==before
            assert all(torch.equal(x,y) for x,y in zip(state,ep.backbone.snapshot_state(ep.state)))
            results.append({'name':'duplicate','repeat':repeat,'pass':True})
            del ep,state;clear(b)
            ep=Episode(b,c,f,override=70+f['m'],capture=False)
            refused=False
            try: ep.run()
            except AdmissionError: refused=True
            assert refused and ep.ledger.dispatched==0 and ep.ledger.b==7 and ep.value is None
            results.append({'name':'one_short','repeat':repeat,'pass':True,'position':ep.state.position})
            del ep;clear(b)
            ep=Episode(b,c,f,capture=False)
            # Intercept the first request step only after its real GPU work.
            original=ep.step
            def cancel_step(embeds,mode,label,kind=None):
                value=original(embeds,mode,label,kind)
                if label=='request':
                    torch.cuda.synchronize();ep.ledger.cancelled=True
                    raise InterruptedError('diagnostic cancellation after started GPU work')
                return value
            ep.step=cancel_step
            cancelled=False
            try: ep.run()
            except InterruptedError: cancelled=True
            assert cancelled and not ep.ledger.consumed and ep.value is None
            assert ep.ledger.b==7+14+f['m']
            assert ep.ledger.h==3 and ep.state.position==ep.ledger.position
            ep.ledger.released+=ep.ledger.holds-(14+f['m']);ep.ledger.holds=0
            results.append({'name':'cancel','repeat':repeat,'pass':True,'position':ep.state.position,'B':ep.ledger.b})
            del ep;clear(b)
    return results


def local_derivatives():
    results=[]
    for length in (1,2,8,16):
        torch.manual_seed(17+length)
        x=(torch.randn(length,3,device='cuda',dtype=torch.float64)*.1).requires_grad_()
        direction=torch.randn_like(x);direction/=direction.norm()
        def fn(z):
            state=torch.zeros(3,3,device='cuda',dtype=torch.float64)
            total=z.sum()*0
            for value in z:
                key=value*torch.rsqrt(value.square().sum()+1e-6)
                state=state*.9
                delta=(value-(state*key[:,None]).sum(0))*.4
                state=state+key[:,None]*delta[None,:]
                total=total+(state*key[:,None]).sum()
            return total
        gradient=torch.autograd.grad(fn(x),x)[0]
        analytic=(gradient*direction).sum()
        numerical=(fn(x.detach()+1e-6*direction)-fn(x.detach()-1e-6*direction))/(2e-6)
        error=(analytic-numerical).abs().item()
        ok=error<=1e-5+1e-3*numerical.abs().item()
        results.append({'steps':length,'pass':ok,'absolute_error':error})
    return results


def certify(b,c,fx,report,out):
    import hashlib
    from pathlib import Path
    from .g1_precision_math import COUNTS
    report['numeric_matrix']=[];report['negative_controls']=[]
    report['source_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in Path(__file__).parent.glob('g1_*.py')}
    report['local_derivatives']=local_derivatives()
    failure=False
    for f in fx:
        baseline=None
        for profile in ('bf16','checkpoint','serving','fp32'):
            for seed in (17,29,43):
                for repeat in range(3):
                    torch.manual_seed(seed)
                    b.checkpointing=profile=='checkpoint'
                    b.set_precision(torch.float32 if profile=='fp32' else torch.bfloat16)
                    before_products=COUNTS.copy()
                    torch.cuda.reset_peak_memory_stats();start=time.time()
                    with torch.set_grad_enabled(profile!='serving'):
                        ep=Episode(b,c,f).run();assert_accounting(ep)
                        g=gradients(ep,c,b,vectors=True) if profile!='serving' else {}
                        if g and not positive(g): failure=True
                        if not .05<=ep.value.item()<=.95: failure=True
                        current=freeze_episode(ep,g)
                    comparison={'pass':True} if baseline is None else compare(current,baseline,profile=='fp32')
                    if baseline is None: baseline=current
                    if not comparison['pass']: failure=True
                    record={'fixture':f['name'],'profile':profile,'seed':seed,'repeat':repeat,
                            'comparison':comparison,'gradient_paths':g,'summary':current['summary'],
                            'seconds':time.time()-start,'peak_allocated':torch.cuda.max_memory_allocated(),
                            'peak_reserved':torch.cuda.max_memory_reserved()}
                    record['matrix_products']={key:COUNTS[key]-before_products[key] for key in COUNTS}
                    report['numeric_matrix'].append(record)
                    emit(stage='numeric',fixture=f['name'],profile=profile,seed=seed,repeat=repeat,
                         passed=comparison['pass'],value=ep.value.item(),seconds=record['seconds'])
                    (out/'progress.json').write_text(__import__('json').dumps(report,indent=2))
                    del ep,g
                    if current is not baseline: del current
                    clear(b)
                    # A measured numerical/path failure blocks certification;
                    # don't burn through repeated cases pretending it can pass.
                    if failure:
                        report['status']='FAILED';report['blocked_at']='numeric/gradient matrix'
                        return
        del baseline;clear(b)
    report['numeric_pass']=True
    report['gradient_paths_pass']=all(positive(x['gradient_paths']) for x in report['numeric_matrix'] if x['gradient_paths'])
    # A collapsed always-Continue readout cannot qualify: both prescribed call
    # operations and post-return Continue must win the original four-way logits.
    gates=[g for x in report['numeric_matrix'] for g in x['summary']['gate_observations']]
    report['gate_supervision']={'pass':len(gates)==144 and all(g['raw_action']==g['target'] for g in gates),
        'boundaries':len(gates),'correct':sum(g['raw_action']==g['target'] for g in gates)}
    b.set_precision(torch.bfloat16);b.checkpointing=False
    for f in fx:
        for negative in ('scalar','packet'):
            ep=Episode(b,c,f,negative=negative,capture=False).run()
            g=gradients(ep,c,b)
            ok=not g.get('scalar_path') if negative=='scalar' else g['slots']['present']==0
            report['negative_controls'].append({'fixture':f['name'],'control':negative,'pass':ok,'gradients':g})
            failure |= not ok
            del ep;clear(b)
    report['negative_controls_pass']=all(x['pass'] for x in report['negative_controls'])
    report['mechanical']=mechanical(b,c,fx[0])
    report['accounting_pass']=all(x['pass'] for x in report['mechanical'])
    emit(stage='controls',negative=report['negative_controls_pass'],accounting=report['accounting_pass'])
    report['continuations']=[]
    for f in fx:
        base=None
        for profile in ('functional','serving'):
            for seed in (17,29,43):
                for repeat in range(3):
                    torch.manual_seed(seed)
                    ep=Episode(b,c,f,capture=False)
                    try:
                        with torch.no_grad(): ep.run(released=True)
                        item=ep.summary()
                        signature=(item['events'],item['generated'],item['B'],item['A'],item['H'],item['terminal'])
                        if base is None: base=signature
                        ok=signature==base
                    except Exception as exc:
                        item={**ep.summary(),'error':str(exc)};ok=False
                    failure |= not ok
                    report['continuations'].append({'profile':profile,'seed':seed,'repeat':repeat,'pass':ok,**item})
                    emit(stage='continuation',fixture=f['name'],profile=profile,passed=ok,text=item['generated'])
                    del ep;clear(b)
                    (out/'progress.json').write_text(__import__('json').dumps(report,indent=2))
    from .g1_evidence import row_checks,frozen_check
    report['row_checks']=row_checks(b,c,out)
    params={id(p):p for p in list(b.parameters())+list(c.parameters()) if p.requires_grad}
    saved={key:p.detach().clone() for key,p in params.items()}
    try:
        report['optimizer_verification']=bootstrap(b,c,fx,out,windows=1,filename='verification.safetensors',verify=True)
        report['optimizer_pass']=all(x['agreement']['pass'] for x in report['optimizer_verification']['verification'])
    except Exception as exc:
        report['optimizer_pass']=False
        report['optimizer_verification']={'error':str(exc)}
    finally:
        with torch.no_grad():
            for key,p in params.items():p.copy_(saved[key])
        del saved;clear(b)
    report['frozen_base']=frozen_check(b,report['snapshot_path'])
    report['continuation_pass']=len(report['continuations'])==36 and all(x['pass'] for x in report['continuations'])
    released=[g for x in report['continuations'] for g in x['gate_observations'] if g['boundary']=='released']
    report['unforced_gate_checks']={'pass':bool(released) and all(
        g['executed_action']==g['raw_action'] or (g.get('resource_mask') and g['executed_action']==0)
        for g in released),'released_boundaries':len(released),
        'resource_masks':sum(bool(g.get('resource_mask')) for g in released)}
    report['status']='PASSED' if all((report['numeric_pass'],report['gradient_paths_pass'],
        report['negative_controls_pass'],report['accounting_pass'],report['row_checks']['pass'],
        report['optimizer_pass'],report['frozen_base']['pass'],report['continuation_pass'],
        report['gate_supervision']['pass'],report['unforced_gate_checks']['pass'],
        all(x['pass'] for x in report['local_derivatives']))) else 'FAILED'
    report['remaining']=[] if report['status']=='PASSED' else ['See failed G1 gates; expansion remains blocked']
