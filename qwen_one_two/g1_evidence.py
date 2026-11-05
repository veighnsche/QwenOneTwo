"""Collect independent G1 evidence after a failed numeric gate; never waive it."""
import gc
import hashlib
import json
import time
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file,load_file
from .g1_proof import Episode,gradients,bootstrap,emit
from .g1_checks import clear,assert_accounting,positive,freeze_episode,compare,mechanical,local_derivatives
from .g1_backbone import DualLinear


def frozen_check(b,snapshot):
    checked=0;bad=[]
    with safe_open(str(Path(snapshot)/'model.safetensors-00001-of-00001.safetensors'),framework='pt',device='cuda') as file:
        for key in file.keys():
            if not key.startswith('model.language_model.'): continue
            relative=key[len('model.language_model.'):]
            parent,leaf=relative.rsplit('.',1)
            module=b.model.model.get_submodule(parent)
            if isinstance(module,DualLinear): module=module.base
            actual=getattr(module,leaf)
            original=file.get_tensor(key)
            if not torch.equal(actual,original): bad.append(relative)
            checked+=1
    return {'pass':not bad,'parameters_checked':checked,'mismatches':bad,
            'base_tied_storage':b.model.lm_head.weight.data_ptr()==b.model.model.embed_tokens.weight.data_ptr()}


def row_checks(b,c,out):
    saved_b=c.budget_row.detach().clone();saved_s=c.score_row.detach().clone()
    opt=torch.optim.AdamW([c.budget_row,c.score_row],lr=1e-4,weight_decay=0,
                         betas=(.9,.95),eps=1e-8,foreach=False,fused=False,capturable=True)
    # Real CUDA Adam momentum on both actual row leaves, outside any episode.
    c.budget_row.grad=torch.ones_like(c.budget_row);c.score_row.grad=torch.ones_like(c.score_row)
    opt.step();opt.zero_grad(set_to_none=True)
    warm_b=c.budget_row.detach().clone();warm_s=c.score_row.detach().clone()
    opt.step()
    inactive=torch.equal(warm_b,c.budget_row) and torch.equal(warm_s,c.score_row)
    c.score_row.grad=torch.ones_like(c.score_row);opt.step();opt.zero_grad(set_to_none=True)
    budget_skipped=torch.equal(warm_b,c.budget_row)
    with torch.no_grad(): c.budget_row.copy_(saved_b);c.score_row.copy_(saved_s)
    path=out/'rows-export.safetensors'
    save_file({'budget_master':saved_b,'score_master':saved_s,
               'budget_effective':saved_b.to(torch.bfloat16),'score_effective':saved_s.to(torch.bfloat16)},str(path))
    loaded=load_file(str(path),device='cuda')
    hidden=torch.linspace(-1,1,1024,device='cuda')
    errors=[]
    for name,index in [('budget',248077),('score',248078)]:
        actual=b.table.scalar(index,hidden)
        exported=(hidden*loaded[name+'_effective'].float()).sum()
        errors.append((actual-exported).abs().item())
    rejected=False
    try: b.embedding([248077])
    except ValueError: rejected=True
    logits,indices=b.ordinary_logits(hidden,1)
    # This controlled native-only objective must not touch either scalar row.
    # Frozen hidden here has no trainable path; use a differentiable hidden probe.
    probe=hidden.detach().requires_grad_()
    lm,_=b.ordinary_logits(probe,1)
    grads=torch.autograd.grad(lm[0],(c.budget_row,c.score_row),allow_unused=True)
    result={'pass':inactive and budget_skipped and rejected and max(errors)<=1e-6 and all(g is None for g in grads),
            'shared_effective_provider':b.table is c.table,'frozen_base_tied':b.table.base.data_ptr()==b.model.lm_head.weight.data_ptr(),
            'inactive_rows_with_momentum_unchanged':bool(inactive),'score_only_budget_skipped':bool(budget_skipped),
            'export_absolute_errors':errors,'private_input_rejected':rejected,'native_row_grad_none':[g is None for g in grads],
            'scalar_rows_in_native_vocabulary':bool(((indices==248077)|(indices==248078)).any())}
    return result


def gather(b,c,fx,report,out):
    report['scope']='G1-core supplementary evidence; numeric failure retained'
    report['status']='RUNNING';report['fixtures']=fx
    report['source_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in Path(__file__).parent.glob('g1_*.py')}
    report['checks']=[];report['comparisons']=[];report['negative_controls']=[]
    report['local_derivatives']=local_derivatives()
    report['note']='Source-path VJPs explicitly pass through request_packet, excluding writer-history bypasses.'
    for f in fx:
        base=None
        for profile in ('bf16','checkpoint','serving','fp32'):
            b.checkpointing=profile=='checkpoint'
            b.set_precision(torch.float32 if profile=='fp32' else torch.bfloat16)
            torch.cuda.reset_peak_memory_stats();start=time.time()
            with torch.set_grad_enabled(profile!='serving'):
                ep=Episode(b,c,f).run();assert_accounting(ep)
                grad=gradients(ep,c,b,vectors=True) if profile!='serving' else {}
                current=freeze_episode(ep,grad)
                budget_own=None
                if profile=='bf16':
                    own=torch.autograd.grad((ep.budget_raw-.25).square(),c.budget_row,allow_unused=True)[0]
                    budget_own={'connected':own is not None,'norm':own.norm().item()}
            comparison={'pass':True} if base is None else compare(current,base,fp32=profile=='fp32')
            if base is None: base=current
            item={'fixture':f['name'],'profile':profile,'summary':ep.summary(),'gradients':grad,
                  'positive_paths':positive(grad) if grad else None,'comparison':comparison,
                  'budget_own_objective':budget_own,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                  'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'seconds':time.time()-start}
            report['checks'].append(item)
            report['comparisons'].append({'fixture':f['name'],'profile':profile,**comparison})
            emit(stage='evidence',fixture=f['name'],profile=profile,passed=comparison['pass'],
                 gradients=item['positive_paths'],value=ep.value.item(),seconds=item['seconds'])
            (out/'evidence-progress.json').write_text(json.dumps(report,indent=2))
            del ep,grad
            if current is not base: del current
            clear(b)
        del base;clear(b)
    b.set_precision(torch.bfloat16);b.checkpointing=False
    for f in fx:
        for negative in ('scalar','packet'):
            ep=Episode(b,c,f,negative=negative,capture=False).run()
            g=gradients(ep,c,b)
            ok=not g.get('scalar_path') if negative=='scalar' else g['slots']['present']==0
            report['negative_controls'].append({'fixture':f['name'],'negative':negative,'pass':ok,'gradients':g})
            emit(stage='negative_control',fixture=f['name'],negative=negative,passed=ok)
            del ep;clear(b)
    report['mechanical']=mechanical(b,c,fx[0])
    emit(stage='mechanical',passed=all(x['pass'] for x in report['mechanical']),cases=len(report['mechanical']))
    report['continuations']=[]
    for f in fx:
        for profile in ('functional','serving'):
            ep=Episode(b,c,f,capture=False)
            try:
                with torch.no_grad(): ep.run(released=True)
                item={'fixture':f['name'],'profile':profile,'pass':True,**ep.summary()}
            except Exception as exc:
                item={'fixture':f['name'],'profile':profile,'pass':False,'error':str(exc),'partial_trace':ep.summary()}
            report['continuations'].append(item)
            emit(stage='generated_continuation',**{k:v for k,v in item.items() if k!='events'})
            del ep
            clear(b)
    report['row_checks']=row_checks(b,c,out)
    # One genuine optimizer verification window; preserve the original diagnostic
    # snapshot, restore live masters afterward, and never touch service artifacts.
    params=list(b.parameters())+list(c.parameters())
    unique={id(p):p for p in params if p.requires_grad}
    saved={key:p.detach().clone() for key,p in unique.items()}
    try:
        verification=bootstrap(b,c,fx,out,windows=1,filename='verification.safetensors',verify=True)
        report['optimizer_verification']=verification
        report['accumulator_reference']='GPU FP64 aggregation of per-example component VJPs using independent manifest denominators'
    except Exception as exc:
        report['optimizer_verification']={'pass':False,'error':str(exc)}
    finally:
        with torch.no_grad():
            for key,p in unique.items(): p.copy_(saved[key])
    clear(b)
    report['frozen_base']=frozen_check(b,report['snapshot_path'])
    report['numeric_pass']=all(x['pass'] for x in report['comparisons'])
    report['gradient_paths_pass']=all(x['positive_paths'] for x in report['checks'] if x['positive_paths'] is not None)
    report['negative_controls_pass']=all(x['pass'] for x in report['negative_controls'])
    report['accounting_pass']=all(x['pass'] for x in report['mechanical'])
    report['status']='FAILED'  # Earlier failed A3 matrix cannot be waived by supplementary checks.
    report['blocking_reason']='BF16/FP32 A3 parity is not certified; see per-fixture measured errors. No expansion.'
    report['remaining']=['Full successful 72-case numeric matrix','Full successful 36-case released-continuation matrix']
