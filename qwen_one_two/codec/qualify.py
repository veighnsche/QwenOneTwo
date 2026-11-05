"""GPU-only prescribed qualification; no optimization, fitting or UI integration."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time
import traceback

import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from safetensors.torch import load_file,save_file
from ..g1_proof import configure
from ..g1_checks import tensor_error,grad_error
from ..g1_evidence import frozen_check
from .backbone import Backbone
from .neural import Codec
from .schema import Port,Status,Kind,Role,ValueKind,Mode,require
from .records import Source,Semantic,Attribute,Value,Span,History,Reason
from .runtime import Runtime
from .session import Session
from .writer import Draft
from .ownership import parameter_groups
from dataclasses import replace


def load(snapshot,checkpoint):
    b=Backbone(snapshot,'data/qev')
    with torch.device('cuda'):
        c=Codec(b.sigma,b.model.model.embed_tokens.weight,load_file('data/qev/pointer.safetensors',device='cuda'))
    original=load_file(checkpoint,device='cuda')
    bp=dict(b.named_parameters());cp=dict(c.named_parameters());copied=[]
    with torch.no_grad():
        for n,v in original.items():
            target=cp.get(n[6:]) if n.startswith('codec.') else bp.get(n)
            if target is not None and target.shape==v.shape:target.copy_(v);copied.append(n)
    b.bind_rows(c);b.set_compensated(True)
    return b,c,copied


def metric(tensors):
    active=[g for g in tensors if g is not None]
    if not active:return {'count':0,'rms':0.,'finite':True}
    return {'count':sum(g.numel() for g in active),
            'rms':(sum(g.double().square().sum() for g in active)/sum(g.numel() for g in active)).sqrt().item(),
            'finite':all(bool(g.isfinite().all()) for g in active)}


def trace(b,c,port,total,gradients):
    prompt='Assess the supplied evidence. Evidence: One step is ready and one is blocked.'
    rendered=b.tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,
                                            add_generation_prompt=True,enable_thinking=True)
    encoded=b.tokenizer(rendered,add_special_tokens=False,return_offsets_mapping=True)
    ids=encoded['input_ids'];excerpt='One step is ready and one is blocked.'
    start=rendered.index(excerpt);end=start+len(excerpt)
    selected=[i for i,(a,z) in enumerate(encoded['offset_mapping']) if a<end and z>start]
    a0=encoded['offset_mapping'][selected[0]][0];z0=encoded['offset_mapping'][selected[-1]][1]
    rt=Runtime('qualification',len(ids),2048,total)
    s=Session(b,c,rt);h=s.prefill(ids)
    offsets=tuple((len(rendered[:a].encode()),len(rendered[:z].encode())) for a,z in encoded['offset_mapping'])
    src=rt.registry.publish(Kind.SOURCE,Source(rendered.encode(),tuple(ids),offsets,h[0].float()))
    s.override=2048
    raw_budget=s.startup()
    statuses=[Status.ASSESSED] if total==128 else ([Status.INSUFFICIENT,Status.ASSESSED] if total==256 else
              [Status.INSUFFICIENT,Status.CONFLICTING,Status.UNRESOLVED]+([Status.NO_FEASIBLE_CHOICE] if port==Port.CHOICE else [])+[Status.ASSESSED])
    tracking=[];metadata=[];last_result=None;selected_slots=[]
    for ordinal,status in enumerate(statuses):
        if ordinal==len(statuses)//2 and rt.phase==0:s.close()
        writer=s.begin_writer(port)
        f=writer.feature(s.hidden[0,-1]);qslots=c.emit(f,Role.QUESTION,8);tslots=c.emit(f,Role.TARGET,2)
        key=c.emit(f,Role.KEY,2,0)
        q=rt.publish(Kind.QUESTION,Semantic(qslots))
        t=rt.publish(Kind.TARGET,Semantic(tslots,(Attribute(key,Value(ValueKind.INT64,True,2,Mode.CONSTRUCT)),)))
        writer.draft=replace(writer.draft,question=q,target=t)
        def tick():
            s.step(writer.tick_vector(s.hidden[0,-1])[None],'R','writer_tick',False)
            return writer.feature(s.hidden[0,-1])
        tick();writer.draft=replace(writer.draft,evidence=(Span(src,selected[0],min(4,len(selected))),));rt.writer_action(0)
        if ordinal==1 and last_result is not None:
            tick();writer.draft=replace(writer.draft,evidence=writer.draft.evidence+(History(last_result),));rt.writer_action(0)
        if port==Port.CHOICE:
            candidates=[];candidate_slots=[]
            for i in range(2):
                f=tick();meaning=c.emit(f,Role.CANDIDATE,4);candidate_slots.append(meaning)
                attrs=(Attribute(c.emit(f,Role.KEY,2,0),Value(ValueKind.UTF8,True,(b'Tuesday' if i==0 else b'Wednesday'),Mode.CONSTRUCT)),)
                candidates.append(rt.publish(Kind.ALTERNATIVE,Semantic(meaning,attrs)))
                writer.draft=replace(writer.draft,candidates=tuple(candidates));rt.writer_action(1)
        if port==Port.SCORE:
            f=tick();rslots=torch.cat([c.emit(f,r,2) for r in (Role.DIMENSION,Role.CONDITIONS,Role.ZERO,Role.ONE)])
            rslots=torch.cat((rslots,rslots.new_zeros(2,1024)))
            rubric=rt.publish(Kind.RUBRIC,Semantic(rslots,direction=1,midpoint=False))
            writer.draft=replace(writer.draft,rubric=rubric);rt.writer_action(2)
        tick();d=writer.draft
        packet=rt.seal(d.question,d.target,d.evidence,d.rubric,d.candidates)
        inputs,evaluated,during=s.evaluate(packet)
        scalar=None;chosen=None
        if status==Status.ASSESSED:
            if port==Port.CHOICE:
                chosen=int(evaluated['pointer_logits'].argmax());selected_slots=candidate_slots[chosen]
                out=rt.prepare_result(rt.call.snapshot,status,selected_index=chosen)
            else:
                scalar=evaluated['value']
                out=rt.prepare_result(rt.call.snapshot,status,scalar=scalar)
        else:out=rt.prepare_result(rt.call.snapshot,status,reason=Reason(evaluated['reason_slots']))
        last_result=rt.call.result;s.deliver(out)
        metadata.append({'status':int(status),'predicted_status':int(evaluated['status_logits'].argmax())+1,
                         'q':len(packet),'r':len(out),'handles':len(out.handles),'selected':chosen,
                         'raw':None if 'raw' not in evaluated else evaluated['raw'].detach().item(),
                         'scalar':None if scalar is None else scalar.detach().item()})
        if scalar is not None:tracking.append((scalar,inputs,qslots,during))
        if port==Port.CHOICE and status==Status.ASSESSED:
            pointer_loss=-evaluated['pointer_logits'].log_softmax(-1)[chosen]
        del writer,packet,out,evaluated,during
    if rt.phase==0:s.close()
    pad=total-rt.position-2
    require(pad>=0,'complete fixture exceeds target length; never truncate')
    if pad:
        rt.debit('A',pad,'prescribed_native_filler')
        s.step(b.embedding([271]*pad)[0],'R','filler')
    token=b.tokenizer.encode('Done',add_special_tokens=False)[0]
    losses,logits=s.native([token],True);loss=torch.stack(losses).mean()
    probe={};vectors={}
    if gradients:
        if tracking:
            scalar,inputs,question,during=tracking[-1]
            dv=torch.autograd.grad(loss,scalar,retain_graph=True)[0]
            packet_grad=torch.autograd.grad(scalar,inputs,grad_outputs=dv,retain_graph=True)[0]
            slot_grad=torch.autograd.grad(inputs,question,grad_outputs=packet_grad,retain_graph=True)[0]
            writer_probe=list(b.adapter_parameters('R'))[1]
            rgrad=torch.autograd.grad(question,writer_probe,grad_outputs=slot_grad,retain_graph=True,allow_unused=True)[0]
            dgrad=torch.autograd.grad(scalar,during,grad_outputs=dv,retain_graph=True,allow_unused=True)
            probe={'dL_dscalar':dv.item(),'packet':metric([packet_grad]),'slots':metric([slot_grad]),
                   'writer_R':metric([rgrad]),'evaluating_D':metric(dgrad)}
            require(abs(probe['dL_dscalar'])>1e-10 and all(probe[k]['finite'] and probe[k]['rms']>1e-10 for k in
                ('packet','slots','writer_R','evaluating_D')),'disconnected scalar/request/writer/evaluator path')
            del packet_grad,slot_grad,rgrad,dgrad
        else:
            selected_grad=torch.autograd.grad(loss,selected_slots,retain_graph=True,allow_unused=True)[0]
            pointer_grad=torch.autograd.grad(pointer_loss,c.qev_q.weight,retain_graph=True,allow_unused=True)[0]
            probe={'selected_contents':metric([selected_grad]),'explicit_pointer_objective':metric([pointer_grad]),
                   'argmax_derivative_claimed':False}
            require(probe['selected_contents']['rms']>1e-10 and probe['explicit_pointer_objective']['rms']>1e-10,'Choice content/pointer gradient')
        groups={'R':list(b.adapter_parameters('R')),'D':list(b.adapter_parameters('D')),
                'wire':list(c.value.parameters())+list(c.meta.parameters())+list(c.tags.parameters())+list(c.atom.parameters()),
                'emitter':list(c.emit1.parameters())+list(c.emit2.parameters())+list(c.emitout.parameters())}
        flat=[p for ps in groups.values() for p in ps]
        grads=torch.autograd.grad(loss,flat,retain_graph=False,allow_unused=True)
        offset=0
        for name,params in groups.items():
            vectors[name]=tuple(None if g is None else g.detach() for g in grads[offset:offset+len(params)])
            offset+=len(params)
    rt.debit('H',1,'chat_end');s.step(b.embedding([248046])[0],'R','end')
    rt.terminate('Completed',emit_end=False)
    require(rt.position==total and s.prefills==1 and rt.summary()['held_positions']==0,'whole-episode accounting')
    require(rt.dispatched==rt.consumptions==len(statuses),'completion counts')
    require(all(p.grad is None for p in list(b.parameters())+list(c.parameters())),'unexpected accumulated parameter gradients')
    return {'summary':s.summary(),'calls':metadata,'probe':probe,'snapshots':s.snapshots,'vectors':vectors,
            'logits':logits,'loss':loss.detach().item(),'budget_raw':raw_budget.detach().item()}


def compare(test,base,fp32):
    errors=[];worst=0.;boundaries=[]
    for key in ('P','B','A','H','total','attempts','dispatched','consumed','phase','terminal','switches','events'):
        if test['summary'][key]!=base['summary'][key]:errors.append('exact:'+key)
    require(len(test['snapshots'])==len(base['snapshots']),'snapshot counts')
    for (label,h,states),(other,rh,reference) in zip(test['snapshots'],base['snapshots']):
        boundary_worst=0.
        if label!=other:errors.append('boundary label')
        for x,y in [(h,rh),*zip(states,reference)]:
            actual,ref=(y,x) if fp32 else (x,y)
            e=tensor_error(actual,ref,.02,.02,.01)
            worst=max(worst,e['normalized_rms'])
            boundary_worst=max(boundary_worst,e['normalized_rms'])
            if not e['pass']:errors.append('state:'+label)
        boundaries.append({'label':label,'worst_state_rms':boundary_worst})
    for x,y in zip(test['logits'],base['logits']):
        actual,ref=(y,x) if fp32 else (x,y)
        if not tensor_error(actual,ref,.05,.02,.01)['pass']:errors.append('logits')
    for a,z in zip(test['calls'],base['calls']):
        for k in ('status','predicted_status','q','r','handles','selected'):
            if a[k]!=z[k]:errors.append('call:'+k)
        if a['scalar'] is not None and abs(a['scalar']-z['scalar'])>.002:errors.append('scalar')
    gradients={}
    for name,v in test['vectors'].items():
        if fp32:e=grad_error(base['vectors'][name],v,.05,.995)
        else:e=grad_error(v,base['vectors'][name],1e-4,.99999)
        gradients[name]=e
        if not e['pass']:errors.append('gradient:'+name)
    if abs(test['budget_raw']-base['budget_raw'])>1e-5:errors.append('startup')
    return {'pass':not errors,'errors':errors[:30],'worst_state_rms':worst,'boundaries':boundaries,'gradients':gradients}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--snapshot',required=True);ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--output',required=True);ap.add_argument('--lengths',type=int,nargs='+',default=[128,256,512])
    ap.add_argument('--ports',type=int,nargs='+',default=[1,2,3]);ap.add_argument('--seeds',type=int,nargs='+',default=[17,29,43])
    ap.add_argument('--repeats',type=int,default=3);ap.add_argument('--profiles',nargs='+',default=['functional','checkpoint','serving','fp32'])
    ap.add_argument('--precision-parts',type=int,choices=(2,3),default=2)
    ap.add_argument('--reserve-state-memory',action='store_true',help='Allocate CUDA moment/component-buffer envelopes; never step an optimizer')
    args=ap.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    report={'status':'RUNNING','scope':'full-codec prescribed runtime qualification; no training/frontend',
            'profile':f'b1-full-v1-compensated{args.precision_parts}-temporal32','cases':[],
            'training_performed':False,'optimizer_updates':0,'frontend_changed':False,
            'prescribed_operations_and_statuses':True,'semantic_ability_claimed':False,
            'requested_matrix':{'lengths':args.lengths,'ports':args.ports,'seeds':args.seeds,'repeats':args.repeats,'profiles':args.profiles},
            'full_A3_matrix_complete':False,
            'checkpoint':args.checkpoint,'checkpoint_sha256':hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
            'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}}
    started=time.time()
    try:
        configure()
        with sdpa_kernel(SDPBackend.MATH):
            b,c,copied=load(args.snapshot,args.checkpoint)
            b.precision_parts=args.precision_parts
            owners=parameter_groups(b,c)
            parameters=[p for g in owners for _,p in g['named_parameters']]
            masters=[p.detach().clone() for p in parameters]
            report['parameter_ownership']=[{'group':g['name'],'parameters':[n for n,_ in g['named_parameters']],
                'elements':sum(p.numel() for _,p in g['named_parameters'])} for g in owners]
            reserved=[]
            if args.reserve_state_memory:
                # Conservative upper envelope: two moment tensors and three
                # component buffers for every live trainable leaf. This is not
                # a trainer or a claim of actual A4 data eligibility.
                reserved=[[torch.zeros_like(p,dtype=torch.float32) for p in parameters] for _ in range(5)]
            report['state_memory_reservation']={'enabled':args.reserve_state_memory,
                'moment_copies':2 if reserved else 0,'component_buffer_copies':3 if reserved else 0,
                'bytes':sum(x.numel()*x.element_size() for group in reserved for x in group),
                'optimizer_created':False,'optimizer_steps':0,'eligibility_scope':'conservative all-parameter envelope, not A4 qualification'}
            report['loaded_diagnostic_tensors']=copied
            report['environment']={'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),'cuda':torch.version.cuda}
            for total in args.lengths:
                for port in args.ports:
                    base=None
                    for profile in args.profiles:
                        for seed in args.seeds:
                            for repeat in range(args.repeats):
                                torch.manual_seed(seed);b.checkpointing=profile=='checkpoint'
                                b.set_precision(torch.float32 if profile=='fp32' else torch.bfloat16)
                                torch.cuda.reset_peak_memory_stats();t=time.time()
                                with torch.set_grad_enabled(profile!='serving'):
                                    current=trace(b,c,Port(port),total,profile!='serving')
                                comparison={'pass':True} if base is None else compare(current,base,profile=='fp32')
                                row={k:current[k] for k in ('summary','calls','probe','loss','budget_raw')}
                                row.update(length=total,port=port,profile=profile,seed=seed,repeat=repeat,
                                           comparison=comparison,seconds=time.time()-t,
                                           allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved())
                                report['cases'].append(row)
                                (out/'progress.json').write_text(json.dumps(report,indent=2))
                                print(json.dumps({k:row[k] for k in ('length','port','profile','seed','repeat','comparison','seconds','allocated')}),flush=True)
                                require(comparison['pass'],'numeric/discrete qualification failed')
                                if base is None:base=current
                                else:del current
                                for d in b.duals:d.evaluation_outputs.clear()
                                gc.collect();torch.cuda.empty_cache()
                    del base;gc.collect();torch.cuda.empty_cache()
            b.set_precision(torch.bfloat16)
            report['frozen_base']=frozen_check(b,args.snapshot)
            require(report['frozen_base']['pass'],'frozen base changed')
            report['all_live_parameters_unchanged']=all(torch.equal(p,old) for p,old in zip(parameters,masters))
            require(report['all_live_parameters_unchanged'],'qualification mutated live parameters')
            report['requested_checks_passed']=True
            report['full_A3_matrix_complete']=(set(args.lengths)=={128,256,512} and set(args.ports)=={1,2,3}
                and set(args.seeds)=={17,29,43} and args.repeats>=3 and
                set(args.profiles)=={'functional','checkpoint','serving','fp32'})
            report['status']='REFERENCE_PATH_PASSED' if args.profiles==['fp32'] else ('PASSED' if report['full_A3_matrix_complete'] else 'PARTIAL_MATRIX_PASSED')
    except Exception as exc:
        report['status']='FAILED';report['error']=str(exc);report['traceback']=traceback.format_exc()
        print(report['traceback'],flush=True)
    finally:
        report['elapsed_seconds']=time.time()-started
        (out/'report.json').write_text(json.dumps(report,indent=2))
        print(json.dumps({'status':report['status'],'cases':len(report['cases']),'output':str(out)}),flush=True)
    if report['status'] not in ('PASSED','PARTIAL_MATRIX_PASSED','REFERENCE_PATH_PASSED'):raise SystemExit(1)


if __name__=='__main__':main()
