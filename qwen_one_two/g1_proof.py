"""Explicit GPU-only, isolated G1-core execution and evidence runner.

Run with CUDA_VISIBLE_DEVICES=<documented 3090 UUID> python -m ...g1_proof.
Never imports Runtime, changes the live checkpoint, or expands to Choice/data/UI.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import gc
import hashlib
import json
import math
import time
import traceback
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from safetensors.torch import load_file, save_file

from .g1_backbone import Backbone, GPU_UUID, norm
from .g1_protocol import Codec, Ledger, AdmissionError
from .decision import verify_qev, MODEL, REVISION, QEV_REVISION


def emit(**event):
    print(json.dumps(event),flush=True)


def configure():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != GPU_UUID:
        raise RuntimeError('CUDA_VISIBLE_DEVICES must explicitly select '+GPU_UUID)
    torch.backends.fp32_precision='ieee'
    torch.backends.cuda.matmul.fp32_precision='ieee'
    torch.backends.cudnn.conv.fp32_precision='ieee'
    # PyTorch 2.15 requires cuBLASLt for A3's explicit split-K prohibition.
    torch.backends.cuda.preferred_blas_library('cublaslt')
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=(False,False)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=(False,False)
    torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
    torch.use_deterministic_algorithms(True,warn_only=False)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.manual_seed(17)


from .g1_accounting import budget


def fixtures(tokenizer):
    definitions=[('noul',1,
        'Access requires an active badge and orientation. Evidence: Jo has an active badge and completed orientation. May Jo enter?',
        'Jo has an active badge and completed orientation.'),
        ('score',3,
        'Two preparation steps count equally. Evidence: One step is ready and one is blocked. Rate the fraction blocked, from 0 (none) to 1 (both), with 0.5 meaning one.',
        'One step is ready and one is blocked.')]
    result=[]
    think=tokenizer.encode('I will assess this',add_special_tokens=False)
    answer=tokenizer.encode('The assessment is available for this request.',add_special_tokens=False)
    assert len(think)==4 and len(answer)==8,(think,answer)
    for name,port,prompt,excerpt in definitions:
        rendered=tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,
                                               add_generation_prompt=True,enable_thinking=True)
        encoded=tokenizer(rendered,add_special_tokens=False,return_offsets_mapping=True)
        start=rendered.index(excerpt);end=start+len(excerpt)
        selected=[i for i,(a,b) in enumerate(encoded['offset_mapping']) if a<end and b>start]
        assert selected and selected==list(range(selected[0],selected[-1]+1))
        assert len(encoded['input_ids'])<=80 and len(selected)<=16,(name,len(encoded['input_ids']),len(selected))
        first=encoded['offset_mapping'][selected[0]][0]
        last_a,last_b=encoded['offset_mapping'][selected[-1]]
        clip_a=len(rendered[first:start].encode())
        clip_b=len(rendered[last_a:min(end,last_b)].encode())
        result.append({'name':name,'port':port,'prompt':prompt,'rendered':rendered,'excerpt':excerpt,
                       'ids':encoded['input_ids'],'source_positions':selected,'clip_a':clip_a,'clip_b':clip_b,
                       'thinking_ids':think,'answer_ids':answer,'P':len(encoded['input_ids']),'m':len(selected),
                       'source_sha256':hashlib.sha256(rendered.encode()).hexdigest(),
                       'review':'engineering fixture; prescribed route; no independent semantic approval claimed'})
    return result


class Episode:
    def __init__(self,backbone,codec,fixture,override=128,negative=None,capture=True):
        self.backbone=backbone;self.codec=codec;self.fixture=fixture
        self.ledger=Ledger(fixture['P'],override)
        self.state=backbone.initial();self.hidden=None;self.switches=[]
        self.snapshots=[];self.losses=[];self.discrete=[];self.value=None;self.raw=None
        self.negative=negative;self.capture=capture;self.mode=None;self.prefills=0
        self.draft={};self.previous=None;self.generated=[];self.terminal=None
        self.logit_snapshots=[];self.reader_outputs=[]
        self.gate_observations=[]
        self.physical_start=backbone.physical_layer_calls
        self.state_version_checks=0

    def step(self,embeds,mode,label,kind=None):
        if kind is not None: self.ledger.commit(label,kind,len(embeds))
        before=self.state.position
        old_state=self.state
        versions=tuple(t._version for pair in old_state.layers for t in pair)
        if self.mode != mode:
            self.switches.append({'from':self.mode,'to':mode,'position':before,'lineage':self.state.lineage})
            self.mode=mode
        self.hidden,self.state=self.backbone(self.state,mode,embeds)
        assert versions==tuple(t._version for pair in old_state.layers for t in pair)
        assert old_state.position==before
        self.state_version_checks+=len(versions)
        assert self.state.position==self.ledger.position
        assert self.state.position==before+len(embeds)
        assert self.state.lineage==1
        if self.capture:
            self.snapshots.append((label,self.state.position,self.hidden.detach().clone(),self.backbone.snapshot_state(self.state)))
        return self.hidden

    def native(self,tokens,kind,ce=False):
        for token in tokens:
            if ce:
                logits,indices=self.backbone.ordinary_logits(self.hidden[0,-1],self.ledger.phase)
                # Every prescribed ordinary token is in the contiguous ordinary vocabulary.
                assert token<248044
                self.losses.append(F.cross_entropy(logits[None,:],torch.tensor([token],device='cuda')))
                if self.capture: self.logit_snapshots.append(logits.detach().clone())
            self.step(self.backbone.embedding([token])[0],'R','native_'+kind,kind)

    def close(self):
        self.step(self.backbone.embedding([248069,271])[0],'R','thinking_close','H')
        self.ledger.phase=1

    def gate_logits(self,boundary,target=None):
        l=self.ledger;c=self.codec
        logits=c.gate(c.feature(self.hidden[0,-1].float(),l.counters()))
        self.gate_observations.append({'boundary':boundary,'position':self.state.position,
            'phase':l.phase,'logits':logits.detach().tolist(),
            'raw_action':int(logits.detach().argmax()),'target':target})
        return logits

    def call(self,port):
        c=self.codec;l=self.ledger
        # Supervise the eligible native boundary inference actually sees,
        # before a new request mutates counters or clears its draft.
        self.discrete.append(F.cross_entropy(self.gate_logits('request',port)[None,:],torch.tensor([port],device='cuda')))
        l.attempts+=1;l.writer=0;l.records=0;l.consumed=False;self.draft={};self.previous=None
        # WRITER_START has no seven-action prediction.
        l.writer+=1;l.commit('writer_start','B')
        payload=c.control_input(35,port,l,self.draft)
        self.step(c.wire(payload,35)[None,:],'R','writer_start')
        feature=c.feature(self.hidden[0,-1].float(),l.counters(),c.draft_memory(self.draft,port))
        self.draft['question']=c.emit(feature,2,8)
        self.draft['target']=c.emit(feature,4,2)
        l.records=2;self.draft['length']=13
        actions=[0,2,5] if port==3 else [0,5]
        for action in actions:
            l.writer+=1;l.commit('writer_tick','B')
            memory=c.draft_memory(self.draft,port)
            feature=c.feature(self.hidden[0,-1].float(),l.counters(),memory)
            payload=c.control_input(36,port,l,self.draft,self.previous)+c.feedback(feature)
            self.step(c.wire(payload,36,previous=self.previous)[None,:],'R','writer_tick')
            feature=c.feature(self.hidden[0,-1].float(),l.counters(),c.draft_memory(self.draft,port))
            self.discrete.append(F.cross_entropy(c.writer_action(feature)[None,:],torch.tensor([action],device='cuda')))
            if action==0:
                self.draft['evidence']=self.source
                self.draft['length']+=1+len(self.source)
            elif action==2:
                self.draft['rubric']=tuple(c.emit(feature,role,2) for role in range(9,14))
                self.draft['length']+=11;l.records+=1
            self.previous=action
        packet=c.packet(self.draft,port,detach_payload=self.negative=='packet',
                        clipping_end=self.fixture['clip_b'],clipping_start=self.fixture['clip_a'],attempt=l.attempts)
        self.request_packet=packet
        q=len(packet);rmax=61 if port==3 else 50
        l.admit(q,rmax,5 if port==3 else 4)
        self.backbone.capture_evaluator(torch.is_grad_enabled())
        self.step(packet,'D','request','B')
        self.eval_outputs=[v for dual in self.backbone.duals for v in dual.evaluation_outputs]
        self.backbone.capture_evaluator(False)
        self.value,self.raw=c.scalar(self.hidden[0,-1],port)
        self.discrete.append(F.cross_entropy(c.status(F.silu(c.eval_trunk(norm(self.hidden[0,-1]))))[None,:],
                                            torch.tensor([0],device='cuda')))
        # Carrier status is prescribed Assessed, not a prediction/correctness claim.
        delivered=self.value.detach() if self.negative=='scalar' else self.value
        ret=c.packet(self.draft,port,result=delivered,attempt=l.attempts)
        self.backbone.capture_evaluator(torch.is_grad_enabled(),mode='R')
        self.step(ret,'R','return','B')
        self.reader_outputs=[v for dual in self.backbone.duals for v in dual.evaluation_outputs]
        self.backbone.capture_evaluator(False)
        l.finish_return(q,len(ret))
        self.discrete.append(F.cross_entropy(self.gate_logits('post_return',0)[None,:],
                                            torch.tensor([0],device='cuda')))

    def run(self,released=False):
        f=self.fixture;c=self.codec;l=self.ledger
        self.prefills+=1
        self.step(self.backbone.embedding(f['ids'])[0],'D','prompt')
        self.source=self.hidden[0,f['source_positions']]
        self.budget_raw=self.backbone.table.scalar(248077,self.hidden[0,-1])
        self.predicted_budget=budget(self.budget_raw.detach().item())
        self.step(c.startup(self.budget_raw),'R','startup','H')
        if l.budget==0:
            self.close()
            self.native(f['answer_ids'],'A',True)
        else:
            self.native(f['thinking_ids'],'B')
            if f['port']==3:
                self.close();self.native(f['answer_ids'][:2],'A')
            self.call(f['port'])
            if released:
                self.release_generation()
                return self
            if l.phase==0: self.close()
            self.native(f['answer_ids'][2:] if f['port']==3 else f['answer_ids'],'A',True)
        self.step(self.backbone.embedding([248046])[0],'R','chat_end','H')
        self.terminal='Completed'
        return self

    def release_generation(self):
        c=self.codec;l=self.ledger
        while True:
            if l.phase==0 and l.b>=l.budget:
                self.close();continue
            if l.phase==1 and l.a>=l.answer:
                self.step(self.backbone.embedding([248046])[0],'R','forced_chat_end','H')
                self.terminal='Limited';break
            logits=self.gate_logits('released')
            if l.b>=l.budget or l.attempts>=16:
                logits=logits.clone();logits[1:]=-torch.inf
                self.gate_observations[-1]['resource_mask']=True
            action=int(logits.argmax())
            self.gate_observations[-1]['executed_action']=action
            if action:
                # G1 is deliberately not a Choice codec expansion. An unsupported
                # learned branch is a reported failure, never silently Continue.
                if action==2: raise RuntimeError('Released gate selected Choice: G1 subset cannot certify this continuation')
                try: self.call(action)
                except AdmissionError:
                    l.commit('admission_notice','H')
                    notice=c.control_input(39,action,l,self.draft,outcome=3)
                    self.step(c.wire(notice,39)[None,:],'R','admission_notice')
                continue
            tokens,indices=self.backbone.ordinary_logits(self.hidden[0,-1],l.phase)
            token=int(indices[tokens.argmax()])
            if token==248069:
                self.close();continue
            if token==248046:
                self.step(self.backbone.embedding([token])[0],'R','chat_end','H')
                self.terminal='Completed';break
            self.generated.append(token)
            self.native([token],'A' if l.phase else 'B')
        if not self.generated or l.a <= (2 if self.fixture['port']==3 else 0):
            raise RuntimeError('No native answer tokens after return')
        if not self.backbone.tokenizer.decode(self.generated).strip():
            raise RuntimeError('Whitespace-only continuation is not a normal answer')

    def summary(self):
        l=self.ledger
        return {'fixture':self.fixture['name'],'P':l.prompt,'m':self.fixture['m'],'B':l.b,'A':l.a,'H':l.h,
                'total':l.position,'override':l.budget,'predicted_budget':self.predicted_budget,
                'raw_budget':self.budget_raw.detach().item(),'value':None if self.value is None else self.value.detach().item(),
                'raw_value':None if self.raw is None else self.raw.detach().item(),
                'prefills':self.prefills,'lineage':self.state.lineage,'switches':self.switches,
                'records':l.records,'attempts':l.attempts,'dispatched':l.dispatched,'consumed':l.consumed,
                'consumed_count':l.consumed_count,'state_version_checks':self.state_version_checks,
                'held_positions':l.holds,'released_positions':l.released,'terminal':self.terminal,
                'generated':self.backbone.tokenizer.decode(self.generated),'events':l.events,
                'gate_observations':self.gate_observations,
                'physical_layer_calls':self.backbone.physical_layer_calls-self.physical_start}


def rms(x):
    return x.detach().double().square().mean().sqrt().item()


def grad_stats(grads):
    active=[x for x in grads if x is not None]
    if not active: return {'present':0,'rms':0.,'finite':True}
    count=sum(x.numel() for x in active)
    energy=sum(x.detach().double().square().sum() for x in active)
    return {'present':len(active),'rms':(energy/count).sqrt().item(),
            'finite':all(bool(torch.isfinite(x).all()) for x in active)}


def gradients(ep,codec,backbone,vectors=False):
    loss=torch.stack(ep.losses).mean()
    slots=[ep.draft['question'],ep.draft['target']]+list(ep.draft.get('rubric',()))
    a=torch.autograd.grad(loss,ep.value,retain_graph=True,allow_unused=True)[0]
    if a is None:
        return {'loss':loss.item(),'scalar_path':False,'negative':ep.negative}
    packet_gradient=torch.autograd.grad(ep.value,ep.request_packet,grad_outputs=a.detach(),retain_graph=True)[0]
    slot_grads=torch.autograd.grad(ep.request_packet,slots,grad_outputs=packet_gradient.detach(),
                                  retain_graph=True,allow_unused=True)
    pairs=[(s,g) for s,g in zip(slots,slot_grads) if g is not None]
    rparams=list(backbone.adapter_parameters('R'))
    through_writer=torch.autograd.grad([s for s,g in pairs],rparams,grad_outputs=[g.detach() for s,g in pairs],
                                       retain_graph=True,allow_unused=True) if pairs else ()
    dparams=list(backbone.adapter_parameters('D'))
    through_d=torch.autograd.grad(ep.value,dparams,grad_outputs=a.detach(),retain_graph=True,allow_unused=True)
    eval_grads=torch.autograd.grad(ep.value,ep.eval_outputs,grad_outputs=a.detach(),retain_graph=True,allow_unused=True)
    native_r=torch.autograd.grad(loss,rparams,retain_graph=True,allow_unused=True)
    reader=torch.autograd.grad(loss,ep.reader_outputs,retain_graph=True,allow_unused=True)
    budget_grad=torch.autograd.grad(loss,codec.budget_row,retain_graph=True,allow_unused=True)[0]
    score_grad=torch.autograd.grad(loss,codec.score_row,retain_graph=True,allow_unused=True)[0]
    result={'loss':loss.item(),'scalar_path':True,'dL_dvalue':a.item(),'negative':ep.negative,
            'request_packet':grad_stats([packet_gradient]),
            'slots':grad_stats(slot_grads),'writer_R_via_slots':grad_stats(through_writer),
            'decision_D_via_value':grad_stats(through_d),'during_evaluation':grad_stats(eval_grads),
            'native_R':grad_stats(native_r),'reader_R_operations':grad_stats(reader),'budget_grad_none':budget_grad is None,
            'score_grad_none':score_grad is None}
    if vectors:
        result['_vectors']={'slots':tuple(g.detach() if g is not None else None for g in slot_grads),
                            'writer':tuple(g.detach() if g is not None else None for g in through_writer),
                            'decision':tuple(g.detach() if g is not None else None for g in through_d),
                            'native_R':tuple(g.detach() if g is not None else None for g in native_r)}
    return result


def optimizer_groups(b,c):
    r=list(b.adapter_parameters('R'));d=list(b.adapter_parameters('D'))
    inherited=list(c.qev_q.parameters())+list(c.qev_k.parameters())
    taken={id(p) for p in r+d+inherited+[c.budget_row,c.score_row]}
    mat=[];other=[]
    for name,p in c.named_parameters():
        if id(p) in taken: continue
        (mat if p.ndim==2 and not any(x in name for x in ('role.','slot.','owner.','attribute.','previous.','prototypes')) else other).append(p)
    return [{'params':r,'lr':1e-4,'weight_decay':0}, {'params':d,'lr':2e-5,'weight_decay':0},
            {'params':inherited,'lr':2e-5,'weight_decay':0}, {'params':mat,'lr':2e-4,'weight_decay':.01},
            {'params':other,'lr':2e-4,'weight_decay':0},
            {'params':[c.budget_row],'lr':1e-4,'weight_decay':0},
            {'params':[c.score_row],'lr':1e-4,'weight_decay':0}]


def bootstrap(b,c,fx,output,windows=2,filename='diagnostic.safetensors',verify=False):
    groups=optimizer_groups(b,c)
    all_params=[p for g in groups for p in g['params']]
    assert len({id(p) for p in all_params})==len(all_params)
    optim=torch.optim.AdamW(groups,betas=(.9,.95),eps=1e-8,foreach=False,fused=False,capturable=True)
    history=[]
    verification=[]
    for window in range(windows):
        # F6 component-wise VJP buffers and eligible-example denominators.
        buffers={};counts={}
        oracle=[None]*len(all_params)
        before=[p.detach().clone() for p in all_params] if verify else None
        for i in range(8):
            fixture=fx[i%2]
            ep=Episode(b,c,fixture,capture=False).run()
            losses={'native':torch.stack(ep.losses).mean(),'discrete':torch.stack(ep.discrete).mean()}
            if fixture['port']==3: losses['calibration']=(ep.raw-.5).square()
            for key,loss in losses.items():
                grads=torch.autograd.grad(loss,all_params,retain_graph=True,allow_unused=True)
                if key not in buffers: buffers[key]=[None]*len(all_params);counts[key]=0
                counts[key]+=1
                for j,g in enumerate(grads):
                    if g is not None:
                        if verify:
                            # Independent FP64 arithmetic oracle for F6's actual
                            # component-VJP algorithm. A weighted BF16 backward
                            # has different rounding and is not an FP32 oracle.
                            contribution=g.detach().double()/(4 if key=='calibration' else 8)
                            if oracle[j] is None: oracle[j]=contribution
                            else: oracle[j].add_(contribution)
                        if buffers[key][j] is None: buffers[key][j]=g.detach().clone()
                        else: buffers[key][j].add_(g.detach())
            row={'window':window,'example':i,'losses':{k:v.item() for k,v in losses.items()},'value':ep.value.item()}
            history.append(row);emit(stage='bootstrap',**row)
            for dual in b.duals: dual.evaluation_outputs.clear()
            del ep,losses,grads,loss
        optim.zero_grad(set_to_none=True)
        for j,p in enumerate(all_params):
            parts=[buf[j]/counts[key] for key,buf in buffers.items() if buf[j] is not None]
            p.grad=sum(parts) if parts else None
        if verify:
            from .g1_checks import grad_error
            assert counts=={'native':8,'discrete':8,'calibration':4}
            agreement=grad_error([p.grad for p in all_params],oracle,1e-4,.99999)
            assert agreement['pass'],agreement
            assert c.budget_row.grad is None
        torch.nn.utils.clip_grad_norm_(all_params,1.,error_if_nonfinite=True)
        optim.step();optim.zero_grad(set_to_none=True)
        if verify:
            changed={id(p):not torch.equal(old,p) for old,p in zip(before,all_params)}
            item={'agreement':agreement,'eligible_examples':counts,
                  'R_changed':any(changed[id(p)] for p in b.adapter_parameters('R')),
                  'D_changed':any(changed[id(p)] for p in b.adapter_parameters('D')),
                  'score_changed':changed[id(c.score_row)],'budget_unchanged':not changed[id(c.budget_row)]}
            assert item['R_changed'] and item['D_changed'] and item['score_changed'] and item['budget_unchanged']
            verification.append(item)
        del buffers
    save_file({name:p.detach().contiguous() for name,p in list(b.named_parameters())+[("codec."+n,p) for n,p in c.named_parameters()] if p.requires_grad},
              str(output/filename))
    return {'episodes':history,'verification':verification}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--snapshot',required=True);ap.add_argument('--qev',default='data/qev')
    ap.add_argument('--output',required=True);ap.add_argument('--stage',choices=['initial','bootstrap','probe','certify','evidence'],default='initial')
    ap.add_argument('--checkpoint',help='Read-only diagnostic checkpoint from an earlier proof run')
    ap.add_argument('--precision-policy',choices=['legacy','residual','compensated'],default='legacy')
    args=ap.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'report.json').exists():
        (out/f'prior-report-{time.time_ns()}.json').write_bytes((out/'report.json').read_bytes())
    report={'status':'RUNNING','scope':'G1-core only','started':time.time(),'requested_stage':args.stage,
            'model':MODEL,'revision':REVISION,'qev_revision':QEV_REVISION,'gpu_uuid':GPU_UUID,
            'snapshot_path':str(Path(args.snapshot).resolve()),
            'precision_policy':args.precision_policy,
            'checks':[],'expansion_performed':False}
    try:
        configure();verify_qev(args.qev)
        report['environment']={'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),'cuda':torch.version.cuda}
        with sdpa_kernel(SDPBackend.MATH):
            b=Backbone(args.snapshot,args.qev)
            weights=load_file(str(Path(args.qev)/'pointer.safetensors'),device='cuda')
            with torch.device('cuda'):
                c=Codec(b.sigma,b.model.model.embed_tokens.weight,weights)
            b.bind_rows(c)
            if args.precision_policy=='residual': b.set_stability(True)
            elif args.precision_policy=='compensated': b.set_compensated(True)
            fx=fixtures(b.tokenizer)
            (out/'fixtures.json').write_text(json.dumps(fx,indent=2))
            emit(stage='loaded',fixtures=[{'name':x['name'],'P':x['P'],'m':x['m']} for x in fx],
                 gpu=torch.cuda.get_device_name(0),allocated=torch.cuda.memory_allocated())
            if args.stage=='bootstrap':
                report['bootstrap']=bootstrap(b,c,fx,out)
            elif args.stage in ('probe','certify','evidence'):
                trained=load_file(args.checkpoint or str(out/'diagnostic.safetensors'),device='cuda')
                report['diagnostic_checkpoint']=args.checkpoint or str(out/'diagnostic.safetensors')
                report['diagnostic_checkpoint_sha256']=hashlib.sha256(Path(report['diagnostic_checkpoint']).read_bytes()).hexdigest()
                bn=dict(b.named_parameters());cn=dict(c.named_parameters())
                with torch.no_grad():
                    for name,value in trained.items():
                        (cn[name[6:]] if name.startswith('codec.') else bn[name]).copy_(value)
            if args.stage=='evidence':
                from .g1_evidence import gather
                gather(b,c,fx,report,out)
                return
            if args.stage=='certify':
                from .g1_checks import certify
                certify(b,c,fx,report,out)
                if report['status']=='FAILED':
                    raise RuntimeError('G1 certification failed; see measured checks in report')
                return
            for f in fx:
                torch.cuda.reset_peak_memory_stats()
                ep=Episode(b,c,f).run()
                item=ep.summary()
                if args.stage!='initial': item['gradients']=gradients(ep,c,b)
                item['peak_allocated']=torch.cuda.max_memory_allocated()
                report['checks'].append(item);emit(stage='episode',**{k:v for k,v in item.items() if k!='events'})
                for dual in b.duals: dual.evaluation_outputs.clear()
                del ep;gc.collect();torch.cuda.empty_cache()
            report['status']='PARTIAL'  # Full matrix and controls must explicitly pass; never inferred.
            report['remaining']=['matched numeric matrix','negative controls','released generation','mechanical controls','optimizer verification']
    except Exception as exc:
        report['status']='FAILED';report['error']=str(exc);report['traceback']=traceback.format_exc()
        emit(stage='failed',error=str(exc),traceback=report['traceback'])
    finally:
        report['elapsed_seconds']=time.time()-report['started']
        (out/'report.json').write_text(json.dumps(report,indent=2))
        emit(stage='report',status=report['status'],path=str(out/'report.json'))
        if report['status']=='FAILED': raise SystemExit(1)
    if report['status']=='FAILED': raise SystemExit(1)


if __name__=='__main__': main()
