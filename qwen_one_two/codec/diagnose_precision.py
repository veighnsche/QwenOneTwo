"""Controlled CUDA-only precision localization; no fitting or runtime overrides."""
import argparse
import json
from pathlib import Path
from types import MethodType
import torch
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel,SDPBackend
from ..g1_proof import configure
from ..g1_backbone import MODE
from ..g1_precision_math import compensated_linear
from .qualify import load,trace
from .session import Session
from .schema import Port


def error(x,y):
    d=x.double()-y.double()
    return (d.square().mean().sqrt()/y.double().square().mean().sqrt().clamp_min(1e-6)).item()


def states(a,b):
    entries=[('hidden',error(a['h'],b['h']))]
    for i,(pair,other) in enumerate(zip(a['state'].layers,b['state'].layers)):
        for j,(x,y) in enumerate(zip(pair,other)):
            entries.append((f'layer{i}/state{j}',error(x,y)))
    name,value=max(entries,key=lambda x:x[1])
    return {'hidden':entries[0][1],'worst':value,'family':name}


def install_ablation(b):
    for name,module in b.model.named_modules():
        if module not in b.duals:continue
        original=module.forward
        def make(original,name):
            def forward(self,x):
                ablation=b.diagnostic_ablation
                selected=b.diagnostic_targets is None or name in b.diagnostic_targets
                if self.base.weight.dtype==torch.float32 or ablation=='none' or not selected:
                    return original(x)
                mode=MODE.get();a,w=(self.R_A,self.R_B) if mode=='R' else (self.D_A,self.D_B)
                base=(F.linear(x.float(),self.base.weight.float(),None if self.base.bias is None else self.base.bias.float())
                      if ablation in ('base','all') else compensated_linear(x,self.base.weight,self.base.bias))
                update=(F.linear(F.linear(x.float(),a),w) if ablation in ('lora','all') else
                        compensated_linear(compensated_linear(x,a),w))*2.
                return base+update
            return forward
        module.forward=MethodType(make(original,name),module)


def capture(b,c,port):
    rows=[];original=Session.step
    def step(session,embeds,mode,label,capture=True):
        old=session.state
        h=original(session,embeds,mode,label,capture)
        rows.append({'label':label,'position':session.state.position,'mode':mode,
                     'input':embeds.detach().clone(),'old':old,'h':h.detach().clone(),'state':session.state})
        return h
    Session.step=step
    try:
        with torch.no_grad():result=trace(b,c,port,512,False)
    finally:Session.step=original
    return rows,result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--snapshot',required=True);ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--output',required=True);ap.add_argument('--port',type=int,default=1)
    args=ap.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    configure();report={'scope':'controlled precision localization, no training','cases':[]}
    with sdpa_kernel(SDPBackend.MATH):
        b,c,_=load(args.snapshot,args.checkpoint)
        b.feedback_precision='compensated'
        b.diagnostic_ablation='none';b.diagnostic_targets=None;install_ablation(b)
        b.set_precision(torch.float32);reference,_=capture(b,c,Port(args.port))
        for ablation in ('none','base','lora','all'):
            b.set_precision(torch.bfloat16);b.diagnostic_ablation=ablation
            actual,result=capture(b,c,Port(args.port))
            rows=[{'label':x['label'],'position':x['position'],'mode':x['mode'],
                   'input_error':error(x['input'].float(),y['input'].float()),**states(x,y)}
                  for x,y in zip(actual,reference)]
            item={'ablation':ablation,'loss':result['loss'],'steps':rows}
            report['cases'].append(item)
            print(json.dumps({'ablation':ablation,'worst':max(x['worst'] for x in rows),
                              'first_above_1pct':next((x for x in rows if x['worst']>.01),None)}),flush=True)
            (out/'diagnosis.json').write_text(json.dumps(report,indent=2))
            del actual,result
        groups={'gates':{'in_proj_a','in_proj_b'},'qkv':{'in_proj_qkv'},
                'linear_attention':{'in_proj_a','in_proj_b','in_proj_qkv','in_proj_z','out_proj'},
                'mlp':{'gate_proj','up_proj','down_proj'},'attention':{'q_proj','k_proj','v_proj','o_proj'},
                'z_out':{'in_proj_z','out_proj'},'layer13':None}
        groups.update({f'bf16_only_{name}':targets for name,targets in list(groups.items()) if name!='layer13'})
        for group,targets in groups.items():
            b.diagnostic_ablation='all'
            b.diagnostic_targets={name for name,m in b.model.named_modules() if m in b.duals and
                ('.layers.13.' in '.'+name if group=='layer13' else
                 (name.rsplit('.',1)[-1] not in targets if group.startswith('bf16_only_') else name.rsplit('.',1)[-1] in targets))}
            actual,result=capture(b,c,Port(args.port))
            rows=[{'label':x['label'],'position':x['position'],'mode':x['mode'],
                   'input_error':error(x['input'].float(),y['input'].float()),**states(x,y)}
                  for x,y in zip(actual,reference)]
            report['cases'].append({'ablation':group,'targets':sorted(b.diagnostic_targets),'loss':result['loss'],'steps':rows})
            print(json.dumps({'ablation':group,'worst':max(x['worst'] for x in rows),
                              'first_above_1pct':next((x for x in rows if x['worst']>.01),None)}),flush=True)
            (out/'diagnosis.json').write_text(json.dumps(report,indent=2))
            del actual,result
        # Same frozen FP32 input/cache isolates local arithmetic from accumulated
        # feedback. This offline experiment never changes the serving cache.
        report['same_state_replays']=[]
        b.diagnostic_targets=None
        for i,event in enumerate(reference):
            if event['label'] not in ('request','return','writer_tick'):continue
            b.diagnostic_ablation='none'
            with torch.no_grad():h,state=b(event['old'],event['mode'],event['input'])
            row={'step':i,'label':event['label'],'position':event['position'],
                 **states({'h':h,'state':state},event)}
            report['same_state_replays'].append(row)
        (out/'diagnosis.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
