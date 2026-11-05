"""CUDA-only controlled precision ablations; original evidence is never edited."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,json,time
from pathlib import Path
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from safetensors.torch import load_file
from .g1_proof import configure,fixtures,Episode,emit,gradients
from .g1_backbone import Backbone
from .g1_protocol import Codec
from .g1_checks import freeze_episode,compare,clear


def load(snapshot,checkpoint):
    b=Backbone(snapshot,'data/qev')
    pointer=load_file('data/qev/pointer.safetensors',device='cuda')
    with torch.device('cuda'): c=Codec(b.sigma,b.model.model.embed_tokens.weight,pointer)
    b.bind_rows(c)
    weights=load_file(str(checkpoint),device='cuda')
    bp=dict(b.named_parameters());cp=dict(c.named_parameters())
    with torch.no_grad():
        for name,value in weights.items():
            (cp[name[6:]] if name.startswith('codec.') else bp[name]).copy_(value)
    return b,c


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--snapshot',required=True);ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--stability',action='store_true')
    ap.add_argument('--compensated',action='store_true')
    ap.add_argument('--profiles',nargs='+',default=['bf16','fp32','fp32_effective_weights','fp32_rounded_inputs','fp32_matched'])
    ap.add_argument('--gradients',action='store_true')
    args=ap.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    configure();report={'status':'DIAGNOSTIC','cases':[]}
    with sdpa_kernel(SDPBackend.MATH):
        b,c=load(args.snapshot,args.checkpoint)
        b.set_stability(args.stability)
        if args.compensated: b.set_compensated(True)
        for f in fixtures(b.tokenizer):
            base=None
            for profile in args.profiles:
                b.set_precision(torch.bfloat16 if profile=='bf16' else torch.float32)
                b.round_reference_inputs=profile in ('fp32_rounded_inputs','fp32_matched')
                for m in b.duals: m.effective_reference=profile in ('fp32_effective_weights','fp32_matched')
                records=[]
                original=b._layer
                def trace(index,mode,h,s0,s1,cos,sin):
                    result=original(index,mode,h,s0,s1,cos,sin)
                    # Prefix localization, with layers matched by absolute position.
                    if h.shape[1]==f['P']:
                        records.append((index,result[0].detach().clone(),result[2].detach().clone()))
                    return result
                b._layer=trace
                with torch.set_grad_enabled(args.gradients):
                    ep=Episode(b,c,f).run()
                    grad=gradients(ep,c,b,vectors=True) if args.gradients else {}
                b._layer=original
                current=freeze_episode(ep,grad)
                if base is None: base=(current,records)
                item={'fixture':f['name'],'profile':profile,'value':ep.value.item(),'gradient_paths':grad}
                if profile!='bf16':
                    item['comparison']=compare(current,base[0],fp32=True)
                    item['prefix_layers']=[]
                    for (idx,x,state),(_,y,old) in zip(records,base[1]):
                        err=(x.double()-y.double()).square().mean().sqrt()/x.double().square().mean().sqrt().clamp_min(1e-6)
                        se=(state.double()-old.double()).square().mean().sqrt()/state.double().square().mean().sqrt().clamp_min(1e-6)
                        item['prefix_layers'].append({'layer':idx,'hidden_rms_error':err.item(),'state_rms_error':se.item()})
                report['cases'].append(item)
                emit(stage='ablation',**{k:v for k,v in item.items() if k!='prefix_layers'})
                (out/'diagnosis.json').write_text(json.dumps(report,indent=2))
                del ep
                if current is not base[0]: del current,records
                clear(b)
            del base;clear(b)


if __name__=='__main__':main()
