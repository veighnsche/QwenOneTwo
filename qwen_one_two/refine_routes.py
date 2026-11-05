"""GPU-only native-answer and both-phase routing supervision.

Train the persistent internal gate on actual served cache/feedback states.
Preserve a separate immutable checkpoint and activate it only after validation.
"""
import argparse
import copy
import json
import shutil
from pathlib import Path

import torch
from torch.nn import functional as F
from safetensors.torch import save_file

from .model import Runtime
from .curriculum import records,encode
from .decision import CHECKPOINT,digest


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cache',required=True);ap.add_argument('--data',default='data')
    ap.add_argument('--output',default='one-two-v1-routes-2');args=ap.parse_args()
    if not torch.cuda.is_available() or '3090' not in torch.cuda.get_device_name(0):raise RuntimeError('RTX 3090 required; no CPU execution')
    root=Path(args.data);out=root/'trained'/args.output
    if out.exists():raise ValueError('Refusing to overwrite a routing checkpoint')
    p=Runtime(root,args.cache);c=p.controls;xs=[];ys=[];training_records=[]
    def add(h, action=0):
        xs.append(h.detach().float().reshape(-1,1024));ys.extend([action]*h.reshape(-1,1024).shape[0])
    with torch.no_grad():
        for i,raw in enumerate(records(seed=361,count=120)):
            e=encode(p.tok,raw);training_records.append(raw)
            for phase in ('thinking','answer') if e['action'] else ('answer',):
                p.select('decision');h,cache=p.forward(e['ids'])
                candidates=h[0,e['candidate_positions']].float() if e['candidate_positions'] else None
                score=c.scalar(h[0,-1].float()).squeeze().clamp(0,1)
                h,cache=p.feedback('budget',score,cache);add(h)
                if phase=='answer':
                    for token in p.tok.encode('</think>\n\n',add_special_tokens=False):h,cache=p.forward([token],cache);add(h)
                if e['prefix_ids']:
                    h,cache=p.forward(e['prefix_ids'],cache)
                    add(h[0,:-1]);add(h[0,-1],e['action'])
                    result,_,cache=p.latent_decision(h[0,-1],e['action'],cache,candidates)
                    if e['action']==3:value=result['score'];selected=None;kind='score'
                    else:value=result['value'];selected=result['selected'];kind='noul' if e['action']==1 else 'choice'
                    h,cache=p.feedback(kind,value,cache,selected);add(h)
                answer=p.tok.encode(e['answer'] or 'I will explain the reasoning.',add_special_tokens=False)
                h,cache=p.forward(answer,cache);add(h)
                # Teach no repeated calls in normal post-decision continuations.
                h,cache=p.forward(p.tok.encode(' That is the answer.',add_special_tokens=False),cache);add(h)
            if i%20==0:print(json.dumps({'stage':'native_and_both_phase_features','done':i}),flush=True)
    x=torch.cat(xs);y=torch.tensor(ys,device=p.device)
    c.route.requires_grad_(True)
    opt=torch.optim.AdamW(c.route.parameters(),lr=.0007,weight_decay=.001)
    for step in range(1000):
        ix=torch.randint(len(x),(512,),device=p.device)
        loss=F.cross_entropy(c.gate(x[ix]),y[ix],weight=torch.tensor([.2,1.,1.,1.],device=p.device))
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    with torch.no_grad():
        pred=c.gate(x).argmax(-1)
        scores={str(k):(pred[y==k]==k).float().mean().item() for k in range(4)}
    out.mkdir(parents=True);src=root/'trained'/CHECKPOINT
    shutil.copytree(src/'router',out/'router')
    save_file({k:v.detach().float().cpu().contiguous() for k,v in c.state_dict().items()},str(out/'controls.safetensors'))
    dataset=out/'routing-curriculum.jsonl';dataset.write_text(''.join(json.dumps(x)+'\n' for x in training_records))
    info=copy.deepcopy(p.training_info)
    info['parent_controls_sha256']=info['files']['controls.safetensors']
    info['files']['controls.safetensors']=digest(out/'controls.safetensors')
    info['training']['routing_refinement']={'gpu':p.gpu_name,'examples':len(training_records),'hidden_states':len(y),
                                            'steps':1000,'accuracy_by_action':scores,'phases':['thinking','answer'],
                                            'dataset_sha256':digest(dataset)}
    (out/'training.json').write_text(json.dumps(info,indent=2))
    print(json.dumps({'stage':'complete','checkpoint':str(out),'accuracy_by_action':scores}),flush=True)

if __name__=='__main__':main()
