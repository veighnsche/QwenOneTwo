"""GPU-only full-continuation training of latent result consumption.

Keep Qev, native routing LoRA, routing gate and initial budget unchanged.
Train the result feedback bridge through the complete normal answer, including
thinking closure and EOS; fit a separate continuous generic Score row.
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
from .decision import digest
from .curriculum import records,encode


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cache',required=True);ap.add_argument('--data',default='data')
    ap.add_argument('--output',default='one-two-v1-feedback-3');ap.add_argument('--steps',type=int,default=240)
    ap.add_argument('--score-feedback-lr',type=float,default=.02)
    ap.add_argument('--matrix-feedback-lr',type=float,default=.001)
    ap.add_argument('--only-operation',choices=['noul','choice','score'])
    ap.add_argument('--first-answer-weight',type=float,default=2.)
    ap.add_argument('--choice-index-only',action='store_true')
    ap.add_argument('--choice-feedback-lr',type=float,default=.03)
    args=ap.parse_args()
    if not torch.cuda.is_available() or '3090' not in torch.cuda.get_device_name(0):raise RuntimeError('RTX 3090 required; no CPU fallback')
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen_impl
    qwen_impl.torch_recurrent_gated_delta_rule=qwen_impl.torch_chunk_gated_delta_rule
    root=Path(args.data);out=root/'trained'/args.output
    if out.exists():raise ValueError('Refusing to overwrite a checkpoint')
    source=(root/'trained/current').resolve() if (root/'trained/current').exists() else root/'trained/one-two-v1'
    p=Runtime(root,args.cache);p.training_mode=True;p.train_router=False;c=p.controls
    examples=[encode(p.tok,x) for x in records(seed=42,count=240) if x['action']]
    def setup(e):
        p.select('decision');h,cache=p.forward(e['ids'])
        candidates=h[0,e['candidate_positions']].float() if e['candidate_positions'] else None
        score=c.scalar(h[0,-1].float()).squeeze().clamp(0,1)
        _,cache=p.feedback('budget',score,cache)
        h,cache=p.forward(e['prefix_ids'],cache)
        result,dh,cache=p.latent_decision(h[0,-1],e['action'],cache,candidates)
        return result,dh,cache
    # Direct scalar regression, not expected categorical levels or option bins.
    sx=[];sy=[]
    with torch.no_grad():
        for e in examples:
            if e['action']!=3:continue
            _,h,_=setup(e);sx.append(h.float());sy.append(e['target'])
        x=torch.stack(sx);y=torch.tensor(sy,device=p.device);scale=x.square().mean().sqrt();xn=x/scale
        w=(xn.T@torch.linalg.solve(xn@xn.T+.01*torch.eye(len(x),device=p.device),y))/scale
        c.tool_score.weight.copy_(w.unsqueeze(0));p.install_scalar()
    if args.only_operation:
        operation={'noul':1,'choice':2,'score':3}[args.only_operation]
        examples=[e for e in examples if e['action']==operation]
    params=list(c.choice_feedback.parameters()) if args.choice_index_only else list(c.feedback.parameters())+list(c.scalar_feedback.parameters())+list(c.choice_feedback.parameters())
    for v in params:v.requires_grad_(True)
    groups=[{'params':list(c.choice_feedback.parameters()),'lr':args.choice_feedback_lr}]
    if not args.choice_index_only:
        groups.extend([{'params':list(c.feedback.parameters()),'lr':args.matrix_feedback_lr}, {'params':list(c.scalar_feedback.parameters()),'lr':args.score_feedback_lr}])
    optimizer=torch.optim.AdamW(groups)
    losses=[]
    for step in range(args.steps):
        e=examples[step%len(examples)]
        with torch.no_grad():result,_,cache=setup(e)
        if e['action']==3:value=result['score'];selected=None;kind='score'
        else:value=result['value'];selected=result['selected'];kind='noul' if e['action']==1 else 'choice'
        h,cache=p.feedback(kind,value,cache,selected)
        target=p.tok.encode('</think>\n\n'+e['answer']+'<|im_end|>',add_special_tokens=False)
        loss=F.cross_entropy(p.logits(h[0,-1]).unsqueeze(0),torch.tensor([target[0]],device=p.device))
        tail,_=p.forward(target[:-1],cache)
        loss+=F.cross_entropy(p.logits(tail[0]),torch.tensor(target[1:],device=p.device))
        boundary=len(p.tok.encode('</think>\n\n',add_special_tokens=False))
        loss+=args.first_answer_weight*F.cross_entropy(p.logits(tail[0,boundary-1]).unsqueeze(0),torch.tensor([target[boundary]],device=p.device))
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(params,1.);optimizer.step()
        losses.append(loss.item())
        if step%20==0:print(json.dumps({'stage':'full_answer_feedback','step':step,'loss':loss.item()}),flush=True)
    out.mkdir(parents=True);shutil.copytree(source/'router',out/'router')
    save_file({k:v.detach().float().cpu().contiguous() for k,v in c.state_dict().items()},str(out/'controls.safetensors'))
    dataset=out/'feedback-curriculum.jsonl';dataset.write_text(''.join(json.dumps({k:v for k,v in e.items() if k not in ('ids','prefix_ids','candidate_positions')})+'\n' for e in examples))
    info=copy.deepcopy(p.training_info);info['parent_controls_sha256']=info['files']['controls.safetensors']
    info['files']['controls.safetensors']=digest(out/'controls.safetensors')
    info['training']['feedback_completion']={'gpu':p.gpu_name,'steps':args.steps,'score_feedback_lr':args.score_feedback_lr,'matrix_feedback_lr':args.matrix_feedback_lr,'only_operation':args.only_operation,'choice_index_only':args.choice_index_only,'choice_feedback_lr':args.choice_feedback_lr,'first_answer_weight':args.first_answer_weight,'loss_first':losses[0],'loss_last':losses[-1],
        'dataset_sha256':digest(dataset),'score_fit':'CUDA scalar ridge regression; generic Score separate from thinking budget',
        'supervision':'Full thinking closure, normal answer and EOS; initial budget and native routing LoRA unchanged'}
    (out/'training.json').write_text(json.dumps(info,indent=2))
    print(json.dumps({'stage':'complete','checkpoint':str(out),'loss_last':losses[-1]}),flush=True)

if __name__=='__main__':main()
