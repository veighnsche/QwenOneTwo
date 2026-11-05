"""GPU-only supervised routing LoRA, scalar regression, latent Qev/feedback training.

No external teacher service, answer grading, or claimed optimal budgets. Qev and
foundation weights are frozen; native routing LoRA and internal controls learn.
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from safetensors.torch import save_file

from .curriculum import records, encode
from .decision import CHECKPOINT, MODEL, REVISION, QEV_REVISION, digest
from .model import Runtime


def emit(**data): print(json.dumps(data), flush=True)


def save(runtime, out, report, dataset):
    out.mkdir(parents=True,exist_ok=True)
    save_file({k:v.detach().float().cpu().contiguous() for k,v in runtime.controls.state_dict().items()}, str(out/'controls.safetensors'))
    runtime.backbone.save_pretrained(out, selected_adapters=['router'], safe_serialization=True)
    # PEFT saves non-default adapters into their named subdirectory.
    paths=['controls.safetensors','router/adapter_model.safetensors','router/adapter_config.json']
    info={'format':CHECKPOINT,'model':MODEL,'model_revision':REVISION,'qev_revision':QEV_REVISION,
          'dataset_sha256':digest(dataset),'label_source':'Reproducible synthetic internal-call demonstrations and manually specified effort preferences; not optimal budget labels',
          'architecture':'native routing LoRA + learned latent gate/query/feedback + Qev decision LoRA/pointer + scalar regression',
          'files':{k:digest(out/k) for k in paths},'training':report,
          'cache_policy':'one prefill; shared mixed-adapter hybrid cache; experimental compatibility',
          'limitations':['Small synthetic curriculum; generalization is not established','Qev latent usage is newly trained, not native Qev prompt scoring','No claim of optimal thinking budgets or quality improvement']}
    (out/'training.json').write_text(json.dumps(info,indent=2))
    return info


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',default='data');ap.add_argument('--cache',required=True)
    ap.add_argument('--examples',type=int,default=240);ap.add_argument('--router-steps',type=int,default=120)
    ap.add_argument('--latent-steps',type=int,default=240);ap.add_argument('--head-steps',type=int,default=400)
    ap.add_argument('--seed',type=int,default=42);ap.add_argument('--device',default='cuda:0')
    args=ap.parse_args()
    if not args.device.startswith('cuda') or not torch.cuda.is_available(): raise RuntimeError('GPU only: accessible CUDA is required before training')
    torch.manual_seed(args.seed);random.seed(args.seed)
    started=time.time();data=Path(args.data);out=data/'trained'/CHECKPOINT
    if (out/'training.json').exists(): raise ValueError('Refusing to overwrite a completed training checkpoint')
    out.mkdir(parents=True,exist_ok=True)
    raw=records(args.seed,args.examples)
    dataset=out/'curriculum.jsonl';dataset.write_text(''.join(json.dumps(x)+'\n' for x in raw))
    # FLA's single-token fused recurrent kernel has no backward implementation.
    # Its differentiable chunk kernel is equivalent for a one-token sequence.
    # This affects this training process only; serving uses the normal kernels.
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen_impl
    qwen_impl.torch_recurrent_gated_delta_rule = qwen_impl.torch_chunk_gated_delta_rule
    p=Runtime(data,args.cache,args.device,training=True)
    if '3090' not in p.gpu_name: raise RuntimeError('Select the explicitly requested RTX 3090 by UUID')
    examples=[encode(p.tok,x) for x in raw]
    c=p.controls;p.train_router=False;p.model.eval()
    features=[];budget_x=[];budget_y=[];candidate_x=[];candidate_y=[]
    with torch.no_grad():
        for i,e in enumerate(examples):
            p.select('decision');h,cache=p.forward(e['ids'])
            budget_x.append(h[0,-1].float());budget_y.append(e['budget'])
            labels=torch.zeros(h.shape[1],device=p.device)
            if e['candidate_positions']: labels[e['candidate_positions']]=1
            candidate_x.append(c.normal(h[0]));candidate_y.append(labels)
            features.append((h[0,-1].float(),0))
            if e['prefix_ids']:
                p.select('router');prefix,cache=p.forward(e['prefix_ids'],cache)
                for j in range(prefix.shape[1]): features.append((prefix[0,j].float(),e['action'] if j==prefix.shape[1]-1 else 0))
            if i%40==0: emit(stage='features',done=i,total=len(examples),gpu=p.gpu_name)
    # Fit the actual continuous budget row on CUDA, not a categorical proxy.
    x=torch.stack(budget_x);y=torch.tensor(budget_y,device=p.device)
    scale=x.square().mean().sqrt();xn=x/scale
    row=(xn.T @ torch.linalg.solve(xn@xn.T+.3*torch.eye(len(x),device=p.device),y))/scale
    with torch.no_grad(): c.scalar.weight.copy_(row.unsqueeze(0))
    p.install_scalar()
    routes=torch.stack([x for x,_ in features]);labels=torch.tensor([y for _,y in features],device=p.device)
    cx=torch.cat(candidate_x);cy=torch.cat(candidate_y)
    optim=torch.optim.AdamW(list(c.route.parameters())+list(c.candidate.parameters()),lr=.002,weight_decay=.001)
    for step in range(args.head_steps):
        indices=torch.randint(len(routes),(min(512,len(routes)),),device=p.device)
        ci=torch.randint(len(cx),(1024,),device=p.device)
        loss=F.cross_entropy(c.gate(routes[indices]),labels[indices],weight=torch.tensor([.15,1.,1.,1.],device=p.device))
        loss+=F.binary_cross_entropy_with_logits(c.candidate(cx[ci]).squeeze(-1),cy[ci],pos_weight=torch.tensor(35.,device=p.device))
        optim.zero_grad(set_to_none=True);loss.backward();optim.step()
        if step%100==0: emit(stage='controls',step=step,loss=loss.item())
    with torch.no_grad(): route_accuracy=(c.gate(routes).argmax(-1)==labels).float().mean().item()
    # Actual Qwen routing LoRA training: gradient reaches backbone adapter weights.
    p.train_router=True;p.select('router')
    params=[v for k,v in p.backbone.named_parameters() if v.requires_grad]
    optim=torch.optim.AdamW(params+list(c.route.parameters()),lr=1e-4)
    positives=[e for e in examples if e['prefix_ids']]
    router_losses=[]
    for step in range(args.router_steps):
        e=positives[step%len(positives)]
        with torch.no_grad():
            p.select('decision');h,cache=p.forward(e['ids'])
        p.select('router')
        hidden,cache=p.forward(e['prefix_ids'],cache)
        target=torch.zeros(hidden.shape[1],device=p.device,dtype=torch.long);target[-1]=e['action']
        loss=F.cross_entropy(c.gate(hidden[0]),target,weight=torch.tensor([.15,1.,1.,1.],device=p.device))
        # Teach the native model the demonstrated reasoning continuation too.
        if hidden.shape[1]>1:
            lm=p.logits(hidden[0,:-1]); expected=torch.tensor(e['prefix_ids'][1:],device=p.device)
            loss+=.2*F.cross_entropy(lm,expected)
        optim.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(params+list(c.route.parameters()),1.);optim.step()
        router_losses.append(loss.item())
        if step%20==0: emit(stage='router_lora',step=step,loss=loss.item())
    p.train_router=False;p.select('router');p.backbone.requires_grad_(False)
    # Train latent query and result feedback through the real Qev branch and
    # original hybrid cache. Only bridge weights and Noul anchors are optimized.
    trained_params=list(c.query.parameters())+list(c.types.parameters())+list(c.feedback.parameters())+list(c.scalar_feedback.parameters())+list(c.budget_feedback.parameters())+list(c.tool_score.parameters())+list(c.choice_feedback.parameters())+[c.noul_options]
    optim=torch.optim.AdamW(trained_params,lr=.001)
    latent_losses=[]
    for step in range(args.latent_steps):
        e=examples[step%len(examples)]
        with torch.no_grad():
            p.select('decision');h,cache=p.forward(e['ids'])
            candidates=h[0,e['candidate_positions']].float() if e['candidate_positions'] else None
            initial_score=c.scalar(h[0,-1].float()).squeeze().clamp(0,1)
        initial,cache=p.feedback('budget',initial_score,cache)
        first_text=e['prefix'] or e['answer'] or 'I'
        first_token=p.tok.encode(first_text,add_special_tokens=False)[0]
        initial_loss=.2*F.cross_entropy(p.logits(initial[0,-1]).unsqueeze(0),torch.tensor([first_token],device=p.device))
        with torch.no_grad():
            if e['prefix_ids']:
                p.select('router');h,cache=p.forward(e['prefix_ids'],cache)
            else:
                h=initial.detach()
        if e['action']:
            result,dh,cache=p.latent_decision(h[0,-1].detach(),e['action'],cache,candidates)
            if e['action']==3:
                loss=F.mse_loss(c.tool_score(dh).squeeze(),torch.tensor(e['target'],device=p.device))
                value=result['score']; selected=None; kind='score'
            else:
                probabilities=result['probabilities']
                loss=-probabilities[int(e['target'])].clamp_min(1e-8).log()
                value=result['value'];selected=result['selected'];kind='noul' if e['action']==1 else 'choice'
        else:
            loss=torch.zeros((),device=p.device);value=torch.tensor(e['budget'],device=p.device);selected=None;kind='budget'
        feedback,_=p.feedback(kind,value,cache,selected)
        loss+=initial_loss
        target_text=e['answer'] or 'I'
        answer_ids=p.tok.encode(target_text,add_special_tokens=False)
        # The decision result changes the next native token through a learned
        # vector. No text result or second prompt is injected.
        loss+=.2*F.cross_entropy(p.logits(feedback[0,-1]).unsqueeze(0),torch.tensor([answer_ids[0]],device=p.device))
        optim.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(trained_params,1.);optim.step()
        latent_losses.append(loss.item())
        if step%20==0: emit(stage='latent_query_feedback',step=step,loss=loss.item())
    # Refit the internal gate on actual post-feedback states after bridge
    # training, including native continuations, so the served path matches it.
    final_x=[];final_y=[]
    with torch.no_grad():
        for i,e in enumerate(examples):
            p.select('decision');h,cache=p.forward(e['ids'])
            initial_score=c.scalar(h[0,-1].float()).squeeze().clamp(0,1)
            h,cache=p.feedback('budget',initial_score,cache)
            final_x.append(h[0,-1].float());final_y.append(0)
            if e['prefix_ids']:
                h,cache=p.forward(e['prefix_ids'],cache)
                for j in range(h.shape[1]):
                    final_x.append(h[0,j].float());final_y.append(e['action'] if j==h.shape[1]-1 else 0)
            if i%80==0:emit(stage='final_route_features',done=i,total=len(examples))
    fx=torch.stack(final_x);fy=torch.tensor(final_y,device=p.device)
    opt=torch.optim.AdamW(c.route.parameters(),lr=.001)
    for step in range(400):
        ix=torch.randint(len(fx),(min(512,len(fx)),),device=p.device)
        loss=F.cross_entropy(c.gate(fx[ix]),fy[ix],weight=torch.tensor([.15,1.,1.,1.],device=p.device))
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    with torch.no_grad():
        predicted=c.gate(fx).argmax(-1)
        final_accuracy=(predicted==fy).float().mean().item()
        per_action={str(i):(predicted[fy==i]==i).float().mean().item() for i in range(4)}
    report={'gpu':p.gpu_name,'device':str(p.device),'seed':args.seed,'examples':len(examples),
            'router_steps':args.router_steps,'latent_steps':args.latent_steps,'head_steps':args.head_steps,
            'route_training_accuracy_before_lora':route_accuracy,'final_route_training_accuracy':final_accuracy,'final_route_accuracy_by_action':per_action,'router_loss_first':router_losses[0],
            'router_loss_last':router_losses[-1],'latent_loss_first':latent_losses[0],'latent_loss_last':latent_losses[-1],
            'elapsed_seconds':round(time.time()-started,1),'peak_cuda_bytes':torch.cuda.max_memory_allocated()}
    info=save(p,out,report,dataset)
    emit(stage='complete',path=str(out),training=report)

if __name__=='__main__': main()
