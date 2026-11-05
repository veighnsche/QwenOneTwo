"""Actual learned internal invocation and one-cache generation. CUDA ONLY."""
import argparse
import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from qwen_one_two.model import Runtime
from qwen_one_two.curriculum import records,encode
from qwen_one_two.thinking import score_to_budget


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cache',required=True);ap.add_argument('--data',default='data')
    ap.add_argument('--thinking',type=int,default=64);args=ap.parse_args()
    if not torch.cuda.is_available() or '3090' not in torch.cuda.get_device_name(0):raise RuntimeError('RTX 3090 required; no CPU fallback')
    p=Runtime(args.data,args.cache)
    assert all(x.device.type=='cuda' for x in p.model.parameters())
    assert p.model.one_two is p.controls
    router_b=[v for k,v in p.backbone.named_parameters() if 'lora_B.router' in k]
    assert router_b and any(torch.count_nonzero(v).item()>0 for v in router_b), 'Native routing LoRA was not trained'
    reports=[];observed=set()
    cases=[
        ('Translate good morning into French. Only the translation.',0),
        ("Mira's application was approved. Is the application approved? Answer yes or no.",args.thinking),
        ('The requested color is blue. Choose the matching option.\nA: red\nB: blue',args.thinking),
        ('A progress indicator shows 70 percent complete. Give its completion score between 0 and 1.',args.thinking),
    ]
    for case_index,(text,budget) in enumerate(cases):
        trace={'prefills':0,'steps':0,'cache':None,'adapters':[]}
        def before(module,args,kwargs):
            cache=kwargs.get('past_key_values');value=kwargs.get('input_ids',kwargs.get('inputs_embeds'))
            assert value.device.type=='cuda'
            trace['adapters'].append(p.active_adapter)
            if cache is None:
                trace['prefills']+=1;assert trace['prefills']==1
            else:
                assert cache is trace['cache'];assert value.shape[1]==1
                trace['steps']+=1
        def after(module,args,out):
            if trace['cache'] is None:trace['cache']=out.past_key_values
            else:assert trace['cache'] is out.past_key_values
        a=p.backbone.register_forward_pre_hook(before,with_kwargs=True);b=p.backbone.register_forward_hook(after)
        try:r=p.thinking.generate({'text':text,'run_id':uuid.uuid4().hex,'budget_override':budget,'answer_tokens':48})
        finally:a.remove();b.remove()
        assert r['prefill_count']==trace['prefills']==1
        assert r['continuation_count']==trace['steps']
        assert r['predicted_budget']==score_to_budget(r['score'])
        assert r['cache_continuous'] and r['answer'] and r['adapter']=='router'
        assert r['answer_end']=='eos', 'Simple demonstration did not complete its normal answer'
        if case_index==2:
            assert r['answer'].strip().lower().startswith('b') and 'blue' in r['answer'].lower(), r['answer']
        assert r['thinking_tokens']<=budget and r['answer_tokens']<=48
        assert trace['adapters'][0]=='decision' and trace['adapters'][1]=='router'
        calls=[x for x in r['decision_calls'] if x['status']=='complete']
        observed.update(x['type'] for x in calls)
        report={k:r[k] for k in ['text','score','predicted_budget','thinking_tokens','answer','thinking','decision_calls','prefill_count','cache_continuous','latent_steps']}
        report['observed_adapters']=trace['adapters'];reports.append(report)
        print(json.dumps(report),flush=True)
    # Separately probe all learned gate operations at held-out demonstrated
    # call boundaries, without assigning/overriding the gate prediction.
    boundaries=[]
    with torch.no_grad():
        for e in [encode(p.tok,x) for x in records(seed=918,count=12) if x['action']]:
            p.select('decision');h,cache=p.forward(e['ids'])
            candidates=h[0,e['candidate_positions']].float() if e['candidate_positions'] else None
            score=p.controls.scalar(h[0,-1].float()).squeeze().clamp(0,1)
            _,cache=p.feedback('budget',score,cache)
            h,cache=p.forward(e['prefix_ids'],cache)
            action=int(p.controls.gate(h[0,-1]).argmax())
            assert action==e['action'],f'Learned gate selected {action}, expected {e["action"]}'
            result,_,_=p.latent_decision(h[0,-1],action,cache,candidates)
            value=float(result['score']) if action==3 else result['probabilities'].tolist()
            boundaries.append({'action':action,'result':value,'target':e['target'],'teacher_forced_prefix':True})
    original=p.thinking.decode
    def cancelled(*args):
        state,publish=args[-2:]
        def update():
            publish()
            if state['thinking_tokens']+state['answer_tokens']>=1:Path(p.cancel_file).touch()
        return original(*args[:-1],update)
    p.thinking.decode=cancelled
    try:
        p.thinking.generate({'text':'Explain binary search.','run_id':uuid.uuid4().hex,'budget_override':128,'answer_tokens':8})
        raise AssertionError('Cancellation did not stop generation')
    except RuntimeError as e:assert 'cancelled' in str(e)
    finally:Path(p.cancel_file).unlink(missing_ok=True);p.thinking.decode=original
    assert p.active_adapter=='router'
    out=Path('artifacts/one-two-gpu.json');out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({'gpu':p.gpu_name,'training':p.training_info['training'],'free_generation_operations':sorted(observed),
                               'runs':reports,'boundary_probes':boundaries,'cancel_restored_native':True},indent=2))
    assert observed=={'noul','choice','score'},f'Free generation did not invoke all operations: {observed}'
    print('GPU single-prefill, learned invocation, latent feedback and normal-answer checks passed.')

if __name__=='__main__':main()
