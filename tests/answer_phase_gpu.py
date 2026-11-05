"""Bounded learned gate probes inside an answer context. CUDA GPU ONLY."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from qwen_one_two.model import Runtime
from qwen_one_two.curriculum import records,encode


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cache',required=True);args=ap.parse_args()
    if not torch.cuda.is_available() or '3090' not in torch.cuda.get_device_name(0):raise RuntimeError('RTX 3090 required')
    p=Runtime('data',args.cache);report=[]
    with torch.no_grad():
        for raw in records(seed=918,count=4):
            if not raw['action']:continue
            e=encode(p.tok,raw);p.select('decision');h,cache=p.forward(e['ids'])
            candidates=h[0,e['candidate_positions']].float() if e['candidate_positions'] else None
            score=p.logits(h[0,-1])[p.scalar_id].clamp(0,1)
            _,cache=p.feedback('budget',score,cache)
            for token in p.tok.encode('</think>\n\n',add_special_tokens=False):_,cache=p.forward([token],cache)
            h,cache=p.forward(e['prefix_ids'],cache)
            predicted=int(p.controls.gate(h[0,-1]).argmax())
            assert predicted==e['action'],(predicted,e['action'])
            result,_,_=p.latent_decision(h[0,-1],predicted,cache,candidates)
            value=result['score'] if predicted==3 else result['value']
            assert torch.isfinite(value).all()
            _,after=p.feedback(('continue','noul','choice','score')[predicted],value,cache,result.get('selected'))
            assert after is cache and p.active_adapter=='router'
            report.append({'phase':'answer','learned_action':predicted,'teacher_forced_prefix':e['prefix'],
                           'value':float(value),'same_cache':True,'adapter_restored':True})
    Path('artifacts/one-two-answer-phase.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
