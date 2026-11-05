"""Single-prefill generation with trained textless Qev calls and scalar budget."""
import hashlib
import json
import math
import os
import time

import torch

from .decision import ACTIONS, CHECKPOINT, MODEL, REVISION, QEV_REVISION
from .policy import MAX_THINKING_TOKENS, TOTAL_CONTEXT_TOKENS, SCORE_CURVE, SCORE_MAPPING


def bounded_score(raw):
    if not math.isfinite(raw): raise ValueError('model produced a non-finite thinking score')
    return max(0.,min(1.,raw))


def score_to_budget(score):
    if not math.isfinite(score) or not 0<=score<=1: raise ValueError('thinking score must be finite and between 0 and 1')
    return math.floor((3*score*score-2*score**3)*MAX_THINKING_TOKENS+.5)


def prompt_text(a):
    text=a.get('text')
    if not isinstance(text,str) or not text.strip() or len(text)>2000: raise ValueError('enter a request of 1–2000 characters')
    return text.strip()


def bounded_int(value,low,high,name):
    if type(value) is not int or not low<=value<=high: raise ValueError(f'{name} must be an integer between {low} and {high}')
    return value


def run_id(value):
    if not isinstance(value,str) or len(value)!=32 or any(c not in '0123456789abcdef' for c in value): raise ValueError('invalid run id')
    return value


def sha(text): return hashlib.sha256(text.encode()).hexdigest()


class ThinkingBudget:
    def __init__(self,runtime): self.p=runtime

    def check_cancel(self):
        if self.p.cancel_file and os.path.exists(self.p.cancel_file): raise RuntimeError('QwenOneTwo operation cancelled')

    def chat(self,text,thinking=True):
        return self.p.tok.apply_chat_template([{'role':'user','content':text}],tokenize=False,
                                             add_generation_prompt=True,enable_thinking=thinking)

    def support(self):
        p=self.p
        if p.device.type!='cuda' or not torch.cuda.is_available(): raise RuntimeError('CUDA GPU required; CPU model execution is forbidden')
        self.open_id,self.close_id=(p.tok.encode(s,add_special_tokens=False)[0] for s in ('<think>','</think>'))
        if not self.chat('check').endswith('<think>\n'): raise ValueError('Unsupported Qwen thinking template')

    def provenance(self):
        p=self.p
        return {'model':MODEL,'revision':REVISION,'qev_revision':QEV_REVISION,'training_checkpoint':CHECKPOINT,
                'output_type':'continuous_score','score_range':[0.,1.],'score_curve':SCORE_CURVE,
                'total_context_tokens':TOTAL_CONTEXT_TOKENS,'max_thinking_tokens':MAX_THINKING_TOKENS,
                'score_mapping':SCORE_MAPPING,'controls_sha256':(p.training_info or {}).get('files',{}).get('controls.safetensors'),
                'cache_policy':'one prefill, same mixed-adapter hybrid cache; experimental compatibility',
                'invocation':'learned hidden-state gate; latent query and feedback; no generated tool arguments'}

    def head(self):
        if self.p.training_info is None: raise ValueError('Train QwenOneTwo on GPU before generation')
        info={**self.provenance(),**self.p.training_info,'rows_sha256':self.p.training_info['files']['controls.safetensors']}
        return info,[self.p.scalar_id],None

    def build(self,args): return self.head()[0]  # Training is explicit and reproducible, never triggered by a page visit.

    def generate(self,a):
        self.support();text=prompt_text(a);rid=run_id(a.get('run_id'))
        if 'decision_id' in a: raise ValueError('Submit a single prompt; separate decision requests are unsupported')
        override=a.get('budget_override')
        if override is not None: override=bounded_int(override,0,MAX_THINKING_TOKENS,'budget_override')
        limit=bounded_int(a.get('answer_tokens',256),1,512,'answer_tokens')
        self.head();self.check_cancel()
        folder=self.p.data_dir/'runs';folder.mkdir(parents=True,exist_ok=True);path=folder/f'{rid}.json'
        if path.exists(): raise ValueError('run id already used')
        for old in sorted(folder.glob('*.json'),key=lambda f:f.stat().st_mtime)[:-31]: old.unlink(missing_ok=True)
        state={'run_id':rid,'text':text,'score':None,'raw_score':None,'predicted_budget':None,'budget':None,
               'manual_override':override is not None,'answer_limit':limit,'thinking_tokens':0,'answer_tokens':0,
               'thinking':'','answer':'','phase':'prefill','thinking_end':None,'answer_end':None,
               'prefill_count':0,'continuation_count':0,'cache_continuous':True,'sampled_control_tokens':0,
               'forced_control_tokens':0,'latent_steps':0,'decision_calls':[],'adapter':'decision',
               'provenance':{**self.provenance(),'prompt_sha256':sha(text)}}
        start=time.perf_counter()
        def publish():
            state['elapsed_ms']=round((time.perf_counter()-start)*1000)
            state['sampled_tokens']=state['thinking_tokens']+state['answer_tokens']+state['sampled_control_tokens']
            temp=path.with_suffix('.tmp');temp.write_text(json.dumps(state));temp.replace(path)
        publish()
        try:
            with torch.no_grad(): self.decode(text,override,limit,state,publish)
            state['phase']='complete'
        except Exception as exc:
            state['phase']='cancelled' if os.path.exists(self.p.cancel_file) else 'error'
            state['error']=str(exc)[:500]
            if 'replaced the continuous inference cache' in str(exc):state['cache_continuous']=False
            publish();raise
        finally:
            self.p.select('router');state['adapter']='router';publish()
        return state

    def decode(self,text,override,answer_limit,state,publish):
        p=self.p;c=p.controls;self.check_cancel()
        ids=p.tok.encode(self.chat(text),add_special_tokens=False)
        p.select('decision');started=time.perf_counter();hidden,cache=p.forward(ids)
        state['prefill_count']=1;state['prompt_tokens']=len(ids)
        # One actual reserved LM-head row, read alongside normal vocabulary logits.
        raw=p.logits(hidden[0,-1])[p.scalar_id].item();score=bounded_score(raw)
        budget=score_to_budget(score)
        state.update(raw_score=raw,score=score,predicted_budget=budget,budget=budget if override is None else override,
                     score_ms=round((time.perf_counter()-started)*1000,1),score_generated_tokens=0,phase='scored')
        candidates_at=(c.candidate(c.normal(hidden[0])).squeeze(-1).sigmoid()>=.5).nonzero().flatten()
        # A learned token-boundary selector chooses candidates. This bound is a
        # resource limit, not semantic keyword routing or a probability threshold on budgets.
        if len(candidates_at)>16:
            rank=c.candidate(c.normal(hidden[0])).squeeze(-1)[candidates_at].topk(16).indices
            candidates_at=candidates_at[rank].sort().values
        candidates=hidden[0,candidates_at].float() if len(candidates_at)>=2 else None
        state['candidate_positions']=candidates_at.tolist()
        publish();self.check_cancel()
        # Budget feedback starts native generation; this is one latent incremental
        # step on the original cache, not a second prompt or a second prefill.
        hidden,cache=p.feedback('budget',torch.tensor(score,device=p.device),cache)
        state['adapter']='router';state['latent_steps']+=1;state['continuation_count']+=1
        h=hidden[0,-1]

        def feed(token):
            nonlocal h
            self.check_cancel();p.select('router')
            output,_=p.forward([token],cache)
            h=output[0,-1];state['continuation_count']+=1

        def internal_call(phase):
            nonlocal h
            self.check_cancel()
            if len(state['decision_calls'])>=16:
                if not state.get('decision_limit_reached'):
                    state['decision_limit_reached']=True;publish()
                return
            logits=c.gate(h);action=int(logits.argmax())
            if action==0:return
            if action==2 and candidates is None:
                # The gate requested Choice, but no complete candidate set was
                # selected. Record it; do not fabricate options or a decision.
                state['decision_calls'].append({'type':'choice','phase':phase,'status':'no_candidates',
                    'at_token':state[f'{phase}_tokens']});publish();return
            event={'type':ACTIONS[action],'phase':phase,'status':'running','at_token':state[f'{phase}_tokens'],
                   'gate_probabilities':logits.softmax(-1).tolist(),'text_tokens':0}
            state['decision_calls'].append(event);state['adapter']='decision';publish()
            result,_,_=p.latent_decision(h,action,cache,candidates)
            state['latent_steps']+=1;state['continuation_count']+=1
            if action==3:
                value=result['score'];selected=None;event['score']=value.item()
            else:
                probs=result['probabilities'];value=result['value'];selected=result['selected']
                event['probabilities']=probs.tolist()
                if action==1:event['noul']=value.item()
                else:event['choice_index']=int(result['index'])
            output,_=p.feedback(ACTIONS[action],value,cache,selected)
            h=output[0,-1];state['latent_steps']+=1;state['continuation_count']+=1
            state['adapter']='router';event['status']='complete';publish()

        def choose(thinking):
            scores=p.logits(h).clone();scores[len(p.tok):]=-torch.inf;scores[self.open_id]=-torch.inf
            if thinking:scores[p.eos]=-torch.inf
            else:scores[self.close_id]=-torch.inf
            return int(scores.argmax())

        if state['budget']:
            state['phase']='thinking';publish();thought=[]
            for _ in range(state['budget']):
                internal_call('thinking');token=choose(True)
                if token==self.close_id:
                    state['sampled_control_tokens']+=1;state['thinking_end']='natural_close';feed(token);break
                thought.append(token);state['thinking_tokens']=len(thought);state['thinking']=p.tok.decode(thought,skip_special_tokens=True)
                publish();feed(token)
            else:
                state['thinking_end']='budget_exhausted';state['forced_control_tokens']+=1;feed(self.close_id)
        else:
            state['thinking_end']='skipped';state['forced_control_tokens']+=1;feed(self.close_id)
        for token in p.tok.encode('\n\n',add_special_tokens=False):feed(token);state['forced_control_tokens']+=1
        state['phase']='answer';publish();answer=[]
        for _ in range(answer_limit):
            internal_call('answer');token=choose(False)
            if token in p.eos:
                state['sampled_control_tokens']+=1;state['answer_end']='eos';break
            answer.append(token);state['answer_tokens']=len(answer);state['answer']=p.tok.decode(answer,skip_special_tokens=True)
            publish();feed(token)
        else:state['answer_end']='answer_limit'
