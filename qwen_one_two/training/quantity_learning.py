"""Matched count-access/Score experiment. No new semantic targets or routing."""
import math
import uuid
from .inputs import canonical, check, digest, Exposure
from .numeric_access import NumericConstruction, count_specs
from .component_producer import component_view
from .behavior_qualification import PROMPT, PROFILE, expected, receiver_hash

PROTOCOL='quantity-receiver-learning-v1'
PASSES=4
COUNT_SCALE=100.


def training_panel(release):
    check(release.purpose=='training', 'training release only')
    panel=[]
    for entry in release.manifest['records']:
        if not entry['selected']:continue
        row=release.record(entry['id'])
        if row['training_category']!='receiver' or row['operation']!='Score':continue
        gold=expected(row)
        if gold['status']!='Assessed':continue
        view,p,index=component_view(release,row)
        excerpts,body=view.materialize(p);specs=count_specs(excerpts)
        counts=[int(s['literal']['text']) for s in specs]
        check(index['direction']==1 and counts[0] in (20,100), 'existing completion training scope')
        lo,hi=gold['payload']['score_interval']
        check(lo==hi and abs(lo-counts[1]/counts[0])<1e-12, 'existing completion label binding')
        panel.append({'id':row['id'],'family':row['family'],'record_sha256':digest(canonical(row)),
                      'view_sha256':digest(body),'counts':counts,'target':lo})
    panel.sort(key=lambda x:digest((PROTOCOL+':17:'+x['id']).encode()))
    check(len(panel)==254 and len({x['family'] for x in panel})==96, 'complete assessed Score training set')
    check(max(v for x in panel for v in x['counts'])==COUNT_SCALE, 'training-only count normalization')
    return panel


def quantity_summary(cases):
    rows=[c['quantities'] for c in cases if 'quantities' in c]
    check(all(len(r['predicted'])==len(r['target'])==2 and
              all(math.isfinite(v) for v in r['predicted']) for r in rows), 'quantity receipt')
    result={'scheduled':len(cases),'measured':len(rows),'missing':len(cases)-len(rows)}
    for i,name in enumerate(('required','completed')):
        result[name]={'mae':sum(abs(r['predicted'][i]-r['target'][i]) for r in rows)/len(rows) if rows else None,
                     'nearest_integer_correct':sum(round(r['predicted'][i])==r['target'][i] for r in rows),
                     'exact_correct':sum(r['predicted'][i]==r['target'][i] for r in rows)}
    result['both_nearest_integer_correct']=sum(all(round(a)==b for a,b in zip(r['predicted'],r['target'])) for r in rows)
    return result


def quantity_receipt(prediction, specs):
    return {'predicted':(prediction.detach()*COUNT_SCALE).tolist(),
            'target':[int(s['literal']['text']) for s in specs],
            'scope':'auxiliary count readout only; never supplied to Score or runtime'}


class QuantityProducer:
    """Actual differentiable latent Session; Score and count objectives only."""
    def __init__(self,backbone,codec,reference,release,arm):
        from .clean import CleanEncoder
        check(release.purpose=='training', 'development cannot enter producer')
        self.b,self.c,self.ref,self.release=backbone,codec,reference,release
        self.construction=NumericConstruction(arm,allow_training=True)
        self.encoder=CleanEncoder(backbone,reference,checkpoint_hash=receiver_hash(backbone,codec),
                                  profile_hash=digest(PROFILE.encode()),cache_references=True)

    def __call__(self,row):
        import torch
        from .clean_packets import CleanPackets
        from .losses import regression
        from ..codec.neural import norm
        from ..codec.schema import Port
        from ..codec.session import Session
        view,p,index=component_view(self.release,row)
        check(row['training_category']=='receiver' and row['operation']=='Score', 'Score training receiver')
        view,p,index=self.construction.prepare(view,p,index)
        self.encoder.identity['checkpoint']=receiver_hash(self.b,self.c)
        run=str(uuid.uuid4());encoded=self.encoder.encode(view,p,run=run)
        factory=CleanPackets(encoded,view,p)
        prompt=self.b.tokenizer.apply_chat_template([{'role':'user','content':PROMPT}],
            tokenize=False,add_generation_prompt=True,enable_thinking=True)
        s=Session.open(self.b,self.c,self.b.tokenizer.encode(prompt,add_special_tokens=False),128,override=512,run=run)
        rt=s.runtime
        report={'producer':PROTOCOL,'record':row['id'],'record_sha256':digest(canonical(row)),
                'input_sha256':encoded['input_hash'],'construction_audit':encoded['audit'],
                'text_tokens_for_invocation':0,'gold_return':False,'diagnostic_reference':False}
        s.episode_report=report;losses={};debit=rt.debit
        def bounded(kind,count,label):
            check(rt.position+count<=512,'complete quantity receiver exceeds 512 positions')
            return debit(kind,count,label)
        rt.debit=bounded
        try:
            spans=[]
            for i,ref in enumerate(factory.sources):
                n=len(factory.registry.get(ref).token_ids)
                spans.extend(factory.span(i,j,min(32,n-j)) for j in range(0,n,32))
            check(len(spans)<=8,'complete evidence; never crop')
            packet=self.construction.request(factory,index)(Port.SCORE,index['question'],index['target'],
                evidence=tuple(spans),candidates=index['candidates'],rubric=index['rubric'],direction=index['direction'],session=s)
            report['input_construction']=dict(self.construction.receipt)
            _,evaluated,_=s.evaluate(packet)
            # Both predictions are computed before targets; the auxiliary output
            # is never fed to the receiver's existing Score row or request.
            prediction=self.c.quantity_probe(norm(s.hidden[0,-1].float()))
            report['head']={'status':('Assessed','Insufficient','Conflicting','Unresolved','NoFeasibleChoice')[int(evaluated['status_logits'].argmax())],
                            'value':float(evaluated['value'].detach()),'raw':float(evaluated['raw'].detach())}
            gold=expected(row)
            check(gold['status']=='Assessed','only reviewed assessed training rows')
            targets=torch.tensor([int(x['literal']['text']) for x in index['numeric_specs']],device='cuda',dtype=torch.float32)/COUNT_SCALE
            score=regression(evaluated['raw'],interval=gold['payload']['score_interval'])
            quantities=(prediction-targets).square().mean()
            # Same eligibility for both terms on every accepted example permits
            # one VJP, exactly the sum of their equally weighted eligible means.
            losses={'quantity_score':score+quantities}
            self.loss_terms=(score,quantities)
            report.update(losses={'score':float(score.detach()),'quantities':float(quantities.detach())},
                          quantities=quantity_receipt(prediction,index['numeric_specs']))
            rt.terminate('Completed',emit_end=False)
        except ValueError as exc:
            losses={};report['failure']=str(exc);rt.terminate('Failed',usable=False,emit_end=False)
        finally:del rt.debit
        report['summary']=s.summary()
        return s,losses,Exposure('isolated_receiver',packet_origin='reviewed_component',decision_forcing=True,
                                semantic_substitution=True,budget_override=512)
