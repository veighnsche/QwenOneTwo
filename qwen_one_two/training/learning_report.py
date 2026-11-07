"""Model-free receiver learning metrics and training-only probe selection."""
from collections import Counter,defaultdict
import math
from .inputs import check,digest,canonical
from .behavior_qualification import expected,grade,fp32

PROTOCOL='receiver-training-probe-v1'


def probe_panel(release):
    check(release.purpose=='training','probe selection requires training release')
    rows=[release.record(e['id']) for e in release.manifest['records'] if e['selected']]
    rows=[r for r in rows if r['training_category']=='receiver']
    rows.sort(key=lambda r:digest((PROTOCOL+':17:'+r['id']).encode()))
    selected=[];families=set()
    def take(candidates,n):
        chosen=0
        for row in candidates:
            if row['family'] in families:continue
            selected.append(row);families.add(row['family']);chosen+=1
            if chosen==n:return
        check(False,'insufficient distinct training families for probe stratum')
    def cell(op,status):return [r for r in rows if r['operation']==op and expected(r)['status']==status]
    noul=cell('Noul','Assessed')
    for truth in (False,True):
        take([r for r in noul if (expected(r)['payload']['p_true']>.5)==truth
              and expected(r)['payload']['p_true']!=.5],4)
    take(cell('Choice','Assessed'),4)
    score=cell('Score','Assessed')
    midpoint=lambda r:sum(expected(r)['payload']['score_interval'])/2
    points=sorted(midpoint(r) for r in score)
    directions=sorted({r['semantics']['rubric']['direction'] for r in score})
    for i in range(10):
        target=points[min(len(points)-1,int((i+.5)*len(points)/10))]
        desired=directions[i%len(directions)]
        take(sorted(score,key=lambda r:(r['semantics']['rubric']['direction']!=desired,abs(midpoint(r)-target),
                                        digest((PROTOCOL+r['id']).encode()))),1)
    for op in ('Noul','Choice','Score'):
        statuses=('Insufficient','Conflicting','Unresolved')+(('NoFeasibleChoice',) if op=='Choice' else ())
        for status in statuses:take(cell(op,status),1)
    selected.sort(key=lambda r:digest((PROTOCOL+':order:'+r['id']).encode()))
    check(len(selected)==len(families)==32,'32 unique-family training records')
    return [{'id':r['id'],'family':r['family'],'operation':r['operation'],'status':expected(r)['status'],
             'record_sha256':digest(canonical(r))} for r in selected]


def summarize_learning(release,cases,ids,*,kind):
    """Failures stay in denominators; head-only metrics never imply delivery."""
    check(kind in ('heads','behavior'),'learning report kind')
    key='record' if kind=='heads' else 'id'
    check(len(ids)==len(set(ids)) and [c[key] for c in cases]==list(ids),'complete ordered learning report coverage')
    ops={}
    for op in ('Noul','Choice','Score'):
        counts=Counter();confusion=Counter();brier=[];distance=[];raws=[];losses=defaultdict(list)
        for case in cases:
            row=release.record(case[key])
            if row['operation']!=op:continue
            check(case['record_sha256']==digest(canonical(row)),'learning report record drift')
            gold=expected(row);assessed=gold['status']=='Assessed';payload=gold['payload'] or {}
            counts['scheduled']+=1;counts['assessed_scheduled']+=int(assessed)
            failed=bool(case.get('failure'))
            head=case.get('head',{}) if kind=='heads' else {
                'status':case.get('predicted_status'),'value':case.get('predicted_value')}
            actual=case.get('actual') or {}
            prediction=head.get('status') if kind=='heads' else actual.get('status')
            status=not failed and prediction==gold['status']
            confusion[gold['status']+' -> '+str(prediction)]+=1
            counts['failed']+=int(failed);counts['status_correct']+=int(status)
            counts['assessed_status_correct']+=int(assessed and status)
            value=head.get('value');valid=type(value) in (int,float) and math.isfinite(value)
            raw=head.get('raw')
            if type(raw) in (int,float) and math.isfinite(raw):raws.append(raw)
            if op=='Noul' and assessed and valid:counts['assessed_probability_ties']+=int(value==.5)
            if op=='Score' and assessed and type(raw) in (int,float):counts['assessed_clipped_readouts']+=int(raw<0 or raw>1)
            head_ok=False
            if assessed:
                if op=='Noul' and valid:
                    head_ok=value!=.5 and (value>.5)==(payload['p_true']>.5);brier.append((value-payload['p_true'])**2)
                elif op=='Score' and valid:
                    lo,hi=map(fp32,payload['score_interval']);head_ok=lo<=value<=hi
                    distance.append(max(lo-value,0,value-hi))
                elif op=='Choice':
                    selected=head.get('selected') if kind=='heads' else actual.get('selected')
                    counts['assessed_selection_measured']+=int(selected is not None)
                    head_ok=selected in payload['acceptable_selection_ids']
                counts['assessed_head_correct']+=int(not failed and head_ok)
                counts['assessed_head_status_correct']+=int(status and head_ok)
            if kind=='behavior':
                score=grade(row,case);check(score==case['score'],'learning report grade drift')
                counts['behavior_passed']+=int(score['passed']);counts['transport_valid']+=int(score['transport_valid'])
                counts['assessed_behavior_passed']+=int(assessed and score['passed'])
                counts['nonassessed_behavior_passed']+=int(not assessed and score['passed'])
            else:
                counts['head_status_correct']+=int(status)
                counts['complete_head_objective']+=int(case.get('summary',{}).get('terminal')=='Completed' and not failed)
                for name,value_loss in case.get('losses',{}).items():losses[name].append(value_loss)
        ops[op]={'counts':dict(counts),'status_confusion':dict(confusion),
            'noul_brier':{'measured':len(brier),'mean':sum(brier)/len(brier) if brier else None},
            'score_interval_distance':{'measured':len(distance),'mean':sum(distance)/len(distance) if distance else None},
            'raw_readout':{'measured':len(raws),'minimum':min(raws) if raws else None,'maximum':max(raws) if raws else None,
                           'mean':sum(raws)/len(raws) if raws else None},
            'losses':{name:{'eligible':len(v),'mean':sum(v)/len(v)} for name,v in losses.items()}}
    return {'kind':kind,'scheduled':len(ids),'operations':ops,
            'scope':'head objectives without return delivery' if kind=='heads' else 'actual decoded/delivered receiver behavior; not learned invocation',
            'calibration_scope':'reviewed assessed cases with finite head value, regardless of predicted status; missing heads counted separately by measured denominator'}
