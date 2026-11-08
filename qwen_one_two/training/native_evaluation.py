"""Typed native-final observations, separate from heads and supplied readers.

No prose parser or new semantic judge is invented. Missing a verified typed
observation of generated native text is unsupported, never successful delivery.
"""
import math
from .inputs import canonical,digest,check
from .behavior_qualification import fp32


def grade_final(row,case):
    check(case['record_sha256']==digest(canonical(row)),'native final record drift')
    if case.get('exposure')!='actual_native':return {'supported':False,'passed':False,'reason':'component/reader exposure is not native-final evidence'}
    final=case.get('final_observation')
    if not final or final.get('origin')!='generated_native_answer':
        return {'supported':False,'passed':False,'reason':'no typed observation of generated native answer; heads/returns cannot substitute'}
    # This grades the supplied typed observation only. The evidence provider
    # must establish its connection to actual generated tokens separately.
    gold=row['supervision'].get('result') or row['labels'];payload=gold.get('payload') or {}
    op=row['operation'];passed=False
    if final.get('status')==gold['status']:
        if gold['status']!='Assessed':passed=final.get('value') is None and final.get('identity') is None
        elif op=='Choice':passed=final.get('value') is None and final.get('identity') in payload['acceptable_selection_ids']
        else:
            value=final.get('value')
            valid=final.get('identity') is None and type(value) in (int,float) and math.isfinite(value) and 0<=value<=1
            if valid and op=='Noul':passed=value!=.5 and (value>.5)==(payload['p_true']>.5)
            if valid and op=='Score':passed=fp32(payload['score_interval'][0])<=value<=fp32(payload['score_interval'][1])
    transport=case.get('actual_return_consumed') is True and case.get('one_prefill') is True and case.get('one_lineage') is True
    return {'supported':True,'passed':bool(passed and transport and not case.get('failure')),
            'scope':'typed final value/identity only; extraction provenance and semantic explanation are separate'}


def grade_change_pairs(release,cases,manifest):
    by_id={c['id']:c for c in cases};results=[]
    check(len(by_id)==len(cases),'duplicate change-pair observations')
    for key,group in sorted(manifest['change_pairs'].items()):
        pair=[by_id.get(k) for k in group['cases']]
        if group['exposure']!='actual_native' or any(c is None for c in pair):
            results.append({'pair':key,'supported':False,'passed':False,'reason':'missing native observations or component/reader-only pair'})
        else:
            grades=[grade_final(release.record(c['id']),c) for c in pair]
            results.append({'pair':key,'supported':all(g['supported'] for g in grades),'passed':all(g['passed'] for g in grades)})
    return results
