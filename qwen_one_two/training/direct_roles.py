"""Direct-role token planning and full-panel coverage. No model library imports."""
import json
from collections import Counter

from ..codec.layout import DIRECT,GPU_CHECKS,packet_costs,admit,field_limit
from ..codec.schema import Port
from .clean import tokenize_excerpt
from .inputs import check,digest,canonical

ROLE_POLICY='frozen-reference-pooled-v1'
REQUEST_ROLES={'question','target','alternative','dimension','conditions','zero','one','midpoint','key'}
RUBRIC_NAMES=('dimension','conditions','zero','one','midpoint')


def plan_role(tokenizer,ref,*,profile=DIRECT):
    check(ref.role in REQUEST_ROLES,'direct annotation role unsupported; Reason remains a return control')
    ids,selected,offsets=tokenize_excerpt(tokenizer,ref.content)
    check(1<=len(selected)<=field_limit(profile),f'direct field exceeds {field_limit(profile)} content states; never crop')
    check(len(ids)<=512,'direct annotation exceeds source encoding envelope')
    return {'role':ref.role,'input_sha256':ref.sha256,'ids':ids,'selected':selected,'offsets':offsets}


def preflight(tokenizer,view,provenance,*,profile=DIRECT):
    check(view.purpose=='receiver','direct profile requires receiver view; supplied-reader support is separate')
    excerpts,body=view.materialize(provenance)
    roles=[plan_role(tokenizer,ref,profile=profile) for ref in view.references]
    sources=[tokenize_excerpt(tokenizer,excerpt) for excerpt in excerpts]
    check(all(len(ids)<=512 for ids,_,_ in sources),'direct source exceeds encoding envelope')
    lengths={}
    for item in roles:lengths.setdefault(item['role'],[]).append(len(item['selected']))
    check(len(lengths.get('question',[]))==len(lengths.get('target',[]))==1,'one direct question/target required')
    rubric=tuple(lengths.get(name,[0])[0] for name in RUBRIC_NAMES) if view.operation=='Score' else ()
    evidence=tuple(min(32,len(selected)-i)+1 for _,selected,_ in sources for i in range(0,len(selected),32))
    port={'Noul':Port.NOUL,'Choice':Port.CHOICE,'Score':Port.SCORE}[view.operation]
    costs=admit(packet_costs(port,lengths['question'][0],lengths['target'][0],rubric=rubric,
        candidates=tuple((n,0) for n in lengths.get('alternative',[])),evidence=evidence,profile=profile))
    return {'profile':profile,'input_sha256':digest(body),'roles':roles,'sources':sources,'costs':costs}


def coverage(release,ids,tokenizer,*,intervention=False,profile=DIRECT):
    from .component_producer import component_view
    records=[]
    for key in ids:
        row=release.record(key);view,p,_=component_view(release,row,intervention=intervention)
        entry={'id':key,'family':row['family'],'operation':row['operation'],
               'record_sha256':digest(canonical(row)),'release_sha256':release.identity,'issues':[],'fields':[]}
        for ordinal,ref in enumerate(view.references):
            try:
                token_ids,selected,offsets=tokenize_excerpt(tokenizer,ref.content)
                entry['fields'].append({'ordinal':ordinal,'role':ref.role,'input_sha256':ref.sha256,'length':len(selected),
                                        'mapping_sha256':digest(canonical([token_ids,selected,offsets]))})
                if len(selected)>field_limit(profile):entry['issues'].append('field_overlength/'+ref.role)
                if len(token_ids)>512:entry['issues'].append('annotation_encoding_overlength/'+ref.role)
            except ValueError as error:entry['issues'].append('field_mapping/'+ref.role+': '+str(error))
        lengths={}
        for field in entry['fields']:lengths.setdefault(field['role'],[]).append(field['length'])
        try:
            excerpts,_=view.materialize(p)
            source_sizes=[len(tokenize_excerpt(tokenizer,text)[1]) for text in excerpts]
            q=lengths['question'][0];t=lengths['target'][0]
            rubric=1+sum(lengths.get(role,[0])[0] for role in RUBRIC_NAMES) if row['operation']=='Score' else 0
            echo=2+q+t+rubric
            candidate_costs=[1+n for n in lengths.get('alternative',[])]
            observed={'request':echo+sum(n+(n+31)//32 for n in source_sizes)+sum(candidate_costs)+1,
                'assessed_return':echo+(max(candidate_costs) if candidate_costs else 1)+1,
                'nonassessed_return':echo+38}
            observed['return_max']=max(observed['assessed_return'],observed['nonassessed_return'])
            if profile!=DIRECT:observed.update(assessed_return=3,nonassessed_return=39,return_max=39)
            entry['observed_costs']=observed
            if observed['request']>256:entry['issues'].append('request_overlength')
            if observed['return_max']>64:entry['issues'].append('return_overlength')
        except (ValueError,KeyError,UnicodeError) as error:entry['issues'].append('cost_mapping: '+str(error))
        try:
            planned=preflight(tokenizer,view,p,profile=profile)
            entry['costs']=planned['costs']
        except (ValueError,UnicodeError) as error:entry['issues'].append('admission: '+str(error))
        entry['supported']=not entry['issues'];records.append(entry)
    return {'records':records,'summary':{'total':len(records),'supported':sum(r['supported'] for r in records),
        'excluded':sum(not r['supported'] for r in records),'issues':dict(Counter(i for r in records for i in set(r['issues']))),
        'operation_support':{op:{'total':sum(r['operation']==op for r in records),
                                'supported':sum(r['operation']==op and r['supported'] for r in records)} for op in ('Noul','Choice','Score')},
        'families_excluded':len({r['family'] for r in records if not r['supported']})}}
