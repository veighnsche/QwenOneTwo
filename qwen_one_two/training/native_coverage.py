"""Full existing-data native/reader supervision inventory; metadata only."""
from collections import Counter,defaultdict
from .inputs import canonical,digest,check
from .binding import annotation
from .clean import tokenize_excerpt
from .native import target_layout
from .producer import native_plan


def audit_record(release,row,original,tokenizer):
    check(original['record_sha256']==digest(canonical(row)) and original['release_sha256']==release.identity,
          'stale native inventory row')
    category=original['native']['category']
    item={'id':row['id'],'release':release.identity,'record_sha256':digest(canonical(row)),
          'family':row['family'],'split':row['split'],'category':category,'operation':row['operation'],
          'calls':[],'issues':[],'native_continuation_eligible':False,'actual_execution':False}
    if category=='reader':
        bound=row['model_inputs'].get('bound_return');continuations=row['supervision'].get('continuations',[])
        unique=bound is not None and (row['operation']!='Choice' or bound['status']!='Assessed'
                                    or len((bound.get('payload') or {}).get('acceptable_selection_ids',[]))==1)
        item.update(legacy_supplied_reader_metadata_eligible=bool(bound and continuations and unique),
            direct_supplied_reader_eligible=False,exposure='supplied_return_reader',
            issues=['direct supplied-reader encoder unsupported','supplied return is not actual native return supervision'])
        if bound and bound['status']!='Assessed':item['issues'].append('new-profile Reason semantics unsupported')
        if not unique:item['issues'].append('set-valued Choice continuation has no unique bound return')
        item['supplied_return_sha256']=None if bound is None else digest(canonical(bound))
        return item
    item['exposure']='actual_native'
    plan=native_plan(release,row,'guided_joint' if release.purpose=='training' else 'free_joint',1,
                     evaluation=release.purpose=='development')
    item['prompt_sha256']=digest(plan.prompt.encode());source_tokens=None
    try:
        ids,selected,offsets=tokenize_excerpt(tokenizer,plan.prompt.encode());source_tokens=len(selected)
        item.update(prompt_tokens=len(ids),source_tokens=source_tokens,source_mapping_sha256=digest(canonical([ids,selected,offsets])))
        if len(ids)+6>=1024:item['issues'].append('native prompt exceeds complete-episode ceiling')
        if source_tokens>256:item['issues'].append('native full source exceeds eight spans')
    except ValueError as exc:item['issues'].append('native source mapping: '+str(exc))
    calls=row.get('trajectory',{}).get('calls',[]) if category=='interleaved' else ([
        {'operation':row['operation'],'request_target':row['supervision']['request_semantics'],
         'result_target':row['supervision']['result']}] if category=='writer' else [])
    for i,call in enumerate(calls):
        old=original['native']['calls'][i];meaning=call['request_target'];op=call['operation']
        check(digest(canonical(meaning))==old['request_sha256'],'native request metadata drift')
        fields=[('question',meaning['question']),('target',meaning['target'])]
        fields += [('alternative',value) for value in meaning.get('alternatives',[])]
        rubric=meaning.get('rubric',{});fields += [(name,rubric[name]) for name in ('dimension','conditions','zero','one','midpoint') if name in rubric]
        refs=tuple(annotation(role,value,op) for role,value in fields)
        entry={'index':i,'operation':op,'status':call['result_target']['status'],'fields':[],'issues':[],
            'reviewed_gaps':old['gaps'],'choice_target_unique':old['choice_target_unique'],
            'formation_eligible':False,'decision_eligible':False,'continuation_eligible':False,
            'binding_blocker':'arbitrary native writer meanings unverified; alignment is not semantic admission'}
        lengths={}
        for ordinal,ref in enumerate(refs):
            try:
                tokens,selected,offsets=tokenize_excerpt(tokenizer,ref.content);n=len(selected)
                lengths.setdefault(ref.role,[]).append(n)
                entry['fields'].append({'index':ordinal,'role':ref.role,'input_sha256':ref.sha256,'length':n,
                    'mapping_sha256':digest(canonical([tokens,selected,offsets]))})
                if n>32:entry['issues'].append('field_overlength/'+ref.role)
            except ValueError as exc:entry['issues'].append('field mapping: '+str(exc))
        if source_tokens is not None and len(entry['fields'])==len(refs):
            echo=2+lengths['question'][0]+lengths['target'][0]
            if op=='Score':echo+=1+sum(lengths.get(name,[0])[0] for name in ('dimension','conditions','zero','one','midpoint'))
            candidates=[1+n for n in lengths.get('alternative',[])]
            costs={'request':echo+source_tokens+(source_tokens+31)//32+sum(candidates)+1,
                   'return_max':max(echo+(max(candidates) if candidates else 1)+1,echo+38)}
            entry['observed_costs']=costs
            if costs['request']>256:entry['issues'].append('request_overlength')
            if costs['return_max']>64:entry['issues'].append('return_overlength')
        try:
            target_layout(refs,op,source_tokens,tokenizer)
            entry['formation_eligible']=not item['issues'] and not entry['issues']
        except ValueError as exc:entry['issues'].append('target admission: '+str(exc))
        item['calls'].append(entry)
    if calls:item['issues'].append('native decision and post-call continuation lack semantic binding')
    item['reviewed_continuation_present']=bool(row['supervision'].get('continuations'))
    item['direct_no_call_continuation_potential']=category in ('direct','language_replay') and item['reviewed_continuation_present']
    return item


def summarize(records):
    groups={}
    for split,category in sorted({(r['split'],r['category']) for r in records}):
        rows=[r for r in records if r['split']==split and r['category']==category]
        calls=[c for r in rows for c in r['calls']];by_op={}
        for op in ('Noul','Choice','Score'):
            selected=[c for c in calls if c['operation']==op]
            by_op[op]={'calls':len(selected),'formation_eligible':sum(c['formation_eligible'] for c in selected),
                'decision_eligible':0,'native_continuation_eligible':0,
                'exclusions':dict(Counter(issue for c in selected for issue in set(c['issues']+c['reviewed_gaps']+[c['binding_blocker']])))}
        groups[split+'/'+category]={'records':len(rows),'calls':len(calls),'operations':by_op,
            'formation_eligible':sum(c['formation_eligible'] for c in calls),'native_continuation_eligible':0,
            'legacy_supplied_reader_metadata_eligible':sum(r.get('legacy_supplied_reader_metadata_eligible',False) for r in rows),
            'direct_supplied_reader_eligible':0,'direct_no_call_continuation_potential':sum(r.get('direct_no_call_continuation_potential',False) for r in rows),
            'issues':dict(Counter(issue for r in rows for issue in set(r['issues']))) }
    return groups


def evaluation_manifest(development,inventory):
    rows={r['id']:r for r in inventory};groups={};cases=[]
    for key in sorted(development.selected_ids):
        row=development.record(key);slice_=row.get('evaluation_slice');info=row.get('intervention',{})
        if slice_ not in ('W','F','R') and slice_!='X':continue
        gold=row['supervision'].get('result') or row['labels']
        exposure='supplied_return_reader' if slice_=='R' or info.get('category')==9 else 'actual_native' if slice_ in ('W','F') else 'prepared_receiver_pair'
        cases.append({'id':key,'record_sha256':digest(canonical(row)),'family':row['family'],'exposure':exposure,
            'operation':row['operation'],'source_assets':row['sources'],'expected_status':gold['status'],
            'expected_final_value_or_identity':gold.get('payload'),
            'native_final_observer':'must read generated native answer; receiver heads or consumed return alone are insufficient',
            'continuation_sha256':[digest(x.encode()) for x in row['supervision'].get('continuations',[])],
            'structured_native_final_extraction':'not provided by existing prose-only continuations; no invented parser/judge'})
        if info:
            group=groups.setdefault(info['pair'],{'category':info['category'],'cases':[],'exposure':exposure})
            group['cases'].append(key)
    check(all(len(g['cases'])==2 for g in groups.values()),'complete existing input-change pairs required')
    return {'protocol':'native-evaluation-plan-v1','cases':cases,'change_pairs':groups,
        'scope':'planned held-out native W/F, supplied-reader R/X9 and prepared receiver X pairs kept separate; not executed',
        'misleading_withheld_reader':'existing R/X9 supplied-return views only; no claimed actual-return intervention labels'}
