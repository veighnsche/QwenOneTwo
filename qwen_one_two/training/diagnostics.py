"""Bounded reference diagnostics and reversible block replacement; no optimizer."""
from .inputs import SLOTS,check,digest

PROTOCOL='reference-block-search-v1'
BLOCKS=('queries','memory','blocks.0','blocks.1','lift')
CONTROL_IDS=tuple(f'train-reference-{i:04d}' for i in (0,10,1,11,8,18))
MASKS=(0,31,*range(1,31))


def block_for(name):
    if not name.startswith('pooler.'):return None
    parts=name.split('.',2)
    check(len(parts)==3 and parts[1] in ('shared','target'),'unknown pooler branch')
    leaf=parts[2]
    matches=[i for i,b in enumerate(BLOCKS) if leaf==b or leaf.startswith(b+'.')]
    check(len(matches)==1,'unknown pooler component')
    return matches[0]


def choose_cases(rows,anchors=CONTROL_IDS):
    """Two unique annotations per role per split, with disjoint source families."""
    rows=tuple(sorted(rows,key=lambda r:r['id']));by_id={r['id']:r for r in rows}
    check(len(by_id)==len(rows) and set(anchors)<=set(by_id),'case identities/anchors')
    unique={};families={}
    for row in rows:
        key=row['annotation_sha256'];unique.setdefault(key,row)
        families.setdefault(key,set()).add(row['family'])
    selected=[];used_hashes=set();used_families=set()
    for split in ('search','screen'):
        current=[]
        def add(row):
            key=row['annotation_sha256']
            check(key not in used_hashes and not families[key]&used_families,'case overlap')
            current.append(row);used_hashes.add(key);used_families.update(families[key])
        if split=='search':
            for key in anchors:add(by_id[key])
        for role in SLOTS:
            need=2-sum(r['role']==role for r in current);check(need>=0,'too many role anchors')
            choices=sorted((r for r in unique.values() if r['role']==role),
                key=lambda r:digest(f'{PROTOCOL}:17:{r["annotation_sha256"]}'.encode()))
            for row in choices:
                if not need:break
                if row['annotation_sha256'] in used_hashes or families[row['annotation_sha256']]&used_families:continue
                add(row);need-=1
            check(need==0,'insufficient disjoint role cases')
        current.sort(key=lambda r:(tuple(SLOTS).index(r['role']),r['id']))
        check(len(current)==20,'case count');selected.append(current)
    return dict(zip(('search','screen'),selected))


def first_difference(a,b):
    check(a and b and a[-1]==b[-1]==248046,'paired target end')
    for i,(x,y) in enumerate(zip(a,b)):
        if x!=y:
            check(x!=248046 and y!=248046,'pair differs only at termination')
            return i
    raise ValueError('identical paired targets')


def ranking(result):
    return (-result['exact'],-result['mean_content_margin'],-result['mean_swap_effect'],
            result['mask'].bit_count(),result['mask'])


def eligible(search,screen,baseline_search,baseline_screen):
    """Strict screening gain with preservation of every baseline recovery."""
    def correct(result):return {r['id'] for r in result['cases'] if r['recovery']['exact']}
    return (correct(baseline_search)<=correct(search) and correct(baseline_screen)<=correct(screen)
            and screen['exact']>baseline_screen['exact'])


def same_bytes(a,b):
    import torch
    return (a.shape==b.shape and a.dtype==b.dtype and
            torch.equal(a.detach().contiguous().reshape(-1).view(torch.uint8),
                        b.detach().contiguous().reshape(-1).view(torch.uint8)))


class BlockBank:
    """Immutable GPU source copies; each application resets all pooler tensors."""
    def __init__(self,baseline,donor):
        import torch
        from ..codec.neural import cuda_only
        check(baseline.keys()==donor.keys(),'source tensor names')
        self.baseline={};self.donor={};self.groups={};self.changed={}
        with torch.no_grad():
            for name,a in baseline.items():
                b=donor[name];cuda_only(a);cuda_only(b)
                check(a.shape==b.shape and a.dtype==b.dtype,'source tensor shape/dtype')
                group=block_for(name)
                check(group is not None or same_bytes(a,b),'decoder/codec differs between sources')
                if group is not None:
                    check(a.dtype==torch.float32,'FP32 pooler required')
                    self.groups[name]=group
                    if not same_bytes(a,b):
                        self.changed[name]=int((a.contiguous().view(torch.uint8).reshape(-1,4)!=
                                                b.contiguous().view(torch.uint8).reshape(-1,4)).any(-1).sum())
                self.baseline[name]=a.detach().clone();self.donor[name]=b.detach().clone()
        check(set(self.groups.values())==set(range(5)),'incomplete component coverage')
        for branch in ('shared','target'):
            check({g for n,g in self.groups.items() if n.startswith('pooler.'+branch+'.')}==set(range(5)),
                  'incomplete branch coverage')

    def apply(self,current,mask):
        import torch
        from ..codec.neural import cuda_only
        check(type(mask) is int and 0<=mask<32,'invalid replacement mask')
        check(current.keys()==self.baseline.keys(),'destination tensor names')
        with torch.no_grad():
            for name,value in current.items():
                cuda_only(value)
                check(value.shape==self.baseline[name].shape and value.dtype==self.baseline[name].dtype,'destination shape/dtype')
                if name not in self.groups:check(same_bytes(value,self.baseline[name]),'nonpooler destination mutation')
            for name,group in self.groups.items():
                current[name].copy_(self.donor[name] if mask&(1<<group) else self.baseline[name])
            changed=[]
            for name,value in current.items():
                source=self.donor if name in self.groups and mask&(1<<self.groups[name]) else self.baseline
                check(same_bytes(value,source[name]),'replacement readback mismatch')
                if name in self.changed and mask&(1<<self.groups[name]):changed.append(name)
        return dict(mask=mask,blocks=[b for i,b in enumerate(BLOCKS) if mask&(1<<i)],
                    changed_tensors=sorted(changed),changed_scalar_values=sum(self.changed[n] for n in changed),
                    unchanged_tensors=len(current)-len(changed),all_other_tensors_byte_identical=True)


def load_resume(path,receipt_path):
    """Load hash-bound completed results and partial recovery records once."""
    import json
    from pathlib import Path
    root=Path(path);receipt=json.loads(Path(receipt_path).read_bytes())
    check(receipt['format']=='qot-reference-block-search-evidence-v1','resume receipt format')
    assets={Path(k).resolve():v for k,v in receipt['assets'].items()}
    def read(name,lines=False):
        file=root/name;body=file.read_bytes()
        check(digest(body)==assets.get(file.resolve()),'resume artifact hash: '+name)
        return [json.loads(line) for line in body.splitlines()] if lines else json.loads(body)
    report=read('report.json');check(report['protocol']==PROTOCOL and report['status']=='INCOMPLETE','resume experiment identity')
    cases=read('cases.json');features=read('feature-identities.json');edits=read('edits.json')
    search={}
    for file in sorted(root.glob('search-*.json')):
        result=read(file.name);mask=result['mask']
        check(mask in MASKS and mask not in search and result['stage']=='search','resume candidate identity')
        check(result['reference_id']==edits[str(mask)]['reference_id'],'resume candidate fingerprint')
        check([c['id'] for c in result['cases']]==[c['id'] for c in cases['splits']['search']],'resume case order')
        check(result['executed']==20 and result['exact']==sum(c['recovery']['exact'] for c in result['cases']),
              'resume complete candidate accounting')
        search[mask]=result
    check(len(search)==report['completed_search_candidates'] and report['completed_screen_candidates']==0,'resume coverage')
    partial={}
    for event in read('events.jsonl',lines=True):
        if event['event']=='recovery' and event['stage']=='search' and event['mask'] not in search:
            part=partial.setdefault(event['mask'],{})
            check(event['id'] not in part and str(event['mask']) in edits,'resume partial identity')
            part[event['id']]=event
    return dict(report=report,cases=cases,features=features,edits=edits,search=search,partial=partial,
                receipt_sha256=digest(Path(receipt_path).read_bytes()),report_sha256=digest((root/'report.json').read_bytes()))


def reused_recovery(row,targets,previous):
    check(all(previous[k]==row[k] for k in ('id','role','family','annotation_sha256')),'reused recovery source identity')
    check(previous['recovery']['target_tokens']==len(targets),'reused recovery target identity')
    return previous['recovery']


class Evaluator:
    """One loaded frozen model, cached W features, fresh decoder state per episode."""
    def __init__(self,reference,prepared,emit):
        import torch
        self.reference=reference;self.prepared=prepared;self.emit=emit
        self.positions=0;self.sensitivity_calls=0;self.recovery_calls=0
        self.reused_recoveries=0;self.reused_positions=0
        b=reference.backbone
        check(b.compensated and b.precision_parts==2 and b.feedback_precision=='fp32' and
              b.recurrence_backend=='graph-v1','v3 graph-v1 required')
        self.indices=torch.cat((torch.arange(248044,device='cuda'),torch.tensor([248046],device='cuda')))
        self.vocabulary=b.model.model.embed_tokens.weight[self.indices]

    def measure(self,mask,stage,rows,*,recoveries=None):
        import time
        import torch
        from .binding import wire_payload
        from ..g1_precision_math import compensated_linear
        ref=self.reference;before=ref.fingerprint();started=time.monotonic();cases=[]
        check(ref.frozen and before==ref.artifact_id and not torch.is_grad_enabled()
              and all(not p.requires_grad for p in ref.parameters()),'frozen no-grad evaluator required')
        start_positions=self.positions;start_reused=self.reused_positions;start_reused_cases=self.reused_recoveries
        recoveries={} if recoveries is None else recoveries
        check(set(recoveries)<={r['id'] for r in rows},'foreign reused recovery case')
        for role in SLOTS:
            pair=[r for r in rows if r['role']==role];check(len(pair)==2,'role pair')
            prepared=[self.prepared[r['id']] for r in pair]
            index=first_difference(prepared[0]['targets'],prepared[1]['targets'])
            tokens=[p['targets'][index] for p in prepared];observations=[];pair_cases=[]
            check(all(0<=t<248044 for t in tokens),'ordinary diagnostic token required')
            for side,(row,p) in enumerate(zip(pair,prepared)):
                payload=ref.pooler(p['states'],*SLOTS[role]);prefix=wire_payload(ref,payload,role)
                # Only the identical preceding token sequence is supplied at this diagnostic position.
                common=ref.backbone.embedding(p['targets'][:index])[0] if index else prefix.new_empty((0,1024))
                inputs=torch.cat((prefix,common));hidden,_=ref(ref.backbone.initial(),'T',inputs)
                scores=compensated_linear(hidden[0,-1],self.vocabulary)
                check(bool(scores.isfinite().all()),'nonfinite diagnostic logits')
                lp=torch.log_softmax(scores.float(),dim=-1)
                selected=lp[torch.tensor(tokens,device='cuda')];observations.append(selected)
                self.positions+=len(inputs);self.sensitivity_calls+=1
                sensitivity=dict(first_difference=index,own_token=tokens[side],foil_token=tokens[1-side],
                    own_token_text=ref.backbone.tokenizer.decode([tokens[side]]),
                    predicted_token=int(self.indices[scores.argmax()]),own_log_probability=float(selected[side]),
                    foil_log_probability=float(selected[1-side]),
                    content_margin=float(selected[side]-selected[1-side]))
                self.emit(dict(event='sensitivity',mask=mask,stage=stage,id=row['id'],**sensitivity))
                del scores,lp,hidden,inputs,common
                reuse=row['id'] in recoveries
                if reuse:
                    recovery=reused_recovery(row,p['targets'],recoveries[row['id']]);self.reused_recoveries+=1
                    self.reused_positions+=len(prefix)+min(recovery['matched_tokens'],len(p['targets'])-1)
                else:
                    recovery=ref.exact_recovery(prefix,p['targets']);self.recovery_calls+=1
                    self.positions+=len(prefix)+min(recovery['matched_tokens'],len(p['targets'])-1)
                case=dict(id=row['id'],role=role,family=row['family'],annotation_sha256=row['annotation_sha256'],
                          recovery=recovery,**sensitivity)
                self.emit(dict(event='recovery_reused' if reuse else 'recovery',mask=mask,stage=stage,**case));pair_cases.append(case)
            for side,case in enumerate(pair_cases):
                case['swap_effect']=float(observations[side][side]-observations[1-side][side]);cases.append(case)
        check(ref.fingerprint()==before,'diagnostics mutated reference')
        result=dict(mask=mask,stage=stage,reference_id=before,cases=cases,executed=len(cases),
            exact=sum(c['recovery']['exact'] for c in cases),
            mean_content_margin=sum(c['content_margin'] for c in cases)/len(cases),
            mean_swap_effect=sum(c['swap_effect'] for c in cases)/len(cases),
            positions=self.positions-start_positions+self.reused_positions-start_reused,
            executed_positions=self.positions-start_positions,reused_recovery_cases=self.reused_recoveries-start_reused_cases,
            seconds=time.monotonic()-started,timing_scope='this invocation; reused recoveries exclude historical time')
        self.emit(dict(event='candidate_finished',**{k:v for k,v in result.items() if k!='cases'}))
        return result
