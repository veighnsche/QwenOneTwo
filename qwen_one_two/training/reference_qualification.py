"""Historical exact-reconstruction diagnostics and semantic evidence handle.

Recovery-v1 reports remain diagnostics; only behavior-v1 can supply acceptance.
"""
from dataclasses import dataclass
from pathlib import Path
from .inputs import canonical,check,digest,SLOTS
from .release import decode,sha256,ReviewedRelease

PROTOCOL='reference-recovery-v1'
PROFILE='b1-full-v3-feedback-fp32-vocab-compensated2-temporal32-recurrence-graph-v1'
_TOKEN=object()


def suite(release):
    check(isinstance(release,ReviewedRelease) and release.manifest['purpose']=='development',
          'reference qualification requires reviewed development data')
    records=release.records
    ids=tuple(e['id'] for e in release.manifest['records'] if e['selected']
              and records[e['id']].get('evaluation_slice')=='RF')
    check(len(ids)==100,'all 100 scheduled reference recovery cases required')
    check({records[key]['reference_role'] for key in ids}==set(SLOTS),'reference role coverage')
    return ids


def relations(release):
    """Only explicitly supplied, release-approved pairs; no batch negatives."""
    descriptor=release.manifest.get('reference_contrasts')
    if descriptor is None:return intervention_relations(release)
    body=release.asset(descriptor['path'])
    check(digest(body)==sha256(descriptor['sha256']),'contrast asset hash')
    groups=decode(body);records=release.records
    ids={e['id'] for e in release.manifest['records'] if e['selected']}
    for group in groups:
        check(set(group)=={'id','role','anchor','positives','negatives'},'contrast group fields')
        a,p,n=group['anchor'],group['positives'],group['negatives']
        check(p and n and a in ids and set(p+n)<=ids and not set(p)&set(n)
              and a not in p+n and len(set(p+n))==len(p+n),'contrast membership')
        check(group['role'] in SLOTS and all(contrast_fields(records[k],group['role']) is not None for k in [a]+p+n),'cross-role contrast')
    check(len({g['id'] for g in groups})==len(groups),'duplicate contrast group')
    return tuple(groups)


def contrast_fields(row,role):
    if row.get('reference_role') is not None:
        return row['model_inputs']['meaning'] if row['reference_role']==role else None
    meanings=row.get('semantics',{})
    return meanings.get(role) if role in {'question','target'} else meanings.get('rubric',{}).get(role)


def intervention_relations(release):
    """Reuse explicit controlled-v1 reviewed interventions, not guessed labels.

    Category 8's zero-anchor wording is a paraphrase, so is deliberately absent
    from the meaning-changing roles. Positive annotations must be byte-identical
    in another reviewed record from the same family. Missing coverage rejects.
    """
    changes={1:('question',),2:('question',),
             7:('question','dimension','conditions','zero','one'),8:('question','conditions','one')}
    records=release.records
    available=[records[e['id']] for e in release.manifest['records'] if e['selected']]
    rows=[row for row in available if row.get('evaluation_slice')=='X']
    pairs={}
    for row in rows:
        info=row.get('intervention',{})
        if info.get('category') not in changes:continue
        check(row['construction'].get('template_version')=='controlled-data-v1'
              and info.get('protocol')=='offline_semantic_intervention','unknown contrast transformation')
        pair=pairs.setdefault(info['pair'],{})
        check(info['side'] in (0,1) and info['side'] not in pair,'contrast side duplication')
        pair[info['side']]=row
    groups=[]
    for key,pair in sorted(pairs.items()):
        check(set(pair)=={0,1},'incomplete reviewed contrast pair')
        a,n=pair[0],pair[1]
        category=a['intervention']['category']
        check(n['intervention']['category']==category and a['family']==n['family'],'contrast family/category')
        for role in changes[category]:
            original,changed=contrast_fields(a,role),contrast_fields(n,role)
            check(original is not None and changed is not None and canonical(original)!=canonical(changed),
                  'missing declared meaning-changing field')
            positives=[r['id'] for r in available if r['family']==a['family'] and r['id'] not in (a['id'],n['id'])
                       and contrast_fields(r,role) is not None and canonical(contrast_fields(r,role))==canonical(original)]
            check(bool(positives),'reviewed equivalent positive missing')
            groups.append({'id':key+'-'+role,'role':role,'anchor':a['id'],
                           'positives':[sorted(positives)[0]],'negatives':[n['id']]})
    return tuple(groups)


def evaluate_contrasts(reference,release):
    import torch
    from .binding import annotation,encode_annotation
    from ..codec.neural import norm
    check(reference.frozen and reference.artifact_id==reference.fingerprint(),'frozen unchanged reference required')
    check(reference.backbone.compensated and reference.backbone.precision_parts==2
          and reference.backbone.feedback_precision=='fp32'
          and reference.backbone.recurrence_backend=='graph-v1','declared v3/graph-v1 arithmetic required')
    records=release.records;payloads={};scored=[]
    before=reference.fingerprint()
    with torch.no_grad():
        def payload(key,role):
            row=records[key];item=annotation(role,contrast_fields(row,role),row['operation'])
            if item.sha256 not in payloads:payloads[item.sha256]=encode_annotation(reference,item)
            return payloads[item.sha256]
        for group in relations(release):
            role=group['role']
            anchor=torch.nn.functional.normalize(norm(payload(group['anchor'],role)).flatten(),dim=0)
            def similarities(keys):
                return [float((anchor*torch.nn.functional.normalize(norm(payload(k,role)).flatten(),dim=0)).sum()) for k in keys]
            positive,negative=similarities(group['positives']),similarities(group['negatives'])
            scored.append(dict(group,positive=positive,negative=negative,passed=min(positive)>max(negative)))
    check(reference.fingerprint()==before,'reference changed during contrast evaluation')
    return scored


def summarize(cases,groups,*,complete,unchanged):
    recovered=sum(c['recovery']['exact'] is True for c in cases)
    contrast=sum(g['passed'] is True for g in groups)
    blockers=[]
    if not complete or len(cases)!=100:blockers.append('incomplete scheduled recovery coverage')
    if recovered<95:blockers.append('fewer than 95/100 exact reviewed recoveries; nonexact outputs unresolved')
    if not groups:blockers.append('no explicit reviewed reference contrast groups in existing release')
    elif contrast/len(groups)<.90:blockers.append('applicable contrast success below 90%')
    if not unchanged:blockers.append('reference changed during qualification')
    return {'status':'BLOCKED' if blockers else 'PASSED','blockers':blockers,
            'recovery':{'scheduled':100,'executed':len(cases),'exact':recovered,'unresolved':100-recovered},
            'contrast':{'scheduled':len(groups),'passed':contrast}}


def evaluate(reference,release):
    """Run the entire existing RF slice, independently of current task outcomes."""
    import torch
    from .component_producer import reference_record
    from .binding import encode_annotation,wire_payload
    from ..codec.neural import cuda_only
    cuda_only()
    check(reference.frozen and all(not p.requires_grad for p in reference.parameters()),'frozen reference required')
    check(reference.backbone.compensated and reference.backbone.precision_parts==2
          and reference.backbone.feedback_precision=='fp32'
          and reference.backbone.recurrence_backend=='graph-v1','declared v3/graph-v1 arithmetic required')
    before=reference.fingerprint();check(before==reference.artifact_id,'reference identity drift')
    ids=suite(release);cases=[]
    with torch.no_grad():
        for key in ids:
            row=release.record(key);annotation,_=reference_record(release,row)
            payload=encode_annotation(reference,annotation)
            target=reference.backbone.tokenizer.encode(annotation.content.decode(),add_special_tokens=False)+[248046]
            recovery=reference.exact_recovery(wire_payload(reference,payload,annotation.role),target)
            cases.append({'id':key,'role':annotation.role,'annotation_sha256':annotation.sha256,'recovery':recovery})
        scored=evaluate_contrasts(reference,release)
    after=reference.fingerprint()
    result=summarize(cases,scored,complete=len(cases)==100,unchanged=before==after)
    return dict(result,protocol=PROTOCOL,profile=PROFILE,reference_sha256=before,reference_after_sha256=after,development_release_sha256=release.identity,
                gpu_uuid='GPU-225509c3-d702-6c27-de03-76b5e99b33b1',cases=cases,groups=scored,
                optimizer_updates=0,source_hashes=source_hashes(),
                diagnostic_untrained=False,
                scope='exact reviewed canonical recovery is sufficient; paraphrases unresolved; no full-flow certification')


def source_hashes():
    root=Path(__file__).resolve().parents[2]
    names=('training/reference.py','training/reference_qualification.py','training/binding.py',
           'training/inputs.py','training/clean.py','training/component_producer.py',
           'codec/neural.py','codec/precision.py','codec/backbone.py','codec/recurrence.py','g1_precision_math.py','g1_backbone.py')
    return {name:digest((root/'qwen_one_two'/name).read_bytes()) for name in names}


@dataclass(frozen=True,init=False)
class ReferenceEvidence:
    identity:str
    reference_sha256:str
    development_release_sha256:str
    receiver_sha256:str

    def __init__(self,identity,reference,development,receiver=None,*,_token=None):
        check(_token is _TOKEN,'load measured reference qualification first')
        for name,value in (('identity',identity),('reference_sha256',reference),('development_release_sha256',development)):
            object.__setattr__(self,name,value)
        object.__setattr__(self,'receiver_sha256',receiver)

    def assert_reference(self,reference):
        check(reference.frozen and all(not p.requires_grad for p in reference.parameters())
              and reference.artifact_id==reference.fingerprint()==self.reference_sha256,'qualified reference changed')

    def assert_receiver(self,backbone,codec):
        from .behavior_qualification import receiver_hash
        check(receiver_hash(backbone,codec)==self.receiver_sha256,'accepted receiver changed; re-evaluate behavior')


def load_reference_evidence(path,release,*,expected_hash):
    """Exact reconstruction never grants semantic/behavioral acceptance."""
    from .behavior_qualification import load_behavior_evidence
    return load_behavior_evidence(path,release,expected_hash=expected_hash)
