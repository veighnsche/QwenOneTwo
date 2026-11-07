"""CPU-only asset lookup, input preparation and saved-audit serialization A/B."""
import argparse
import ast
import cProfile
import dataclasses
import importlib.util
import json
import os
from pathlib import Path
import pstats
import statistics
import sys
import tempfile
import time
from scripts.benchmark_manifest import BLOCKED,NoModels,RELEASES

BASELINE='68cf7d1edf1f834053815ec46a28b11633e4b1b816eda7703b005305b9ad3bb6'


def main():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    assert not BLOCKED & set(sys.modules)
    sys.meta_path.insert(0,NoModels())
    from qwen_one_two.training.inputs import check,canonical,digest
    from qwen_one_two.training.release import load_release
    from qwen_one_two.training.component_producer import component_view,reference_record
    from qwen_one_two.training.producer import native_plan
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline',required=True)
    p.add_argument('--profile-before',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    sha=lambda path:digest(Path(path).read_bytes())
    check(sha(a.baseline)==BASELINE,'literal baseline source')
    spec=importlib.util.spec_from_file_location('qwen_one_two.training._asset_legacy',a.baseline)
    old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
    before=json.loads(Path(a.profile_before).read_bytes())
    check(before['release_source_sha256']==BASELINE,'profile used another baseline')
    prior_path=Path('docs/evidence/manifest-access-v1.json');prior=json.loads(prior_path.read_bytes())
    for name,value in prior['sources'].items():
        if name!='qwen_one_two/training/release.py':check(sha(name)==value,'prior input construction changed')
    def validation(path):
        tree=ast.parse(Path(path).read_text())
        return ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='load_release'),include_attributes=False)
    check(validation(a.baseline)==validation('qwen_one_two/training/release.py'),'loader validation changed')
    def normalized(x):
        if dataclasses.is_dataclass(x):return normalized(dataclasses.asdict(x))
        if isinstance(x,bytes):return {'bytes_sha256':digest(x),'length':len(x)}
        if isinstance(x,dict):return {k:normalized(v) for k,v in x.items()}
        if isinstance(x,(set,frozenset)):return sorted(normalized(v) for v in x)
        if isinstance(x,(tuple,list)):return [normalized(v) for v in x]
        return x
    def prepared(release,row,**kw):
        view,prov,indices=component_view(release,row,**kw)
        return normalized((view,prov.nodes,prov.sources,indices,view.materialize(prov)))
    pairs={};snapshots={};parity=[]
    for split,identity in RELEASES.items():
        root=Path('data/training-collections-20261105/releases')/split
        new=load_release(root,expected_hash=identity);previous=old.load_release(root,expected_hash=identity)
        check(new._manifest==previous._manifest and new._records==previous._records and
              new._assets==previous._assets and new._summary==previous._summary,'snapshot byte drift')
        check(dict(new._asset_index)==dict(previous._assets),'asset index drift')
        for name,body in previous._assets:check(new.asset(name)==body,'asset bytes drift')
        pairs[split]=(new,previous)
        snapshots[split]={'identity':identity,'assets':len(new._assets),'records':len(new._records),
                         'all_asset_bytes_equal':True,'all_snapshot_bytes_equal':True}
        for case in [v for v in prior['cases'] if v['split']==split]:
            row=new.record(case['record'])
            if case['kind']=='component':
                kw={'intervention':True} if row.get('evaluation_slice') in {'E','X'} else {}
                result=prepared(new,row,**kw)
            elif case['kind']=='reference':
                item,prov=reference_record(new,row);result=normalized((item,prov.nodes,prov.sources))
            else:result=normalized(native_plan(new,row,'guided_joint',1))
            check(digest(canonical(result))==case['matched_sha256'],'prior input/provenance mismatch')
            parity.append(case)
    log=Path('data/receiver-learning-20261106/continued/training.jsonl')
    check(sha(log)==before['training_log_sha256'],'saved audit log drift')
    entries=[json.loads(line) for line in log.read_text().splitlines()]
    entries=[v for v in entries if str(v['window']['window']) in before['cases']]
    new,previous=pairs['train'];measurements={};repeats=25
    def profile(fn,name):
        pr=cProfile.Profile();pr.enable()
        for _ in range(repeats):fn()
        pr.disable();pr.dump_stats(out/(name+'.pstats'))
        return [{'file':f,'line':line,'function':n,'calls':nc,'self_seconds':tt,'cumulative_seconds':ct}
            for (f,line,n),(cc,nc,tt,ct,_) in sorted(pstats.Stats(pr).stats.items(),key=lambda v:v[1][3],reverse=True)[:20]]
    for entry in entries:
        batch=str(entry['window']['window']);rows=[new.record(v['record']) for v in entry['episodes']]
        paths=[s['path'] for row in rows for s in row['sources']]
        for row in rows:check(prepared(new,row)==prepared(previous,row),'direct paired input/provenance mismatch')
        def prepare(release):
            for row in rows:
                view,prov,_=component_view(release,row);view.materialize(prov)
        def lookup(release):
            for path in paths:release.asset(path)
        samples=[]
        for name,fn in [('prepare_eight',prepare),('asset_lookups',lookup)]:
            for release in (previous,new):fn(release)
            for repeat in range(5):
                for label,release in ((('legacy',previous),('current',new)) if repeat%2==0 else (('current',new),('legacy',previous))):
                    start=time.perf_counter()
                    for _ in range(repeats):fn(release)
                    samples.append({'operation':name,'pair':repeat,'path':label,
                                    'seconds_per_eight':(time.perf_counter()-start)/repeats})
        audit=(json.dumps(entry)+'\n').encode()
        check(digest(audit)==before['cases'][batch]['audit_sha256'],'audit JSON bytes changed')
        serialization=[]
        with tempfile.TemporaryFile(mode='w+',encoding='utf-8') as stream:
            def serialize():return json.dumps(entry)+'\n'
            def serialize_flush():stream.write(serialize());stream.flush()
            for name,fn in [('audit_json',serialize),('audit_json_write_flush',serialize_flush)]:
                fn()
                for repeat in range(25):
                    start=time.perf_counter();fn();serialization.append({'operation':name,'seconds':time.perf_counter()-start})
            audit_profile=profile(serialize_flush,'audit-'+batch)
        med={name:{label:statistics.median(v['seconds_per_eight'] for v in samples if v['operation']==name and v['path']==label)
                   for label in ('legacy','current')} for name in ('prepare_eight','asset_lookups')}
        measurements[batch]={'records':[r['id'] for r in rows],'asset_calls':len(paths),'samples':samples,'medians':med,
            'serialization':serialization,'audit_sha256':digest(audit),'audit_bytes':len(audit),
            'profile_after':profile(lambda:prepare(new),'prepare-'+batch),'audit_profile':audit_profile}
        print(json.dumps({'batch':batch,'medians':med}),flush=True)
    sources={str(p):sha(p) for p in [Path(__file__).relative_to(Path.cwd()),Path('scripts/benchmark_manifest.py'),
        *(Path('qwen_one_two/training')/name for name in ('release.py','component_producer.py','clean.py','inputs.py')),
        Path('scripts/train_receiver.py')]}
    for name in sources:
        target=out/'source-snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(name).read_bytes())
    check(not BLOCKED & set(sys.modules),'model runtime imported')
    report={'protocol':'asset-audit-v1','scope':'CPU input preparation and existing audit JSON only; no training speed measurement',
        'baseline_commit':'c10c163e2572ead63e4d65c358c184cd2f222ffc','baseline_source_sha256':BASELINE,
        'sources':sources,'before_profile_sha256':sha(a.profile_before),'prior_input_evidence_sha256':sha(prior_path),
        'snapshots':snapshots,'input_cases':parity,'direct_paired_component_cases':16,'measurements':measurements,
        'loader_validation_ast_identical':True,'cuda_visible_devices':'','model_runtimes_loaded':[],
        'python':sys.version,'passed':True}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
