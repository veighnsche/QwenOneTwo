"""CPU-only release/input A/B; importing a model runtime is forbidden."""
import argparse
import dataclasses
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time

BLOCKED={'torch','transformers','safetensors','triton','cupy'}
LEGACY={'release.py':'2d9e0d216cf13bac3e57c652bb486dfea6880b6a5e1ff51e3d0ac5c699af9da3',
        'component_producer.py':'e24fea359c8cfd5ce1b05efc4705543369fa60fc0eecf49ae70602a09167a0cc',
        'producer.py':'4b6bfade492f256af38d0493b168cbc640310943bc145a43b3c3d0bd2fbe15a1'}
RELEASES={'train':'c70d4f44135a34c00e7fa0348833b1ec56613def15f5753231589492ed35478f',
          'development':'5b8f4cffdc4f33e255d3a34306cac03a68c50dcd9dab905ae8d8adcf89c81e77'}


class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in BLOCKED:
            raise AssertionError('Model runtime forbidden in CPU benchmark: '+fullname)


def main():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    assert not BLOCKED & set(sys.modules),'model runtime already loaded'
    sys.meta_path.insert(0,NoModels())
    from qwen_one_two.training.inputs import canonical,digest,check
    from qwen_one_two.training.release import load_release,ReviewedRelease
    from qwen_one_two.training.component_producer import component_view,reference_record
    from qwen_one_two.training.producer import native_plan
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline',required=True)
    p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    def sha(path):return digest(Path(path).read_bytes())
    def legacy(name):
        path=Path(a.baseline)/name;check(sha(path)==LEGACY[name],'baseline source hash')
        spec=importlib.util.spec_from_file_location('qwen_one_two.training._manifest_legacy_'+path.stem,path)
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        return module
    old_release=legacy('release.py');old_component=legacy('component_producer.py');old_producer=legacy('producer.py')
    # native_plan's isinstance check must use its own literal baseline class.
    old_producer.ReviewedRelease=old_release.ReviewedRelease
    def normalized(value):
        if dataclasses.is_dataclass(value):return normalized(dataclasses.asdict(value))
        if isinstance(value,bytes):return {'bytes_sha256':digest(value),'length':len(value)}
        if isinstance(value,dict):return {k:normalized(v) for k,v in value.items()}
        if isinstance(value,(set,frozenset)):return sorted(normalized(v) for v in value)
        if isinstance(value,(tuple,list)):return [normalized(v) for v in value]
        return value
    def view_result(fn,release,row,**kwargs):
        view,provenance,indices=fn(release,row,**kwargs)
        return normalized((view,provenance.nodes,provenance.sources,indices,view.materialize(provenance)))
    def reference_result(fn,release,row):
        item,provenance=fn(release,row)
        return normalized((item,provenance.nodes,provenance.sources))
    sources={str(f):sha(f) for f in [Path(__file__).relative_to(Path.cwd()),
        *(Path('qwen_one_two/training')/n for n in ('release.py','component_producer.py','producer.py','binding.py','receiver_learning.py'))]}
    for name in sources:
        target=out/'source-snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(name).read_bytes())
    cases=[];snapshots={};loaded={}
    for split,identity in RELEASES.items():
        root=Path('data/training-collections-20261105/releases')/split
        current=load_release(root,expected_hash=identity);previous=old_release.load_release(root,expected_hash=identity)
        loaded[split]=(current,previous)
        check(current.identity==previous.identity and current._manifest==previous._manifest,'manifest bytes')
        check(current._records==previous._records and current._assets==previous._assets and current._summary==previous._summary,'snapshot bytes')
        manifest=previous.manifest
        check(current.purpose==manifest['purpose'] and current.selected_ids==frozenset(e['id'] for e in manifest['records'] if e['selected']),'admission index')
        snapshots[split]={'identity':identity,'manifest_bytes':len(current._manifest),'records':len(current._records),
            'assets':len(current._assets),'selected':len(current.selected_ids),'all_snapshot_bytes_equal':True,
            'records_sha256':digest(canonical({k:digest(v) for k,v in current._records.items()})),
            'assets_sha256':digest(canonical({k:digest(v) for k,v in current._assets}))}
        seen=set()
        for entry in manifest['records']:
            if not entry['selected']:continue
            row=current.record(entry['id']);cat=row.get('training_category')
            if cat=='reference' or row.get('evaluation_slice')=='RF':
                key=('reference',row['reference_role'],row['operation']);kind='reference'
            elif cat in {'receiver','reader','decision_replay'} or row.get('evaluation_slice') in {'E','X'}:
                key=('component',cat,row['operation'],row.get('evaluation_slice'),row['labels'].get('status'));kind='component'
            elif cat in {'writer','interleaved','direct','language_replay'}:
                key=('native',cat,row['operation']);kind='native'
            else:continue
            if key in seen:continue
            seen.add(key)
            if kind=='reference':
                before=reference_result(old_component.reference_record,previous,row)
                after=reference_result(reference_record,current,row)
            elif kind=='component':
                kw={'intervention':True} if row.get('evaluation_slice') in {'E','X'} else {}
                before=view_result(old_component.component_view,previous,row,**kw)
                after=view_result(component_view,current,row,**kw)
            else:
                before=normalized(old_producer.native_plan(previous,row,'guided_joint',1))
                after=normalized(native_plan(current,row,'guided_joint',1))
            check(before==after,'input/provenance parity: '+row['id'])
            cases.append({'split':split,'record':row['id'],'kind':kind,'stratum':key,'matched_sha256':digest(canonical(after))})
        print(json.dumps({'stage':'parity','split':split,'cases':len(seen)}),flush=True)
    current,previous=loaded['train'];ids=[]
    for line in Path('data/receiver-learning-20261106/continued/training.jsonl').read_text().splitlines():
        entry=json.loads(line)
        if entry['window']['window']==105:ids=[e['record'] for e in entry['episodes']];break
    check(len(ids)==8,'fixed existing eight-record batch');rows=[current.record(k) for k in ids]
    timings=[]
    def prepare(label):
        fn,release=(old_component.component_view,previous) if label=='legacy' else (component_view,current)
        start=time.perf_counter()
        for row in rows:
            view,provenance,_=fn(release,row);view.materialize(provenance)
        return time.perf_counter()-start
    for label in ('legacy','current'):prepare(label)
    for repeat in range(5):
        for label in (('legacy','current') if repeat%2==0 else ('current','legacy')):
            timings.append({'repeat':repeat,'path':label,'seconds':prepare(label)})
    from unittest.mock import patch
    # Count actual full-property reads across eight complete CPU view builders.
    counts={}
    for label,cls in [('legacy',old_release.ReviewedRelease),('current',ReviewedRelease)]:
        getter=cls.manifest.fget;calls=[]
        def counted(self):calls.append(1);return getter(self)
        with patch.object(cls,'manifest',property(counted)):prepare(label)
        counts[label]=len(calls)
    med={k:statistics.median(t['seconds'] for t in timings if t['path']==k) for k in ('legacy','current')}
    check(not BLOCKED & set(sys.modules),'model runtime imported')
    result={'protocol':'manifest-access-v1','scope':'CPU metadata/input preparation only; no GPU/model workload or training speed measurement',
        'baseline_commit':'8459e8441887b6621aa2ceec6ddcbb2206d017b0','baseline_sources':LEGACY,'sources':sources,
        'snapshots':snapshots,'cases':cases,'fixed_timing_records':ids,'timings':timings,'medians':med,
        'seconds_saved_per_eight_views':med['legacy']-med['current'],'manifest_reads_per_eight_views':counts,
        'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],'model_runtimes_loaded':[],
        'python':sys.version,'passed':True}
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'stage':'complete','cases':len(cases),'medians':med,'manifest_reads':counts}),flush=True)


if __name__=='__main__':main()
