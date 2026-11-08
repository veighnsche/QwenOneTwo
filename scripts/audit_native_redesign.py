"""RD-06 existing native/reader targets and evaluation plan; no model execution."""
import argparse
from pathlib import Path
from qwen_one_two.training.inputs import canonical,check
from qwen_one_two.training.release import load_release,decode
from qwen_one_two.training.manifest import file_hash
from qwen_one_two.training.native_coverage import audit_record,summarize,evaluation_manifest
from scripts.audit_direct_roles import MetadataTokenizer
from scripts.fit_reference import TRAIN_SHA,DEV_SHA


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--snapshot',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1];output=Path(args.output).resolve()
    check(output.is_relative_to(root/'data') and not output.exists(),'fresh local data output required')
    snapshot=Path(args.snapshot);tokenizer=MetadataTokenizer(snapshot)
    coverage=decode((root/'docs/evidence/redesign-coverage-v1.json').read_bytes())
    path=next(p for p in coverage['artifacts'] if p.endswith('inventory.jsonl'))
    check(file_hash(root/path)==coverage['artifacts'][path],'native inventory identity drift')
    inventory=[decode(line) for line in (root/path).read_bytes().splitlines()]
    releases={'train':load_release(root/'data/training-collections-20261105/releases/train',expected_hash=TRAIN_SHA),
              'development':load_release(root/'data/training-collections-20261105/releases/development',expected_hash=DEV_SHA)}
    records=[]
    for item in inventory:
        if item['native']['category'] not in ('writer','interleaved','reader','direct','language_replay'):continue
        release=releases['train' if item['split']=='train' else 'development']
        records.append(audit_record(release,release.record(item['id']),item,tokenizer))
    evaluation=evaluation_manifest(releases['development'],inventory)
    output.mkdir(parents=True);(output/'records.json').write_bytes(canonical(records));(output/'evaluation.json').write_bytes(canonical(evaluation))
    report={'protocol':'direct-native-eligibility-v1','releases':{'training':TRAIN_SHA,'development':DEV_SHA},
        'profile':'direct-role-states-v1','summary':summarize(records),'records':len(records),
        'records_sha256':file_hash(output/'records.json'),'evaluation_sha256':file_hash(output/'evaluation.json'),
        'tokenizer_files':{p.name:file_hash(p) for p in snapshot.iterdir() if p.suffix in ('.json','.jinja')},
        'scope':'BPE/position/provenance metadata only; no features, model forwards, real Sessions or eligible gradient measurements',
        'actual_execution':{'sessions':0,'prefills':0,'optimizer_updates':0},'gpu_qualification':'pending RD-07'}
    (output/'report.json').write_bytes(canonical(report));print('records',len(records));print(report['summary'])


if __name__=='__main__':main()
