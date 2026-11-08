"""RD-04 token/position coverage only. No neural model or tensor imports."""
import argparse
import json
from pathlib import Path

from qwen_one_two.training.inputs import canonical,check
from qwen_one_two.training.manifest import file_hash
from qwen_one_two.training.release import load_release,decode
from qwen_one_two.training.direct_roles import coverage
from scripts.fit_reference import TRAIN_SHA,DEV_SHA


class MetadataTokenizer:
    """Pinned Rust BPE and Jinja framing for the one-user-message audit regime."""
    def __init__(self,snapshot):
        from tokenizers import Tokenizer
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        self.backend=Tokenizer.from_file(str(snapshot/'tokenizer.json'))
        env=ImmutableSandboxedEnvironment(trim_blocks=True,lstrip_blocks=True)
        def reject(message):raise ValueError(message)
        env.globals['raise_exception']=reject
        self.template=env.from_string((snapshot/'chat_template.jinja').read_text())

    def apply_chat_template(self,messages,**kwargs):
        check(len(messages)==1 and messages[0]['role']=='user' and isinstance(messages[0]['content'],str),
              'metadata tokenizer supports only the actual offline one-user regime')
        return self.template.render(messages=messages,tools=None,add_vision_id=False,**kwargs)

    def __call__(self,text,**kwargs):
        encoded=self.backend.encode(text,add_special_tokens=False)
        return {'input_ids':encoded.ids,'offset_mapping':encoded.offsets}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=('direct-role-states-v1','direct-role-states-v2'),default='direct-role-states-v1');parser.add_argument('--snapshot',required=True);parser.add_argument('--output',required=True)
    a=parser.parse_args();root=Path(__file__).resolve().parents[1];output=Path(a.output).resolve()
    check(output.is_relative_to(root/'data') and not output.exists(),'fresh local data output required')
    snapshot=Path(a.snapshot).resolve();tokenizer=MetadataTokenizer(snapshot)
    evidence=decode((root/'docs/evidence/redesign-coverage-v1.json').read_bytes())
    path=next(root/p for p in evidence['artifacts'] if p.endswith('/panels.json'))
    check(file_hash(path)==evidence['artifacts'][str(path.relative_to(root))],'coverage panel identity')
    panels=decode(path.read_bytes())['panels']
    train=load_release(root/'data/training-collections-20261105/releases/train',expected_hash=TRAIN_SHA)
    dev=load_release(root/'data/training-collections-20261105/releases/development',expected_hash=DEV_SHA)
    results={name:coverage(train if name=='train_receiver' else dev,panels[name],tokenizer,
        intervention=name!='train_receiver',profile=a.profile) for name in ('train_receiver','development_E','development_X')}
    output.mkdir(parents=True)
    (output/'records.json').write_bytes(canonical({name:r['records'] for name,r in results.items()}))
    report={'profile':a.profile,'scope':'BPE/position metadata only; no features or model execution',
        'releases':{'training':TRAIN_SHA,'development':DEV_SHA},
        'tokenizer_files':{p.name:file_hash(p) for p in snapshot.iterdir() if p.suffix in {'.json','.jinja'}},
        'panels':{name:r['summary'] for name,r in results.items()},
        'records_sha256':file_hash(output/'records.json'),'model_execution':False,'training':False}
    (output/'report.json').write_bytes(canonical(report));print(json.dumps(report))


if __name__=='__main__':main()
