"""Fetch immutable public Qev artifacts as data; never execute remote code."""
import argparse
import json
import urllib.request
from pathlib import Path

from .decision import QEV_REPO,QEV_REVISION,verify_qev


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',default='data');args=parser.parse_args()
    out=Path(args.data)/'qev';out.mkdir(parents=True,exist_ok=True)
    for name in ('qev_config.json','adapter/adapter_config.json','adapter/adapter_model.safetensors',
                 'pointer.safetensors','release_manifest.json','LICENSE','NOTICE'):
        dest=out/name;dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():
            with urllib.request.urlopen(f'https://huggingface.co/{QEV_REPO}/resolve/{QEV_REVISION}/{name}',timeout=90) as response:
                temp=dest.with_suffix(dest.suffix+'.tmp');temp.write_bytes(response.read());temp.replace(dest)
    verify_qev(out)
    (out/'upstream.json').write_text(json.dumps({'repo':QEV_REPO,'revision':QEV_REVISION}))
    print('Pinned Qev adapter and pointer verified.')

if __name__=='__main__':main()
