"""GPU-only G1 control-readout diagnosis; never masks or replaces a gate action."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from safetensors.torch import load_file

from .g1_backbone import Backbone
from .g1_protocol import Codec
from .g1_proof import configure, fixtures, Episode


def load(snapshot, checkpoint):
    b = Backbone(snapshot, 'data/qev')
    pointer = load_file('data/qev/pointer.safetensors', device='cuda')
    with torch.device('cuda'):
        c = Codec(b.sigma, b.model.model.embed_tokens.weight, pointer)
    initial = {n: p.detach().clone() for n, p in c.gate.named_parameters()}
    b.bind_rows(c)
    weights = load_file(str(checkpoint), device='cuda')
    bp, cp = dict(b.named_parameters()), dict(c.named_parameters())
    with torch.no_grad():
        for name, value in weights.items():
            (cp[name[6:]] if name.startswith('codec.') else bp[name]).copy_(value)
    b.set_compensated(True)
    return b, c, initial


def trace(b, c, fixture, released):
    ep = Episode(b, c, fixture, capture=False)
    records = []

    def record(module, args, output):
        feature = args[0].detach().float()
        event = ep.ledger.events[-1]['label']
        target = 0 if event == 'return' else fixture['port']
        # Labels only describe the two supervised boundaries, not arbitrary
        # free-running boundaries; Choice remains a visible competing logit.
        if len(records) >= 2:
            target = None
        item = {'event': event, 'position': ep.state.position,
                'phase': ep.ledger.phase, 'B': ep.ledger.b,
                'logits': output.detach().tolist(),
                'probabilities': output.detach().softmax(-1).tolist(),
                'argmax': int(output.argmax()), 'target': target}
        if target is not None:
            item['cross_entropy'] = F.cross_entropy(
                output[None], torch.tensor([target], device='cuda')).item()
        records.append(item)

    handle = c.gate.register_forward_hook(record)
    error = None
    try:
        with torch.no_grad():
            ep.run(released=released)
    except Exception as exc:
        error = str(exc)
    finally:
        handle.remove()
    return {'fixture': fixture['name'], 'released': released,
            'gate_boundaries': records, 'error': error, 'summary': ep.summary()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--snapshot', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    configure()
    report = {'checkpoint_sha256': hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
              'action_ids': ['Continue', 'Noul', 'Choice', 'Score'], 'cases': []}
    with sdpa_kernel(SDPBackend.MATH):
        b, c, initial = load(args.snapshot, args.checkpoint)
        report['gate_change_from_seeded_initialization'] = {
            n: {'maximum_absolute': (p - initial[n]).abs().max().item(),
                'normalized_rms': ((p - initial[n]).double().square().mean().sqrt()
                                   / initial[n].double().square().mean().sqrt().clamp_min(1e-10)).item()}
            for n, p in c.gate.named_parameters()}
        for f in fixtures(b.tokenizer):
            for released in (False, True):
                case = trace(b, c, f, released)
                report['cases'].append(case)
                print(json.dumps({k: v for k, v in case.items() if k != 'summary'}), flush=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
