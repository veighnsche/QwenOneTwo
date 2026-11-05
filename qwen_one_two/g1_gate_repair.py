"""Bounded diagnostic gate-readout fit on GPU; no codec or serving changes.

Only the existing four-way Linear(256,4) readout is optimized. The fixed
teacher trajectory supplies call labels and post-return native-boundary labels.
Free-running continuation never substitutes a target action for an argmax.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from safetensors.torch import load_file, save_file

from .g1_gate_diagnosis import load, trace
from .g1_proof import configure, fixtures, Episode


def collect(b, c, fixture):
    ep = Episode(b, c, fixture, capture=False)
    rows, annotations = [], []

    def add(feature, target, boundary):
        assert feature.is_cuda
        rows.append(feature.detach().clone())
        annotations.append({'fixture': fixture['name'], 'boundary': boundary,
                            'position': ep.state.position, 'phase': ep.ledger.phase,
                            'target': target})

    # The two existing supervised gate uses: request operation, then Continue
    # after the completed return. The fixture's operation is a training target,
    # never a feature or a free-running control override.
    def supervised(module, args, output):
        returning = ep.ledger.events[-1]['label'] == 'return'
        add(args[0], 0 if returning else fixture['port'],
            'post_return' if returning else 'request')

    handle = c.gate.register_forward_hook(supervised)
    original_step = ep.step

    def step(embeds, mode, label, kind=None):
        h = original_step(embeds, mode, label, kind)
        # Cover the already-declared teacher-forced continuation, rather than
        # supervising only RETURN_END and ignoring later native boundaries.
        if ep.ledger.consumed and label in ('thinking_close', 'native_A'):
            add(c.feature(h[0, -1].float(), ep.ledger.counters()), 0, label)
        return h

    ep.step = step
    try:
        with torch.no_grad():
            ep.run()
    finally:
        handle.remove()
    return rows, annotations


def fit(gate, features, labels, steps=256):
    """Fixed full-batch, class-balanced CE schedule. No test-driven stopping."""
    assert features.is_cuda and labels.is_cuda
    optimizer = torch.optim.AdamW(gate.parameters(), lr=.01, betas=(.9, .95),
                                 eps=1e-8, weight_decay=0., foreach=False, fused=False)
    history = []
    classes = labels.unique()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        per_row = F.cross_entropy(gate(features), labels, reduction='none')
        loss = torch.stack([per_row[labels == k].mean() for k in classes]).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gate.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if step in (0, steps - 1):
            history.append({'step': step, 'loss': loss.item()})
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        logits = gate(features)
        correct = logits.argmax(-1) == labels
        selected = logits.gather(1, labels[:, None]).squeeze(1)
        other = logits.clone().scatter_(1, labels[:, None], -torch.inf).max(-1).values
        result = {'pass': bool(correct.all()), 'rows': labels.numel(),
                  'correct': int(correct.sum()), 'minimum_margin': (selected - other).min().item(),
                  'per_class': {str(int(k)): {'rows': int((labels == k).sum()),
                                            'correct': int(correct[labels == k].sum())}
                                for k in classes}, 'history': history}
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--snapshot', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    configure()
    source_hash = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    report = {'scope': 'G1 diagnostic gate readout only', 'parent_sha256': source_hash,
              'schedule': {'steps': 256, 'lr': .01, 'loss': 'class-balanced four-way CE',
                           'optimizer': 'AdamW(.9,.95), eps=1e-8, wd=0, clip=1'},
              'no_runtime_action_overrides': True, 'codec_expanded': False}
    with sdpa_kernel(SDPBackend.MATH):
        b, c, _ = load(args.snapshot, args.checkpoint)
        rows, annotations = [], []
        for f in fixtures(b.tokenizer):
            x, a = collect(b, c, f)
            rows.extend(x)
            annotations.extend(a)
        features = torch.stack(rows)
        labels = torch.tensor([a['target'] for a in annotations], device='cuda')
        report['annotations'] = annotations
        report['before'] = {'correct': int((c.gate(features).argmax(-1) == labels).sum()),
                            'rows': labels.numel()}
        # All features are detached CUDA tensors. The backbone, control trunk,
        # writer, evaluator, scalar rows and adapters cannot receive this loss.
        report['fit'] = fit(c.gate, features, labels)
        if not report['fit']['pass']:
            raise RuntimeError('Fixed gate-readout fit failed; do not certify')
        original = load_file(args.checkpoint, device='cuda')
        child = dict(original)
        child['codec.gate.weight'] = c.gate.weight.detach().contiguous()
        child['codec.gate.bias'] = c.gate.bias.detach().contiguous()
        changed = [n for n in child if not torch.equal(original[n], child[n])]
        assert set(changed) == {'codec.gate.weight', 'codec.gate.bias'}, changed
        assert all(p.grad is None for p in b.parameters())
        assert all(p.grad is None for p in c.parameters())
        report['changed_tensors'] = changed
        save_file({'features': features, 'labels': labels}, str(out / 'gate-training.safetensors'))
        save_file(child, str(out / 'diagnostic.safetensors'))
        report['checkpoint_sha256'] = hashlib.sha256((out / 'diagnostic.safetensors').read_bytes()).hexdigest()
        report['unforced_validation'] = [trace(b, c, f, True) for f in fixtures(b.tokenizer)]
        report['pass'] = all(x['error'] is None for x in report['unforced_validation'])
    (out / 'repair.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('annotations', 'unforced_validation')}), flush=True)
    if not report['pass']:
        raise RuntimeError('Unforced continuation still fails; preserve evidence, no certification claim')


if __name__ == '__main__':
    main()
