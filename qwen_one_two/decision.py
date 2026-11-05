"""Qev pointer readout and trained, textless internal controls.

PointerHead follows loadchange/qev (Apache-2.0), qev/model.py. Qev weights
remain separately switchable; this module supplies the new latent protocol.
"""
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
from safetensors.torch import load_file

MODEL = 'Qwen/Qwen3.5-0.8B'
REVISION = '2fc06364715b967f1860aea9cf38778875588b17'
QEV_REPO = 'twainsk/qev-0.8b'
QEV_REVISION = '1d2a6d7f891daab7483a49c3abe507c0c9d7ea62'
ACTIONS = ('continue', 'noul', 'choice', 'score')
CHECKPOINT = 'one-two-v1'


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def verify_qev(folder):
    folder = Path(folder)
    config = json.loads((folder / 'qev_config.json').read_text())
    if (config['base_model'], config['base_revision']) != (MODEL, REVISION):
        raise ValueError('Qev adapter does not match the pinned Qwen backbone')
    if config.get('hidden_size') != 1024 or config.get('native_generation') != 'adapter_disabled':
        raise ValueError('Unsupported Qev architecture')
    manifest = json.loads((folder / 'release_manifest.json').read_text())
    for name in ('adapter/adapter_config.json', 'adapter/adapter_model.safetensors', 'pointer.safetensors', 'qev_config.json'):
        if digest(folder / name) != manifest['files'][name]['sha256']:
            raise ValueError(f'Qev checksum mismatch: {name}')
    return config


class PointerHead(nn.Module):
    def __init__(self, hidden=1024, dim=256):
        super().__init__()
        self.q = nn.Linear(hidden, dim, bias=False)
        self.k = nn.Linear(hidden, dim, bias=False)
        self.scale = 1 / math.sqrt(dim)

    def forward(self, query, candidates):
        return (self.k(candidates.float()) * self.q(query.float()).unsqueeze(-2)).sum(-1) * self.scale


class Controls(nn.Module):
    """Persistent learned routing, a latent query and feedback bridge.

    Budget/Score are regression. Only operation routing and Choice/Noul use
    categorical losses. No decision arguments or results are generated as text.
    """
    def __init__(self, hidden=1024):
        super().__init__()
        self.route = nn.Sequential(nn.Linear(hidden, 128), nn.Tanh(), nn.Linear(128, 4))
        self.candidate = nn.Linear(hidden, 1)
        self.query = nn.Sequential(nn.Linear(hidden, 256), nn.Tanh(), nn.Linear(256, hidden))
        self.types = nn.Embedding(4, hidden)
        self.feedback = nn.Linear(hidden, hidden, bias=False)
        self.scalar_feedback = nn.Linear(1, hidden)
        self.budget_feedback = nn.Linear(1, hidden)
        self.choice_feedback = nn.Linear(1, hidden)
        nn.init.zeros_(self.choice_feedback.weight)
        nn.init.zeros_(self.choice_feedback.bias)
        self.tool_score = nn.Linear(hidden, 1, bias=False)
        self.scalar = nn.Linear(hidden, 1, bias=False)
        self.noul_options = nn.Parameter(torch.randn(2, hidden) * 0.02)
        nn.init.zeros_(self.query[-1].weight)
        nn.init.zeros_(self.query[-1].bias)

    def normal(self, hidden):
        value = hidden.float()
        return value / value.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)

    def gate(self, hidden):
        return self.route(self.normal(hidden))

    def latent_query(self, hidden, action, embedding_scale):
        h = self.normal(hidden)
        # A bounded residual query, in the model's input-embedding scale.
        return (h + self.query(h) + self.types.weight[action]).tanh() * embedding_scale

    def feedback_vector(self, kind, value, selected=None):
        if kind == 'budget':
            return self.budget_feedback(value.reshape(1, 1).float())
        if kind == 'score':
            return self.scalar_feedback(value.reshape(1, 1).float())
        context = self.feedback(selected.reshape(1, -1).float())
        if kind == 'choice':
            return context.tanh() + self.choice_feedback(value.reshape(1, 1).float())
        return context


def load_controls(folder, device):
    folder = Path(folder)
    info = json.loads((folder / 'training.json').read_text())
    if info.get('format') != CHECKPOINT or info.get('model_revision') != REVISION or info.get('qev_revision') != QEV_REVISION:
        raise ValueError('Incompatible QwenOneTwo training checkpoint')
    for name in ('controls.safetensors', 'router/adapter_model.safetensors', 'router/adapter_config.json'):
        if digest(folder / name) != info['files'][name]: raise ValueError(f'Trained artifact checksum mismatch: {name}')
    with torch.device(device):
        controls = Controls()
    weights = load_file(str(folder / 'controls.safetensors'), device=str(device))
    # Explicit migration of the first trained prototype: separate budget
    # feedback from generic Score feedback without changing its initial vector.
    if 'budget_feedback.weight' not in weights:
        weights['budget_feedback.weight'] = weights['scalar_feedback.weight'].clone()
        weights['budget_feedback.bias'] = weights['scalar_feedback.bias'].clone()
    if 'tool_score.weight' not in weights:
        weights['tool_score.weight'] = weights['scalar.weight'].clone()
    if 'choice_feedback.weight' not in weights:
        weights['choice_feedback.weight'] = torch.zeros_like(controls.choice_feedback.weight)
        weights['choice_feedback.bias'] = torch.zeros_like(controls.choice_feedback.bias)
    controls.load_state_dict(weights, strict=True)
    controls.eval().requires_grad_(False)
    return controls, info
