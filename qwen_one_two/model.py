"""One CUDA backbone, switchable Qev, always-active trained internal controls."""
import json
from pathlib import Path

import torch
from peft import PeftModel, LoraConfig
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from .decision import (MODEL, REVISION, CHECKPOINT, Controls, PointerHead,
                       verify_qev, load_controls, digest)
from .policy import check_model_context

file_hash = digest


def tensor_hash(tensor):
    import hashlib
    return hashlib.sha256(tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


class Runtime:
    def __init__(self, data_dir, cache_dir, device='cuda:0', training=False):
        self.device = torch.device(device)
        if self.device.type != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError('CUDA GPU required. CPU model execution is disabled. Check device selection and access.')
        self.gpu_name = torch.cuda.get_device_name(self.device)
        if 'RTX 3090' not in self.gpu_name:
            raise RuntimeError('Select the RTX 3090 explicitly by UUID; refusing a different GPU')
        self.data_dir = Path(data_dir)
        self.cancel_file = str(self.data_dir / '.cancel')
        self.model_id, self.revision = MODEL, REVISION
        folder = self.data_dir / 'qev'
        self.qev_config = verify_qev(folder)
        self.tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, cache_dir=str(cache_dir))
        self.model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION,
            cache_dir=str(cache_dir), dtype=torch.bfloat16, attn_implementation='sdpa').to(self.device).eval()
        self.model.requires_grad_(False)
        self.context_length = check_model_context(self.model.config)
        self.base_hash = REVISION
        self.model.model = PeftModel.from_pretrained(self.model.model, folder / 'adapter', adapter_name='decision')
        self.backbone = self.model.model
        with torch.device(self.device):
            self.pointer = PointerHead()
        self.pointer.load_state_dict(load_file(str(folder / 'pointer.safetensors'), device=str(self.device)), strict=True)
        self.pointer.eval().requires_grad_(False)
        self.model.qev_pointer = self.pointer
        self.training_mode = training
        self.train_router = training
        trained_root = self.data_dir / 'trained'
        trained = trained_root / 'current' if (trained_root / 'current').exists() else trained_root / CHECKPOINT
        if training:
            self.backbone.add_adapter('router', LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0,
                target_modules=['q_proj','v_proj','in_proj_qkv','in_proj_z','gate_proj','down_proj'],
                task_type='FEATURE_EXTRACTION', bias='none'))
            with torch.device(self.device):
                self.controls = Controls()
            with torch.no_grad():
                for i, word in enumerate(['false','true']):
                    self.controls.noul_options[i].copy_(self.model.get_input_embeddings().weight[self.tok.encode(word,add_special_tokens=False)[0]].float())
                self.controls.types.weight.copy_(self.model.get_input_embeddings().weight[
                    [self.tok.encode(w,add_special_tokens=False)[0] for w in ['continue','boolean','choice','score']]].float())
            self.training_info = None
        else:
            self.controls, self.training_info = load_controls(trained, self.device)
            self.backbone.load_adapter(trained / 'router', adapter_name='router', is_trainable=False)
            if not all(torch.isfinite(v).all().item() for v in self.controls.parameters()):
                raise ValueError('Non-finite trained controls')
        # Register controls in the Qwen module tree: one model, resident learned weights.
        self.model.one_two = self.controls
        self.embedding_scale = self.model.get_input_embeddings().weight.detach().float().square().mean().sqrt().item()
        self.scalar_id = len(self.tok)
        self.budget_original = self.model.get_output_embeddings().weight[self.scalar_id].detach().clone()
        if not training: self.install_scalar()
        self.eos = sorted({t for t in [self.tok.eos_token_id,self.tok.convert_tokens_to_ids('<|im_end|>')] if isinstance(t,int)})
        self.select('router')
        if not training:
            self.model.eval().requires_grad_(False)
        from .thinking import ThinkingBudget
        self.thinking = ThinkingBudget(self)
        self.thinking.support()

    def select(self, adapter):
        if adapter not in ('router','decision'): raise ValueError('invalid adapter mode')
        if not self.training_mode and getattr(self, 'active_adapter', None) == adapter:
            return
        self.backbone.set_adapter(adapter)
        self.active_adapter = adapter
        if self.training_mode:
            # PEFT changes requires_grad on selection. Decision remains frozen.
            for name, value in self.backbone.named_parameters():
                value.requires_grad_(self.train_router and adapter == 'router' and '.router.' in name and 'lora_' in name)
        else:
            self.backbone.requires_grad_(False)

    def install_scalar(self):
        with torch.no_grad():
            self.model.get_output_embeddings().weight[self.scalar_id].copy_(self.controls.scalar.weight[0].to(torch.bfloat16))
            self.model.get_output_embeddings().weight[self.scalar_id+1].copy_(self.controls.tool_score.weight[0].to(torch.bfloat16))

    def forward(self, ids=None, cache=None, embeds=None):
        args = {'use_cache':True, 'past_key_values':cache, 'return_dict':True}
        if embeds is None:
            args['input_ids'] = torch.as_tensor(ids, device=self.device, dtype=torch.long).reshape(1,-1)
        else:
            args['inputs_embeds'] = embeds.to(device=self.device, dtype=torch.bfloat16).reshape(1,-1,1024)
        if self.training_mode and cache is not None:
            # Transformers' inference cache copies recurrent states in-place.
            # During autograd use replacement tensors, retaining the cache object.
            import types
            for layer in cache.layers:
                if hasattr(layer, 'recurrent_states'):
                    def update(layer, value, state_idx=0, **kwargs):
                        layer.recurrent_states[state_idx] = value
                        layer.is_recurrent_states_initialized[state_idx] = True
                        return value
                    layer.update_recurrent_state = types.MethodType(update, layer)
                    layer.conv_states = {k:v.clone() if v is not None else None for k,v in layer.conv_states.items()}
        out = self.backbone(**args)
        if cache is not None and out.past_key_values is not cache:
            raise RuntimeError('Qwen replaced the continuous inference cache')
        return out.last_hidden_state, out.past_key_values

    def logits(self, hidden):
        return self.model.get_output_embeddings()(hidden.to(torch.bfloat16)).float()

    def latent_decision(self, hidden, action, cache, candidates):
        query = self.controls.latent_query(hidden.reshape(-1), action, self.embedding_scale)
        self.select('decision')
        try:
            out, cache = self.forward(embeds=query, cache=cache)
            h = out[0,-1].float()
            if action == 3:
                raw = self.controls.tool_score(h).squeeze() if self.training_mode else self.logits(h)[self.scalar_id+1]
                return {'score':raw.clamp(0,1)}, h, cache
            options = self.controls.noul_options if action == 1 else candidates
            if options is None or len(options) < 2:
                raise ValueError('Choice requires at least two internally selected candidate representations')
            logits = self.pointer(h, options)
            probabilities = logits.softmax(-1)
            index = logits.argmax()
            selected = options[index] if action == 2 else (probabilities.unsqueeze(-1) * options).sum(0)
            value = index.float() / (len(options)-1) if action == 2 else probabilities[-1]
            return {'probabilities':probabilities, 'index':index, 'selected':selected, 'value':value}, h, cache
        finally:
            self.select('router')

    def feedback(self, kind, value, cache, selected=None):
        self.select('router')
        vector = self.controls.feedback_vector(kind, value, selected)
        # Bounded learned feedback in the foundation's embedding scale.
        return self.forward(embeds=vector.tanh() * self.embedding_scale * (1 if kind == 'budget' else 3), cache=cache)
