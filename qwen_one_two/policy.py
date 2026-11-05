"""Token policy for the pinned checkpoint; contains no model execution."""

HEAD = "one-two-v1"
TOTAL_CONTEXT_TOKENS = 262_144
MAX_THINKING_TOKENS = TOTAL_CONTEXT_TOKENS // 2
SCORE_CURVE = "smoothstep"
SCORE_MAPPING = f"s = clamp(raw_logit, 0, 1); budget = floor((3*s*s - 2*s*s*s) * {MAX_THINKING_TOKENS} + 0.5)"


def check_model_context(config):
    text = getattr(config, "text_config", config)
    context = text.get("max_position_embeddings") if isinstance(text, dict) else getattr(text, "max_position_embeddings", None)
    if context != TOTAL_CONTEXT_TOKENS:
        raise ValueError(f"Pinned model context must be {TOTAL_CONTEXT_TOKENS}, got {context}")
    return context
