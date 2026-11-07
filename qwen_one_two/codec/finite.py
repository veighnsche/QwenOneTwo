"""Read-only CUDA finite predicates, with one host decision per boundary."""
import torch


def all_finite(values):
    """Check every present tensor without synchronizing once per tensor.

    Infinity norms cannot overflow on finite real inputs (unlike L2 norms).
    The multi-tensor reduction propagates NaNs. Exotic tensors use exact GPU
    flags instead. No arithmetic here participates in the model's graph.
    CPU tensors are rejected before any tensor computation.
    """
    values = tuple(t for t in values if t is not None)
    if any(not t.is_cuda for t in values):
        return False
    values = tuple(t.detach() for t in values if t.numel())
    if not values:
        return True
    with torch.no_grad():
        if all(t.layout == torch.strided and t.is_floating_point() for t in values):
            return bool(torch.stack(torch._foreach_norm(values, float('inf'))).isfinite().all())
        else:
            flags = [t.isfinite().all() for t in values]
        return bool(torch.stack(flags).all())
