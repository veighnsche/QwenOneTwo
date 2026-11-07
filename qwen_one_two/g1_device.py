"""Model-free device policy; secondary GPU is read-only diagnostics only."""
from contextlib import contextmanager
import os
import threading

PRIMARY = 'GPU-225509c3-d702-6c27-de03-76b5e99b33b1'
SECONDARY = 'GPU-744c1cf5-9578-6aad-95c1-7a6f1c9d11db'
DEVICES = {PRIMARY:'RTX 3090', SECONDARY:'RTX 3060'}
# Autograd checkpoint recomputation can run on engine threads. Admission and
# optimizer prohibition belong to the entire dedicated diagnostic process.
_diagnostic = None
_scope_lock = threading.Lock()


def require_visibility():
    selected = _diagnostic or PRIMARY
    if os.environ.get('CUDA_VISIBLE_DEVICES') != selected:
        raise RuntimeError('explicit single GPU UUID required: '+selected)
    return selected


@contextmanager
def diagnostic_device(uuid, *, execute_gpu=False):
    """Explicit process-local opt-in; never discovers or initializes a GPU."""
    global _diagnostic
    if not execute_gpu or uuid not in DEVICES:
        raise RuntimeError('explicit non-nested read-only diagnostic device required')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != uuid:
        raise RuntimeError('select diagnostic UUID before importing model runtimes')
    with _scope_lock:
        if _diagnostic is not None:raise RuntimeError('explicit non-nested read-only diagnostic device required')
        _diagnostic=uuid
    try:
        yield
    finally:
        with _scope_lock:_diagnostic=None


def require_cuda(cuda, tensor=None):
    selected = require_visibility()  # Reject selection before any device query.
    if tensor is not None and (not tensor.is_cuda or tensor.device.index != 0):
        raise RuntimeError('model tensors require selected local cuda:0; no CPU fallback')
    if not cuda.is_available():
        raise RuntimeError('selected CUDA GPU required; no CPU fallback')
    if not cuda.get_device_name(0).endswith(DEVICES[selected]):
        raise RuntimeError('selected GPU model mismatch: expected '+DEVICES[selected])


def require_training_context():
    if _diagnostic is not None:
        raise ValueError('optimizer windows forbidden in read-only diagnostic workers')
