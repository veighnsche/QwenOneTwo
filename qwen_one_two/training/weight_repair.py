"""One fixed input-calibrated rank-one reference repair; never an optimizer."""
from .inputs import SLOTS,check,digest

PROTOCOL='reference-common-mode-v1'
MAX_PER_ROLE=8


def calibration_views(views):
    views=tuple(views)
    check(len({v.identity for v in views})==len(views),'duplicate calibration identity')
    check({v.role for v in views}==set(SLOTS),'calibration role coverage')
    return tuple(view for role in SLOTS for view in sorted(
        (v for v in views if v.role==role),
        key=lambda v:digest(f'{PROTOCOL}:17:{v.identity}'.encode()))[:MAX_PER_ROLE])


def deflate_affine(weight,bias,direction):
    """Return projected FP32 tensors without changing inputs or building a VJP."""
    import torch
    from ..codec.neural import cuda_only
    for value in (weight,bias,direction):cuda_only(value)
    check(weight.dtype==bias.dtype==torch.float32,'FP32 reference lift required')
    check(weight.ndim==2 and bias.shape==direction.shape==(weight.shape[0],),'affine repair shape')
    with torch.no_grad():
        check(all(bool(x.isfinite().all()) for x in (weight,bias,direction)),'nonfinite repair input')
        u=direction.double();length=torch.linalg.vector_norm(u)
        check(float(length)>1e-12,'zero common direction')
        u=u/length
        w=weight.double();b=bias.double()
        repaired_w=(w-u[:,None]*(u@w)[None,:]).float()
        repaired_b=(b-u*torch.dot(u,b)).float()
        check(bool(repaired_w.isfinite().all()) and bool(repaired_b.isfinite().all()),'nonfinite repair result')
    return repaired_w,repaired_b
