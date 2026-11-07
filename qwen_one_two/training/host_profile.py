"""Optional worker-local call attribution; imports model code only on entry."""
from contextlib import contextmanager


@contextmanager
def model_hooks(recorder,backbone,codec,reference,encoder):
    from . import component_producer,clean,reference as reference_module
    from ..codec import session,writer,neural,recurrence
    targets=[(component_producer,'component_view','prepare_view'),
        (clean.CleanView,'materialize','materialize_view'),
        (clean,'tokenize_excerpt','source_tokenize'),
        (clean.CleanEncoder,'encode','clean_encode'),
        (clean.CleanEncoder,'_states','source_model_encode'),
        (clean.CleanEncoder,'reference_payload','reference_payload'),
        (type(reference),'annotation_states','reference_model_encode'),
        (type(reference.pooler),'forward','reference_pooler'),
        (reference_module,'state_fingerprint','weight_fingerprint'),
        (session.Session,'step','session_step'),(session.Session,'evaluate','session_evaluate'),
        (session.Session,'deliver','session_deliver'),
        (type(backbone),'forward','backbone_forward'),
        (type(codec),'encode','codec_encode'),(type(codec),'evaluate','codec_evaluate'),
        (writer,'decode_result','result_decode'),
        (neural,'finite','codec_finite_guard'),(neural,'cuda_only','codec_device_guard'),
        (recurrence._Graphs,'__init__','recurrence_capture_setup'),
        (recurrence._Graphs,'prepare_backward','recurrence_backward_prepare'),
        (recurrence._Graphs,'copy','recurrence_input_copy'),
        # Session imported the predicate directly; wrap that actual call site.
        (session,'all_finite','session_finite_guard')]
    with recorder.patch(targets):yield
