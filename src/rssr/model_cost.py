"""Partial Conv/Linear/MHA MAC accounting, not a full FLOP profiler."""
import math
import torch
from torch import nn

EXCLUDED = ['bias additions', 'normalization', 'activations', 'pooling',
            'softmax', 'interpolation', 'elementwise operations', 'data movement']


def measure(model, shape):
    """Count actual executed layers on a CPU synthetic input, restoring modes/hooks.

    One MAC is one multiply-accumulate. MHA includes Q/K/V/output projections
    and both attention matrix products; child out_proj is not counted twice.
    This implementation requires batch-first, same-width self-attention.
    """
    if len(shape) != 4 or any(type(v) is not int or v < 1 for v in shape):
        raise ValueError('Expected positive integer NCHW shape')
    if math.prod(shape) > 4*1024*1024:
        raise ValueError('Preflight fixture exceeds configured input-element cap')
    modules = list(model.modules())
    if any(p.device.type != 'cpu' for p in model.parameters()):
        raise ValueError('CPU preflight only')
    excluded_children = {child for m in modules if isinstance(m, nn.MultiheadAttention)
                         for child in list(m.modules())[1:]}
    counts = {'conv': 0, 'linear': 0, 'self_attention': 0}
    modes = {m:m.training for m in modules}
    handles = []

    def count(module, inputs, output):
        if isinstance(module, nn.Conv2d):
            counts['conv'] += output.numel()*(module.in_channels//module.groups)*math.prod(module.kernel_size)
        elif isinstance(module, nn.Linear):
            counts['linear'] += output.numel()*module.in_features
        else:
            q,k,v = inputs[:3]
            if not module.batch_first or q.shape != k.shape or q.shape != v.shape or not module._qkv_same_embed_dim:
                raise ValueError('Only batch-first same-width self-attention is supported')
            batch,tokens,width = q.shape
            counts['self_attention'] += batch*(4*tokens*width*width+2*tokens*tokens*width)

    try:
        for module in modules:
            if module not in excluded_children and isinstance(module,(nn.Conv2d,nn.Linear,nn.MultiheadAttention)):
                handles.append(module.register_forward_hook(count))
        model.eval()
        parameter = next(model.parameters())
        with torch.inference_mode():
            output = model(torch.zeros(shape,dtype=parameter.dtype))
        return {'parameters':sum(p.numel() for p in model.parameters()),
                'partial_macs':sum(counts.values()),'breakdown':counts,
                'input_shape':list(shape),'output_shape':list(output.shape),
                'excluded':EXCLUDED,'full_flops':None}
    finally:
        for handle in handles:
            handle.remove()
        for module,mode in modes.items():
            module.training = mode


def relative_difference(value, reference):
    if reference <= 0:
        raise ValueError('Positive reference required')
    return abs(value-reference)/reference


def match(candidate, reference):
    """Protocol-v2 thresholds relative to the fixed reference architecture."""
    p = relative_difference(candidate['parameters'],reference['parameters'])
    m = relative_difference(candidate['partial_macs'],reference['partial_macs'])
    return {'parameter_fraction':p,'partial_mac_fraction':m,
            'eligible_partial_budget':p <= .05 and m <= .10,
            'limitation':'Eligibility covers this partial operation count only; not full compute or training budget.'}
