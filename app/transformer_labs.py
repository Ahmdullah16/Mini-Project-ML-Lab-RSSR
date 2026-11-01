"""Bounded native lesson computations; canonical operators, no datasets or training."""
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from stage_07_transformers.MP27_self_attention_from_scratch.attention_tools import attention, entropy, score_bytes
from stage_07_transformers.MP28_multi_head_self_attention.multihead_tools import multihead, projection_parameters
from stage_07_transformers.MP29_transformer_encoder_block.encoder_tools import Encoder, positions
from stage_07_transformers.MP30_image_patches_and_window_attention.window_tools import patchify, unpatchify, connectivity, score_counts
from stage_07_transformers.MP31_transformer_based_sr_baseline.benchmark_tools import comparison
from rssr.models import SRModel


def single(scale=1., masked=False):
    if scale not in (.1, 1., 10., 100.):
        raise ValueError('Choose a bounded canonical feature multiplier')
    q, k, v = np.array([[1., 0.]]), np.eye(2), np.array([[10.], [20.]])
    mask = np.array([[True, False]]) if masked else None
    output, probability = attention(q*scale, k*scale, v, mask)
    x = np.random.default_rng(27).normal(size=(64, 3))
    rows = []
    for c in (.1, 1., 10., 100.):
        for scaled in (False, True):
            _, a = attention(x*c, x*c, x, scaled=scaled)
            rows.append(dict(multiplier=c, sqrt_divisor=scaled,
                             entropy_nats=float(entropy(a).mean())))
    return dict(input_Q=q.tolist(), input_K=k.tolist(), input_V=v.tolist(),
                output=output.tolist(), probability=probability.tolist(), scale_rows=rows,
                memory=[dict(tokens=n, float32_score_bytes=score_bytes(n)) for n in (16, 64, 1024, 4096)])


def heads(count=2):
    # Reproduce MP28's synthetic fixture and fixed matrices, including RNG draw order.
    rng = np.random.default_rng(28)
    rng.normal(size=(3, 4))  # canonical unused RGB embedding draw
    x = rng.normal(size=(16, 4))
    projections = [rng.normal(size=(4, 4))/2 for _ in range(4)]
    out, maps = multihead(x, projections, count)
    hand, hand_maps = multihead(np.eye(2), [np.eye(2)]*4, 2)
    return dict(input=x.tolist(), projections=[p.tolist() for p in projections],
                output=out.tolist(), maps=maps.tolist(), hand=hand.tolist(), hand_maps=hand_maps.tolist(),
                parameters=projection_parameters(4), head_width=4//count, map_bytes=maps.nbytes)


def encoder(count=2, positional=False):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(29)
        torch.randn(3, 4, dtype=torch.float64)  # canonical embedding draw
        x = torch.randn(1, 16, 4, dtype=torch.float64)
        torch.manual_seed(29)
        model = Encoder(4, count).double().eval()
        pe = positions(16, 4)
        actual = x+pe if positional else x
        with torch.inference_mode():
            out, trace = model(actual)
            ln = F.layer_norm(torch.tensor([1., 3.], dtype=torch.float64), (2,), eps=1e-5)
    return dict(input=x[0].tolist(), actual_input=actual[0].tolist(), output=out[0].tolist(),
                positions=pe.tolist(), trace=trace, layernorm_example=ln.tolist(),
                parameters=sum(p.numel() for p in model.parameters()),
                residual_change_rms=float((out-actual).square().mean().sqrt()))


def windows(size=2, shift=1):
    image = np.arange(16).reshape(4, 4, 1)
    tokens = patchify(image, size)
    restored = unpatchify(tokens, image.shape, size)
    local = connectivity(4, 4, 2)
    shifted = connectivity(4, 4, 2, shift)
    reached = (shifted.astype(int) @ local.astype(int)) > 0
    return dict(input=image[:, :, 0].tolist(), tokens=tokens.tolist(), restored=restored[:, :, 0].tolist(),
                exact_roundtrip=bool(np.array_equal(image, restored)),
                maps={k:a.astype(int).tolist() for k,a in [('local',local), ('shifted',shifted), ('two_layers',reached)]},
                pairs=dict(global_pairs=256, local=int(local.sum()), shifted=int(shifted.sum()), two_layers=int(reached.sum())),
                memory=[dict(side=n, **score_counts(n, n, 2)) for n in (4, 8, 16, 32, 64, 128)])


def baseline(root=ROOT, height=5, width=7):
    if any(isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= 16 for v in (height, width)):
        raise ValueError('Synthetic demonstration sides must be integers in [1,16]')
    report = comparison(root)  # read-only reconciliation and source hashes
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(31)
        model = SRModel(bands=4, width=16, kind='transformer', scale=2, zero_residual=True).eval()
        x = torch.arange(4*height*width, dtype=torch.float32).reshape(1,4,height,width)/(4*height*width)
        with torch.inference_mode():
            out = model(x)
            skip = F.interpolate(x, scale_factor=2, mode='bicubic', align_corners=False)
    return dict(historical=report, input=x[0].tolist(), output=out[0].tolist(),
                input_shape=list(x.shape), output_shape=list(out.shape),
                max_difference_from_bicubic=float((out-skip).abs().max()))
