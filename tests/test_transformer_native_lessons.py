"""Independent numerical expectations for the native Transformer worked examples."""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'app'))
from transformer_labs import single, heads, encoder, windows, baseline, attention, positions, patchify


def test_attention_hand_mask_scale_and_memory():
    r = single()
    p = np.exp(1/np.sqrt(2))/(np.exp(1/np.sqrt(2))+1)
    assert np.allclose(r['probability'], [[p,1-p]])
    assert np.allclose(r['output'], [[10*p+20*(1-p)]])
    assert single(masked=True)['output'] == [[10.]]
    assert single(100.)['probability'] == [[1.,0.]]
    assert r['memory'][2]['float32_score_bytes'] == 4*1024**2
    with pytest.raises(ValueError):
        attention([[1,0]], np.eye(2), [[10],[20]], np.zeros((1,2),dtype=bool))
    with pytest.raises(ValueError):
        attention([[1e308]], [[1e308]], [[1.]])


def test_heads_fixed_projections_hand_and_invalid_partition():
    p = np.exp(1)/(np.exp(1)+1)
    reference = heads(1)
    for n in (1,2,4):
        r = heads(n)
        assert r['input'] == reference['input']
        assert r['projections'] == reference['projections']
        assert np.asarray(r['maps']).shape == (n,16,16)
        assert np.allclose(np.asarray(r['maps']).sum(-1),1)
        assert np.allclose(r['hand'], [[p,.5],[.5,p]])
        assert r['parameters'] == 64 and r['map_bytes'] == n*16*16*8
    with pytest.raises(ValueError):
        heads(3)


def test_encoder_examples_trace_determinism_and_rng_isolation():
    rng = torch.random.get_rng_state().clone()
    r = encoder()
    assert torch.equal(rng, torch.random.get_rng_state())
    assert r == encoder()
    assert np.allclose(r['layernorm_example'], np.array([-1,1])/np.sqrt(1+1e-5))
    assert r['positions'][0] == [0,1,0,1]
    assert r['trace']['ffn_hidden'] == [1,16,8]
    assert r['trace']['output'] == [1,16,4]
    assert not np.allclose(r['output'], encoder(positional=True)['output'])
    for n in (1,2,4):
        assert np.isfinite(encoder(n)['output']).all()
    with pytest.raises(ValueError):
        positions(4,3)


def test_window_hand_example_clipped_paths_and_roundtrip():
    r = windows()
    assert r['tokens'][:2] == [[0,1,4,5],[2,3,6,7]]
    assert r['pairs'] == dict(global_pairs=256,local=64,shifted=36,two_layers=144)
    assert r['maps']['shifted'][0][-1] == 0
    for size in (1,2,4):
        assert windows(size)['exact_roundtrip']
    assert r['memory'][0]['local_float32_bytes'] == 256
    with pytest.raises(ValueError):
        patchify(np.ones((3,4,1)),2)


@pytest.mark.parametrize('shape', [(1,1),(5,7),(16,16)])
def test_baseline_readonly_hashes_and_exact_shape(shape):
    paths = [ROOT/'reports/submission-study.json', ROOT/'reports/development-benchmark.json']
    before = [p.read_bytes() for p in paths]
    r = baseline(height=shape[0], width=shape[1])
    assert r['output_shape'] == [1,4,2*shape[0],2*shape[1]]
    assert r['max_difference_from_bicubic'] == 0
    assert before == [p.read_bytes() for p in paths]
    for p, data in zip(paths,before):
        assert r['historical']['source_hashes'][p.relative_to(ROOT).as_posix()] == hashlib.sha256(data).hexdigest()
    assert len(r['historical']['rows']) == 6
    summary = r['historical']['summary']
    assert summary[0]['mean_delta_db'] > summary[1]['mean_delta_db']


def test_baseline_failure_bounds_and_missing_evidence(tmp_path):
    with pytest.raises(ValueError):
        baseline(height=17)
    with pytest.raises(FileNotFoundError):
        baseline(tmp_path)
