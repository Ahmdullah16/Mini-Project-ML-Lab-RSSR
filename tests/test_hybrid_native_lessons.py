"""Canonical Stage-09 algebra, shapes and historical evidence isolation."""
import sys
import json
from pathlib import Path
import numpy as np
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'app'))
from hybrid_labs import fusion,hybrid,ablation,robustness,fuse_features


@pytest.mark.parametrize('mode,output,gradient',[('mean',[2,3],.5),('sum',[4,6],1),('concat',[1,2,3,4],1)])
def test_fusion_and_projection(mode,output,gradient):
    r=fusion(mode)
    assert r['output']==output
    assert r['mean_projection']==[2,3]
    assert r['gradient_a']==[gradient]*2 and r['gradient_b']==[gradient]*2
    assert r['cost']=={'parameters':36,'macs':960}
    with pytest.raises(ValueError):fuse_features(torch.ones(1,2,1,1),torch.ones(1,1,1,1))


@pytest.mark.parametrize('bands',[3,4])
@pytest.mark.parametrize('gate',[True,False])
def test_hybrid_zero_residual_and_rng(bands,gate):
    state=torch.random.get_rng_state().clone()
    r=hybrid(bands,gate)
    assert torch.equal(state,torch.random.get_rng_state())
    assert r['output_shape']==[1,bands,10,14]
    assert r['difference_from_bicubic']==0
    assert ('gate' in r['trace'])==gate
    assert np.isclose(r['scalar_gate_example'],.45)


def test_ablation_historical_and_toy_ci():
    r=ablation()
    assert r['source_runs']==24 and len(r['historical'])==8
    assert all(x['geographic_ci'] is None for x in r['historical'])
    assert r['toy_mean']==1 and r['toy_interval95']==[1,1]
    with pytest.raises(ValueError,match='At least two'):ablation(units=1)


@pytest.mark.parametrize('bands',[3,4])
@pytest.mark.parametrize('axis',['blur','noise'])
def test_robustness_does_not_mix_protocols(bands,axis):
    r=robustness(bands=bands,axis=axis)
    assert r['rows'] and r['efficiency']
    assert all(x['bands']==bands for x in r['rows']+r['efficiency'])
    for x in r['rows']:
        assert (x['noise_std']==0 if axis=='blur' else x['sigma_hr_pixels']==.8)
        assert np.isclose(x['mean_delta_bicubic_db'],np.mean(x['paired_scene_deltas_db']))
    assert r['toy_effect_db']==[2,-1]


def test_historical_files_unchanged_and_missing_rejected(tmp_path):
    paths=[ROOT/'reports/ablation-pilot-runs.json',ROOT/'reports/development-benchmark.json']
    before=[p.read_bytes() for p in paths]
    ablation();robustness()
    assert before==[p.read_bytes() for p in paths]
    with pytest.raises(FileNotFoundError):ablation(tmp_path)
    with pytest.raises(ValueError):robustness(bands=5)


def test_rejects_substituted_pilot_and_empty_protocol(tmp_path):
    (tmp_path/'reports').mkdir()
    (tmp_path/'reports/ablation-pilot-runs.json').write_text('[]',encoding='utf-8')
    with pytest.raises(ValueError,match='24-run'):ablation(tmp_path)
    (tmp_path/'reports/development-benchmark.json').write_text(json.dumps({'robustness':[]}),encoding='utf-8')
    with pytest.raises(ValueError,match='No recorded conditions'):robustness(tmp_path)
