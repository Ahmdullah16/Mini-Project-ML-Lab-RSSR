"""Numerical expectations for native Stage-08 synthetic worked examples."""
import sys
from pathlib import Path
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'app'))
from remote_sensing_labs import (raster,normalization,pairing,geography,feasibility,
    fit_standardization,georeferenced_pair,TRANSFORM,current_mp36_prose)
from rssr.learning_content import load_lesson


def test_coordinates_masks_and_unavailable():
    r = raster()
    assert r['center'] == [135,175] and r['corner'] == [130,180]
    assert [b['mean'] for b in r['statistics']] == [3,4]
    assert raster('include all')['statistics'][0]['mean'] == 27
    assert all(b['mean'] is None and b['valid_pixels']==0 for b in raster('all invalid')['statistics'])


def test_training_only_population_statistics_and_masks():
    r = normalization()
    assert np.isclose(r['training_statistics']['bands'][0]['std'],np.sqrt(8/3))
    assert np.allclose(np.asarray(r['normalized'])[0][np.asarray(r['mask'])[0]],[-np.sqrt(1.5),0,np.sqrt(1.5)])
    assert not r['normalized_mask'][0][1][0] and r['normalized'][0][1][0] == 0
    assert r['training_statistics']['bands'][1]['constant_band']
    assert r['validation_output'][1][0][0] == 2
    assert r['accepted_patches'] == []
    assert normalization(threshold=.75)['accepted_patches'][0]['valid_fraction'] == .75
    assert r['translated_patch_corner'] == [120,200]
    with pytest.raises(ValueError,match='training data only'):
        normalization('validation')
    with pytest.raises(ValueError,match='empty valid band'):
        fit_standardization(np.ones((1,2,2)),np.zeros((1,2,2),bool),split='train')


@pytest.mark.parametrize('scale,expected',[(2,[[2.5,4.5],[10.5,12.5]]),(4,[[7.5]])])
def test_pair_values_shape_footprint_and_repeat(scale,expected):
    r = pairing(scale)
    assert r == pairing(scale)
    assert r['lr'][0] == expected
    assert r['metadata']['lr_shape'] == [1,4//scale,4//scale]
    assert r['metadata']['hr_corners'] == r['metadata']['lr_corners']
    assert r['metadata']['lr_transform'][0] == 10*scale
    blurred = pairing(scale,.8)
    assert np.isfinite(blurred['lr']).all()
    assert blurred['metadata']['operator']['blur'].startswith('normalized Gaussian')


def test_pair_rejects_nodata_and_indivisible_shape():
    with pytest.raises(ValueError,match='fully valid HR'):
        pairing(invalid=True)
    with pytest.raises(ValueError,match='divisible'):
        georeferenced_pair(torch.ones(1,3,4),TRANSFORM,'EPSG:32611')


def test_angular_clearance_strict_boundary_and_region_reuse():
    assert geography()['audit']['cross_split_pairs'][0]['max_axis_clearance_degrees'] == 1
    with pytest.raises(ValueError):
        geography(1.)
    with pytest.raises(ValueError):
        geography(repeated_region=True)


def test_feasibility_hand_examples_are_not_sensor_scores():
    r = feasibility()
    assert [x['ratio'] for x in r['ratios']] == [2,4,10/1.5]
    assert r['offset_hr_pixels'] == 1 and feasibility(2.5)['offset_hr_pixels'] == .5
    assert np.allclose(r['residual'],[.03,.05]) and np.isclose(r['toy_mse'],.0017)
    assert r['toy_responses'] == [.2,.8,.5]
    with pytest.raises(ValueError):
        feasibility(-1)


def test_mp36_historical_prose_updated_only_for_site():
    lesson = load_lesson(ROOT,'MP36')
    for field in ('overview','explanations','validation'):
        revised = current_mp36_prose(lesson[field])
        assert 'leaving registration-method review open' not in revised
        assert 'official paper/correction detail is unavailable' not in revised
        assert 'Next: complete official registration/correction review' not in revised
        assert current_mp36_prose(revised) == revised
    assert 'version-specific' in current_mp36_prose(lesson['overview'])
