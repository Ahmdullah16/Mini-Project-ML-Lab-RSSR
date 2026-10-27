from pathlib import Path
import sys
ROOT=next(p for p in (Path(__file__).resolve().parent,*Path(__file__).resolve().parents) if (p/'src/rssr').is_dir())
sys.path.insert(0,str(ROOT/'src'))
import pytest
import torch
from mp39_tools import *
import copy


def rows():
    return [dict(variant='cnn',seed=s,psnr=float(i+1),bicubic_psnr=2.,
        parameters=10,manifest_sha256='fixture',run_id=str(s)) for i,s in enumerate((17,23,41))]


def test_seed_statistics_not_geographic_interval():
    result=summarize_pilot(rows())[0]
    assert result['mean_psnr_db']==2 and result['sample_seed_sd_db']==1
    assert result['mean_delta_bicubic_db']==0 and result['geographic_ci'] is None


@pytest.mark.parametrize('field,value',[('seed',17),('manifest_sha256','different'),
    ('parameters',11),('psnr',float('nan')),('run_id','17')])
def test_invalid_pilot_pooling(field,value):
    records=copy.deepcopy(rows());records[1][field]=value
    with pytest.raises(ValueError):summarize_pilot(records)

def test_paired_constant_effect_and_invalid_pairs():
    mean,ci=paired_scene_ci([2,3,4],[1,2,3],seed=39)
    assert mean==1 and ci.tolist()==[1,1]
    with pytest.raises(ValueError):paired_scene_ci([1],[0])
    with pytest.raises(ValueError):paired_scene_ci([1,2],[1])
