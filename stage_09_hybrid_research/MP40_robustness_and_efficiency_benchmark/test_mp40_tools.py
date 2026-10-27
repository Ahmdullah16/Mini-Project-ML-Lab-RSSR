from pathlib import Path
import sys
ROOT=next(p for p in (Path(__file__).resolve().parent,*Path(__file__).resolve().parents) if (p/'src/rssr').is_dir())
sys.path.insert(0,str(ROOT/'src'))
import pytest
import torch
from mp40_tools import *
import copy


def fixture():
    return dict(bands=4,variant='fixture',seed=17,sigma_hr_pixels=.8,noise_std=0,
        scenes=[dict(scene_id='a',scores={'model':{'psnr':3},'bicubic':{'psnr':2}}),
                dict(scene_id='b',scores={'model':{'psnr':1},'bicubic':{'psnr':3}})])


def test_pair_first_then_average():
    result=summarize_robustness([fixture()])[0]
    assert result['paired_scene_deltas_db']==[1,-2]
    assert result['mean_delta_bicubic_db']==-.5


def test_band_groups_remain_separate():
    a=fixture();b=copy.deepcopy(a);b['bands']=3
    assert len(summarize_robustness([a,b]))==2
    with pytest.raises(ValueError):summarize_robustness([a,a])


def test_duplicate_scene_rejected():
    a=fixture();a['scenes'][1]['scene_id']='a'
    with pytest.raises(ValueError):summarize_robustness([a])


def test_nonfinite_score_rejected():
    a=fixture();a['scenes'][0]['scores']['model']['psnr']=float('nan')
    with pytest.raises(ValueError):summarize_robustness([a])

def test_timer_counts_and_restores_mode():
    class Counter(torch.nn.Module):
        def __init__(self):super().__init__();self.calls=0
        def forward(self,x):self.calls+=1;return x+1
    model=Counter();r=benchmark_cpu(model,torch.zeros(1),warmups=2,repeats=3)
    assert model.calls==5 and model.training and len(r['seconds'])==3 and r['median_seconds']>=0
    with pytest.raises(ValueError):benchmark_cpu(model,torch.zeros(1),repeats=0)
