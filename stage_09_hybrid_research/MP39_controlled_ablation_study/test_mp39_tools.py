from pathlib import Path
import sys
ROOT=next(p for p in (Path(__file__).resolve().parent,*Path(__file__).resolve().parents) if (p/'src/rssr').is_dir())
sys.path.insert(0,str(ROOT/'src'))
import pytest
import torch
from mp39_tools import *

def test_paired_constant_effect_and_invalid_pairs():
    mean,ci=paired_scene_ci([2,3,4],[1,2,3],seed=39)
    assert mean==1 and ci.tolist()==[1,1]
    with pytest.raises(ValueError):paired_scene_ci([1],[0])
    with pytest.raises(ValueError):paired_scene_ci([1,2],[1])
