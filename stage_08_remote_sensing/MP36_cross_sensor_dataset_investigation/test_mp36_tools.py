from pathlib import Path
import sys
ROOT=next(p for p in (Path(__file__).resolve().parent,*Path(__file__).resolve().parents) if (p/'src/rssr').is_dir())
sys.path.insert(0,str(ROOT/'src'))
import pytest
import torch
from mp36_tools import *

def test_ratios_do_not_assume_integer_scale():
    assert resolution_ratio(10,5)==2
    assert resolution_ratio(20,5)==4
    assert resolution_ratio(10,1.5)==pytest.approx(20/3)
    for value in (0,-1,float('nan')):
        with pytest.raises(ValueError):resolution_ratio(10,value)
