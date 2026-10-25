from pathlib import Path
import sys
ROOT=next(p for p in (Path(__file__).resolve().parent,*Path(__file__).resolve().parents) if (p/'src/rssr').is_dir())
sys.path.insert(0,str(ROOT/'src'))
import pytest
import torch
from mp35_tools import *

def test_geographic_separation_and_leakage():
    scenes=[{'region_id':'a','split':'train','wgs84_bounds':[0,0,1,1]},{'region_id':'b','split':'test','wgs84_bounds':[3,3,4,4]}]
    check_splits(scenes)
    with pytest.raises(ValueError):check_splits([scenes[0],dict(scenes[1],region_id='a')])
    with pytest.raises(ValueError):check_splits([scenes[0],dict(scenes[1],wgs84_bounds=[.5,.5,1.5,1.5])])
