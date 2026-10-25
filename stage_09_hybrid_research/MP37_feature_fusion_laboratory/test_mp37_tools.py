from pathlib import Path
import sys
ROOT=next(p for p in (Path(__file__).resolve().parent,*Path(__file__).resolve().parents) if (p/'src/rssr').is_dir())
sys.path.insert(0,str(ROOT/'src'))
import pytest
import torch
from mp37_tools import *

def test_fusion_values_and_contract():
    a=torch.ones(1,2,3,4);b=3*a
    torch.testing.assert_close(fuse_features(a,b),2*a)
    c=fuse_features(a,b,'concat');assert c.shape==(1,4,3,4)
    torch.testing.assert_close(c[:,:2],a);torch.testing.assert_close(c[:,2:],b)
    with pytest.raises(ValueError):fuse_features(a,b[:,:,:2])
    with pytest.raises(ValueError):fuse_features(a,b,'unknown')
