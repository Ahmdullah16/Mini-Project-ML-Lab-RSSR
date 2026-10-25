from pathlib import Path
import sys
ROOT=next(p for p in (Path(__file__).resolve().parent,*Path(__file__).resolve().parents) if (p/'src/rssr').is_dir())
sys.path.insert(0,str(ROOT/'src'))
import pytest
import torch
from mp32_tools import *

def test_affine_coordinates_and_centres():
    from affine import Affine
    a=Affine(10,0,100,0,-10,200)
    assert pixel_to_map(a,0,0)==(105,195)
    assert pixel_to_map(a,2,3,False)==(130,180)
    assert pixel_to_map(a*Affine.scale(.5),0,0)==(102.5,197.5)
    with pytest.raises(ValueError):pixel_to_map(a,float('nan'),0)
