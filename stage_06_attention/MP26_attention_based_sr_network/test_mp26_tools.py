from pathlib import Path
import sys
ROOT=next(p for p in (Path(__file__).resolve().parent,*Path(__file__).resolve().parents) if (p/'src/rssr').is_dir())
sys.path.insert(0,str(ROOT/'src'))
import pytest
import torch
from mp26_tools import SRModel

def test_model_shape_and_declared_zero_path():
    torch.manual_seed(26);model=SRModel(bands=3,width=8,kind='attention').eval()
    for parameter in model.parameters():parameter.data.zero_()
    x=torch.rand(1,3,7,9);y=model(x)
    assert y.shape==(1,3,14,18) and torch.isfinite(y).all()
    expected=torch.zeros_like(y) if 'attention'=='srcnn' else torch.nn.functional.interpolate(x,scale_factor=2,mode='bicubic',align_corners=False)
    torch.testing.assert_close(y,expected)
    with pytest.raises(ValueError):model(torch.ones(1,4,7,9))
