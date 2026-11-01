"""MP18-22 canonical model parity, no trained outputs or external data."""
import pytest
import torch
from rssr.cnn_labs import Reconstruction, UpsamplingCNN
from rssr.models import SRModel
from torch.nn import functional as F
from stage_05_cnn_sr.MP18_cnn_edge_detection_laboratory.mp18_tools import sobel_components
def test_mp18_fixed_sobel_and_boundary():
    a=torch.zeros(8,10);a[:,5:]=1
    g=sobel_components(a)
    assert g.shape==(2,8,10) and g[0,4,4]==pytest.approx(.5)
    assert sobel_components(torch.ones(8,10)).abs().max()==0
def test_mp19_mp21_direct_and_skip_zero_weights():
    a=torch.ones(1,3,8,10)
    for residual in (False,True):
        model=Reconstruction(residual=residual)
        for p in model.parameters():p.data.zero_()
        y=model(a)
        torch.testing.assert_close(y,a if residual else torch.zeros_like(a))
def test_mp20_srcnn_zero_weights_not_bicubic():
    a=torch.ones(1,3,8,10)
    model=SRModel(bands=3,kind="srcnn")
    for p in model.parameters():p.data.zero_()
    y=model(a)
    assert y.shape==(1,3,16,20) and torch.count_nonzero(y)==0
def test_mp22_three_architectures_residual_identity():
    a=torch.rand(1,3,8,10)
    target=F.interpolate(a,scale_factor=2,mode="bicubic",align_corners=False)
    for mode in ("shuffle","transpose","interpolate"):
        model=UpsamplingCNN(mode)
        for p in model.parameters():p.data.zero_()
        torch.testing.assert_close(model(a),target)
