import numpy as np
import pytest
from rssr.inference_diagnostics import difference_summary

def test_seam_error_is_separate_from_background():
    x=np.zeros((3,8,8));y=x.copy();y[:,3:5,:]=.1;y[:,:,3:5]=.1
    result=difference_summary(x,y,tile=2,scale=2,radius=1)
    assert result['seam_spatial_pixels']==28
    assert result['seam_mae_normalized']==pytest.approx(.1)
    assert result['nonseam_mae_normalized']==0
    assert result['mae_normalized']==pytest.approx(.1*28/64)
    assert result['uint8_disagreement_fraction']==pytest.approx(28/64)

def test_clipping_and_missing_seams_are_explicit():
    x=np.array([[[-.1,1.1],[0,1]]]);r=difference_summary(x,x)
    assert r['mae_normalized']==0 and r['uint8_disagreement_fraction']==0
    assert r['tiled_out_of_range_fraction']==.5 and r['seam_mae_normalized'] is None
    with pytest.raises(ValueError):difference_summary(x,np.full_like(x,np.nan))
    with pytest.raises(ValueError):difference_summary(x,x,tile=0)
