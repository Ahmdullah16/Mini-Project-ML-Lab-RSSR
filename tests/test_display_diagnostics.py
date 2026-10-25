import numpy as np
import torch
import pytest
from rssr.display import predict_rgb
from rssr.models import SRModel

def test_rgb_order_scaling_and_clipping_metadata():
    model=SRModel(bands=3,kind='cnn',width=4,zero_residual=True).eval()
    image=np.broadcast_to(np.array([10,80,200],np.uint8),(5,7,3)).copy()
    base,sr,meta=predict_rgb(image,model,{'run_id':'fixture','provenance':'synthetic test'},'a'*64)
    assert np.array_equal(sr,np.broadcast_to(image[0,0],(10,14,3)))
    assert np.array_equal(base,sr)
    assert meta['input_channel_means_uint8']==[10,80,200]
    assert meta['display_clipped_scalar_fraction']==0
    assert meta['raw_output_range_normalized']==pytest.approx([10/255,200/255])
    with torch.no_grad():model.tail[0].bias.fill_(2)
    _,sr,meta=predict_rgb(image,model,{'run_id':'fixture','provenance':'synthetic test'},'a'*64)
    assert (sr==255).all() and meta['display_clipped_scalar_fraction']==1

def test_nonfinite_model_output_rejected():
    model=SRModel(bands=3,kind='cnn',width=4,zero_residual=True).eval()
    with torch.no_grad():model.tail[0].bias.fill_(float('nan'))
    with pytest.raises(ValueError,match='nonfinite'):
        predict_rgb(np.zeros((3,4,3),np.uint8),model,{'run_id':'fixture','provenance':'test'},'a'*64)
