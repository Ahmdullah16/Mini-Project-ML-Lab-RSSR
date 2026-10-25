from pathlib import Path
import sys
ROOT=next(p for p in (Path(__file__).resolve().parent,*Path(__file__).resolve().parents) if (p/'src/rssr').is_dir())
sys.path.insert(0,str(ROOT/'src'))
import pytest
import torch
from mp33_tools import *

def test_training_only_stats_mask_and_constant_band():
    import numpy as np
    x=np.array([[[1.,3.],[99.,5.]],[[2.,2.],[99.,2.]]]);mask=np.ones_like(x,dtype=bool);mask[:,1,0]=False
    stats=fit_standardization(x,mask,split='train')
    assert stats['bands'][0]['mean']==3
    assert stats['bands'][0]['std']==pytest.approx(np.sqrt(8/3))
    out,valid=standardize(x,mask,stats)
    assert out.dtype==np.float32 and out[:,1,0].tolist()==[0,0] and not valid[:,1,0].any()
    assert out[0][valid[0]].mean()==pytest.approx(0,abs=1e-6)
    assert stats['bands'][1]['constant_band'] and np.all(out[1]==0)
    shifted,_=standardize(x+10,mask,stats)
    assert shifted[0][valid[0]].mean()>0 # apply frozen train stats, never refit
    with pytest.raises(ValueError):fit_standardization(x,mask,split='validation')
    with pytest.raises(ValueError):fit_standardization(x,np.zeros_like(mask),split='train')

def test_band_order_and_patch_georeferencing():
    import numpy as np
    from affine import Affine
    x=np.arange(2*5*5).reshape(2,5,5);mask=np.ones_like(x,dtype=bool);mask[:,0,0]=False
    selected,valid=select_valid_bands(x,mask,[1,0]);assert selected[0,1,1]==x[1,1,1]
    transform=Affine(10,0,100,0,-10,200)
    patches,report=extract_patches(selected,valid,transform,size=2)
    assert len(patches)==3 and report=={'rejected_full_patches':1,'bottom_rows_dropped':1,'right_columns_dropped':1}
    assert tuple(patches[0]['transform'])[:6]==(10,0,120,0,-10,200)
    assert np.array_equal(patches[0]['data'],selected[:,:2,2:4])
    with pytest.raises(ValueError):select_valid_bands(x,mask,[0,0])
    with pytest.raises(ValueError):extract_patches(x,mask,transform,size=0)

def test_nodata_mask_and_empty_rejection():
    mean,mask=valid_mean([1,2,-9999,float('nan')])
    assert mean==1.5 and mask.tolist()==[True,True,False,False]
    with pytest.raises(ValueError):valid_mean([-9999,float('nan')])
