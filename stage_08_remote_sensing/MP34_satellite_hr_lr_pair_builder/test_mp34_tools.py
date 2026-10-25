from pathlib import Path
import sys
ROOT=next(p for p in (Path(__file__).resolve().parent,*Path(__file__).resolve().parents) if (p/'src/rssr').is_dir())
sys.path.insert(0,str(ROOT/'src'))
import pytest
import torch
from mp34_tools import *

def test_pair_shape_constant_and_rejection():
    hr=torch.ones(4,16,20);old=hr.clone();lr=make_pair(hr)
    assert lr.shape==(4,8,10)
    torch.testing.assert_close(lr,torch.ones_like(lr));torch.testing.assert_close(hr,old)
    with pytest.raises(ValueError):make_pair(torch.ones(3,15,16))
    with pytest.raises(ValueError):make_pair(hr,1)

def test_hand_average_and_rotated_extent_alignment():
    from affine import Affine
    x=torch.arange(16,dtype=torch.float32).reshape(1,4,4)
    transform=Affine(10,2,100,1,-10,200)
    lr,meta=georeferenced_pair(x,transform,'EPSG:32632',scale=2,sigma=0)
    torch.testing.assert_close(lr,torch.tensor([[[2.5,4.5],[10.5,12.5]]]))
    assert meta['hr_corners']==meta['lr_corners']
    assert meta['lr_transform']==[20,4,100,2,-20,200]
    lr4,meta4=georeferenced_pair(x,transform,'EPSG:32632',scale=4,sigma=0)
    assert lr4.item()==7.5 and meta4['lr_shape']==[1,1,1]

def test_nodata_crs_and_noise_seed_contract():
    from affine import Affine
    x=torch.ones(3,8,8);valid=torch.ones(8,8,dtype=torch.bool);valid[0,0]=False
    with pytest.raises(ValueError,match='fully valid'):georeferenced_pair(x,Affine.identity(),'EPSG:32632',valid=valid)
    with pytest.raises(ValueError):georeferenced_pair(x,Affine.identity(),None)
    with pytest.raises(ValueError):georeferenced_pair(x,Affine.scale(0),'EPSG:32632')
    a=make_pair(x,noise=.01,seed=17);b=make_pair(x,noise=.01,seed=17)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert not torch.equal(a,make_pair(x,noise=.01,seed=18))
