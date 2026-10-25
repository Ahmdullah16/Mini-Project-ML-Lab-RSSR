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

def test_masked_statistics_empty_and_nonfinite():
    import numpy as np
    x=np.array([[[1,3],[99,np.nan]],[[2,4],[99,8]]])
    result=masked_statistics(x,np.array([[True,True],[False,True]]))
    assert result[0]['mean']==2 and result[0]['valid_pixels']==2
    assert result[1]['mean']==pytest.approx(14/3)
    assert masked_statistics(x,np.zeros((2,2),bool))[0]['mean'] is None
    with pytest.raises(ValueError):masked_statistics(x,np.ones((3,3)))

def test_geotiff_metadata_window_and_nodata(tmp_path):
    import numpy as np
    import rasterio
    from affine import Affine
    from rasterio.windows import Window
    path=tmp_path/'fixture.tif';data=np.arange(24,dtype=np.uint16).reshape(2,3,4)
    with rasterio.open(path,'w',driver='GTiff',width=4,height=3,count=2,dtype='uint16',crs='EPSG:32632',transform=Affine(10,0,100,0,-10,200),nodata=0) as dst:dst.write(data)
    values,valid,meta=inspect_raster(path)
    assert values.shape==(2,3,4) and not valid[0,0,0] and valid[1,0,0]
    assert meta['first_pixel_center']==[105,195] and meta['crs']=='EPSG:32632'
    assert meta['statistics'][0]['mean']==6
    assert inspect_raster(path,Window(1,1,2,2))[2]['first_pixel_center']==[115,185]
    with pytest.raises(ValueError):inspect_raster(path,Window(0,0,257,1))
    with pytest.raises(ValueError):inspect_raster(path,Window(-1,0,1,1))
