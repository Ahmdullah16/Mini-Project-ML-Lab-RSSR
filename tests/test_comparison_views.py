import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rssr.comparison_views import comparison_views,crop_and_wipe

def fixture(w,h,value,dtype='uint8',pixel=10.,crs='EPSG:32632'):
 with rasterio.io.MemoryFile() as mem:
  with mem.open(driver='GTiff',width=w,height=h,count=3,dtype=dtype,transform=from_origin(100,200,pixel,pixel),crs=crs) as ds:
   ds.write(np.full((3,h,w),value,dtype=dtype))
  return mem.read()

def test_common_grid_and_zero_disagreement():
 v=comparison_views(fixture(5,3,51),fixture(10,6,.2,'float32',5),2)
 assert v['model'].shape==(6,10,3) and v['valid'].all()
 assert np.max(v['disagreement'])<1e-6
 a=crop_and_wipe(v,.5,.5,2,1)
 np.testing.assert_array_equal(a['wipe'],a['baseline'])
 b=crop_and_wipe(v,.5,.5,2,0)
 np.testing.assert_array_equal(b['wipe'],b['model'])
 assert a['preview_crop_xywh']==b['preview_crop_xywh']

@pytest.mark.parametrize('output',[fixture(9,6,.2,'float32',5),fixture(10,6,.2,'float32',6),fixture(10,6,.2,'float32',5,'EPSG:32633')])
def test_reject_misaligned_grid(output):
 with pytest.raises(ValueError):comparison_views(fixture(5,3,51),output,2)

def test_output_resource_limit_before_allocation():
 with pytest.raises(ValueError,match='allocation'):
  comparison_views(fixture(1024,1024,51),b'',64)

def test_preview_bound_and_same_spatial_crop():
 v=comparison_views(fixture(129,65,51),fixture(258,130,.3,'float32',5),2)
 assert max(v['valid'].shape)<=256
 c=crop_and_wipe(v,1,0,4,.5)
 assert c['model'].shape==c['baseline'].shape
 assert c['disagreement'].shape==c['model'].shape[:2]
 assert np.mean(c['disagreement'])==pytest.approx(.1,abs=1e-6)
