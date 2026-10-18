import numpy as np
import rasterio
from rasterio.transform import from_origin
import torch
from rssr.models import SRModel
from rssr.inference import geotiff


def test_export_preserves_extent_and_mask(tmp_path):
    source=tmp_path/'source.tif'; target=tmp_path/'output.tif'
    transform=from_origin(100,200,10,10)
    x=np.ones((4,8,8),dtype=np.uint8)*100; x[:,0,0]=0
    with rasterio.open(source,'w',driver='GTiff',width=8,height=8,count=4,dtype='uint8',transform=transform,crs='EPSG:32632',nodata=0) as ds: ds.write(x)
    cfg={'model':{'bands':4,'width':8,'kind':'cnn','scale':2}}
    checkpoint=tmp_path/'model.pt'; torch.save({'model':SRModel(**cfg['model']).state_dict(),'config':cfg,'run_id':'fixture','provenance':'random test fixture'},checkpoint)
    geotiff(checkpoint,source,target,255)
    with rasterio.open(source) as a,rasterio.open(target) as b:
        assert a.bounds==b.bounds and a.crs==b.crs
        assert b.width==16 and b.height==16 and b.transform.a==5
        assert np.isnan(b.read()[:,:2,:2]).all()
