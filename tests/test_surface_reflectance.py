import copy
import json
import numpy as np
import pytest
import rasterio
from rssr.surface_reflectance import prepare,export


def fixture():
    grid={'crs':'EPSG:32643','transform':[10,0,500000,0,-10,3000000],'shape':[3,5]}
    meta={'product_id':'synthetic-L2A-fixture','acquisition_time':'2026-10-28T00:00:00Z',
          'processing_level':'L2A','processing_baseline':'04.00','processor':'synthetic test only',
          'provider':'offline fixture','license':'test fixture','radiometry_source':'synthetic metadata',
          'radiometry_review':'synthetic offset arithmetic; not a real product review','mask_source':'synthetic SCL/quality',
          'harmonization_status':'not_applied','input_encoding':'encoded_dn',
          'grid':grid,'scl_grid':copy.deepcopy(grid),'quality_grid':copy.deepcopy(grid),
          'bands':[{'name':'B04','units':'1','native_resolution_m':10,'center_wavelength_nm':665,
                    'scale':.0001,'offset':-.1,'nodata':0,'source_asset':'synthetic','resampling':'none',
                    'grid':copy.deepcopy(grid)}]}
    return np.full((1,3,5),2000,dtype=np.uint16),np.full((3,5),4,dtype=np.uint8),np.ones((1,3,5),bool),np.zeros((1,3,5),bool),meta


def test_offset_once_no_clipping_and_source_unchanged():
    x,scl,valid,sat,meta=fixture();x[0,0,:3]=[500,2000,15000]
    out,mask,record=prepare(x,scl,valid,sat,meta)
    np.testing.assert_allclose(out[0,0,:3],[-.05,.1,1.4],atol=1e-7)
    assert out.dtype==np.float32 and mask.all() and 'conversion_applied' not in meta
    with pytest.raises(ValueError,match='already applied'):prepare(out,scl,valid,sat,record)


@pytest.mark.parametrize('code',range(12))
def test_scl_policy(code):
    x,scl,valid,sat,meta=fixture();scl[0,0]=code
    out,mask,_=prepare(x,scl,valid,sat,meta)
    assert bool(mask[0,0])==(code in (4,5,6))
    assert np.isfinite(out[0,0,0]) if code in (4,5,6) else np.isnan(out[0,0,0])


def test_masks_nodata_and_no_valid():
    x,scl,valid,sat,meta=fixture();valid[0,0,0]=False;sat[0,0,1]=True;x[0,0,2]=0
    out,mask,_=prepare(x,scl,valid,sat,meta)
    assert not mask[0,:3].any() and np.isnan(out[0,0,:3]).all()
    scl[:]=9
    with pytest.raises(ValueError,match='No valid'):prepare(x,scl,valid,sat,meta)


@pytest.mark.parametrize('field,value',[('processing_level','L1C'),('harmonization_status','unknown'),('processor',''),('input_encoding','display_rgb')])
def test_missing_or_unsupported_metadata(field,value):
    args=list(fixture());args[-1][field]=value
    with pytest.raises(ValueError):prepare(*args)


def test_grid_mismatch_and_unknown_classes():
    args=list(fixture());args[-1]['bands'][0]['grid']['transform'][0]=20
    with pytest.raises(ValueError,match='grid mismatch'):prepare(*args)
    args=list(fixture());args[1][0,0]=12
    with pytest.raises(ValueError,match='SCL'):prepare(*args)


def test_decoded_reflectance_requires_identity():
    x,scl,valid,sat,meta=fixture();x=x.astype(np.float32)*.0001-.1
    meta['input_encoding']='surface_reflectance'
    with pytest.raises(ValueError,match='identity'):prepare(x,scl,valid,sat,meta)
    meta['bands'][0].update(scale=1,offset=0,nodata=-9999)
    out,_,_=prepare(x,scl,valid,sat,meta);np.testing.assert_array_equal(x,out)


def test_round_trip_atomic_float_export(tmp_path):
    x,scl,valid,sat,meta=fixture();scl[0,0]=9
    out,mask,record=prepare(x,scl,valid,sat,meta)
    path=export(tmp_path/'scientific.tif',out,mask,record)
    with rasterio.open(path) as ds:
        np.testing.assert_allclose(ds.read(),out,equal_nan=True)
        np.testing.assert_array_equal(ds.dataset_mask()>0,mask)
        assert ds.crs.to_epsg()==32643 and ds.transform.a==10 and ds.transform.e==-10
        assert ds.dtypes==('float32',) and json.loads(ds.tags()['provenance'])==record
        assert ds.scales==(1.,) and ds.offsets==(0.,)
    before=path.read_bytes()
    with pytest.raises(FileExistsError):export(path,out,mask,record)
    assert path.read_bytes()==before and len(list(tmp_path.iterdir()))==1

@pytest.mark.parametrize('shape',[(1,1,7),(1,1024,1024)])
def test_bounded_extreme_shapes(shape):
    _,_,_,_,meta=fixture()
    grid=meta['grid'];grid['shape']=list(shape[1:])
    meta['scl_grid']=copy.deepcopy(grid);meta['quality_grid']=copy.deepcopy(grid)
    meta['bands'][0]['grid']=copy.deepcopy(grid)
    out,valid,_=prepare(np.full(shape,2000,np.uint16),np.full(shape[1:],4,np.uint8),
                        np.ones(shape,bool),np.zeros(shape,bool),meta)
    assert out.shape==shape and valid.all() and np.isfinite(out).all()


def test_oversize_input_rejected():
    x,scl,valid,sat,meta=fixture()
    with pytest.raises(ValueError,match='spatial'):
        prepare(np.zeros((1,1,1025),np.uint16),scl,valid,sat,meta)
