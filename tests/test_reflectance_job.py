import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
import rasterio
from affine import Affine
from rssr.reflectance_job import inspect,run,sha256


def request_fixture(tmp_path):
    bands=[]
    def asset(name,data,nodata):
        path=tmp_path/(name+'.tif')
        with rasterio.open(path,'w',driver='GTiff',height=4,width=6,count=data.shape[0],dtype=str(data.dtype),
                           crs='EPSG:32643',transform=Affine(10,0,500000,0,-10,3000000),nodata=nodata) as ds:ds.write(data)
        return {'path':path.name,'sha256':sha256(path)}
    for name in ('B04','B08'):bands.append(asset(name,np.full((1,4,6),2000,np.uint16),0))
    scl=asset('scl',np.full((1,4,6),4,np.uint8),0)
    valid=asset('valid',np.ones((2,4,6),np.uint8),255)
    sat=asset('saturation',np.zeros((2,4,6),np.uint8),255)
    meta={'product_id':'synthetic','acquisition_time':'2026-10-28T00:00:00Z','processing_level':'L2A',
          'processing_baseline':'04.00','processor':'synthetic fixture','provider':'offline fixture','license':'test',
          'radiometry_source':'synthetic declared values','radiometry_review':'fixture arithmetic only',
          'mask_source':'synthetic flags','harmonization_status':'not_applied','input_encoding':'encoded_dn',
          'bands':[{'name':name,'units':'1','native_resolution_m':10,'center_wavelength_nm':wave,
                    'scale':.0001,'offset':-.1,'nodata':0,'source_asset':'synthetic:'+name,'resampling':'none'}
                   for name,wave in [('B04',665),('B08',842)]]}
    request={'window':[1,1,3,2],'band_assets':bands,'scl_asset':scl,'validity_asset':valid,
             'saturation_asset':sat,'metadata':meta}
    path=tmp_path/'request.json';path.write_text(json.dumps(request),encoding='utf-8');return path,request


def save(path,request):path.write_text(json.dumps(request),encoding='utf-8')


def test_plan_and_real_window_export(tmp_path):
    path,req=request_fixture(tmp_path)
    plan=inspect(path)[3]
    assert plan['output_shape']==[2,2,3] and not plan['checksums_verified']
    assert plan['output_grid']['transform']==[10,0,500010,0,-10,2999990]
    output=tmp_path/'result.tif';result=run(path,output)
    assert result['checksums_verified'] and result['output_sha256']==sha256(output)
    with rasterio.open(output) as ds:
        np.testing.assert_allclose(ds.read(),.1)
        assert ds.read().dtype==np.float32 and ds.dataset_mask().min()==255
        provenance=json.loads(ds.tags()['provenance'])
        assert len(provenance['local_input_assets'])==5 and provenance['requested_window']==[1,1,3,2]
    with pytest.raises(FileExistsError):run(path,output)


@pytest.mark.parametrize('window',[[0,0,1025,1],[-1,0,1,1],[5,3,2,2],[0,0,1.5,2]])
def test_invalid_window(tmp_path,window):
    path,req=request_fixture(tmp_path);req['window']=window;save(path,req)
    with pytest.raises(ValueError):inspect(path)


def test_checksum_failure_no_output(tmp_path):
    path,req=request_fixture(tmp_path);req['band_assets'][0]['sha256']='0'*64;save(path,req)
    with pytest.raises(ValueError,match='checksum'):run(path,tmp_path/'out.tif')
    assert not (tmp_path/'out.tif').exists()


def test_misaligned_quality_rejected(tmp_path):
    path,req=request_fixture(tmp_path)
    with rasterio.open(tmp_path/'scl.tif','r+') as ds:ds.transform=Affine(20,0,500000,0,-20,3000000)
    with pytest.raises(ValueError,match='same grid'):inspect(path)


def test_nonbinary_mask_rejected(tmp_path):
    path,req=request_fixture(tmp_path)
    with rasterio.open(tmp_path/'valid.tif','r+') as ds:ds.write(np.full((2,4,6),128,np.uint8))
    req['validity_asset']['sha256']=sha256(tmp_path/'valid.tif');save(path,req)
    with pytest.raises(ValueError,match='binary'):run(path,tmp_path/'out.tif')


def test_quality_mask_coverage_is_intersected(tmp_path):
    path,req=request_fixture(tmp_path)
    with rasterio.open(tmp_path/'saturation.tif','r+') as ds:
        mask=np.full((4,6),255,np.uint8);mask[1,1]=0;ds.write_mask(mask)
    req['saturation_asset']['sha256']=sha256(tmp_path/'saturation.tif');save(path,req)
    result=run(path,tmp_path/'out.tif')
    assert result['valid_fraction']==5/6


def test_cli_preflight_no_export_and_execute(tmp_path):
    path,req=request_fixture(tmp_path);output=tmp_path/'cli.tif'
    command=[sys.executable,'-m','rssr.cli','prepare-reflectance',str(path),str(output)]
    planned=subprocess.run(command,capture_output=True,text=True,check=True)
    assert json.loads(planned.stdout)['output_shape']==[2,2,3] and not output.exists()
    executed=subprocess.run(command+['--execute'],capture_output=True,text=True,check=True)
    assert 'completed local conversion' in executed.stdout and output.exists()


def test_external_mask_requires_pinning(tmp_path):
    path,req=request_fixture(tmp_path)
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=False):
        with rasterio.open(tmp_path/'saturation.tif','r+') as ds:ds.write_mask(np.full((4,6),255,np.uint8))
    with pytest.raises(ValueError,match='sidecars'):inspect(path)


def test_disk_guard(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import rssr.reflectance_job as job
    path,_=request_fixture(tmp_path)
    monkeypatch.setattr(job.shutil,'disk_usage',lambda _:SimpleNamespace(free=0))
    with pytest.raises(ValueError,match='disk space'):run(path,tmp_path/'out.tif')
    assert not (tmp_path/'out.tif').exists()
