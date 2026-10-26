import io,json,zipfile
import numpy as np
import pytest
import rasterio
import torch
from PIL import Image
from rssr.scaling import plan,cascade,job_bundle,cache_key,SCALES
from rssr.models import SRModel,tiled
from rssr.uploads import decode_image

@pytest.mark.parametrize('width,height',[(1,1),(7,5),(256,128),(1024,1024),(1024,17)])
@pytest.mark.parametrize('scale',SCALES)
def test_dimensions_and_resources(width,height,scale):
    p=plan(width,height,scale)
    assert (p['output_width'],p['output_height'])==(width*scale,height*scale)
    assert p['output_pixels']==width*height*scale**2
    assert p['estimated_disk_bytes']>p['output_rgb_bytes']
    assert p['host_available']==(width*height*scale**2<=1024**2)

@pytest.mark.parametrize('args',[(0,2,2),(1025,1,2),(1,1025,2),(2,2,3),(2.,2,2)])
def test_invalid_plan(args):
    with pytest.raises(ValueError):plan(*args)

@pytest.mark.parametrize('scale',SCALES)
def test_exact_disk_export_and_quantized_cascade(tmp_path,scale):
    model=SRModel(bands=3,kind='cnn',width=4,zero_residual=True).eval()
    a=np.broadcast_to(np.array([30,90,180],np.uint8),(3,5,3)).copy()
    result=cascade(a,model,tmp_path/'out',scale,{'fixture':True})
    with rasterio.open(tmp_path/'out'/result['output']) as ds:
        assert (ds.width,ds.height)==(5*scale,3*scale)
        assert np.array_equal(ds.read()[:,0,0],a[0,0])
    assert len(result['stage_outputs'])==int(np.log2(scale))
    with Image.open(tmp_path/'out'/'preview.png') as preview:assert max(preview.size)<=512
    with pytest.raises(ValueError,match='new output'):cascade(a,model,tmp_path/'out',scale,{})


def test_tile_seams_match_existing_pipeline_and_full_cnn(tmp_path):
    torch.manual_seed(9)
    model=SRModel(bands=3,kind='cnn',width=4,zero_residual=False).eval()
    a=np.random.default_rng(7).integers(0,256,(133,137,3),dtype=np.uint8)
    result=cascade(a,model,tmp_path/'out',2,{})
    with rasterio.open(tmp_path/'out'/result['output']) as ds:actual=ds.read()
    x=torch.from_numpy(a.transpose(2,0,1).astype(np.float32)/255)[None]
    with torch.inference_mode():
        reference=np.rint(tiled(model,x)[0].numpy().clip(0,1)*255).astype(np.uint8)
        full=np.rint(model(x)[0].numpy().clip(0,1)*255).astype(np.uint8)
    assert np.array_equal(actual,reference)
    assert np.max(np.abs(actual.astype(int)-full.astype(int)))<=1


def test_max_decode_bundle_and_identity():
    a=np.zeros((17,1024,3),np.uint8);buffer=io.BytesIO();Image.fromarray(a).save(buffer,format='PNG')
    assert decode_image(buffer.getvalue()).shape==a.shape
    spec={'version':'fixture','checkpoint':{'sha256':'a'*64}}
    with zipfile.ZipFile(io.BytesIO(job_bundle(a,spec,64))) as z:
        assert np.array_equal(decode_image(z.read('input.png')),a)
        assert json.loads(z.read('request.json'))['output_width']==65536
        assert json.loads(z.read('model.json'))==spec
    assert cache_key(a,spec,2)!=cache_key(a,spec,4)


def test_insufficient_disk_fails_before_directory(tmp_path,monkeypatch):
    import shutil
    monkeypatch.setattr(shutil,'disk_usage',lambda p:type('Usage',(),{'free':0})())
    model=SRModel(bands=3,kind='cnn',width=4).eval()
    with pytest.raises(ValueError,match='disk'):cascade(np.zeros((3,4,3),np.uint8),model,tmp_path/'out',2,{})
    assert not (tmp_path/'out').exists()
