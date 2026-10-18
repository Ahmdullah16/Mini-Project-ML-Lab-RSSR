import sqlite3
import numpy as np
import pytest
import torch
from rssr.core import degrade,metrics,resize
from rssr.models import SRModel,tiled
from rssr.data import check_splits
from rssr.db import connect
from rssr.ingest import refresh

torch.set_num_threads(2)

def test_degradation_constant_and_seed():
    x=torch.ones(4,32,32)
    assert torch.allclose(degrade(x),torch.ones(4,16,16))
    assert torch.equal(degrade(x,noise=.1,seed=2),degrade(x,noise=.1,seed=2))
    assert not torch.equal(degrade(x,noise=.1,seed=2),degrade(x,noise=.1,seed=3))
    with pytest.raises(ValueError): degrade(torch.ones(3,13,14))

def test_metric_known_errors_and_masks():
    x=torch.ones(3,12,12)
    m=metrics(x,x)
    assert m['mse']==0 and m['psnr']==float('inf') and m['ssim']==pytest.approx(1)
    assert metrics(x*.5,x)['mse']==pytest.approx(.25)
    assert metrics(x*.5,x)['psnr']==pytest.approx(6.0205999)
    assert metrics(torch.zeros_like(x),x)['sam_rad'] is None
    with pytest.raises(ValueError): metrics(x,x,mask=np.zeros((12,12),bool))
    with pytest.raises(ValueError): metrics(x,x,border=6)
    with pytest.raises(ValueError): metrics(x.long(),x)
    mask=np.ones((12,12),bool); mask[0]=False
    y=x.clone(); y[:,0]=100
    assert metrics(y,x,mask=mask)['mse']==0

@pytest.mark.parametrize('kind',['srcnn','cnn','attention','transformer','hybrid'])
@pytest.mark.parametrize('scale',[2,4])
def test_shapes_and_gradients(kind,scale):
    model=SRModel(bands=4,scale=scale,width=8,kind=kind,gate=kind=='hybrid')
    x=torch.rand(2,4,9,13,requires_grad=True); y=model(x)
    assert y.shape==(2,4,9*scale,13*scale)
    y.square().mean().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum()>0

def test_tiling_cnn_boundaries():
    model=SRModel(width=8,kind='cnn').eval(); x=torch.rand(1,3,33,41)
    with torch.inference_mode(): expected=model(x)
    actual=tiled(model,x,tile=16,halo=16)
    assert torch.allclose(actual,expected,atol=1e-6)

def test_geographic_leakage():
    scenes=[{'region_id':'a','split':'train','wgs84_bounds':[0,0,1,1]},{'region_id':'b','split':'test','wgs84_bounds':[2,2,3,3]}]
    check_splits(scenes)
    scenes[1]['region_id']='a'
    with pytest.raises(ValueError): check_splits(scenes)
    scenes[1].update(region_id='b',wgs84_bounds=[1.01,0,2,1])
    with pytest.raises(ValueError): check_splits(scenes)

def test_schema_foreign_keys_and_migration(tmp_path):
    db=connect(tmp_path/'db.sqlite'); db.close(); db=connect(tmp_path/'db.sqlite')
    assert db.execute('SELECT count(*) FROM schema_versions').fetchone()[0]==1
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("INSERT INTO pair_sources VALUES('absent','absent')")
    assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'

def test_ingestion_idempotence_and_failure(tmp_path):
    config={'endpoint':'https://example.invalid','max_pages':1,'limit':1}
    item={'id':'1','collection':'s2','properties':{'datetime':'2024-01-01T00:00:00Z'},'bbox':[1,2,3,4],'assets':{}}
    fetch=lambda *args:{'features':[item],'links':[]}
    assert refresh(config,tmp_path,fetch)['items_added']==1
    assert refresh(config,tmp_path,fetch)['items_added']==0
    def fail(*args): raise OSError('offline')
    with pytest.raises(OSError): refresh(config,tmp_path,fail)
    db=connect(tmp_path/'catalog.sqlite')
    assert db.execute("SELECT count(*) FROM ingestion_jobs WHERE status='failed'").fetchone()[0]==1
