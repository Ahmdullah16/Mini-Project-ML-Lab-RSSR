import io,urllib.request,urllib.error
import pytest
from rssr.cog import bounded_cog
from rssr.db import connect


def test_bounded_cog_ranges_and_budget(monkeypatch):
    real=urllib.request.urlopen
    content=b'0123456789'*100
    class Response(io.BytesIO):
        def __init__(self,data,status,headers): super().__init__(data); self.status=status; self.headers=headers
    def fake(request,**kwargs):
        if request.full_url.startswith('http://127.0.0.1'): return real(request,**kwargs)
        if request.get_method()=='HEAD': return Response(b'',200,{'Content-Length':'1000','ETag':'test'})
        start,end=map(int,request.get_header('Range').split('=')[1].split('-'))
        return Response(content[start:end+1],206,{'Content-Range':f'bytes {start}-{end}/1000'})
    monkeypatch.setattr(urllib.request,'urlopen',fake)
    with bounded_cog('https://sentinel-cogs.s3.us-west-2.amazonaws.com/test.tif',max_bytes=20) as (url,ledger):
        with real(urllib.request.Request(url,headers={'Range':'bytes=5-14'})) as response:
            assert response.read()==content[5:15]
        with pytest.raises(urllib.error.HTTPError): real(urllib.request.Request(url,headers={'Range':'bytes=0-19'}))
        assert ledger['bytes_received']==10 and ledger['errors']


def test_migration_preserves_v1_rows(tmp_path):
    p=tmp_path/'catalog.sqlite'; db=connect(p)
    db.execute('UPDATE schema_versions SET version=1')
    db.execute("INSERT INTO datasets VALUES('saved','name','1','url','license','citation','hash')")
    db.commit(); db.close()
    db=connect(p)
    assert db.execute('SELECT version FROM schema_versions').fetchone()[0]==2
    assert db.execute("SELECT name FROM datasets WHERE id='saved'").fetchone()[0]=='name'
    assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
    db.close()


def test_zero_initialization_preserves_bicubic():
    import torch
    from rssr.models import SRModel
    x=torch.rand(1,3,8,8); model=SRModel(width=8,zero_residual=True)
    torch.testing.assert_close(model(x),torch.nn.functional.interpolate(x,scale_factor=2,mode='bicubic',align_corners=False))
    model(x).sum().backward()
    assert model.tail[0].weight.grad.abs().sum()>0

