import hashlib
import io
import pytest
from rssr import hosted_assets


def test_download_verification_cache_and_corruption(tmp_path,monkeypatch):
    content=b'actual fixture'; calls=[]
    def fetch(*args,**kwargs): calls.append(args); return io.BytesIO(content)
    monkeypatch.setattr(hosted_assets.urllib.request,'urlopen',fetch)
    asset={'url':'https://raw.githubusercontent.com/owner/repo/commit/model','sha256':hashlib.sha256(content).hexdigest(),'max_bytes':100}
    path=hosted_assets.fetch_asset(asset,tmp_path)
    assert path.read_bytes()==content
    assert hosted_assets.fetch_asset(asset,tmp_path)==path and len(calls)==1
    path.write_bytes(b'corrupt')
    assert hosted_assets.fetch_asset(asset,tmp_path).read_bytes()==content and len(calls)==2


def test_wrong_hash_and_budget_leave_no_partial(tmp_path,monkeypatch):
    monkeypatch.setattr(hosted_assets.urllib.request,'urlopen',lambda *a,**k:io.BytesIO(b'12345'))
    asset={'url':'https://raw.githubusercontent.com/owner/repo/commit/model','sha256':'0'*64,'max_bytes':10}
    with pytest.raises(ValueError,match='checksum'): hosted_assets.fetch_asset(asset,tmp_path)
    assert not list(tmp_path.iterdir())
    with pytest.raises(ValueError,match='budget'): hosted_assets.fetch_asset(dict(asset,max_bytes=2),tmp_path)
    assert not list(tmp_path.iterdir())
