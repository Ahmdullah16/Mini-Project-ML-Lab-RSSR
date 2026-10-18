from unittest.mock import patch
import hashlib
import io
import pytest
from rssr.data import download

class Response(io.BytesIO):
    status=206
    headers={'Content-Range':'bytes 3-5/6'}

def test_resume_checksum_and_idempotence(tmp_path):
    target=tmp_path/'asset.tif'; target.with_suffix('.tif.part').write_bytes(b'abc')
    sha=hashlib.sha256(b'abcdef').hexdigest()
    with patch('urllib.request.urlopen',return_value=Response(b'def')) as fetch:
        assert download('https://example.invalid/asset',target,sha)==3
        assert fetch.call_args[0][0].headers['Range']=='bytes=3-'
    assert target.read_bytes()==b'abcdef'
    assert download('https://example.invalid/asset',target,sha)==0
    target.write_bytes(b'bad')
    with pytest.raises(ValueError): download('https://example.invalid/asset',target,sha)

def test_retry_and_budget(tmp_path):
    sha=hashlib.sha256(b'abc').hexdigest()
    response=Response(b'abc'); response.status=200
    with patch('urllib.request.urlopen',side_effect=[OSError('network'),response]),patch('time.sleep'):
        assert download('https://example.invalid/asset',tmp_path/'a',sha)==3
    response=Response(b'abcdef'); response.status=200
    with patch('urllib.request.urlopen',return_value=response):
        with pytest.raises(ValueError): download('https://example.invalid/asset',tmp_path/'b',sha,max_bytes=3)
