"""Comparison identity and one exact-source georeferenced execution."""
import hashlib,json,tempfile
from pathlib import Path
from .inference import geotiff,geotiff_plan

def identity(source,spec,scale):
    settings={'source_sha256':hashlib.sha256(source).hexdigest(),'model':spec,'scale':scale,'divisor':255.,'pipeline':'scientific-disk-cascade-v1'}
    return hashlib.sha256(json.dumps(settings,sort_keys=True).encode()).hexdigest()

def execute(source,checkpoint,spec,scale,acquisition):
    with tempfile.TemporaryDirectory(prefix='rssr-satellite-compare-') as temp:
        folder=Path(temp);src=folder/'source.tif';src.write_bytes(source)
        request=geotiff_plan(src,scale)
        if not request['host_available']:raise ValueError('Exact output exceeds configured host limit; use local job')
        target=folder/'output.tif';geotiff(checkpoint,src,target,255.,scale)
        metadata=json.loads(target.with_suffix('.json').read_text(encoding='utf-8'))
        metadata.update(acquisition=acquisition,model=spec,comparison_key=identity(source,spec,scale),source_bytes_sha256=hashlib.sha256(source).hexdigest())
        return {'geotiff':target.read_bytes(),'metadata':metadata}
