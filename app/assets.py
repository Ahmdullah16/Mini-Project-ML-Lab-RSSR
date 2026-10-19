"""Portable operator-configured demo assets and expiring exports."""
import json,os,tempfile,threading
from pathlib import Path
from PIL import Image


def models():
    configured=os.environ.get('RSSR_MODELS')
    result=json.loads(Path(configured).read_text()) if configured else {'RGB pilot':os.environ.get('RSSR_CHECKPOINT','')}
    if not isinstance(result,dict) or not result: raise ValueError('Provide a nonempty operator model map')
    resolved={}
    for name,path in result.items():
        p=Path(path)
        if configured and not p.is_absolute(): p=Path(configured).resolve().parent/p
        if not p.is_file(): raise ValueError(f'Missing checkpoint for {name}')
        resolved[name]=p
    return resolved


def export_result(sr,metadata):
    directory=tempfile.TemporaryDirectory(prefix='rssr-export-')
    image=Path(directory.name)/'sr-estimate.png'; provenance=Path(directory.name)/'provenance.json'
    Image.fromarray(sr).save(image); provenance.write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    timer=threading.Timer(60,directory.cleanup); timer.daemon=True; timer.start()
    return str(image),str(provenance)
