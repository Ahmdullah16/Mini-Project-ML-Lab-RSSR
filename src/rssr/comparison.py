"""Identity and export contracts for actual RGB display comparisons."""
import hashlib
import io
import json
import zipfile
import numpy as np
from PIL import Image

SETTINGS={'pipeline':'rgb-display-v2-diagnostics','input':'uint8 RGB /255','scale':2,'tile':128,'halo':32,
          'output':'fixed [0,1] clipping and uint8 quantization; no sharpening','max_side':1024}

def input_digest(image):
    image=np.asarray(image)
    if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3 or min(image.shape[:2])<1 or max(image.shape[:2])>1024:
        raise ValueError('Comparison requires uint8 RGB up to 512×512')
    return hashlib.sha256(str(image.shape).encode()+image.tobytes()).hexdigest()

def result_key(image,specification):
    payload={'input_sha256':input_digest(image),'checkpoint_sha256':specification['checkpoint']['sha256'],
             'version':specification['version'],'settings':SETTINGS}
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()

def validate_result(image,baseline,sr,metadata,specification):
    input_digest(image)
    expected=(image.shape[0]*2,image.shape[1]*2,3)
    if baseline.shape!=expected or sr.shape!=expected or baseline.dtype!=np.uint8 or sr.dtype!=np.uint8:
        raise ValueError('Native output must be uint8 RGB at exactly ×2; comparison never resizes mismatched outputs')
    if metadata.get('scale')!=2 or metadata.get('checkpoint_sha256')!=specification['checkpoint']['sha256']:
        raise ValueError('Checkpoint/scale identity does not match selected model')

def png_bytes(image):
    out=io.BytesIO();Image.fromarray(image).save(out,format='PNG');return out.getvalue()

def comparison_bundle(batch):
    """Export computed results with immutable run labels; no inferred HR."""
    out=io.BytesIO()
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('input-lr.png',png_bytes(batch['input']))
        provenance={'input_sha256':input_digest(batch['input']),'source':batch['source'],'settings':SETTINGS,'results':[],'errors':batch['errors']}
        if batch['results']:
            archive.writestr('bicubic.png',png_bytes(batch['results'][0]['baseline']))
        for i,result in enumerate(batch['results']):
            name=f'model-{i+1}-{result["key"][:12]}.png'
            archive.writestr(name,png_bytes(result['sr']))
            provenance['results'].append({'file':name,'key':result['key'],**result['metadata']})
        archive.writestr('provenance.json',json.dumps(provenance,indent=2,allow_nan=False))
    return out.getvalue()
