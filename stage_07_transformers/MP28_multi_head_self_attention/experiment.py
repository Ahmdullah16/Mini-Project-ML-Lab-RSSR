"""Hold total width and projection matrices fixed while changing head partitions."""
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import rasterio
from rasterio.windows import Window
from multihead_tools import multihead,projection_parameters
from attention_tools import entropy


def run(root):
    started=time.perf_counter();root=Path(root)
    manifest=json.loads((root/'configs/naip-rgb-development-manifest.json').read_text())
    scene=next(s for s in manifest['scenes'] if s['split']=='train');source=root/'data'/scene['path']
    if hashlib.sha256(source.read_bytes()).hexdigest()!=scene['sha256']:raise ValueError('Source hash mismatch')
    with rasterio.open(source) as ds:
        window=Window(ds.width//2-2,ds.height//2-2,4,4)
        rgb=ds.read([1,2,3],window=window).astype(np.float64)/manifest['divisor']
        if not (ds.read_masks([1,2,3],window=window)>0).all():raise ValueError('Invalid source crop')
    rng=np.random.default_rng(28);embedding=rng.normal(size=(3,4))/np.sqrt(3)
    inputs={'synthetic_normal':rng.normal(size=(16,4)),'naip_train_rgb':rgb.reshape(3,-1).T@embedding}
    weights=[rng.normal(size=(4,4))/2 for _ in range(4)]
    rows=[]
    for name,x in inputs.items():
        for heads in [1,2,4]:
            out,maps=multihead(x,weights,heads)
            rows.append({'input':name,'heads':heads,'head_width':4//heads,'output_shape':list(out.shape),
                         'attention_shape':list(maps.shape),'projection_parameters':projection_parameters(4),
                         'float64_attention_bytes':maps.nbytes,'head_entropy_nats':[float(entropy(m).mean()) for m in maps],
                         'max_row_sum_error':float(np.abs(maps.sum(axis=-1)-1).max()),'finite':bool(np.isfinite(out).all()),
                         'maps':maps.tolist()})
    return {'seed':28,'device':'CPU','python':platform.python_version(),'numpy':np.__version__,
            'protocol':'Width4; same bias-free Q/K/V/O weights across head counts; untrained; no positions/masks/dropout',
            'source':{'scene':scene['id'],'sha256':scene['sha256'],'split':'train','window':list(window.flatten()),
                      'license':manifest['license'],'units':manifest['units'],'embedding_parameters':12},
            'rows':rows,'seconds':time.perf_counter()-started,
            'limitations':'Different maps are not semantic specialization or accuracy. Embedding adds12 parameters, excluded from the64 attention-projection entries. No SR or held-out evaluation.'}


if __name__=='__main__':
    root=Path(__file__).resolve().parents[2];result=run(root)
    (root/'reports/mp28-multihead.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Six controlled conditions recorded; all finite:',all(r['finite'] for r in result['rows']))
