import hashlib,json,platform
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window
from window_tools import patchify,unpatchify,connectivity,score_counts


def run(root):
    root=Path(root);manifest=json.loads((root/'configs/naip-rgb-development-manifest.json').read_text())
    scene=next(s for s in manifest['scenes'] if s['split']=='train');path=root/'data'/scene['path']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=scene['sha256']:raise ValueError('Source hash mismatch')
    with rasterio.open(path) as ds:
        window=Window(ds.width//2-4,ds.height//2-4,8,8);image=ds.read([1,2,3],window=window).transpose(1,2,0)
        if not (ds.read_masks([1,2,3],window=window)>0).all():raise ValueError('Invalid crop')
    rows=[]
    for name,x in [('synthetic_ramp',np.arange(8*8*3).reshape(8,8,3)),('naip_train_rgb',image)]:
        for size in [1,2,4]:
            tokens=patchify(x,size);restored=unpatchify(tokens,x.shape,size)
            rows.append({'input':name,'patch_size':size,'token_shape':list(tokens.shape),'exact_roundtrip':bool(np.array_equal(restored,x))})
    local=connectivity(4,4,2);shifted=connectivity(4,4,2,1);reached=(shifted.astype(int)@local.astype(int))>0
    return {'python':platform.python_version(),'numpy':np.__version__,'device':'CPU; deterministic integer ordering, no RNG',
            'source':{'scene':scene['id'],'sha256':scene['sha256'],'window':list(window.flatten()),'split':'train','license':manifest['license'],'dtype':str(image.dtype)},
            'rows':rows,'connectivity':{k:v.astype(int).tolist() for k,v in [('global',np.ones_like(local)),('local',local),('shifted',shifted),('two_layers',reached)]},
            'connected_pairs':{'global':int(local.size),'local':int(local.sum()),'shifted':int(shifted.sum()),'two_layers':int(reached.sum())},
            'counts':[dict(grid_side=n,**score_counts(n,n,2)) for n in [4,8,16,32,64,128]],
            'limitations':'Clipped shifted boundary windows, no cyclic wrap or Swin mask implementation; connectivity only, not learned attention. Memory counts analytical. No test data or SR metrics.'}


if __name__=='__main__':
    root=Path(__file__).resolve().parents[2];r=run(root)
    (root/'reports/mp30-windows.json').write_text(json.dumps(r,indent=2)+'\n')
    print('Exact round trips:',sum(v['exact_roundtrip'] for v in r['rows']),'connected pairs:',r['connected_pairs'])
