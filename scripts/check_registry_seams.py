"""Synthetic full-frame/tiled discrepancy, explicitly not trained quality."""
import hashlib,json,platform,time
from pathlib import Path
import numpy as np
import torch
from rssr.architecture_registry import REGISTRY,build
from rssr.models import tiled

ROOT=Path(__file__).resolve().parents[1]


def check(key, shape=(1,3,65,97)):
    with torch.random.fork_rng():
        torch.manual_seed(76)
        model=build(key,width=4).eval()
        x=torch.rand(shape)
        with torch.inference_mode():
            full=model(x); patch=tiled(model,x,tile=32,halo=32)
    difference=(full-patch).abs()
    mask=torch.zeros(full.shape[-2:],dtype=torch.bool)
    for axis,length in enumerate(shape[-2:]):
        for point in range(32,length,32):
            at=point*2
            if axis==0:mask[max(0,at-2):at+2,:]=True
            else:mask[:,max(0,at-2):at+2]=True
    assert full.shape==patch.shape==(1,3,shape[2]*2,shape[3]*2)
    assert torch.isfinite(patch).all()
    maximum=float(difference.max())
    if not REGISTRY[key].a:
        assert maximum<=1e-6, (key,maximum)
    return {'architecture_id':key,'input_shape':list(shape),'output_shape':list(full.shape),
            'width':4,'native_scale':2,'tile':32,'halo':32,'seed':76,
            'maximum_absolute_difference':maximum,'mean_absolute_difference':float(difference.mean()),
            'seam_strip_mean_absolute_difference':float(difference[...,mask].mean()),
            'interior_mean_absolute_difference':float(difference[...,~mask].mean()),
            'full_output_sha256':hashlib.sha256(full.numpy().tobytes()).hexdigest(),
            'tiled_output_sha256':hashlib.sha256(patch.numpy().tobytes()).hexdigest(),
            'criterion':'finite and exact dimensions; no equivalence claim for global feature attention' if REGISTRY[key].a else 'maximum absolute discrepancy <=1e-6 on this fixture',
            'status':'passed stated synthetic criterion','checkpoint':None}


def main():
    torch.set_num_threads(2); started=time.perf_counter()
    rows=[check(key) for key in REGISTRY]
    paths=['src/rssr/models.py','src/rssr/architecture_registry.py','scripts/check_registry_seams.py']
    report={'schema':'cat-tiled-discrepancy-v1','command':'python scripts/check_registry_seams.py','environment':{'python':platform.python_version(),'torch':torch.__version__,'device':'CPU','threads':2},'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},'rows':rows,'seconds':time.perf_counter()-started,'exit_code':0,'limitations':['Untrained weights and deterministic random fixture, not representative-image or perceptual quality.','Seam strips extend two output pixels on either side of each internal tile boundary; discrepancy is full-frame vs tiled, not HR error.','A global pooling sees different tile context; no universal tolerance/equivalence claimed.','Native2 only: repeated cascades, maximum input and trained checkpoint seam tests remain pending.']}
    (ROOT/'reports/architecture-seam-smoke.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    for row in rows:print(row['architecture_id'],row['maximum_absolute_difference'])


if __name__=='__main__':main()
