"""Bounded untrained architecture smoke matrix; never scientific SR evidence."""
import hashlib
import json
import platform
import tempfile
import time
from pathlib import Path
import numpy as np
import psutil
import rasterio
import torch
from rssr.architecture_registry import REGISTRY, build, describe
from rssr.scaling import SCALES, cascade

ROOT = Path(__file__).resolve().parents[1]


def main():
    start = time.perf_counter()
    torch.set_num_threads(2)
    image = np.random.default_rng(73).integers(0,256,(3,5,3),dtype=np.uint8)
    rows = []
    process = psutil.Process()
    for key in REGISTRY:
        torch.manual_seed(73)
        model = build(key, width=4).eval()
        for scale in SCALES:
            before = process.memory_info().rss
            with tempfile.TemporaryDirectory(prefix='rssr-cat-smoke-') as directory:
                target = Path(directory)/'output'
                result = cascade(image,model,target,scale,{'architecture_id':key,'weights':'random initialization','seed':73})
                with rasterio.open(target/result['output']) as raster:
                    shape = [raster.height,raster.width,raster.count]
                    assert shape == [3*scale,5*scale,3]
                    values = raster.read()
                    assert np.isfinite(values).all() and values.size == 3*5*scale*scale*3
                rows.append({'architecture_id':key,'scale':scale,'input_shape':[3,5,3],
                    'actual_output_shape':shape,'method':'untrained x2 block, repeated log2(scale) times; bicubic residual; RGB clipping/quantization',
                    'checkpoint_sha256':None,'execution_status':'passed synthetic plumbing only',
                    'runtime_seconds':result['runtime_seconds'],
                    'rss_before_bytes':before,'rss_after_bytes':process.memory_info().rss,
                    'memory_method':'process RSS snapshots before/after; NOT peak memory',
                    'output_pixel_sha256':hashlib.sha256(values.tobytes()).hexdigest(),
                    'stage_count':len(result['stage_outputs']),
                    'artifacts':'temporary TIFFs verified then removed; no distributable checkpoint or trained output'})
    report = {'schema':'cat-smoke-v1','seed':73,'command':'python scripts/check_architecture_matrix.py',
        'environment':{'python':platform.python_version(),'torch':torch.__version__,'platform':platform.platform(),'device':'cpu','threads':2},
        'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['src/rssr/models.py','src/rssr/architecture_registry.py','src/rssr/scaling.py','scripts/check_architecture_matrix.py']},
        'registry':[describe(k,width=4) for k in REGISTRY], 'rows':rows,
        'duration_seconds':time.perf_counter()-start,'exit_status':0,
        'limitations':['Random weights: no trained quality, checkpoint compatibility provenance binding, representative-image or geographic evaluation.',
          'Maximum-size execution and seam validation for new variants remain pending; historical checks do not establish these.',
          'The 42 rows are bounded synthetic disk-export checks, not the final trained capability matrix.']}
    (ROOT/'reports/architecture-smoke-matrix.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(f'{len(rows)} synthetic export combinations passed in {report["duration_seconds"]:.2f}s')


if __name__ == '__main__':
    main()
