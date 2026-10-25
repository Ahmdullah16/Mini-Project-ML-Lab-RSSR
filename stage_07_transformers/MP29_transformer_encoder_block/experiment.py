import hashlib,json,platform,time
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window
import torch
from encoder_tools import Encoder,positions


def run(root):
    root=Path(root);started=time.perf_counter();torch.set_num_threads(1);torch.manual_seed(29)
    manifest=json.loads((root/'configs/naip-rgb-development-manifest.json').read_text())
    scene=next(s for s in manifest['scenes'] if s['split']=='train');path=root/'data'/scene['path']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=scene['sha256']:raise ValueError('Source hash mismatch')
    with rasterio.open(path) as ds:
        window=Window(ds.width//2-2,ds.height//2-2,4,4)
        rgb=ds.read([1,2,3],window=window).astype(np.float64)/manifest['divisor']
        if not (ds.read_masks([1,2,3],window=window)>0).all():raise ValueError('Invalid crop')
    projection=torch.randn(3,4,dtype=torch.float64)/3**.5
    fixtures={'synthetic_normal':torch.randn(1,16,4,dtype=torch.float64),
              'naip_train_rgb':torch.from_numpy(rgb.reshape(3,-1).T)[None]@projection}
    torch.manual_seed(29);reference=Encoder(4,1).double();rows=[]
    with torch.no_grad():
        for name,x in fixtures.items():
            for heads in [1,2,4]:
                model=Encoder(4,heads).double();model.load_state_dict(reference.state_dict())
                for positional in [False,True]:
                    actual=x+positions(16,4) if positional else x
                    output,trace=model(actual)
                    rows.append({'input':name,'heads':heads,'positions':positional,'parameters':sum(p.numel() for p in model.parameters()),
                                 'trace':trace,'finite':bool(torch.isfinite(output).all()),
                                 'output_rms':float(output.square().mean().sqrt()),
                                 'residual_change_rms':float((output-actual).square().mean().sqrt()),
                                 'output':output[0].tolist()})
    return {'seed':29,'python':platform.python_version(),'torch':torch.__version__,'device':'CPU float64; one thread',
            'protocol':'Fixed width4 and identical state across head counts; pre-LN; FFN width8; GELU; eps1e-5; biases; dropout0; no optimization',
            'source':{'scene':scene['id'],'sha256':scene['sha256'],'window':list(window.flatten()),'split':'train','license':manifest['license'],'units':manifest['units']},
            'rows':rows,'seconds':time.perf_counter()-started,
            'limitations':'Response magnitude is not accuracy. Fixed sinusoidal positions are added before the block. Untrained; no SR or test evaluation.'}


if __name__=='__main__':
    root=Path(__file__).resolve().parents[2];report=run(root)
    (root/'reports/mp29-encoder.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Encoder conditions:',len(report['rows']),'all finite:',all(r['finite'] for r in report['rows']))
