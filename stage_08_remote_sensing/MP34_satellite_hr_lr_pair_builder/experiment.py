"""Create bounded synthetic x2/x4 pairs from a pinned cached training crop."""
import hashlib,json,platform
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
import rasterio
from rasterio.windows import Window
from affine import Affine
from mp34_tools import georeferenced_pair
ROOT=Path(__file__).resolve().parents[2]

def main():
    torch.set_num_threads(2)
    ramp=torch.arange(16,dtype=torch.float32).reshape(1,4,4)
    tiny,_=georeferenced_pair(ramp,Affine(10,0,100,0,-10,200),'EPSG:32632',sigma=0)
    assert tiny.tolist()==[[[2.5,4.5],[10.5,12.5]]]
    manifest=json.loads((ROOT/'configs/naip-pilot-manifest.json').read_text(encoding='utf-8'))
    scene=next(s for s in manifest['scenes'] if s['split']=='train');path=ROOT/'data'/scene['path']
    if not path.is_file():raise FileNotFoundError('Pinned training raster unavailable; no download attempted')
    assert hashlib.sha256(path.read_bytes()).hexdigest()==scene['sha256']
    with rasterio.open(path) as raster:
        window=Window(238,238,64,64);raw=raster.read(window=window,masked=True)
        transform=raster.window_transform(window);crs=raster.crs
    hr=torch.from_numpy(raw.data.astype(np.float32)/manifest['divisor']);mask=~np.ma.getmaskarray(raw)
    destination=ROOT/'data/teaching/mp34';destination.mkdir(parents=True,exist_ok=True)
    np.save(destination/'hr.npy',hr.numpy(),allow_pickle=False)
    rows=[]
    for scale in [2,4]:
        lr,metadata=georeferenced_pair(hr,transform,crs,valid=mask,scale=scale)
        output=destination/f'lr-x{scale}.npy';np.save(output,lr.numpy(),allow_pickle=False)
        metadata.update(hr_file='data/teaching/mp34/hr.npy',lr_file=output.relative_to(ROOT).as_posix(),
                        hr_sha256=hashlib.sha256((destination/'hr.npy').read_bytes()).hexdigest(),
                        lr_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                        dtype=str(lr.numpy().dtype),hr_range=[float(hr.min()),float(hr.max())],lr_range=[float(lr.min()),float(lr.max())])
        rows.append(metadata)
    result={'recorded_at':datetime.now(timezone.utc).isoformat(),'source':scene,'license':manifest['license'],
            'bands':manifest['bands'],'units':manifest['units'],'crop':[238,238,64,64],'rows':rows,
            'environment':{'python':platform.python_version(),'torch':torch.__version__,'rasterio':rasterio.__version__},
            'limits':'Synthetic pairs from one aerial training crop, not a real paired cross-sensor dataset. No model training/test evaluation; paired arrays are ignored rebuildable outputs.'}
    (ROOT/'reports/mp34-pairing-manifest.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print([(r['scale'],r['hr_shape'],r['lr_shape'],r['lr_transform']) for r in rows])

if __name__=='__main__':main()
