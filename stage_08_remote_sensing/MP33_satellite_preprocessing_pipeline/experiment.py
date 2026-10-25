"""Bounded cached-training preprocessing experiment; does not alter SR inputs."""
import hashlib,json,platform
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window
from mp33_tools import select_valid_bands,fit_standardization,standardize,extract_patches
ROOT=Path(__file__).resolve().parents[2]

def main():
    tiny=np.array([[[1.,3.],[99.,5.]]]);valid=np.array([[[True,True],[False,True]]])
    stats=fit_standardization(tiny,valid,split='train');z,mask=standardize(tiny,valid,stats)
    assert np.isclose(z[mask].mean(),0) and np.isclose(z[mask].std(),1)
    manifest=json.loads((ROOT/'configs/naip-pilot-manifest.json').read_text(encoding='utf-8'))
    scene=next(s for s in manifest['scenes'] if s['split']=='train');path=ROOT/'data'/scene['path']
    if not path.is_file():raise FileNotFoundError('Restore pinned training data; no automatic download')
    assert hashlib.sha256(path.read_bytes()).hexdigest()==scene['sha256']
    with rasterio.open(path) as raster:
        window=Window(238,238,128,128);raw=raster.read(window=window,masked=True)
        transform=raster.window_transform(window);crs=str(raster.crs)
    image,mask=select_valid_bands(raw.data,~np.ma.getmaskarray(raw),[0,1,2,3])
    statistics=fit_standardization(image,mask,split='train')
    standardized,mask=standardize(image,mask,statistics)
    patches,dropped=extract_patches(standardized,mask,transform,size=32)
    record={'recorded_at':datetime.now(timezone.utc).isoformat(),'source':scene,'license':manifest['license'],'sensor':manifest['sensor'],
            'window':[238,238,128,128],'crs':crs,'band_order':manifest['bands'],'statistics':statistics,
            'normalized_dtype':str(standardized.dtype),'normalized_units':'dimensionless z-score; not reflectance; not used by existing frozen SR models',
            'normalized_band_means':[float(b[m].mean()) for b,m in zip(standardized,mask)],
            'normalized_band_stds':[float(b[m].std()) for b,m in zip(standardized,mask)],
            'patch_size':32,'accepted_patches':len(patches),'edge_policy':dropped,
            'patches':[{'row':p['row'],'column':p['column'],'transform':list(p['transform'])[:6],'valid_fraction':p['valid_fraction'],'shape':list(p['data'].shape)} for p in patches],
            'environment':{'python':platform.python_version(),'numpy':np.__version__,'rasterio':rasterio.__version__},
            'limits':'Training crop only; no learned preprocessing from validation/test, no cloud mask or reflectance calibration. Per-band z-scoring changes spectral vector angles; do not use these z-scores for physical SAM. Existing DN/10000 inference remains unchanged.'}
    (ROOT/'reports/mp33-preprocessing.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('Accepted patches:',len(patches),'means:',record['normalized_band_means'],'stds:',record['normalized_band_stds'])

if __name__=='__main__':main()
