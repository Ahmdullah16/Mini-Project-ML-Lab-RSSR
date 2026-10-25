"""MP33 local numerical API; full lesson acceptance remains in README."""
from pathlib import Path
import sys
ROOT=next(p for p in (Path(__file__).resolve().parent,*Path(__file__).resolve().parents) if (p/'src/rssr').is_dir())
sys.path.insert(0,str(ROOT/'src'))
import numpy as np

def valid_mean(values,nodata=-9999.):
    values=np.asarray(values,dtype=float)
    valid=np.isfinite(values)&(values!=nodata)
    if not valid.any():raise ValueError('No valid pixels; no replacement mean invented')
    return float(values[valid].mean()),valid

def select_valid_bands(image,mask,indices):
    """Select unique zero-based CHW bands; validity includes finite samples."""
    image=np.asarray(image);mask=np.asarray(mask,dtype=bool)
    if image.ndim!=3 or mask.shape!=image.shape or not all(image.shape):raise ValueError('CHW image and per-band mask required')
    if not indices or len(set(indices))!=len(indices) or any(isinstance(i,bool) or not isinstance(i,int) or not 0<=i<image.shape[0] for i in indices):raise ValueError('Unique in-range integer band indices required')
    out=image[indices].astype(np.float64)
    return out,mask[indices]&np.isfinite(out)

def fit_standardization(image,mask,*,split):
    """Population statistics from explicitly training-tagged input only."""
    if split!='train':raise ValueError('Fit normalization on training data only')
    image,mask=select_valid_bands(image,mask,list(range(len(image))))
    result=[]
    for band,valid in zip(image,mask):
        values=band[valid]
        if not values.size:raise ValueError('Cannot fit an empty valid band')
        mean=float(values.mean());std=float(values.std(ddof=0))
        result.append({'mean':mean,'std':std,'scale':std if std>0 else 1.,'count':int(values.size),'constant_band':std==0})
    return {'fit_split':split,'bands':result,'convention':'population std; constant bands use scale1'}

def standardize(image,mask,statistics):
    image,mask=select_valid_bands(image,mask,list(range(len(image))))
    if statistics.get('fit_split')!='train' or len(statistics['bands'])!=len(image):raise ValueError('Compatible training statistics required')
    out=np.zeros(image.shape,dtype=np.float32)
    for i,stats in enumerate(statistics['bands']):
        if not np.isfinite([stats['mean'],stats['scale']]).all() or stats['scale']<=0:raise ValueError('Finite positive normalization scale required')
        out[i,mask[i]]=(image[i,mask[i]]-stats['mean'])/stats['scale']
    return out,mask

def extract_patches(image,mask,transform,*,size=16,min_valid_fraction=1.):
    """Nonoverlapping full patches only; CHW validity, common-valid HW acceptance."""
    from affine import Affine
    image,mask=select_valid_bands(image,mask,list(range(len(image))))
    if isinstance(size,bool) or not isinstance(size,int) or size<1 or not np.isfinite(min_valid_fraction) or not 0<=min_valid_fraction<=1:raise ValueError('Invalid patch size/validity threshold')
    patches=[];rejected=0;h,w=image.shape[1:]
    for row in range(0,h-size+1,size):
        for col in range(0,w-size+1,size):
            valid=mask[:,row:row+size,col:col+size];fraction=float(valid.all(0).mean())
            if fraction<min_valid_fraction:rejected+=1;continue
            patches.append({'data':image[:,row:row+size,col:col+size].copy(),'mask':valid.copy(),
                            'row':row,'column':col,'transform':transform*Affine.translation(col,row),'valid_fraction':fraction})
    return patches,{'rejected_full_patches':rejected,'bottom_rows_dropped':h%size,'right_columns_dropped':w%size}
