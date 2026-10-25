"""Implementation differences, never reference-image accuracy metrics."""
import numpy as np

def difference_summary(full,tiled,*,tile=128,scale=2,radius=2):
    """CHW finite float outputs; seam strips are +/-radius HR pixels from tile edges."""
    full=np.asarray(full);tiled=np.asarray(tiled)
    if full.ndim!=3 or not all(full.shape) or full.shape!=tiled.shape or not np.isfinite(full).all() or not np.isfinite(tiled).all():
        raise ValueError('Matching finite nonempty CHW outputs required')
    if any(isinstance(v,bool) or not isinstance(v,int) or v<1 for v in (tile,scale,radius)):
        raise ValueError('Positive integer tile, scale and radius required')
    delta=np.abs(full.astype(np.float64)-tiled.astype(np.float64));h,w=full.shape[1:]
    seam=np.zeros((h,w),bool)
    for edge in range(tile*scale,h,tile*scale):seam[max(0,edge-radius):min(h,edge+radius),:]=True
    for edge in range(tile*scale,w,tile*scale):seam[:,max(0,edge-radius):min(w,edge+radius)]=True
    quantize=lambda x:np.rint(np.clip(x,0,1)*255).astype(np.uint8)
    return {'mae_normalized':float(delta.mean()),'max_abs_normalized':float(delta.max()),
            'seam_mae_normalized':float(delta[:,seam].mean()) if seam.any() else None,
            'nonseam_mae_normalized':float(delta[:,~seam].mean()) if (~seam).any() else None,
            'seam_spatial_pixels':int(seam.sum()),'output_shape_chw':list(full.shape),
            'tiled_out_of_range_fraction':float(((tiled<0)|(tiled>1)).mean()),
            'full_out_of_range_fraction':float(((full<0)|(full>1)).mean()),
            'uint8_disagreement_fraction':float((quantize(full)!=quantize(tiled)).mean())}
