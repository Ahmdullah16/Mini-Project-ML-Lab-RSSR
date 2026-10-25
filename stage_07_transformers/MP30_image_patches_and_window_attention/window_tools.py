"""Row-major patch ordering and explicit non-wrapping window connectivity."""
import numpy as np


def positive(value):
    if isinstance(value,bool) or not isinstance(value,int) or value<=0:raise ValueError('Expected positive integer')


def patchify(image,size):
    positive(size);image=np.asarray(image)
    if image.ndim!=3 or min(image.shape)==0 or not np.isfinite(image).all():raise ValueError('Expected finite nonempty HWC image')
    h,w,c=image.shape
    if h%size or w%size:raise ValueError('Dimensions must divide patch size; no implicit padding')
    return image.reshape(h//size,size,w//size,size,c).transpose(0,2,1,3,4).reshape(-1,size*size*c).copy()


def unpatchify(tokens,shape,size):
    positive(size)
    if len(shape)!=3:raise ValueError('Expected HWC shape')
    for value in shape:positive(value)
    h,w,c=shape;tokens=np.asarray(tokens)
    if h%size or w%size or tokens.shape!=((h//size)*(w//size),size*size*c) or not np.isfinite(tokens).all():
        raise ValueError('Token shape does not match declared image/patch geometry')
    return tokens.reshape(h//size,w//size,size,size,c).transpose(0,2,1,3,4).reshape(shape).copy()


def connectivity(height,width,window,shift=0):
    """Dense teaching adjacency only. Shifted boundary windows are clipped, never wrapped."""
    for value in (height,width,window):positive(value)
    if isinstance(shift,bool) or not isinstance(shift,int) or not 0<=shift<window:raise ValueError('Shift must be in [0,window)')
    if height*width>4096:raise ValueError('Teaching adjacency capped at4096 tokens; use analytical counts for larger grids')
    row,col=np.indices((height,width));groups=np.stack(((row+shift)//window,(col+shift)//window),axis=-1).reshape(-1,2)
    return (groups[:,None,:]==groups[None,:,:]).all(axis=-1)


def score_counts(height,width,window):
    for value in (height,width,window):positive(value)
    if height%window or width%window:raise ValueError('Analytical complete windows require divisible grid')
    n=height*width
    return {'tokens':n,'global_pairs':n*n,'local_pairs':n*window*window,
            'global_float32_bytes':4*n*n,'local_float32_bytes':4*n*window*window}
