"""MP12 matrix approximation; float64 grayscale matrices, no implicit scaling."""
import numpy as np


def approximate(matrix, rank):
    x=np.asarray(matrix,dtype=np.float64)
    if x.ndim!=2 or min(x.shape)<1 or not np.isfinite(x).all():
        raise ValueError('Finite nonempty two-dimensional matrix required')
    if isinstance(rank,bool) or not isinstance(rank,int) or not 0<=rank<=min(x.shape):
        raise ValueError('Integer rank must lie between zero and min(H,W)')
    u,s,vh=np.linalg.svd(x,full_matrices=False)
    y=(u[:,:rank]*s[:rank])@vh[:rank,:]
    return y,{'singular_values':s.tolist(),'frobenius_error':float(np.linalg.norm(x-y)),
              'tail_energy':float(np.sum(s[rank:]**2)),
              'dense_bytes':x.nbytes,'factor_bytes':8*rank*(x.shape[0]+x.shape[1]+1)}
