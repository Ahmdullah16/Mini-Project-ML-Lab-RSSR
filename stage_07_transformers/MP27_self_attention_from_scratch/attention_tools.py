"""Single-head NumPy attention, float64; no training or positional encoding."""
import numpy as np


def matrix(value, name):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or min(value.shape) == 0 or not np.isfinite(value).all():
        raise ValueError(f'{name} must be a nonempty finite matrix')
    return value


def softmax(logits, mask=None):
    """Normalize rows; True mask entries are allowed keys, never padding weights."""
    logits = matrix(logits, 'logits')
    if mask is None:
        mask = np.ones(logits.shape, dtype=bool)
    mask = np.asarray(mask)
    if mask.dtype != bool or mask.shape != logits.shape or not mask.any(axis=1).all():
        raise ValueError('Boolean mask must match logits and allow a key in each row')
    allowed = np.where(mask, logits, -np.inf)
    with np.errstate(under='ignore'):
        weights = np.exp(allowed - allowed.max(axis=1, keepdims=True))
    return weights / weights.sum(axis=1, keepdims=True)


def attention(query, key, value, mask=None, scaled=True):
    """Q[N,d], K[M,d], V[M,dv] -> output[N,dv], probabilities[N,M]."""
    query, key, value = (matrix(x, n) for x, n in [(query, 'Q'), (key, 'K'), (value, 'V')])
    if query.shape[1] != key.shape[1] or key.shape[0] != value.shape[0]:
        raise ValueError('Q/K feature widths and K/V token counts must agree')
    with np.errstate(over='ignore', invalid='ignore'):
        logits = query @ key.T
    if scaled:
        logits = logits / np.sqrt(query.shape[1])
    weights = softmax(logits, mask)
    return weights @ value, weights


def score_bytes(tokens, heads=1, itemsize=4):
    """One dense score tensor only; excludes Q/K/V, gradients and temporaries."""
    for v in (tokens, heads, itemsize):
        if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
            raise ValueError('Sizes must be positive integers')
    return heads * tokens * tokens * itemsize


def entropy(weights):
    """Row entropy in nats, including the exact 0 log 0 limit."""
    weights = matrix(weights, 'weights')
    if (weights < 0).any() or not np.allclose(weights.sum(axis=1), 1):
        raise ValueError('Expected probability rows')
    return -(weights * np.log(np.maximum(weights, np.finfo(float).tiny))).sum(axis=1)
