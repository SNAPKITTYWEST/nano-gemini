from __future__ import annotations

import numpy as np

def context_pad(length, patch):
    pad = (patch - length % patch) % patch
    return length+pad, pad

def patch_series(values, patch):
    v,t = values.shape
    if t%patch: raise ValueError(f"length {t} not multiple of {patch}")
    return values.reshape(v, t//patch, patch)

def stitch_patches(patch_preds, patch_len):
    if patch_preds.ndim == 3:
        patch_preds = patch_preds[:, :, :, None]
    v,N,total_len,qdim = patch_preds.shape
    overlap = total_len - patch_len
    if N == 1:
        return patch_preds[:, 0, :patch_len]
    weights = np.linspace(1.0,0.0,overlap) if overlap else np.array([])
    pieces = [patch_preds[:,0,:patch_len]]
    for k in range(N-1):
        prev = patch_preds[:,k,patch_len:]
        nxt = patch_preds[:,k+1,:overlap]
        if overlap:
            mixed = weights[None,:,None]*prev + (1-weights[None,:,None])*nxt
            pieces.append(mixed)
        pieces.append(patch_preds[:,k+1,overlap:patch_len])
    pieces.append(patch_preds[:,-1,patch_len:])
    return np.concatenate(pieces, axis=1)
