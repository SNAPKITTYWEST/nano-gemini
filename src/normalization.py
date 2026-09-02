from __future__ import annotations

import numpy as np
from .config import SIGMA_EPS

def linear_interpolate(row: np.ndarray) -> np.ndarray:
    row = np.asarray(row, dtype=np.float64).copy()
    nan = ~np.isfinite(row)
    if not nan.any(): return row
    if nan.all(): return np.zeros_like(row)
    idx = np.arange(row.shape[0])
    row[nan] = np.interp(idx[nan], idx[~nan], row[~nan])
    return row

def revin(x, mu, sigma, reverse=False):
    s = np.where(sigma < SIGMA_EPS, 1.0, sigma)
    return x*s+mu if reverse else (x-mu)/s
