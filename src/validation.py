from __future__ import annotations

import numpy as np

class ShapeError(ValueError):
    pass

def validate_forecast_request(target, horizon, past_only, past_future, cfg):
    if target.size == 0 or target.shape[-1] == 0:
        raise ShapeError("Empty input rejected.")
    if target.ndim != 2:
        raise ShapeError("target must be (V, T)")
    v, c = target.shape
    if v > cfg.max_variates:
        raise ShapeError(f"variates {v} exceed {cfg.max_variates}")
    if horizon <= 0:
        raise ShapeError("Invalid horizon.")
    if horizon > cfg.max_horizon:
        raise ShapeError("horizon exceeds max")
    if c > cfg.max_context:
        raise ShapeError("context exceeds max")
    if past_only is not None:
        po = np.asarray(past_only)
        if po.ndim != 2 or po.shape[-1] != c:
            raise ShapeError("past-only covariates must have length context")
    if past_future is not None:
        pf = np.asarray(past_future)
        if pf.ndim != 2 or pf.shape[-1] != c + horizon:
            raise ShapeError("past-future covariates must have length context+horizon")
