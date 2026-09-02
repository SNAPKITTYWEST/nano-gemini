# Nano-Gemini

[![License: Sovereign](https://img.shields.io/badge/License-Sovereign%20Source%20v1.0%20%2B%20BSL--1.1%20%2B%20AGPL--3.0-critical.svg)](#license)
[![Model](https://img.shields.io/badge/Model-Meridian--Nano%204×64%200.3M-blue.svg)](#architecture)
[![Hardware](https://img.shields.io/badge/Hardware-RTX%203080%20%2B%20Artix--7-green.svg)](#hardware)

> **Gemma decoder, time-series skin. 0.3M that runs on your 3080.**

Executable Nano-class time-series foundation model. Not Google TimesFM. No Google weights.

## Quick Start

```bash
pip install -r requirements.txt
python -m src.inference  # numpy reference forecast
python -c "from src.model import GemmaTimeSeriesTorch; from src.config import NanoConfig; m=GemmaTimeSeriesTorch(NanoConfig); print(sum(p.numel() for p in m.parameters()))"  # 0.3M
```

```python
import numpy as np
from src.inference import forecast

target = np.random.randn(3,128)  # [V, T]
past_only = np.random.randn(1,128)
past_future = np.random.randn(2,136)  # [C, T+H]
out = forecast(target, horizon=8, past_only_covariates=past_only, past_future_covariates=past_future)
# out.forecast (3,8), out.quantiles (3,8,9)
```

## Architecture

| | Nano (live) | G6 (spec) |
|---|---|---|
| d_model | 64 | 4096 |
| layers | 4 | 28 |
| GQA | 4/4 | 32/8 |
| head_dim | 16 | 128 |
| GeGLU | 128 | 11008 |
| params | 0.3M | 6.15B |

`INPUT → NaN interpolate → RevIN → patch 32 → adapter 192→64 → 4× GemmaMixingBlock (GQA+GeGLU, RoPE, QK-Norm) → quantile head 64×9 → inverse RevIN → stitch → FORECAST`

## Files

| File | What |
|------|------|
| `src/config.py` | `NanoConfig` / `G6Config`, `QUANTILES` 0.1–0.9 |
| `src/model.py` | `RMSNorm`, `GeGLU`, `GQA`, `GemmaTimeSeriesTorch` |
| `src/inference.py` | `forecast()` numpy reference, `_norm_inv` Acklam |
| `src/normalization.py` | `linear_interpolate`, `revin` |
| `src/patches.py` | `patch_series`, `stitch_patches`, `context_pad` |
| `src/validation.py` | shape contract for covariates |

Cherry-picked from `sovereign-gemini-gguf` + `ahmad-foundations`, zero-sorry.

## License

Tri-licensed: Sovereign Source v1.0 + BSL-1.1 + AGPL-3.0. See `LICENSE`.

Contact: Ahmad Ali Parr <ahmedparr93@gmail.com> · Bel Esprit D'Accord Trust
