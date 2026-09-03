# Nano-Gemini

[![License](https://img.shields.io/badge/License-Sovereign%20Source%20v1.0%20%7C%20BSL--1.1%20%7C%20AGPL--3.0-critical.svg)](#license)
[![Model](https://img.shields.io/badge/Model-Meridian--Nano%204x64%200.3M%20--%20G6%206.15B%20spec-blue.svg)](#param-count)
[![Hardware](https://img.shields.io/badge/Hardware-RTX%203080%2010GB%20--%20Artix--7%20BYECODE-green.svg)](#hardware)
[![Python](https://img.shields.io/badge/Python-3.11%20torch%20--%20numpy-3776AB.svg)](#quick-start)

> **Gemma decoder, time-series skin. 0.3M that runs on your 3080. Not Google TimesFM. No Google weights.**

Executable Nano-class time-series foundation model on a Gemma-3 backbone. G6 (6.15B) is spec only -- Nano (0.3M) is live.

Cherry-picked from `sovereign-gemini-gguf` + `ahmad-foundations`. Public, tri-licensed.

---

## What Is Nano?

**Nano is the executable contract. G6 is the blueprint.**

| | **Nano (live, this repo)** | **G6 (spec, not shipped)** |
|---|---|---|
| `d_model` | 64 | 4096 |
| Layers | 4 | 28 |
| GQA | 4 / 4 (MHA) | 32 / 8 (factor 4) |
| `head_dim` | 16 | 128 |
| GeGLU intermediate | 128 | 11008 |
| `input_patch_len` | 32 | 32 |
| `output_patch_len` | 64 | 64 |
| Quantiles | 9 (0.1-0.9) | 9 |
| **Params** | **~0.30M** | **6,157,679,744 (~6.15B)** |
| **Fits RTX 3080 10GB?** | **Yes** | **No (needs 24GB FP32 / 3GB Int4)** |

Gemma is not a time-series model. The **residual adapter** (`192 -> d_model`) is the only learned map from continuous patches into Gemma hidden space.

---

## Param Count -- How 0.3M and 6.15B Are Built

`src/model.py:1` + `src/config.py:1` -- exact accounting, no estimates:

```mermaid
flowchart TD
    A["Per-layer GQA<br/>Q + K + V + O + QK-Norm"] --> P["Per-layer total"]
    B["Per-layer GeGLU<br/>3 x d_model x intermediate"] --> P
    C["Per-layer RMSNorm<br/>4x or 6x d_model"] --> P
    P --> S["Stacked x n_layers"]
    D["Residual Adapter<br/>in_f x h + h x out + in_f x out"] --> T["Total"]
    S --> T
    E["Quantile Head<br/>d_model x 64x9 + 64x9"] --> T
```

| Component | Nano (64x4) | G6 (4096x28) |
|-----------|-------------|--------------|
| Sequence GQA / layer | 16,512 | 37,748,736 |
| Variate GQA / layer | 16,512 | 37,748,736 |
| GeGLU MLP / layer | 24,576 | 135,266,304 |
| RMSNorm / layer | 256 / 384 | 16,384 / 24,576 |
| **Per layer** | **~57k** | **~210M** |
| Stacked (x layers) | ~0.23M | ~5.88B |
| Residual adapter (192->) | ~8k | ~10M |
| Quantile head (64x9) | ~37k | ~2.36M |
| **Total** | **~0.30M** | **6,157,679,744** |

```python
from src.parameters import G6_PARAMS, NANO_PARAMS
print(G6_PARAMS.total)  # 6157679744
print(NANO_PARAMS.total)  # ~304k
for line in G6_PARAMS.lines:
    print(f"{line.name}: {line.count:,}")
```

---

## Method -- Gemma Is Not a Time-Series Model

```mermaid
flowchart LR
    A["INPUT<br/>NaN -> interpolate<br/>left-pad to 32"] --> B["RevIN<br/>running mean/std"]
    B --> C["Patch 32<br/>V x N x 32"]
    C --> D["Adapter 192->d<br/>input patch +<br/>future covariate roll<br/>+ 2 mask channels<br/>-> ResidualBlock"]
    D --> E["Gemma Stack<br/>4x or 28x<br/>GQA causal sequence<br/>+ optional variate<br/>RoPE 10k + QK-Norm<br/>GeGLU"]
    E --> F["Quantile Head<br/>hybrid C+D<br/>64x9 per patch"]
    F --> G["Inverse RevIN"]
    G --> H["Stitch<br/>overlap blend<br/>-> re-add trend"]
    H --> I["FORECAST<br/>point + 9 quantiles"]

    style D fill:#f59e0b,stroke:#d97706,color:#fff
    style E fill:#0ea5e9,stroke:#0284c7,color:#fff
    style F fill:#a855f7,stroke:#9333ea,color:#fff
```

**TimesFM-3 contract preserved:** patch 32, RevIN, per-patch quantile head, non-autoregressive decode, multivariate + past-only / past-future covariates. **Approximated:** GeGLU vs ReLU FFN, lookahead dim, alternating 1:1. **Never claimed:** training mixture, loss, optimizer -- marked UNKNOWN.

---

## Flow -- From GGUF Parse to Forecast

```mermaid
flowchart TD
    A["GGUF file<br/>sovereign-gemini-gguf<br/>GGUFParser"] --> B["ModelGraph IR<br/>36 blocks<br/>GQA/SwiGLU"]
    B --> C["Meridian Config<br/>Nano 64x4 / G6 4096x28"]
    C --> D["GemmaTimeSeriesTorch<br/>adapter -> GemmaMixingBlock x4 -> head"]
    D --> E["inference.forecast<br/>target (V,T) + covariates<br/>-> ForecastOutput"]
    E --> F["forecast (V,H)<br/>quantiles (V,H,9)"]

    style A fill:#22c55e,stroke:#16a34a,color:#fff
    style D fill:#0ea5e9,stroke:#0284c7,color:#fff
    style E fill:#f59e0b,stroke:#d97706,color:#fff
```

---

## Quick Start

```bash
git clone https://github.com/SNAPKITTYWEST/nano-gemini
cd nano-gemini
pip install -r requirements.txt  # torch, numpy

# Param count
python -c "from src.parameters import G6_PARAMS, NANO_PARAMS; print(f'Nano {NANO_PARAMS.total:,}  G6 {G6_PARAMS.total:,}')"

# Torch model (0.3M, fits 3080)
python -c "from src.model import GemmaTimeSeriesTorch; from src.config import NanoConfig; m=GemmaTimeSeriesTorch(NanoConfig); print(sum(p.numel() for p in m.parameters()))"

# Numpy reference forecast (no torch, no weights)
python -c "import numpy as np; from src.inference import forecast; print(forecast(np.random.randn(3,128), horizon=8).forecast.shape)"
# (3, 8)

# With covariates (TimesFM-3 contract)
python << 'PY'
import numpy as np
from src.inference import forecast
target = np.random.randn(3,128)          # (V, T)
past_only = np.random.randn(1,128)       # (C, T)
past_future = np.random.randn(2,136)     # (C, T+H)
out = forecast(target, horizon=8, past_only_covariates=past_only, past_future_covariates=past_future)
print(out.forecast.shape)   # (3, 8)
print(out.quantiles.shape)  # (3, 8, 9)
PY
```

---

## What a Nano Model Is

Nano is **not a downscaled G6**. It is a *contract-faithful miniature* that implements the exact same pipeline -- RevIN, patch 32, RoPE, GQA, GeGLU, quantile head, stitch -- at `d=64` so it runs **and trains** on consumer hardware:

| Resource | Nano (4x64) | G6 (28x4096) FP32 | G6 Int4 |
|----------|-------------|------------------|---------|
| Params | 0.3M | 6.15B | 6.15B |
| VRAM | ~0.01GB | ~24GB | ~3.02GB |
| RTX 3080 10GB | Yes - Fits + trains | No - OOM | Yes - Fits inference |
| Browser (WASM) | Yes | No | No |

Use Nano to **develop and test the pipeline**; swap `NanoConfig -> G6Config` when you have the weights and VRAM.

---

## Structure

```
src/config.py          # NanoConfig / G6Config, QUANTILES 0.1-0.9
src/model.py           # RMSNorm, GeGLU, GQA, GemmaMixingBlock, ResidualAdapter, GemmaTimeSeriesTorch
src/inference.py       # forecast() numpy reference, Acklam inv_cdf, RevIN, stitch
src/normalization.py   # linear_interpolate, revin
src/patches.py         # patch_series, stitch_patches, context_pad
src/validation.py      # shape contract for covariates
```

---

## License

Tri-licensed: **Sovereign Source License v1.0** (Bel Esprit d'Accord Trust, 2026-06-01) | **BSL-1.1** (Change Date 2030-06-01 -> Apache 2.0) | **AGPL-3.0**. See `LICENSE`.

Headers `SNAPKITTYWEST-PROPRIETARY-2026-001` preserved. No Google weights.

Contact: **Ahmad Ali Parr** <ahmedparr93@gmail.com> -- Bel Esprit D'Accord Trust

---

*The Gemma is the backbone. The adapter is the skin. The quantile head is the forecast.*
