from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

QUANTILES = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
MEDIAN_INDEX = 4
SIGMA_EPS = 1e-6

@dataclass
class GemmaConfig:
    model_dims: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    intermediate_size: int
    qk_norm: bool = True
    sliding_window: int = 0
    global_every: int = 6

@dataclass
class ResidualConfig:
    hidden_dims: int = 128
    output_dims: int = 64

@dataclass
class MeridianConfig:
    name: str
    size: str
    gemma: GemmaConfig
    n_layers: int
    input_feature_dim: int = 192
    input_patch_len: int = 32
    output_patch_len: int = 64
    quantiles: List[float] = field(default_factory=lambda: QUANTILES.copy())
    max_variates: int = 128
    max_context: int = 15360
    max_horizon: int = 512
    use_variate_attention: bool = True
    residual: ResidualConfig = field(default_factory=ResidualConfig)

NanoConfig = MeridianConfig("Meridian-Nano","nano",
    GemmaConfig(64,4,4,16,128), n_layers=4,
    residual=ResidualConfig(128,64))

G6Config = MeridianConfig("Meridian-G6","g6",
    GemmaConfig(4096,32,8,128,11008,sliding_window=64,global_every=6),
    n_layers=28, residual=ResidualConfig(512,4096))
