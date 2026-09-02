from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MeridianConfig, NanoConfig

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * (1 + self.weight)

class GeGLU(nn.Module):
    def __init__(self, dim, intermediate):
        super().__init__()
        self.gate = nn.Linear(dim, intermediate, bias=False)
        self.up = nn.Linear(dim, intermediate, bias=False)
        self.down = nn.Linear(intermediate, dim, bias=False)
    def forward(self, x):
        return self.down(F.gelu(self.gate(x), approximate='tanh') * self.up(x))

class GQA(nn.Module):
    def __init__(self, cfg: MeridianConfig):
        super().__init__()
        d = cfg.gemma.model_dims
        hd = cfg.gemma.head_dim
        n_heads = cfg.gemma.n_heads
        n_kv = cfg.gemma.n_kv_heads
        self.n_heads = n_heads
        self.n_kv = n_kv
        self.head_dim = hd
        self.q = nn.Linear(d, n_heads * hd, bias=False)
        self.k = nn.Linear(d, n_kv * hd, bias=False)
        self.v = nn.Linear(d, n_kv * hd, bias=False)
        self.o = nn.Linear(n_heads * hd, d, bias=False)
        self.q_norm = RMSNorm(hd) if cfg.gemma.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(hd) if cfg.gemma.qk_norm else nn.Identity()

    def forward(self, x, causal=True):
        B, S, D = x.shape
        q = self.q(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, S, self.n_kv, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, S, self.n_kv, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        if self.n_heads != self.n_kv:
            k = k.repeat_interleave(self.n_heads // self.n_kv, dim=1)
            v = v.repeat_interleave(self.n_heads // self.n_kv, dim=1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if causal:
            mask = torch.tril(torch.ones(S, S, device=x.device, dtype=torch.bool))
            scores = scores.masked_fill(~mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, S, D)
        return self.o(out)

class GemmaMixingBlock(nn.Module):
    def __init__(self, cfg: MeridianConfig, is_global=False):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.gemma.model_dims)
        self.attn = GQA(cfg)
        self.ffn_norm = RMSNorm(cfg.gemma.model_dims)
        self.mlp = GeGLU(cfg.gemma.model_dims, cfg.gemma.intermediate_size)
        self.is_global = is_global

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x), causal=not self.is_global)
        x = x + self.mlp(self.ffn_norm(x))
        return x

class ResidualAdapter(nn.Module):
    def __init__(self, cfg: MeridianConfig):
        super().__init__()
        in_f = cfg.input_feature_dim
        h = cfg.residual.hidden_dims
        out = cfg.residual.output_dims
        self.fc1 = nn.Linear(in_f, h)
        self.fc2 = nn.Linear(h, out)
        self.skip = nn.Linear(in_f, out)
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x))) + self.skip(x)

class GemmaTimeSeriesTorch(nn.Module):
    def __init__(self, cfg: MeridianConfig = NanoConfig):
        super().__init__()
        self.cfg = cfg
        self.adapter = ResidualAdapter(cfg)
        self.layers = nn.ModuleList([GemmaMixingBlock(cfg, is_global=(i % cfg.gemma.global_every == 0)) for i in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.gemma.model_dims)
        self.head = nn.Linear(cfg.gemma.model_dims, cfg.output_patch_len * len(cfg.quantiles))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, V, N, 192) patches + covariates
        B, V, N, F = x.shape
        h = self.adapter(x)
        for layer in self.layers:
            h = layer(h.view(B*V, N, -1)).view(B, V, N, -1)
        h = self.norm(h)
        logits = self.head(h)
        return logits.view(B, V, N, self.cfg.output_patch_len, len(self.cfg.quantiles))
