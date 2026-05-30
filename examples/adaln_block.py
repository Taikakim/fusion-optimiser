"""Minimal adaLN-zero block with the optional `mods=...` kwarg.

The TimeConditioningCache needs every block to accept precomputed
modulators. The pattern is: if `mods is None`, compute live; otherwise
unpack the cached tuple. The live path stays bit-identical to the
no-cache case.

Run:  python examples/adaln_block.py
"""

from __future__ import annotations

import torch
import torch.nn as nn

from fusion_optimiser import TimeConditioningCache


class AdaLNBlock(nn.Module):
    """A DiT-style block with adaLN-zero conditioning.

    Note the optional `mods` kwarg on forward(). The cache injects
    precomputed (g1, b1, a1, g2, b2, a2) tuples here.
    """
    def __init__(self, dim: int = 256, num_heads: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(),
                                 nn.Linear(4 * dim, dim))
        # The cache REQUIRES this attribute name: adaLN_mod.
        self.adaLN_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        # Zero-init the modulator's output projection (the "zero" in adaLN-zero).
        nn.init.zeros_(self.adaLN_mod[-1].weight)
        nn.init.zeros_(self.adaLN_mod[-1].bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor,
                mods: tuple | None = None) -> torch.Tensor:
        if mods is None:
            g1, b1, a1, g2, b2, a2 = self.adaLN_mod(t_emb).chunk(6, dim=-1)
        else:
            g1, b1, a1, g2, b2, a2 = mods

        # adaLN-zero: scale-shift the LayerNorm output, gate the residual.
        h = self.norm1(x) * (1 + g1.unsqueeze(1)) + b1.unsqueeze(1)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + a1.unsqueeze(1) * attn_out

        h = self.norm2(x) * (1 + g2.unsqueeze(1)) + b2.unsqueeze(1)
        x = x + a2.unsqueeze(1) * self.mlp(h)
        return x


class AdaLNModel(nn.Module):
    """A DiT-style model with `t_embedder` and `blocks`.

    The cache REQUIRES these attribute names: `t_embedder`, `blocks`.
    """
    def __init__(self, dim: int = 256, depth: int = 3):
        super().__init__()
        self.t_embedder = nn.Sequential(
            nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.blocks = nn.ModuleList([AdaLNBlock(dim) for _ in range(depth)])

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Cache lookup: only valid if t is uniform across the batch.
        cache = getattr(self, "_time_cache", None)
        cache_entry = None
        if cache is not None and t.numel() > 0:
            t_first = t.flatten()[0]
            if t.numel() == 1 or torch.all(t == t_first):
                cache_entry = cache.get(float(t_first.item()))

        if cache_entry is not None:
            t_emb = cache_entry["t_emb"]                 # (1, dim) — broadcasts
            block_mods = cache_entry["modulators"]
        else:
            t_emb = self.t_embedder(t.unsqueeze(-1))
            block_mods = [None] * len(self.blocks)

        for i, block in enumerate(self.blocks):
            x = block(x, t_emb, mods=block_mods[i])
        return x


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AdaLNModel().to(device).eval()

    # Warm the cache for the sampler's t values.
    cache = TimeConditioningCache(model, device=device)
    cache.warm_for_schedule(n_steps=40, include_zero=True)
    model._time_cache = cache

    # Run a few "sampler steps" — cache should hit 100 %.
    x = torch.randn(4, 16, 256, device=device)
    schedule = torch.linspace(1.0, 0.0, 41)[:-1].tolist()
    with torch.no_grad():
        for t_val in schedule[:5]:
            t = torch.full((4,), t_val, device=device)
            _ = model(x, t)
    stats = cache.hit_stats
    print(f"cache stats: hits={stats['hits']} misses={stats['misses']}")
    print(f"cached values: {len(cache.cached_values)}")


if __name__ == "__main__":
    main()
