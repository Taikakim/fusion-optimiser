"""Parameter-group routing for FusionOpt.

Splits a model's trainable parameters into two groups:
- "spectral": 2D weight matrices with both dims >= 128 (Muon NS5 + KL-Shampoo path)
- "scalar":   1D params, biases, LayerNorm gains/betas, and small/odd 2D layers
              (latent_proj 256x64, out_proj F x256) — ScheduleFree-AdamW path

The threshold min(shape) >= 128 keeps the spectral path on matmul-favourable
tile sizes while excluding rank-deficient projections where Muon explicitly
says NS5 is inappropriate.

See README.md for the recipe; docs/best_practices.md for the rationale on
the routing threshold and `force_scalar` escape hatch.
"""

from __future__ import annotations

import re
from typing import Iterable

import torch
import torch.nn as nn


MIN_SPECTRAL_DIM = 128


def build_fusion_param_groups(
    model: nn.Module,
    force_scalar: Iterable[str] = (),
    spectral_lr: float | None = None,
    scalar_lr: float | None = None,
    spectral_wd: float = 0.01,
    scalar_wd: float = 0.0,
) -> list[dict]:
    """Return torch.optim-compatible param groups for FusionOpt.

    Args:
        model: the nn.Module whose parameters to route.
        force_scalar: iterable of regex patterns; matching param names are
            forced onto the scalar path (escape hatch for layers that misbehave
            under spectral updates).
        spectral_lr, scalar_lr: optional per-group LR overrides; if None, the
            optimiser's default `lr` is used.
        spectral_wd, scalar_wd: weight-decay on the FAST iterate z_t for each
            group (SF-NorMuon convention; not on the averaged iterate).
    """
    spectral, scalar = [], []
    force_patterns = [re.compile(pat) for pat in force_scalar]

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(pat.search(name) for pat in force_patterns):
            scalar.append((name, p))
            continue
        if p.ndim == 2 and min(p.shape) >= MIN_SPECTRAL_DIM:
            spectral.append((name, p))
        else:
            scalar.append((name, p))

    spectral_group = {
        "params": [p for _, p in spectral],
        "param_names": [n for n, _ in spectral],
        "group_type": "spectral",
        "weight_decay": spectral_wd,
    }
    scalar_group = {
        "params": [p for _, p in scalar],
        "param_names": [n for n, _ in scalar],
        "group_type": "scalar",
        "weight_decay": scalar_wd,
    }
    if spectral_lr is not None:
        spectral_group["lr"] = spectral_lr
    if scalar_lr is not None:
        scalar_group["lr"] = scalar_lr

    return [spectral_group, scalar_group]


def summarise_groups(groups: list[dict]) -> str:
    """Human-readable summary, for logging at training start-up."""
    lines = []
    for g in groups:
        n_params = sum(p.numel() for p in g["params"])
        lines.append(
            f"  [{g['group_type']:>8}] {len(g['params']):>3} tensors  "
            f"{n_params:>10,d} params  wd={g['weight_decay']}"
        )
        for name, p in zip(g.get("param_names", []), g["params"]):
            lines.append(f"      {name:<48} {tuple(p.shape)}")
    return "\n".join(lines)
