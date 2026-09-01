"""Parameter-group routing for FusionOpt.

Splits a model's trainable parameters into two groups:
- "spectral": 2D weight matrices with both dims >= 128 (Muon NS5 + KL-Shampoo path)
- "scalar":   1D params, biases, LayerNorm gains/betas, and small/odd 2D layers
              (latent_proj 256x64, out_proj F x256) — ScheduleFree-AdamW path

The threshold min(shape) >= 128 keeps the spectral path on the RDNA4 256-grid
fast tile while excluding rank-deficient projections where Muon explicitly
says NS5 is inappropriate.

See docs/superpowers/specs/2026-05-29-fusion-optimiser-design.md §2.
"""

from __future__ import annotations

import re
from typing import Iterable

import torch
import torch.nn as nn


MIN_SPECTRAL_DIM = 128

# Fused ATTENTION up-projections whose output rows stack multiple functional projections
# (q,k,v,q_diff,k_diff / q,q_diff / k,v,k_diff — the SA3 differential-attention variant).
# Restricted to attention names so MLP up-projections (also k*dim wide, but ONE projection)
# are NOT split. Matches the OUTPUT-side param (lora_B / full weight); lora_A is input-side
# (out=rank<dim) so it naturally gets block_count 1. See TASK C, C's 2026-07-30 ruling.
_FUSED_ATTN_RE = re.compile(r"\.(to_qkv|to_kv|to_q)\.")
_TO_OUT_RE = re.compile(r"\.to_out\.")

# CMuon / AdaLN: the fused modulation projection that emits scale/shift/gate (self + ff).
# In SA3 this is the second Linear of `global_cond_embedder` (Sequential(Linear, SiLU, Linear)),
# a [k*dim, dim] tensor whose k output row-blocks are functionally-distinct sub-matrices fused
# for efficiency. Applying NS5 to the whole fused tensor couples those independent subspaces; the
# fix (arXiv 2608.02502) is to orthogonalise each dim-row block independently. The input side IS
# the model dim (== shape[1]), so k = shape[0] // shape[1] with no inferred-dim needed. Matches
# the OUTPUT-side emitter (`.2.` of the Sequential); the input Linear `.0.` maps global_cond_dim
# -> dim and is not a multiple-block fusion. See TASK CMuon, arXiv 2608.02502.
_ADALN_RE = re.compile(r"\.global_cond_embedder\.2\.")


def _infer_attn_dim(named_specs) -> int | None:
    """Model attention dim, for the qkv row-block split. to_out maps (heads*head_dim)->dim,
    so its OUTPUT-side param (lora_B (dim,rank) or full weight (dim,dim)) has shape[0]==dim.
    Fallback: GCD of the fused up-projection out-dims (all exact multiples of dim)."""
    import math
    for n, p in named_specs:
        if _TO_OUT_RE.search(n) and (n.endswith("lora_B") or n.endswith(".weight")):
            return int(p.shape[0])
    outs = [int(p.shape[0]) for n, p in named_specs
            if _FUSED_ATTN_RE.search(n) and (n.endswith("lora_B") or n.endswith(".weight"))]
    if outs:
        return math.gcd(*outs) if len(outs) > 1 else outs[0]
    return None


def _block_count(name: str, p, dim: int | None) -> int:
    """How many equal dim-row blocks to orthogonalise this spectral param in. >1 only for a
    fused attention up-projection whose out rows are an exact multiple of dim; else 1 (whole)."""
    if dim and _FUSED_ATTN_RE.search(name) and p.shape[0] % dim == 0 and p.shape[0] // dim > 1:
        return p.shape[0] // dim
    return 1


def _adaln_block_count(name: str, p) -> int:
    """How many equal dim-row blocks to orthogonalise a fused AdaLN modulation projection in
    (CMuon chunking). >1 only for the `global_cond_embedder.2` emitter, whose out rows are an
    exact multiple k of its input dim (k = scale/shift/gate x {self, ff}, k=6 in SA3); else 1.
    The input side IS the model dim (== shape[1]), so k = shape[0] // shape[1] — no inferred dim."""
    if (_ADALN_RE.search(name) and p.ndim == 2 and p.shape[1] > 0
            and p.shape[0] % p.shape[1] == 0 and p.shape[0] // p.shape[1] > 1):
        return p.shape[0] // p.shape[1]
    return 1


def build_fusion_param_groups(
    model: nn.Module,
    force_scalar: Iterable[str] = (),
    spectral_lr: float | None = None,
    scalar_lr: float | None = None,
    spectral_wd: float = 0.01,
    scalar_wd: float = 0.0,
    split_qkv: bool = False,
    split_adaln: bool = False,
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
    if split_qkv or split_adaln:
        # tag each spectral param with its NS5 block count (>1 = a fused up-projection to
        # orthogonalise per dim-row block; the optimiser reads this parallel list per param).
        # split_qkv covers fused attention up-projections (q/k/v + diff blocks); split_adaln
        # covers the fused AdaLN modulation emitter (scale/shift/gate x {self,ff}). Both feed
        # the SAME chunked-NS5 path via spectral_split; they compose (a param is at most one
        # kind, so the max is just "whichever matched"). OFF by default => no key => unchanged.
        dim = _infer_attn_dim(spectral) if split_qkv else None
        splits = []
        for n, p in spectral:
            bc = _block_count(n, p, dim) if split_qkv else 1
            if bc == 1 and split_adaln:
                bc = _adaln_block_count(n, p)
            splits.append(bc)
        spectral_group["spectral_split"] = splits
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
