"""Hyperball must not silently freeze a ZERO-INITIALISED parameter at zero.

Why (ported from the live implementation, CONTINUITY 2026-09-09, found while scoping a
Lion+hyperball morph-conditioner run): hyperball constrains W to the sphere of radius
R = ‖W0‖_F, captured on the first step. Every adapter trained against this optimiser is
typically zero-init by construction (e.g. LoRA/DoRA's `lora_B`, or a zero_module()'d
output projection). For those, R = 0, so the update −γ·R·û is zero AND the
renormalisation R·W̃/‖W̃‖ is zero: the parameter is pinned at exactly zero for the whole
run. The loss looks like the unconditioned model and an A/B reads "no control" — a
broken instrument that reports a null instead of an error.

So a zero-norm parameter falls back to the unconstrained update and says so, loudly.
"""
import torch

from fusion_optimiser import FusionOpt


def _step_n(p, n=6, **kw):
    opt = FusionOpt([{"params": [p], "group_type": "spectral"}],
                    lr=1e-2, warmup_steps=0, components={"ns5", "normuon"}, **kw)
    for _ in range(n):
        opt.zero_grad()
        # constant, full-rank gradient: any working optimizer moves p away from init
        p.grad = torch.ones_like(p)
        opt.step()
    return p


def test_zero_init_param_still_trains_under_hyperball():
    p = torch.nn.Parameter(torch.zeros(8, 8))
    _step_n(p, hyperball=True)
    assert p.data.norm() > 0, "zero-init param stayed pinned at zero under hyperball"


def test_nonzero_param_norm_is_still_frozen_under_hyperball():
    """The guard must not weaken hyperball where it legitimately applies."""
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(8, 8))
    R0 = p.data.norm().item()
    _step_n(p, hyperball=True)
    assert abs(p.data.norm().item() - R0) < 1e-3, "hyperball stopped preserving ‖W‖"


def test_zero_init_matches_the_unconstrained_update():
    """The fallback is the ordinary update, not some third behaviour."""
    torch.manual_seed(0)
    a = torch.nn.Parameter(torch.zeros(8, 8))
    b = torch.nn.Parameter(torch.zeros(8, 8))
    _step_n(a, hyperball=True)
    _step_n(b, hyperball=False)
    assert torch.allclose(a.data, b.data, atol=1e-6)
