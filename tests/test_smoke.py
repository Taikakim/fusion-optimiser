"""Smoke tests — does the optimiser run without crashing?

Not a quality validation. Just confirms the API + param routing +
all components compose without exceptions on CPU.

Run:  python -m pytest tests/ -v
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from fusion_optimiser import (
    FusionOpt,
    build_fusion_param_groups,
    summarise_groups,
    newton_schulz_5,
)


class TinyMLP(nn.Module):
    def __init__(self, dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def _step(optimizer, model, n_steps: int = 3):
    for _ in range(n_steps):
        x = torch.randn(8, 16, 256)
        y = torch.randn(8, 16, 256)
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        optimizer.set_loss(loss.detach())
        optimizer.step()
        optimizer.zero_grad()


def test_param_routing_basic():
    model = TinyMLP()
    groups = build_fusion_param_groups(model)
    assert len(groups) == 2
    spectral, scalar = groups
    assert spectral["group_type"] == "spectral"
    assert scalar["group_type"] == "scalar"
    assert len(spectral["params"]) >= 1, "expected at least one 2D weight in spectral"
    assert len(scalar["params"]) >= 1, "expected biases in scalar"
    # summarise_groups returns a human-readable string.
    summary = summarise_groups(groups)
    assert isinstance(summary, str) and "spectral" in summary and "scalar" in summary


def test_sf_normuon_runs():
    model = TinyMLP()
    optimizer = FusionOpt(
        params=build_fusion_param_groups(model),
        lr=1e-3,
        components={"ns5", "normuon", "sf"},
        hot_dtype="fp32",
    )
    _step(optimizer, model)


def test_full_fusion_runs():
    model = TinyMLP()
    optimizer = FusionOpt(
        params=build_fusion_param_groups(model),
        lr=1e-3,
        components={"ns5", "normuon", "sf", "mona", "shampoo"},
        hot_dtype="fp32",
    )
    _step(optimizer, model)


@pytest.mark.parametrize("hot_dtype", ["fp32", "bf16"])
def test_hot_dtypes(hot_dtype):
    model = TinyMLP()
    optimizer = FusionOpt(
        params=build_fusion_param_groups(model),
        lr=1e-3,
        components={"ns5", "normuon", "sf"},
        hot_dtype=hot_dtype,
    )
    _step(optimizer, model)


def test_schedule_free_eval_train_swap():
    model = TinyMLP()
    optimizer = FusionOpt(
        params=build_fusion_param_groups(model),
        lr=1e-3,
        components={"ns5", "normuon", "sf"},
        hot_dtype="fp32",
    )
    _step(optimizer, model)
    w_train = model.fc1.weight.detach().clone()
    optimizer.eval()
    w_eval = model.fc1.weight.detach().clone()
    optimizer.train()
    w_train2 = model.fc1.weight.detach().clone()
    # eval-swap should give a different weight; train-swap should restore.
    assert not torch.allclose(w_train, w_eval), "eval() didn't swap weights"
    assert torch.allclose(w_train, w_train2), "train() didn't restore weights"


def test_newton_schulz_orthogonalises():
    """NS5 should produce a matrix with singular values clustered near 1."""
    torch.manual_seed(0)
    A = torch.randn(128, 128)
    U = newton_schulz_5(A / A.norm())
    s = torch.linalg.svdvals(U)
    # NS5 from a Frobenius-normalised input -> singular values clustered
    # near 1 (much tighter than the input's wide spread of ~[0, 0.2]).
    assert s.min() > 0.2, f"smallest singular value too small: {s.min().item()}"
    assert s.max() < 1.5, f"largest singular value too large: {s.max().item()}"


def test_force_scalar_overrides_routing():
    model = TinyMLP()
    groups = build_fusion_param_groups(model, force_scalar=[r"fc1\.weight"])
    spectral_params = sum(p.numel() for p in groups[0]["params"])
    scalar_params = sum(p.numel() for p in groups[1]["params"])
    # fc1.weight (the 256 x 512 matrix) should now be in the scalar group.
    assert scalar_params > spectral_params, "force_scalar didn't move fc1.weight"
