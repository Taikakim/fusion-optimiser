"""Minimal end-to-end usage of FusionOpt.

Trains a tiny MLP on random data for a few steps. The point is to
demonstrate the API, not to learn anything meaningful.

Run:  python examples/basic_usage.py
"""

from __future__ import annotations

import torch
import torch.nn as nn

from fusion_optimiser import FusionOpt, build_fusion_param_groups, summarise_groups


class TinyTransformerBlock(nn.Module):
    """A 2-layer MLP that looks transformer-ish (2D weights ≥ 128×128)."""
    def __init__(self, dim: int = 256, hidden: int = 1024):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return x + self.fc2(self.act(self.fc1(self.norm(x))))


class TinyModel(nn.Module):
    def __init__(self, dim: int = 256, depth: int = 3):
        super().__init__()
        self.embed = nn.Linear(64, dim)
        self.blocks = nn.ModuleList([TinyTransformerBlock(dim) for _ in range(depth)])
        self.head = nn.Linear(dim, 1)

    def forward(self, x):
        x = self.embed(x)
        for blk in self.blocks:
            x = blk(x)
        return self.head(x).squeeze(-1)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    model = TinyModel().to(device)

    # Inspect the spectral/scalar split.
    groups = build_fusion_param_groups(model)
    print("param routing:")
    print(summarise_groups(groups))

    optimizer = FusionOpt(
        params=groups,
        lr=3e-4,
        components={"ns5", "normuon", "sf"},   # SF-NorMuon
        hot_dtype="bf16" if device == "cuda" else "fp32",
    )

    # Random data — point is to exercise the optimiser, not to learn.
    for step in range(20):
        x = torch.randn(32, 16, 64, device=device)
        y = torch.randn(32, 16, device=device)
        pred = model(x)
        loss = ((pred - y) ** 2).mean()
        loss.backward()
        optimizer.set_loss(loss.detach())
        optimizer.step()
        optimizer.zero_grad()
        if step % 5 == 0:
            print(f"step {step:3d}  loss={loss.item():.4f}")

    # Schedule-Free: swap to the averaged iterate before saving.
    optimizer.eval()
    torch.save(model.state_dict(), "/tmp/fusion_demo_ckpt.pt")
    print("checkpoint saved (using averaged iterate x_t)")
    optimizer.train()


if __name__ == "__main__":
    main()
