# FusionOpt

A composed PyTorch optimiser that fuses **Muon** (spectral orthogonalisation),
**NorMuon** (per-neuron row scaling), **Schedule-Free** averaging,
**MONA** (curvature-aware momentum), **KL-Shampoo** (Kronecker preconditioner)
and **ScheduleFree+** (Polyak step size) — plus a `TimeConditioningCache`
for diffusion-model inference speedup.

Targets small-to-medium DiT-style networks (~5 M – ~10 B parameters,
adaLN-zero conditioning, transformer-style 2D weight matrices). Public
domain (CC0 1.0) — see [`LICENSE`](LICENSE).

> ⚠️ **Research code.** The composition is novel; the components are
> well-cited. Treat numbers from the case study as evidence, not as
> guarantees for your workload. See [`docs/best_practices.md`](docs/best_practices.md)
> and [`docs/open_questions.md`](docs/open_questions.md).

---

## TL;DR — production recipe

```python
import torch
from fusion_optimiser import FusionOpt, build_fusion_param_groups

param_groups = build_fusion_param_groups(model)
optimizer = FusionOpt(
    params=param_groups,
    lr=3e-4,
    components={"ns5", "normuon", "sf"},  # SF-NorMuon: the load-bearing subset
    hot_dtype="bf16",                     # do NOT use "fp16" — NS5 overflows
)
```

That's **SF-NorMuon** at bf16. Empirically this captures ~95 % of the
quality lift of the full composition at ~50 % of the wall-clock cost,
on small-to-medium DiT-style models with adaLN-zero conditioning.

**For diversity-incentivised training** (penalising similarity to a frozen
reference) use the full composition warm-started from an SF-NorMuon
checkpoint instead — see the "Diversity training" section below.

---

## Install

```bash
pip install -e .
```

Requires: `torch >= 2.2`. No other runtime dependencies.

---

## Applicability — when does FusionOpt help?

✓ **Yes**:
- Transformer-style 2D weight matrices (qkv, projections, MLPs ≥ 128 × 128)
- adaLN-zero conditioning (gives the time cache its leverage; not required)
- Mid-range scales (5 M – 10 B params per the paper validations)
- BF16- or FP16-friendly hardware with tuned matmul kernels

✗ **No / unclear**:
- Pure conv nets — spectral methods like Muon are matrix-aware, less
  appropriate for 4-D conv kernels
- Very small networks (< 1 M params) — overhead may dominate
- Workloads where per-element adaptivity matters more than spectral
  geometry (e.g. very sparse gradients)

---

## What FusionOpt fuses

All seven building blocks are published optimisers. The novelty is the
**composition**, not the components.

| Component | Mechanism | Source |
|---|---|---|
| **Muon** | Newton-Schulz quintic orthogonalisation on 2D weights | Keller Jordan |
| **NorMuon** | Per-neuron row-norm normalisation after NS5 | arXiv:2510.05491 |
| **MONA** | EMA of gradient differences as curvature proxy | arXiv:2605.26842 |
| **KL-Shampoo** | Two-sided Kronecker preconditioner via KL divergence | arXiv:2509.03378 |
| **Schedule-Free** | Averaged-iterate framework, no LR schedule | Defazio et al. |
| **ScheduleFree+** | Polyak step size on top of Schedule-Free | arXiv:2605.19095 |
| **SF-NorMuon** | Schedule-Free + NorMuon + WD on the **fast** iterate z_t | arXiv:2605.23061 |

Full citations in [`docs/references.md`](docs/references.md).

### Stabilisers added since the first release

Four mechanisms landed after the initial packaging (2026-06-20) and are now
part of `optimizer.py`. All are **off by default** — the recipe above is
unchanged without them.

| Control | What it does | Why it exists |
|---|---|---|
| **`hyperball`** | Freezes each spectral matrix's Frobenius norm at ‖W₀‖_F, optimizer-side | Full-fine-tune latent-scale runaway: weights grow, the decoder is driven out of distribution, output degenerates into a spectral drone. A norm-freeze bounds it without weight decay — the two are alternatives, and `weight_decay` is **inert** when hyperball is on. |
| **`autoscale`** | Scales the finalized spectral update by its own signal-to-noise ratio | Removes a hand-tuned LR multiplier on the spectral path. |
| **`snr_gate`** | Row/tensor-wise SNR gate on the update | Same family as autoscale; gates rather than scales. |
| **`apply_cautious`** | Masks update elements that disagree in sign with the gradient | Cautious-optimizer style; cheap variance reduction. |

**Weight decay is a GROUP property, not an optimizer flag.** `build_fusion_param_groups`
takes `spectral_wd` (default 0.01) and `scalar_wd` (default 0.0) and sets each group's
`weight_decay`; the optimizer just consumes it. Look in `groups.py`, not `optimizer.py`,
when tuning it.

---

## How it works (one screen)

**Bifurcated routing.** Parameters are split into two groups:

- **Spectral path** — 2D matrices with both dims ≥ 128. Gets the full
  composed update (NS5 + NorMuon + optional MONA / KL-Shampoo,
  wrapped in Schedule-Free with WD on the fast iterate).
- **Scalar path** — biases, LayerNorm, embeddings, small/odd matrices.
  Gets ScheduleFree-AdamW.

Both paths share a **Polyak step size**

```
γ_t = γ_base · clamp(loss_ema / gnorm_ema, 0.1, 10)
```

(all reductions on-device, no host syncs in the optimiser hot loop).

**Time cache for adaLN-zero models.** At fixed sampler step counts, the
time embedding `t_emb` and per-block modulators `(g1, b1, a1, g2, b2, a2)`
are pure functions of `t` and weights → cacheable. Saves ~5–10 % render
latency on small models with many sampler steps.

---

## Quickstart

The shortest possible training loop:

```python
from fusion_optimiser import FusionOpt, build_fusion_param_groups

model = YourModel().to("cuda")
optimizer = FusionOpt(
    params=build_fusion_param_groups(model),
    lr=3e-4,
    components={"ns5", "normuon", "sf"},
    hot_dtype="bf16",
)

for batch in loader:
    loss = model(batch).mean()
    loss.backward()
    optimizer.set_loss(loss.detach())    # feeds the Polyak γ; call BEFORE step()
    optimizer.step()
    optimizer.zero_grad()

# Schedule-Free deploys from the averaged iterate x_t:
optimizer.eval()        # swap live weights -> averaged x_t
torch.save(model.state_dict(), "model.pt")
optimizer.train()       # restore live weights to keep training
```

See [`examples/basic_usage.py`](examples/basic_usage.py) for a runnable
end-to-end script.

---

## Inference acceleration (adaLN-zero models)

For a DiT-style model with `adaLN_mod` per block, sampled at a fixed
step count:

```python
from fusion_optimiser import TimeConditioningCache, get_or_build_cache

cache = get_or_build_cache(model, model_path="model.pt", n_steps=40, device="cuda")
model._time_cache = cache    # forward() picks it up

# render as usual — first call warms the cache, subsequent calls hit 100 %.
```

The block forward needs to accept an optional `mods=...` kwarg so the cache
can inject precomputed modulators. See
[`examples/adaln_block.py`](examples/adaln_block.py) for the wiring pattern.

---

## Diversity training (parallel "personality" heads)

For training a head to be *different* from a frozen reference (negative
loss component on `MSE(pred, ref_pred)`):

- **Recommended**: warm-start from an SF-NorMuon checkpoint, switch to
  the **full** composition (`{"ns5", "normuon", "sf", "mona", "shampoo"}`),
  apply the diversity penalty.
- **Not recommended**: bare SF-NorMuon under a diversity penalty drifts
  into incoherence; AdamW under a diversity penalty NaNs.

The KL-Shampoo + MONA components dropped from production for cost are
**load-bearing stabilisers** under a magnitude-unbounded negative loss
term. See [`docs/results.md`](docs/results.md) for the case-study
evidence.

---

## Docs

- [`docs/results.md`](docs/results.md) — empirical case-study findings
- [`docs/references.md`](docs/references.md) — full paper citations + arXiv links
- [`docs/best_practices.md`](docs/best_practices.md) — recipes and gotchas
- [`docs/open_questions.md`](docs/open_questions.md) — things we don't know yet,
  and how you could help verify them
- [`docs/porting_notes.md`](docs/porting_notes.md) — verification checklist for new projects

---

## Citation

If FusionOpt helps your work, citing the underlying papers (see
`docs/references.md`) is appreciated. Citing this repo is optional —
the project is dedicated to the public domain.

---

## License

[CC0 1.0 Universal](LICENSE) — public domain dedication. No rights
reserved. See `LICENSE` for the project-relevant notes about underlying
research.
