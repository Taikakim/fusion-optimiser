# Open questions — things to verify

Everything in this file is **untested by this repo** or **specific to
our case-study setup and not validated elsewhere**. If you have evidence
either way, please open an issue or PR.

## Open question 1 — does SF-NorMuon's small-scale dominance extend to LLM scale?

Our case study is 5 M-param control heads. The MONA paper validates at
1 B – 68 B MoE pretraining. We do **not** know whether:

- MONA's curvature deflection starts paying off at some param-count
  threshold (and where), or whether
- SF-NorMuon's "ditch MONA" recommendation continues to dominate at LLM
  scale.

**What would help**: a side-by-side at the 100 M – 1 B scale on a
standard pretraining benchmark.

## Open question 2 — does the bf16 mandate hold on NVIDIA Hopper / Blackwell?

NS5 overflow in fp16 is a mathematical property of the polynomial
intermediates, so we expect fp16 to NaN on NVIDIA too. But we have not
tested. The interesting frontier is fp16 with the rescale-restore trick
fused into the matmul epilogue — on RDNA4 we tested and rejected it
because the rescale overhead dominated. On NVIDIA the cost ratios are
different and a fused Triton kernel could be a real win.

**What would help**: someone with H100/Blackwell access trying
`hot_dtype="fp16"` on a small model and reporting (a) the matrix size
at which it NaNs, and (b) whether a fused rescale kernel changes the
picture.

## Open question 3 — is the diversity-training "Full Fusion for negative losses" finding workload-specific?

We saw clearly in our case study that:
- AdamW + diversity penalty → NaN
- SF-NorMuon + diversity → incoherent
- Full Fusion warm-started + diversity → drifted-but-structured ✓

This was one workload (audio control heads). We **suspect** it
generalises — MONA's curvature term and KL-Shampoo's preconditioner do
the work of bounding the negative-loss component's destabilising
direction — but we have no second data point.

**What would help**: a repro on any other architecture with a
similarity-penalty / contrastive-divergence-style negative loss term.

## Open question 4 — does the depth-vs-width tradeoff for inference cost generalise?

In our case study, depth 6 → 4 cost 1.6 % quality for 33 % inference
savings; halving width was much worse. **Width contributes squared,
depth linear** in the FLOP count, which is a general statement, but
the quality-per-FLOP tradeoff is workload-dependent.

**What would help**: anyone testing the same shrink ratios on a
different DiT-style architecture.

## Open question 5 — Polyak γ clamp range

The default `clamp(loss_ema / gnorm_ema, 0.1, 10)` was tuned on our
workload's loss/gradient magnitudes. If your workload has very different
absolute scales (e.g. very small losses or very large gradients), the
clamp may saturate one side and the γ will be constant — defeating the
point of the Polyak step.

**What would help**: instrument `fusion/gnorm_ema`, `fusion/loss_ema`,
`fusion/gamma_curr` for the first ~1 k steps on your workload and
report whether γ floats freely between the clamps.

## Open question 6 — does the TimeConditioningCache invalidation handle EMA-of-weights properly?

The cache invalidates by checking `data_ptr()` of the first trainable
2-D parameter. If you use Schedule-Free's `eval()`/`train()` swap, the
data_ptr changes and the cache invalidates correctly. If you do
something more exotic (an external EMA of weights swapped in for
inference), check that the cache invalidates when expected.

**What would help**: a test case demonstrating cache behaviour under
external weight EMA, and a fix if it misbehaves.

## Open question 7 — pure-convolutional models

Muon is matrix-aware; the natural application to 4-D conv kernels is to
reshape the kernel into 2-D (out_channels, in_channels × kH × kW) and
treat it as a 2-D matrix. We have **not** tested this. The reshape may
or may not respect the spectral structure that NS5 is bounding.

**What would help**: a small CNN experiment with the reshape-and-route
pattern, comparing to AdamW on the same model.

## Open question 8 — Schedule-Free with gradient accumulation

Schedule-Free's z_t / x_t bookkeeping assumes one optimiser step per
gradient. With gradient accumulation, you take N forward passes for
each `optimizer.step()`. We believe this is fine (the averaging just
happens at the slower cadence) but have not stress-tested.

**What would help**: anyone running this under aggressive gradient
accumulation (≥ 16 micro-batches/step) reporting whether train/val
curves look normal.

---

## How to contribute findings

Open an issue at the repo with:
1. Your hardware + dtype
2. The model size / architecture
3. The specific question above your evidence addresses
4. Numbers (the rougher the better — point estimates with confidence
   notes beat polished but underspecified results)

Public domain repo, public domain contributions. Nothing fancy.
