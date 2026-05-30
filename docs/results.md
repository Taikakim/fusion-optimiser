# Case-study results

These numbers come from one specific deployment: small audio-control heads
(~5 M params, LatCH heads on Stable Audio Open's small DiT latent space)
trained on AMD RX 9070 XT (RDNA4, gfx1201) with ROCm 7.2.x and hipBLASLt
build `dabb6df2b98`. They are evidence, not promises.

If you reproduce these on different hardware / scales, please open an issue —
the porting evidence is exactly what this repo needs more of.

## Headline findings (what transferred-out evidence is strongest)

| Finding | Confidence | Why |
|---|---|---|
| **bf16 is mandatory for NS5** | Very high | Mathematical: NS5 polynomial intermediates blow through fp16's 65 504 ceiling on any matrix bigger than ~256×256. NaN within a few steps. |
| **SF-NorMuon ≈ 95 % of full-Fusion lift, ~50 % the cost** | High in the small-DiT regime | Per-component ablation. The bigger pieces (MONA, KL-Shampoo) don't earn their wall-clock at small scale. |
| **WD on the fast iterate z_t (not the average x_t) is the stability mechanism** | High | Matches SF-NorMuon paper theory + we observed z_t drift to ∞ when WD was applied to x_t over thousands of steps. |
| **`torch.compile` is critical** | High | Without it, the spectral path overhead dominates. With it, per-step cost matches AdamW. |
| **adaLN-zero modulators are cacheable at fixed step counts** | High | Pure functions of t and weights. ~5–10 % render-latency reduction in our setup. |

## Quality lift over AdamW (case-study)

5 M-param control heads, 30 epochs, same wall-clock budget:

| Optimiser | val_MAE | Δ vs AdamW | it/s vs AdamW |
|---|---|---|---|
| AdamW | 3.23 | — | 1.00× |
| Muon (bare) | 3.14 | −2.7 % | 0.92× |
| Muon + MONA | 3.05 | −5.8 % | 0.84× |
| **SF-NorMuon** (ns5+normuon+sf) | **2.99** | **−7.4 %** | **1.00×** *(after compile + depth co-tune)* |
| Full Fusion (ns5+normuon+sf+mona+shampoo) | 2.96 | −8.4 % | 0.51× |

Read: **SF-NorMuon dominates the Pareto frontier** on this workload. Full
Fusion buys 1 % more quality for 2× the wall-clock.

## Findings that may or may not transfer

These were specific to our setup (5 M-param models, RDNA4). Test before
shipping at your scale:

1. **MONA at small scale**: in our case-study, MONA + Muon *underperforms*
   pure Muon. The MONA paper validates the recipe at 1 B–68 B MoE scale.
   The curvature deflection may help at LLM scale but didn't help here.

2. **KL-Shampoo standalone diverges to NaN.** The SPD covariance
   preconditioner has no spectral-norm bound; without NS5 to clip the
   update magnitude, the rescaling blows up. This is the classical
   Shampoo instability — KL-Shampoo *must be composed with NS5* (or
   grafted with Adam) to be stable. We expect this to hold at any scale.

3. **Depth-shrink vs width-shrink tradeoff for inference cost**: in our
   case study, depth 6 → 4 cost 1.6 % quality for 33 % inference savings;
   halving width was much worse. Width contributes squared, depth linear
   — **depth is the cheaper axis** when you need to claw back FLOPs.

## Hardware-specific (RDNA4 / ROCm 7.2.x case study)

These will probably differ on NVIDIA or other AMD generations:

1. **TunableOp autotuning matters.** On a fresh cache, the bf16 path may
   have no tuned kernels; first run takes ~30 % longer. Set
   `PYTORCH_TUNABLEOP_TUNING="1"` BEFORE `import torch` — late
   application is silently ignored.

2. **fp16 with fp32 polynomial accumulation (`hot_dtype="fp16_safe"`)
   is a quality knob, not a speed knob.** On RDNA4 it ran 0.5× bf16
   throughput but delivered val_MAE slightly better than fp32. Useful
   for refinement passes where 1 % quality matters more than wall-clock.

3. **fp16 with fused rescale-restore inside NS5** was tested and
   rejected: per-tensor rescale overhead (~6 max-reductions × matmuls
   per step) dominates the fp16 throughput win on RDNA4. On NVIDIA
   H100/Blackwell with different reduction-vs-matmul throughput ratios
   this *might* be different — testable.

## Diversity training (negative-loss penalty against a frozen reference)

Tested in the case study. Single most surprising finding:

> **The production SF-NorMuon recipe is NOT the right choice for
> diversity training.** The KL-Shampoo + MONA components we dropped
> from production for cost are load-bearing stabilisers under a
> magnitude-unbounded negative loss term.

| Optimiser | Init | Result |
|---|---|---|
| AdamW + diversity | fresh OR warm-start | **NaN** within ~1 k steps |
| SF-NorMuon + diversity | fresh init | "alien coherent" — val_MAE 6.04, deriv-corr 0.01 (direction-uncorrelated) |
| **Full Fusion + diversity** | **warm-start from SF-NorMuon ship head** | **"drifted but structured"** — val_MAE 4.19 vs 3.03 reference, deriv-corr 0.33. ✓ usable variant. |

Two recipes for two purposes: SF-NorMuon for score-driven heads, Full
Fusion (warm-started) for parallel "personality" variants.

## Inference cache hit rates (case study)

At 40 sampler steps, n_iter=4 mean-guidance loop:

- t=0 cache hits: ~4 calls/step × 40 steps = 160 hits
- t=t_curr cache hits: ~1 call/step × 40 steps = 40 hits
- ~400 ms saved per render (most of which is from the t=0 path)

Observed 100 % cache hit rate after the first render at a given
(checkpoint, n_steps) pair.
