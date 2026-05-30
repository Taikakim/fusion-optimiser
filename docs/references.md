# References

The seven inspirations that FusionOpt composes. The novelty in this
repo is the **composition**, the **bifurcated routing** (spectral vs
scalar groups), the **shared Polyak step size** across both groups,
and the **single inference cache abstraction** for adaLN-zero models.
The building blocks are all published.

## Muon

Keller Jordan. *Muon: An optimizer for hidden layers in neural networks.*
[github.com/KellerJordan/Muon](https://github.com/KellerJordan/Muon).

Newton-Schulz quintic orthogonalisation applied to the momentum of 2-D
weight tensors. Bounds the spectral norm of the update; aligns the
geometry with the weight matrix's natural scaling.

The NS5 polynomial coefficients (3.4445, −4.7750, 2.0315) come from
Jordan's empirical tuning to maximise the smallest singular value after
5 iterations from a Frobenius-normalised input.

## NorMuon

*NorMuon: Per-neuron norm correction for Muon-style optimisers.*
[arXiv:2510.05491](https://arxiv.org/abs/2510.05491).

Fixed-iteration NS5 leaves uneven row magnitudes; NorMuon renormalises
each output neuron's row before the update is applied. Cheap, helps
consistently.

## MONA

*Momentum with Online-Newton Adjustment.*
[arXiv:2605.26842](https://arxiv.org/abs/2605.26842).

An EMA of consecutive gradient differences is injected into momentum
as a curvature proxy, biasing the trajectory away from sharp minima.
Validated at 1 B – 68 B MoE pretraining scale.

## KL-Shampoo

*KL-Shampoo: Two-sided Kronecker covariance preconditioner via KL
divergence.*
[arXiv:2509.03378](https://arxiv.org/abs/2509.03378).

Matches Shampoo's geometry without requiring Adam grafting. The
covariance matrices L (left) and R (right) are accumulated via EMA;
their inverse fourth-roots become the per-side preconditioners. **The
eigendecomp must run in fp32** — fp16 eigendecomp on SPD covariance
matrices produces garbage.

## Schedule-Free

Defazio et al. *The Road Less Scheduled.*
[github.com/facebookresearch/schedule_free](https://github.com/facebookresearch/schedule_free).

Anytime-stopping framework: gradients are computed at the
**evaluation point** y_t = (1−β)·z_t + β·x_t, where z_t is the fast
iterate (updated by the optimiser step) and x_t is the slow average
(deployed at inference). No learning-rate schedule.

## ScheduleFree+

*ScheduleFree+: Polyak step size on top of Schedule-Free.*
[arXiv:2605.19095](https://arxiv.org/abs/2605.19095).

Removes the last hyperparameter (learning rate) by deriving the step
size from the running ratio of loss-EMA to gradient-norm-EMA:

```
γ_t = γ_base · clamp(loss_ema / gnorm_ema, 0.1, 10)
```

LR-free training at LLM scale. We share this γ between the spectral
and scalar paths.

## SF-NorMuon

*SF-NorMuon: Schedule-Free + NorMuon + weight decay on the fast iterate.*
[arXiv:2605.23061](https://arxiv.org/abs/2605.23061).

The load-bearing stability insight: apply weight decay to the fast
iterate z_t, **not** to the deployed average x_t. The latter lets z_t
drift unboundedly over thousands of steps; the former keeps the entire
trajectory inside a regularised ball.

This is the recipe we ended up wanting to ship (we discovered it
empirically via per-component ablation; it turns out to be precisely
the SF-NorMuon paper).

## Diffusion model conditioning context

Peebles & Xie. *Scalable Diffusion Models with Transformers* (DiT).
[arXiv:2212.09748](https://arxiv.org/abs/2212.09748).

The adaLN-zero conditioning scheme used in DiT is what makes the
`TimeConditioningCache` work — per-block modulators are pure functions
of t and weights, so they can be precomputed once and reused across
every sampler step at the same t.
