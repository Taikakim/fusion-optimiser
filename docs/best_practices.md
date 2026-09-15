# Best practices

Lessons from one deployment. Generalises with caution — see
[`open_questions.md`](open_questions.md) for what we don't know.

## Choosing the recipe

| Goal | Components | Rationale |
|---|---|---|
| **Standard training** | `{"ns5", "normuon", "sf"}` — SF-NorMuon | Captures ~95 % of full-Fusion quality at ~50 % the wall-clock cost in the small-DiT regime. |
| **Diversity / adversarial-loss training** | `{"ns5", "normuon", "sf", "mona", "shampoo"}` — Full Fusion, warm-start from SF-NorMuon ship | MONA + KL-Shampoo are load-bearing stabilisers under magnitude-unbounded negative loss components. |
| **LLM-scale pretraining** | Probably full set, but **untested by this repo** — see references. | The MONA and ScheduleFree+ papers validate at 1 B – 68 B scales individually; we have no joint evidence. |

## Choosing `hot_dtype`

| Value | When | Why |
|---|---|---|
| `"bf16"` | **Default for training.** | Avoids fp16 overflow in NS5 polynomial intermediates. ~1 % quality cost vs fp32, ~2× the wall-clock of fp32 on bf16-tuned hardware. |
| `"fp32"` | Debugging, very small models. | Reference dtype. No surprises. |
| `"fp16"` | **Never** unless your matrices stay under ~256×256. | NS5 produces values that exceed fp16's 65 504 ceiling on larger matrices. NaN within a few steps. |
| `"fp16_safe"` | Quality refinement (last-mile training). | fp16 matmuls + fp32 polynomial accumulation + per-tensor rescale. On our RDNA4 setup this ran 0.5× bf16 throughput but delivered val_MAE slightly better than fp32. |

## Param routing — what goes where

The default `build_fusion_param_groups()` routes:

- **2D matrices with both dims ≥ 128** → spectral path (NS5 + the
  composition)
- **Everything else** (biases, LayerNorm, embeddings, small/odd matrices)
  → scalar path (ScheduleFree-AdamW)

If you have an unusual layer that should go to the scalar path (e.g. an
output projection that's narrow but you want adaptive per-element steps),
use the `force_scalar` regex list:

```python
build_fusion_param_groups(model, force_scalar=[r"output_proj"])
```

## Damping — when the spectral path won't stop wandering

NS5 + NorMuon are magnitude-blind (every update gets normalised to unit
spectral norm / unit-RMS rows), so on a flat loss landscape — a fine-tune
near a good base, or a small effective batch — the walk continues at full
speed forever instead of settling. Two independent brakes, combinable:

| Mechanism | Use when | Knob |
|---|---|---|
| `decay_schedule` | You know roughly how long the run is (`total_steps`) and want a predictable envelope | `cosine`/`linear` for a smooth taper, `wsd` to stay at full strength until `decay_start_frac` then decay — good for "train hard, then settle" schedules |
| `snr` component | You want a **data-driven** brake that reacts to whatever the gradient is actually doing, no `total_steps` needed | `snr_source="grad"` (leave at default — see below), tune `snr_beta`/`snr_floor` |

**`snr_source` must stay `"grad"`.** Gating on the finalized update
(`snr_source=None`, measuring `U` itself) reads momentum's own smoothing as
"signal" — after NS5/NorMuon, a post-momentum update's row-consistency at
`beta≈0.9` is close to 1 by construction, so the gate stays open regardless
of whether the underlying gradient is noise. Gating on the raw gradient is
the version that actually engages the brake.

## Hyperball with zero-init adapters

If you train **hyperball** on a LoRA/DoRA or other zero-init adapter, check
that the optimizer's stderr does NOT report `hyperball DISABLED for a
zero-init param`. If it does for params you expected to be constrained,
your routing put an unexpectedly-zero-init tensor on the spectral path —
worth investigating, since the whole point of hyperball there was the norm
constraint. For LoRA/DoRA's `lora_B` and similar zero-init projections this
message is *expected and correct*: hyperball would otherwise pin them at
exactly zero for the entire run (a null result reported as if it were a
real one), so the optimizer falls back to the ordinary update for those
params instead. Hyperball's constraint is a full-fine-tune tool — it needs
a real pretrained `W0` to freeze the norm of.

## Schedule-Free deployment

Schedule-Free's whole trick is that you train with the fast iterate z_t
but **deploy** from the averaged x_t. Call `optimizer.eval()` before
saving the checkpoint or running validation; call `optimizer.train()`
before resuming training.

```python
optimizer.eval()
val_loss = evaluate(model, val_loader)
torch.save(model.state_dict(), f"ckpt_step{step}.pt")
optimizer.train()
```

Forgetting `optimizer.eval()` is the single most common Schedule-Free
mistake — your saved weights are then the fast iterate, not the
intended average, and quality drops noticeably.

## torch.compile

**Run with `torch.compile`.** Without it, the spectral-path overhead
dominates per-step cost; with it, per-step cost matches AdamW on the
same model size. Don't benchmark FusionOpt without compile.

The optimiser step itself isn't graph-captured (it's outside the model's
forward), but the model forward + backward must be compiled for the
end-to-end story to add up.

## Inference cache (adaLN-zero only)

Three requirements for the time cache to apply:

1. The model has a `t_embedder` attribute (a `nn.Module` mapping
   t → t_emb of shape `(B, dim)`).
2. The model has a `blocks` attribute (`nn.ModuleList` of transformer
   blocks).
3. Each block has an `adaLN_mod` attribute (a `nn.Sequential`
   producing 6×dim modulators) AND each block's `forward()` accepts an
   optional `mods=...` kwarg for injection.

See [`../examples/adaln_block.py`](../examples/adaln_block.py) for a
minimal block adapter pattern.

Cache lifetime: `get_or_build_cache(model, ckpt_path, n_steps)` registers
a persistent cache keyed by `(ckpt_path, n_steps)` — the second render
at the same key hits 100 %. Call `clear_cache_registry()` after a model
swap in a long-running process.

## Gotchas

- **`PYTORCH_TUNABLEOP_TUNING="1"` must be set BEFORE `import torch`.**
  Late application is silently ignored.
- **Bake in a warmup epoch on a fresh TunableOp cache** (~30 % slowdown
  on the first run while matmul kernels are autotuned).
- **KL-Shampoo eigendecomp must run in fp32.** The optimiser handles
  this internally, but if you adapt the code, don't move the eigendecomp
  into the hot dtype.
- **Diversity training NaNs on AdamW.** Adam's EMA-of-moments can't
  bound a magnitude-unbounded negative loss term. Use Full Fusion.

## Logging hooks

The optimiser writes the following running diagnostics to its state for
tensorboard logging:

- `fusion/loss_ema` — EMA of training loss
- `fusion/gnorm_ema` — EMA of gradient norm
- `fusion/gamma_curr` — current Polyak step size γ_t

Watch these during the first ~500 steps. If γ_t saturates the upper
clamp (10) immediately, your `gamma_base` is probably too small (or
your loss is anomalously small relative to gradient norm).
