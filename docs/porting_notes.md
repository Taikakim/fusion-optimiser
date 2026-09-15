# Porting checklist

When applying FusionOpt to a new project, work through this list before
declaring success.

## Before the first run

- [ ] **`PYTORCH_TUNABLEOP_TUNING="1"` set before `import torch`.**
      (Skip if not on AMD.) Late application is silently ignored.
- [ ] **Model has 2-D weight matrices ≥ 128 × 128.** If not, FusionOpt
      degrades to mostly-scalar-path; you may as well use ScheduleFree-AdamW.
- [ ] **Param routing inspected.** Run `summarise_groups(model)` and
      eyeball the spectral / scalar split. If LayerNorm weights ended up
      in the spectral group, your model's parameter naming surprised the
      router; use `force_scalar=[...]` to fix.

## Smoke test (the cheapest signal)

- [ ] **Try `hot_dtype="fp16"` deliberately on one short run.** If val
      loss is NaN within a few hundred steps, you've confirmed the fp16
      overflow pathway is hot on your model. Switch to bf16. If fp16
      is *stable*, your matrices are all small (< 256×256) and you
      should consider whether FusionOpt's spectral path is even worth
      it for you.
- [ ] **Bake in a warmup epoch** before recording any wall-clock
      numbers (on a fresh TunableOp cache, the first run is ~30 %
      slower).

## If using `hyperball`

- [ ] **Confirm the constrained params are NOT zero-init.** Hyperball
      freezes each spectral matrix's Frobenius norm at ‖W₀‖_F; a zero-init
      tensor (LoRA/DoRA's `lora_B`, a `zero_module()`'d projection) has
      ‖W₀‖ = 0, which the optimizer detects and works around by falling
      back to the ordinary update (logged on stderr) — but that means
      hyperball wasn't actually constraining that param. If you *wanted*
      the norm constraint there, hyperball is the wrong tool for a
      zero-init adapter; it's a full-fine-tune mechanism.

## Recipe selection

- [ ] **Default to `{"ns5", "normuon", "sf"}`** (SF-NorMuon). Test
      adding `mona` and `shampoo` only if you have wall-clock to burn.
- [ ] **A/B SF-NorMuon vs SF-NorMuon-plus-MONA on one feature** before
      committing. The MONA paper validates at LLM scale; in our small-DiT
      case study, MONA at small scale *underperformed* bare Muon. Your
      scale may differ.

## Schedule-Free hygiene

- [ ] **Call `optimizer.eval()` before validation and checkpoint save.**
      Forgetting this is the #1 Schedule-Free mistake. Your saved
      weights are then the fast iterate z_t, not the averaged x_t,
      and quality drops noticeably.
- [ ] **Call `optimizer.train()` before resuming training.**
- [ ] **Initial `gamma_base`**: start at the same numeric value you'd
      use for an AdamW `lr`. The Polyak clamp adapts; the base is just
      a scale.

## Polyak step monitoring

- [ ] **Log `fusion/gamma_curr`, `fusion/loss_ema`, `fusion/gnorm_ema`**
      for the first ~1 k steps. If γ saturates the upper clamp (10)
      immediately, raise `gamma_base`. If it pins to the lower clamp
      (0.1), lower `gamma_base`.

## Inference cache (optional, adaLN-zero only)

- [ ] **Model has `t_embedder`, `blocks`, and each block has
      `adaLN_mod`.** If your conditioning scheme is different
      (cross-attention conditioning, FiLM, concat) the cache won't
      apply.
- [ ] **Block forward accepts `mods=None` kwarg.** Without this hook,
      the cache has nowhere to inject precomputed modulators.
      See `examples/adaln_block.py`.
- [ ] **Verify bit-equivalence**: warm the cache, run one sampler step
      with cache active and one without, diff the outputs. Should
      match to ~1e-4 (tensor-core noise).

## Quality regression check

- [ ] **One short AdamW baseline run** on the same model + data. If
      SF-NorMuon doesn't at least match AdamW val_loss, something is
      wrong with the recipe — usually the param-routing split or the
      Schedule-Free deploy step.

## If it's slow

Common culprits, in order:

1. **`torch.compile` not enabled** → spectral overhead dominates.
2. **TunableOp cache cold** → matmuls running autotune.
3. **Wrong hot_dtype** (`fp32` instead of `bf16`).
4. **Wrong batch size for your hardware's matmul sweet spot.**
5. **Param routing puts too much in the scalar path** → spectral
   path's amortisation never kicks in.

## If it diverges

Common culprits, in order:

1. **`hot_dtype="fp16"`** with matrices > 256×256 → NS5 overflow.
2. **`gamma_base` too high** → Polyak clamp can't save you on the first
   few steps.
3. **KL-Shampoo without NS5** (`components={"shampoo"}` alone) →
   unbounded rescaling. Always compose with NS5.
4. **Diversity penalty without Full Fusion** (just SF-NorMuon or
   AdamW) → see `open_questions.md` #3.
