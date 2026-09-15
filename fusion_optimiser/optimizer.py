"""FusionOpt — Muon + MONA + KL-Shampoo + ScheduleFree+ optimiser.

Bifurcated optimiser for LatCH heads:

- Spectral group (2D matrices, min(shape) >= 128):
    1. KL-Shampoo two-sided Kronecker covariance from RAW gradient
    2. MONA curvature-augmented momentum
    3. KL-Shampoo preconditioner applied to augmented momentum
    4. Muon Newton-Schulz quintic spectral normalisation
    5. SF-NorMuon per-neuron row-norm scaling
    6. Schedule-Free averaging with weight decay on the FAST iterate z_t

- Scalar group (1D params, biases, LayerNorm, small/odd 2D):
    Standard ScheduleFree-AdamW with shared Polyak step size.

Outer loop:
    Polyak step size gamma_t = gamma_base * clamp(loss_ema / gnorm_ema, 0.1, 10).
    All reductions on-device; no host syncs in the hot loop.

Train / eval semantics:
    - optimizer.train() writes the Schedule-Free eval point y = (1-beta)z + beta*x
      into params for the next forward.
    - optimizer.eval() writes the averaged iterate x into params for validation
      and checkpoint serialisation.

See docs/superpowers/specs/2026-05-29-fusion-optimiser-design.md §3.
"""

from __future__ import annotations

from typing import Any, Iterable

import re

import math
import torch
from torch.optim import Optimizer

# Matches a DiT layer index in a param name (e.g. "…transformer.layers.6.attn…" -> 6) for the
# optional per-layer update-weight schedule. Non-transformer params (no match) stay at 1.0.
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


# ---------- Newton-Schulz quintic (Muon) ----------

_NS5_COEFFS = (3.4445, -4.7750, 2.0315)


def newton_schulz_5(G: torch.Tensor, steps: int = 5, eps: float = 1e-12) -> torch.Tensor:
    """Muon's NS5 orthogonalisation.

    Frobenius-normalises G so its singular values land in [0, 1], then runs
    `steps` iterations of the quintic phi(X) = a*X + b*X(X^T X) + c*X(X^T X)^2
    with coefficients (3.4445, -4.7750, 2.0315). After 5 iterations the SVs
    are pulled into approximately [0.7, 1.3].

    For tall matrices (rows > cols) we transpose to keep the inner X @ X^T
    small (cols x cols rather than rows x rows). Result is transposed back.

    Dtype handling: this is the standard implementation. The input dtype is
    preserved throughout — at fp16 the iterated polynomial overflows on
    big matrices (see §20D). Use newton_schulz_5_fp16_safe() instead for
    a mixed-precision path that keeps polynomial accumulation in fp32.
    """
    a, b, c = _NS5_COEFFS
    X = G / (G.norm() + eps)
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.transpose(-1, -2).contiguous()
    for _ in range(steps):
        A = X @ X.transpose(-1, -2)
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.transpose(-1, -2)
    return X


def newton_schulz_5_fp16_safe(G: torch.Tensor, steps: int = 5,
                              eps: float = 1e-12) -> torch.Tensor:
    """NS5 with FP16 matmuls and FP32 polynomial accumulation.

    Splits the iteration into "matmul" steps (cast to fp16, use tensor cores)
    and "polynomial" steps (in fp32, where intermediate values can exceed
    fp16's 65 504 ceiling without issue). Each matmul is preceded by a
    per-tensor max-abs rescale so the fp16 inputs always fit in range; the
    result is rescaled back to fp32 immediately.

    Predicted speedup over bf16 on RDNA4: 1.3-1.5x (matmul throughput from
    the well-tuned fp16 hipBLASLt path; per-tensor rescale costs <1 % of
    matmul time when implemented as a max + division kept on-device).

    Compared to the naive fp16 NS5: avoids the overflow that diverges
    training in ~2 steps (§20D), because the polynomial term `b*A + c*A²`
    is computed in fp32 and the next matmul gets a rescaled fp16 input.

    Memory: no extra allocations beyond the standard path (one fp16 cast
    per matmul, immediately consumed).
    """
    a, b, c = _NS5_COEFFS
    # All accumulation in fp32; matmul inputs are explicitly cast to fp16.
    X = G.float() / (G.float().norm() + eps)
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.transpose(-1, -2).contiguous()

    def _safe_mm(P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        """fp16 tensor-core matmul with rescale-and-restore; result in fp32."""
        scale_p = P.abs().max().clamp_min(1.0)
        scale_q = Q.abs().max().clamp_min(1.0)
        out = (P / scale_p).to(torch.float16) @ (Q / scale_q).to(torch.float16)
        return out.float() * (scale_p * scale_q)

    for _ in range(steps):
        A = _safe_mm(X, X.transpose(-1, -2))
        AA = _safe_mm(A, A)
        B = b * A + c * AA                  # fp32 accumulation, no overflow
        BX = _safe_mm(B, X)
        X = a * X + BX
    if transposed:
        X = X.transpose(-1, -2)
    return X


# ---------- KL-Shampoo SPD inverse quarter ----------

def _inv_quarter(M: torch.Tensor, delta: float = 1e-4) -> torch.Tensor:
    """Compute (M + delta*I)^(-1/4) via eigendecomposition.

    Runs entirely in FP32 (the "FP32 island" per spec §3). Caller is
    expected to downcast the result for the hot-path matmul if desired.
    M must be square, symmetric, on the same device as the model.
    """
    M = M.float()
    n = M.shape[-1]
    eye = torch.eye(n, device=M.device, dtype=M.dtype)
    Mr = 0.5 * (M + M.transpose(-1, -2)) + delta * eye  # symmetrise + ridge
    eigvals, eigvecs = torch.linalg.eigh(Mr)
    inv_q = eigvals.clamp_min(1e-12).pow(-0.25)
    return eigvecs @ torch.diag_embed(inv_q) @ eigvecs.transpose(-1, -2)


# ---------- Cautious masking (Liang et al., "Cautious Optimizers", 2024) ----------

def apply_cautious(update: torch.Tensor, grad: torch.Tensor,
                   eps: float = 1e-8) -> torch.Tensor:
    """Mask out the update coordinates that fight the gradient, rescale survivors
    to preserve the UPDATE NORM (not the mean magnitude).

    The weight moves by ``-gamma * update``; along a coordinate the first-order
    loss change is ``grad * (-gamma * update)``, so the step only descends where
    ``update * grad > 0``. Zero the rest, then rescale the survivors by
    ``||U|| / ||U*mask||`` so the masked step is exactly as large as the unmasked one.

    WHY norm- and not mean-preserving (2026-07-02, the DoRA r128 divergence): the
    C-AdamW-style ``1/keep_frac`` rescale inflates the norm by ``1/sqrt(keep)`` —
    negligible at sign-consistent keeps (~0.9) but a hidden +37% effective LR at the
    keep≈0.53 near-random masks NS5-orthogonalized updates produce (keep_frac telemetry,
    wandb iu1bmlyj), which NaN'd the r128 DoRA full-fusion run between ep2 and ep3
    while the identical-minus-cautious baseline trained clean.

    All-agree -> identity. All-disagree -> ~0 (no blow-up). Pure, stateless;
    `update` and `grad` must be the same shape (read in fp32 in the hot loop).
    """
    mask = (update * grad > 0).to(update.dtype)
    masked = update * mask
    return masked * (update.norm() / (masked.norm() + eps))


# ---------- FusionOpt ----------


def decay_factor(schedule: str, step: int, total_steps: int, warmup: int, decay_min: float,
                 decay_start_frac: float = 0.8) -> float:
    """LR multiplier at optimizer step `step` (0-based) for the in-optimizer decay schedule.

    WHY THIS EXISTS (C 2026-08-19, from the step-resolution trajectory data): the spectral path is
    magnitude-blind by construction — NS5 sets every singular value of the update to 1 and NorMuon
    then makes every output row unit-RMS — so once the gradient turns to noise (small effective
    batch, fine-tune near a good base) the walk continues at full speed forever. Schedule-Free's
    premise ("the x-average IS the decay") holds inside a basin, not on a flat landscape: the average
    of a random walk is a random walk with 1/sqrt(3) the variance. And weight decay at our lr binds
    with time constant 1/(lr*wd) ~ 3e5 steps, longer than any run. A decay schedule is the one
    mechanism that stops the wandering by construction.
        none    -> 1.0
        cosine  -> after warmup, 1 -> decay_min along a half-cosine over the remaining steps
        linear  -> after warmup, 1 -> decay_min linearly
        wsd     -> 1.0 until decay_start_frac*total, then linear to decay_min at total
    Past total_steps the factor stays at decay_min (never re-warms)."""
    if schedule in (None, "none"):
        return 1.0
    if total_steps <= 0:
        raise ValueError(f"decay_schedule={schedule!r} needs total_steps > 0 (got {total_steps})")
    if step < warmup:
        return 1.0
    span = max(1, total_steps - warmup)
    prog = min(1.0, max(0.0, (step - warmup) / span))
    if schedule == "cosine":
        f = 0.5 * (1.0 + math.cos(math.pi * prog))
    elif schedule == "linear":
        f = 1.0 - prog
    elif schedule == "wsd":
        # progress measured on the whole run (warmup counts toward the stable phase)
        p_all = min(1.0, max(0.0, step / max(1, total_steps)))
        if p_all < decay_start_frac:
            f = 1.0
        else:
            f = 1.0 - (p_all - decay_start_frac) / max(1e-12, 1.0 - decay_start_frac)
    else:
        raise ValueError(f"unknown decay_schedule {schedule!r}")
    return float(decay_min + (1.0 - decay_min) * max(0.0, f))


def snr_gate(U: torch.Tensor, state: dict, mode: str = "row", beta: float = 0.9,
             floor: float = 0.0, power: float = 1.0, step: int = 1, ref: "torch.Tensor | None" = None):
    """Scale the finalized spectral update by its own signal-to-noise ratio.

    Keeps a bias-corrected EMA of U (first moment, m) and of U^2 (second moment, v) and gates
        row : g_row = clamp(||m_row|| / sqrt(sum_j v_row_j), max 1)   (one gain per output neuron)
        elem: g_ij  = clamp(|m_ij| / sqrt(v_ij), max 1)                (Adam-like, per weight)
    then U <- U * max(g^power, floor).  `ref` (default U) is the tensor the SNR is MEASURED on:
    pass the raw gradient to gate on gradient consistency — measuring on the post-momentum update
    reads momentum's own smoothing as "signal" and the gate never closes (bs1 arms, 2026-08-19).
    On iid noise the ratio settles at sqrt((1-beta)/(1+beta))
    (0.23 at beta 0.9 — exactly Adam's built-in brake on noise); on a consistent direction it is 1.
    This re-introduces the one thing NS5+NorMuon strip: whether the step is repeatable. It is
    "decay per weight" as a data-driven brake rather than a schedule. Extra state: m (full) and v
    (row vector or full) — one to two update-sized buffers per matrix.
    Returns (gated_U, gate)."""
    R = U if ref is None else ref
    if "snr_m" not in state:
        state["snr_m"] = torch.zeros_like(R)
        state["snr_v"] = (torch.zeros(R.shape[0], device=R.device, dtype=R.dtype) if mode == "row"
                          else torch.zeros_like(R))
    m, v = state["snr_m"], state["snr_v"]
    m.mul_(beta).add_(R, alpha=(1.0 - beta))
    if mode == "row":
        v.mul_(beta).add_((R * R).sum(dim=-1), alpha=(1.0 - beta))
    elif mode == "elem":
        v.mul_(beta).add_(R * R, alpha=(1.0 - beta))
    else:
        raise ValueError(f"snr mode must be row|elem, got {mode!r}")
    bc = 1.0 - beta ** max(1, int(step))
    m_hat = m / bc
    v_hat = v / bc
    if mode == "row":
        g = (m_hat.norm(dim=-1) / v_hat.clamp_min(1e-30).sqrt()).clamp(max=1.0)
        if power != 1.0:
            g = g.pow(power)
        g = g.clamp_min(floor)
        return U * g.unsqueeze(-1), g
    g = (m_hat.abs() / v_hat.clamp_min(1e-30).sqrt()).clamp(max=1.0)
    if power != 1.0:
        g = g.pow(power)
    g = g.clamp_min(floor)
    return U * g, g


class FusionOpt(Optimizer):
    """The fused optimiser. Takes param groups from build_fusion_param_groups."""

    def __init__(
        self,
        params: Iterable[dict],
        lr: float = 3e-4,
        # Schedule-Free / Polyak
        beta: float = 0.9,
        beta_p: float = 0.98,
        gamma_min: float = 0.1,
        gamma_max: float = 10.0,
        # Spectral path
        mu: float = 0.95,
        mona_alpha: float = 0.2,
        beta_n: float = 0.9,
        beta_k: float = 0.99,
        eigen_period: int = 100,
        shampoo_delta: float = 1e-4,
        beta_r: float = 0.95,
        # Scalar path
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        # Weight decay (set per-group via param_groups; these are fallbacks)
        weight_decay: float = 0.0,
        # Hot-path dtype: cast preconditioned momentum + NS5 inputs to this dtype
        # (kept FP32 by default for safety; FP16 on RDNA4 if profiling shows need).
        # Valid: "fp32", "fp16", "bf16".
        hot_dtype: str = "fp32",
        warmup_steps: int = 0,
        # FP32 audit: every N steps, recompute NS5 in FP32 alongside the hot_dtype
        # path and log relative error stats. 0 = disabled. Useful for verifying that
        # fp16/bf16 quantisation isn't quietly destabilising the spectral updates.
        # Cost: roughly 2x optimizer step on audit steps only (~0.5% if N=200).
        fp32_audit_period: int = 0,
        # Component flags — controls which mechanisms run in the spectral path.
        # Use this for the per-component ablation (one optimiser at a time).
        # Default = full Fusion (everything on). Valid components:
        #   "mona"     - MONA curvature-augmented momentum
        #   "shampoo"  - KL-Shampoo two-sided preconditioner
        #   "ns5"      - Muon Newton-Schulz quintic spectral norm
        #   "normuon"  - SF-NorMuon per-neuron row-norm scaling
        #   "sf"       - Schedule-Free averaging (z_t fast iterate + x_t average,
        #                Polyak step, WD on z_t). When disabled, weight decay
        #                applies to live weights p directly.
        components: "set[str] | None" = None,
        # Optional per-DiT-layer update-weight schedule: {dit_layer_index: multiplier}.
        # Scales the finalized spectral step of params in transformer.layers.N by curve[N]
        # (post-NS5/NorMuon/cautious — pure step-size modulation, orthogonalisation intact);
        # non-transformer params and unlisted layers stay 1.0. None = no-op (default).
        layer_update_weights: "dict | None" = None,
        # HYPERBALL (arXiv 2606.16899, Alg. 1) — constrain each spectral 2D weight matrix to
        # the hypersphere of radius R = ‖W0‖_F (captured ONCE, on the first step for that
        # param). Replaces the DIRECT-to-p step with the Hyperball retraction:
        #   u_hat   = U / ‖U‖         (normalized spectral update)
        #   W_tilde = W - gamma_t * R * u_hat
        #   W       = R * W_tilde / ‖W_tilde‖
        # Hyperball is its OWN iterate → INCOMPATIBLE with Schedule-Free ('sf'); constructing
        # with both raises ValueError. Weight decay is IGNORED under hyperball (the norm
        # constraint replaces it). Applies ONLY in the spectral DIRECT-to-p path; the scalar
        # path is untouched. Off by default => byte-identical to today.
        hyperball: bool = False,
        # DAMPING (C 2026-08-19; Kim's "muon damping tests"). Both off by default => byte-identical.
        #   decay_schedule none|cosine|linear|wsd over total_steps (after warmup) -> multiplies gamma_t
        #   'snr' component: gate the finalized spectral update by |EMA(U)|/RMS(U) per row|elem
        decay_schedule: str = "none",
        total_steps: int = 0,
        decay_min: float = 0.0,
        decay_start_frac: float = 0.8,
        snr_mode: str = "row",
        # WHAT the gate measures (C 2026-08-19, from the bs1 damping arms): 'update' gates on the
        # consistency of the finalized spectral update U — but U is post-MOMENTUM, so its row-
        # consistency at beta 0.9 is ~1 by construction and the gate was inert (|update| 0.094 vs
        # control 0.091, every trajectory statistic identical). 'grad' (default now) gates on the RAW
        # gradient's row SNR — the quantity that is actually 1e-3 at bs1 — so the brake engages.
        snr_source: str = "grad",
        snr_beta: float = 0.9,
        snr_floor: float = 0.0,
        snr_power: float = 1.0,
        # AUTOSCALE (Kim 2026-09-01, "snatch a principle from Prodigy for Fusion"): a
        # global, model-wide D-Adaptation (Defazio & Mishchenko) step-size multiplier,
        # folded into gamma_t alongside the Polyak ratio. Off by default. See
        # _update_autoscale for the mechanism and the memory argument (why this is
        # cheap where prodigyopt.Prodigy is not).
        autoscale_d0: float = 1e-6,
        autoscale_coef: float = 1.0,
        autoscale_growth_rate: float = float("inf"),
        autoscale_slice_p: int = 16,
        autoscale_beta3: "float | None" = None,
    ):
        all_components = {"mona", "shampoo", "ns5", "normuon", "sf", "cautious", "snr", "autoscale"}
        if components is None:
            components = all_components
        components = set(components)
        unknown = components - all_components
        if unknown:
            raise ValueError(f"FusionOpt: unknown components: {sorted(unknown)} "
                             f"(valid: {sorted(all_components)})")
        defaults = dict(
            lr=lr,
            beta=beta, beta_p=beta_p, gamma_min=gamma_min, gamma_max=gamma_max,
            mu=mu, mona_alpha=mona_alpha, beta_n=beta_n, beta_k=beta_k,
            eigen_period=eigen_period, shampoo_delta=shampoo_delta, beta_r=beta_r,
            beta1=beta1, beta2=beta2, eps=eps,
            weight_decay=weight_decay,
            hot_dtype=hot_dtype, warmup_steps=warmup_steps,
            fp32_audit_period=int(fp32_audit_period),
            decay_schedule=decay_schedule, total_steps=int(total_steps), decay_min=float(decay_min),
            decay_start_frac=float(decay_start_frac),
            snr_mode=snr_mode, snr_source=snr_source, snr_beta=float(snr_beta), snr_floor=float(snr_floor),
            snr_power=float(snr_power),
        )
        super().__init__(params, defaults)
        if decay_schedule not in (None, "none") and int(total_steps) <= 0:
            raise ValueError("FusionOpt: decay_schedule needs total_steps > 0")

        self._components = frozenset(components)
        # HYPERBALL — norm-constrained iterate (arXiv 2606.16899). Its own iterate, so it
        # cannot coexist with Schedule-Free averaging (both own the update of p/z).
        self._hyperball = bool(hyperball)
        if self._hyperball and "sf" in self._components:
            raise ValueError(
                "FusionOpt: hyperball=True is incompatible with the 'sf' (Schedule-Free) "
                "component — Hyperball is its own norm-constrained iterate. Drop 'sf' from "
                "components (e.g. use ['mona','ns5','normuon']) when enabling hyperball."
            )
        # per-DiT-layer update-weight schedule ({int layer: float mult}); cache name->mult
        self._layer_update_weights = (
            {int(k): float(v) for k, v in layer_update_weights.items()}
            if layer_update_weights else None)
        self._layer_mult_cache: "dict[str, float]" = {}
        self._telem_on = False          # per-component telemetry gate; trainer sets it per step
        self._comp_telem = {}           # last instrumented step's per-stage update-magnitude profile
        self._comp_acc = None           # accumulator (reset each instrumented step)
        self._mode = "train"  # "train" or "eval"
        self._step_count = 0
        # External trust multiple on the effective step (written by training.sonar's
        # radial probes; 1.0 = neutral). Multiplies gamma_t in BOTH paths.
        self.gamma_scale = 1.0
        # Pending loss for Polyak (set by train loop before step)
        self._current_loss: torch.Tensor | None = None
        # On-device EMAs; lazily created on first step()
        self._loss_ema: torch.Tensor | None = None
        self._gnorm_ema: torch.Tensor | None = None
        # FP32 audit records; trimmed to recent N to bound memory
        self._audit_stats: list[dict] = []
        self._audit_keep = 200

        # AUTOSCALE (D-Adaptation) global running state — see _update_autoscale.
        # d only ever grows (bounded by autoscale_growth_rate); d/d0 is the multiplier
        # folded into gamma_t. Inert (returns 1.0 every step) unless "autoscale" is
        # in components, so absent => byte-identical to today.
        self._auto_d0 = float(autoscale_d0)
        self._auto_coef = float(autoscale_coef)
        self._auto_growth_rate = float(autoscale_growth_rate)
        self._auto_slice_p = max(1, int(autoscale_slice_p))
        self._auto_beta3 = float(autoscale_beta3) if autoscale_beta3 is not None else math.sqrt(0.999)
        self._auto_d = self._auto_d0
        self._auto_d_max = self._auto_d0
        self._auto_numerator = 0.0

        # Sanity: groups must declare group_type
        for g in self.param_groups:
            if g.get("group_type") not in ("spectral", "scalar"):
                raise ValueError(
                    "FusionOpt param groups must set group_type=spectral|scalar"
                )

    @property
    def uses_sf_averaging(self) -> bool:
        """Whether this optimiser uses Schedule-Free averaging (the .train()/.eval()
        toggle behaviour). False when 'sf' is excluded from components."""
        return "sf" in self._components

    @property
    def components(self) -> "frozenset[str]":
        """Currently enabled components. Useful for logging / introspection."""
        return self._components

    # ---- public ---------------------------------------------------------

    def set_loss(self, loss: torch.Tensor | None) -> None:
        """Pass the most recent training loss for the Polyak step.

        Call this BEFORE optimizer.step() (or before scaler.step(optimizer) when
        using GradScaler). The loss tensor is detached and kept on-device.
        If never called, Polyak falls back to a constant ratio of 1.0.
        """
        self._current_loss = loss.detach() if loss is not None else None

    def train(self) -> None:
        """Switch params to the eval-point y = (1-beta)*z + beta*x.
        No-op when SF averaging is disabled (live weights are always in p)."""
        if self._mode == "train":
            return
        self._mode = "train"
        if "sf" not in self._components:
            return
        for group in self.param_groups:
            beta = group["beta"]
            for p in group["params"]:
                st = self.state.get(p, None)
                if st is None or "z" not in st:
                    continue
                p.data.copy_((1 - beta) * st["z"] + beta * st["x"])

    def eval(self) -> None:
        """Switch params to the averaged iterate x (deployable model).
        No-op when SF averaging is disabled."""
        if self._mode == "eval":
            return
        self._mode = "eval"
        if "sf" not in self._components:
            return
        for group in self.param_groups:
            for p in group["params"]:
                st = self.state.get(p, None)
                if st is None or "x" not in st:
                    continue
                p.data.copy_(st["x"])

    def average_state_dict(self) -> dict[str, torch.Tensor]:
        """Return {param_name: x_t} for serialisation as a deployable model.
        When SF averaging is disabled, returns the live weights (which ARE the
        deployable model in that case)."""
        out: dict[str, torch.Tensor] = {}
        for group in self.param_groups:
            names = group.get("param_names", [])
            for p, name in zip(group["params"], names):
                st = self.state.get(p, None)
                if "sf" not in self._components or st is None or "x" not in st:
                    out[name] = p.detach().clone()
                else:
                    out[name] = st["x"].detach().clone()
        return out

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
            if self._current_loss is None and loss is not None:
                self._current_loss = loss.detach()

        # Polyak step size (global, on-device)
        gamma_ratio = self._update_polyak()
        # D-Adaptation autoscale multiplier (global, python float; 1.0 when off)
        auto_mult = self._update_autoscale()

        if self._telem_on:                      # per-component instrumentation (this step only)
            self._comp_acc = {"sp_n": 0, "mom_sq": 0.0, "monaA_sq": 0.0, "mpre_sq": 0.0,
                              "ns5_sq": 0.0, "final_sq": 0.0, "gamma_t": 0.0,
                              "sc_n": 0, "sc_grad_sq": 0.0,
                              "caut_kept": 0.0, "caut_n": 0.0,
                              "snr_gain": 0.0, "snr_n": 0.0, "decay": 1.0}
        for group in self.param_groups:
            if group["group_type"] == "spectral":
                self._spectral_group_step(group, gamma_ratio, auto_mult)
            else:
                self._scalar_group_step(group, gamma_ratio, auto_mult)
        if self._telem_on:
            self._comp_telem = self._finalize_comp_telem()

        # After updating z and x, write y back to p.data for next forward.
        # Skipped when SF averaging is disabled: live weights are already in p.
        if self._mode == "train" and "sf" in self._components:
            for group in self.param_groups:
                beta = group["beta"]
                for p in group["params"]:
                    st = self.state.get(p, None)
                    if st is None or "z" not in st:
                        continue
                    p.data.copy_((1 - beta) * st["z"] + beta * st["x"])

        self._step_count += 1
        self._current_loss = None
        return loss

    def _finalize_comp_telem(self):
        """Turn the per-stage accumulators into a per-component update-magnitude profile.
        Each *gain* is the ratio of update magnitude across one pipeline stage (≈1 when the
        component is off), so you can read off what each enabled component actually does:
        momentum -> [shampoo precondition] -> [ns5 orthonormalize] -> [normuon row-scale] -> step."""
        a, d = self._comp_acc, {}
        if a["sp_n"]:
            mom = a["mom_sq"] ** 0.5
            mpre = a["mpre_sq"] ** 0.5
            ns5 = a["ns5_sq"] ** 0.5
            fin = a["final_sq"] ** 0.5
            d["comp/momentum_norm"] = mom
            if "mona" in self._components:
                d["comp/mona_curvature_norm"] = a["monaA_sq"] ** 0.5
            if "shampoo" in self._components:
                d["comp/shampoo_gain"] = mpre / (mom + 1e-12)      # preconditioner scaling
            if "ns5" in self._components:
                d["comp/ns5_gain"] = ns5 / (mpre + 1e-12)          # spectral orthonormalization
            if "normuon" in self._components:
                d["comp/normuon_gain"] = fin / (ns5 + 1e-12)       # per-neuron row scaling
            d["comp/spectral_update_norm"] = a["gamma_t"] * fin    # actual spectral weight delta
            d["comp/gamma_t"] = a["gamma_t"]                        # effective step = lr*polyak*warmup
        if a.get("caut_n", 0.0) > 0:
            # Fraction of update coords kept by cautious masking. Falls toward 0.5
            # = update fighting the gradient half the time = wandering (drift signal).
            d["comp/cautious_keep_frac"] = a["caut_kept"] / a["caut_n"]
        if a.get("snr_n", 0) > 0:
            d["comp/snr_gain"] = a["snr_gain"] / a["snr_n"]       # mean SNR gate (1 = signal, ~0.23 = noise)
        d["comp/decay"] = a.get("decay", 1.0)
        if "autoscale" in self._components:
            d["comp/autoscale_mult"] = a.get("autoscale_mult", 1.0)
            d["comp/autoscale_d"] = self._auto_d
        if a["sc_n"]:
            d["comp/scalar_grad_norm"] = a["sc_grad_sq"] ** 0.5
        return d

    # ---- internals ------------------------------------------------------

    def _update_polyak(self) -> torch.Tensor:
        """Update loss/gnorm EMAs and return gamma_ratio = clamp(L/G, gamma_min, gamma_max).

        Returned as a scalar tensor on the device of the first available grad,
        so downstream multiplications stay GPU-resident.
        """
        # Find a reference device from gradients
        device = None
        total_abs = None
        total_count = 0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if device is None:
                    device = p.grad.device
                ga = p.grad.detach().abs().sum()
                total_abs = ga if total_abs is None else total_abs + ga
                total_count += p.grad.numel()

        if device is None or total_count == 0:
            # No grads — return identity ratio
            return torch.tensor(1.0)

        gnorm_now = total_abs / total_count
        beta_p = self.param_groups[0]["beta_p"]

        if self._gnorm_ema is None:
            self._gnorm_ema = gnorm_now.clone().detach()
        else:
            self._gnorm_ema = (
                self._gnorm_ema.to(device) * beta_p + gnorm_now * (1 - beta_p)
            )

        if self._current_loss is not None:
            loss_now = self._current_loss.to(device).float()
            if self._loss_ema is None:
                self._loss_ema = loss_now.clone().detach()
            else:
                self._loss_ema = self._loss_ema.to(device) * beta_p + loss_now * (1 - beta_p)
            ratio = self._loss_ema / (self._gnorm_ema + 1e-12)
        else:
            # No loss provided — degenerate to constant 1.0
            ratio = torch.ones((), device=device)

        gmin = self.param_groups[0]["gamma_min"]
        gmax = self.param_groups[0]["gamma_max"]
        return ratio.clamp(gmin, gmax)

    def _update_autoscale(self) -> float:
        """Global, model-wide Prodigy/D-Adaptation step-size multiplier (Kim
        2026-09-01, "snatch a principle from Prodigy for Fusion"). Off unless
        "autoscale" is in components (returns 1.0, no state touched -> free).

        Mechanism (Defazio & Mishchenko's D-Adaptation, as implemented in
        prodigyopt.Prodigy): `d` starts at d0 and only ever grows, driven by how
        much the observed gradient correlates with the DISPLACEMENT FROM INIT
        (dot(grad, p0 - p)) relative to an EMA of gradient magnitude (`s`) — a
        provable lower bound on the true optimal step size under online-convex-
        optimization theory. Returned as d/d0, a pure growth multiplier that starts
        at 1.0 and rises as evidence accumulates, folded into gamma_t alongside the
        existing Polyak ratio — Fusion's own update geometry (NS5/NorMuon/MONA/
        Shampoo/SF) is untouched; this only modulates overall magnitude.

        MEMORY, why this is cheap where prodigyopt.Prodigy is not: real Prodigy
        keeps a FULL-SIZE clone of the initial weights (`p0`) plus a full-size
        accumulator (`s`) per parameter — on top of Adam's own exp_avg/exp_avg_sq,
        that's up to 4 full tensors per param, roughly 2x AdamW's state. Here both
        `p0` and `s` are kept at every `autoscale_slice_p`-th coordinate only
        (identical in spirit to Prodigy's own documented `slice_p` knob) — state
        cost is O(numel / slice_p), and it rides in the SAME per-param `self.state`
        dict Fusion already allocates rather than a second parallel one. The
        reduction pass below also only touches the sliced elements, so the extra
        compute is O(numel / slice_p) too, not a second full-size pass.

        Simplifications vs. real Prodigy: this fork's own `lr`/weight-decay/bias-
        correction already live in gamma_t and the group step functions, so here
        "lr" is fixed at 1.0 and bias_correction/decoupled-WD are not reproduced —
        this module ONLY computes and returns a scalar multiplier, it never writes
        to `p.data` itself.
        """
        if "autoscale" not in self._components:
            return 1.0

        d = self._auto_d
        d0 = self._auto_d0
        beta3 = self._auto_beta3
        slice_p = self._auto_slice_p
        factor = (d / d0) * d  # == (d/d0)*dlr with our own "lr" fixed at 1.0

        d_numerator = self._auto_numerator * beta3
        delta_numerator = 0.0
        d_denom = 0.0
        any_grad = False

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                any_grad = True
                state = self.state[p]
                if "auto_p0" not in state:
                    state["auto_p0"] = p.detach().flatten()[::slice_p].clone().float()
                    state["auto_s"] = torch.zeros_like(state["auto_p0"])
                p0 = state["auto_p0"]
                s = state["auto_s"]

                g_sliced = p.grad.detach().flatten()[::slice_p].float()
                p_sliced = p.detach().flatten()[::slice_p].float()

                delta_numerator += float(factor * torch.dot(g_sliced, p0 - p_sliced))
                s.mul_(beta3).add_(g_sliced, alpha=factor)
                d_denom += float(s.abs().sum())

        if not any_grad or d_denom == 0.0:
            return d / d0

        d_numerator += delta_numerator
        d_hat = self._auto_coef * d_numerator / d_denom
        if d == d0:
            d = max(d, d_hat)
        d_max = max(self._auto_d_max, d_hat)
        d = min(d_max, d * self._auto_growth_rate)

        self._auto_numerator = d_numerator
        self._auto_d = d
        self._auto_d_max = d_max
        return d / d0

    # ---- spectral path --------------------------------------------------

    def _layer_mult(self, name):
        """Per-layer step-size multiplier for a param name (cached). 1.0 when the schedule
        is off, the name is None, it carries no `layers.N`, or that layer isn't listed."""
        if self._layer_update_weights is None or not name:
            return 1.0
        c = self._layer_mult_cache
        v = c.get(name)
        if v is None:
            m = _LAYER_RE.search(name)
            v = self._layer_update_weights.get(int(m.group(1)), 1.0) if m else 1.0
            c[name] = v
        return v

    def _spectral_group_step(self, group, gamma_ratio, auto_mult=1.0):
        lr = group["lr"]
        beta = group["beta"]
        mu = group["mu"]
        alpha = group["mona_alpha"]
        beta_n = group["beta_n"]
        beta_k = group["beta_k"]
        beta_r = group["beta_r"]
        eigen_period = group["eigen_period"]
        delta = group["shampoo_delta"]
        wd = group["weight_decay"]
        hot_dtype_name = group["hot_dtype"]
        warmup = group.get("warmup_steps", 0)

        # Warmup factor (linear ramp 0 -> 1 over `warmup` steps)
        if warmup > 0 and self._step_count < warmup:
            warm = (self._step_count + 1) / warmup
        else:
            warm = 1.0

        decay = decay_factor(group.get("decay_schedule", "none"), self._step_count,
                             group.get("total_steps", 0), warmup, group.get("decay_min", 0.0),
                             group.get("decay_start_frac", 0.8))
        gamma_t = lr * gamma_ratio * auto_mult * warm * decay * float(getattr(self, "gamma_scale", 1.0))
        if self._telem_on:
            self._comp_acc["gamma_t"] = float(gamma_t)
            self._comp_acc["decay"] = float(decay)
            self._comp_acc["autoscale_mult"] = float(auto_mult)

        # hot_dtype_name controls how the NS5 hot path runs:
        #   "fp32"     - safe, slowest. Standard NS5 in fp32.
        #   "bf16"     - safe, ~1.65x faster than fp32. NS5 in bf16 (fp32-range
        #                exponent, can't overflow).
        #   "fp16"     - UNSAFE for NS5; iterated quintic overflows. Will diverge.
        #                Kept as a choice for users who want to confirm divergence.
        #   "fp16_safe"- predicted ~1.3-1.5x over bf16. Uses fp16 matmuls with
        #                fp32 polynomial accumulation and per-tensor rescale-restore
        #                around each matmul (see newton_schulz_5_fp16_safe).
        # The non-NS5 ops (preconditioner P_L @ m @ P_R) still cast to the
        # name's literal dtype mapping below; "fp16_safe" maps to fp16 there
        # since the preconditioner matmul is single-shot (no iterated overflow).
        hot_dtype = {
            "fp16": torch.float16,
            "fp16_safe": torch.float16,
            "bf16": torch.bfloat16,
        }.get(hot_dtype_name, torch.float32)
        use_safe_ns5 = (hot_dtype_name == "fp16_safe")
        _pnames = group.get("param_names") or [None] * len(group["params"])
        _splits = group.get("spectral_split") or [1] * len(group["params"])

        for p, _pname, _nblk in zip(group["params"], _pnames, _splits):
            if p.grad is None:
                continue

            grad = p.grad.detach().float()
            if grad.ndim != 2:
                # Should not happen — fusion_groups guarantees spectral params are 2D
                continue

            state = self.state[p]
            if "step" not in state:
                # Lazy init — allocate ONLY the buffers the active components use. The modular
                # compute path already gates every op by component; gating the allocation to match
                # means e.g. SF-NorMuon (no shampoo) no longer carries the (in_dim x in_dim) Shampoo
                # matrices, which dominate memory on large adapters. Sentinel is "step" (always set).
                state["step"] = 0
                state["m"] = torch.zeros_like(grad)           # momentum (spectral/muon path, always)
                if "sf" in self._components:
                    state["z"] = p.detach().clone().float()   # SF: copy p as z and x (eval==train at init)
                    state["x"] = p.detach().clone().float()
                if "mona" in self._components:
                    state["A"] = torch.zeros_like(grad)       # MONA curvature EMA
                    state["g_prev"] = torch.zeros_like(grad)  # last gradient
                if "shampoo" in self._components:
                    out_dim, in_dim = grad.shape
                    state["L"] = torch.zeros(out_dim, out_dim, device=grad.device, dtype=torch.float32)
                    state["R"] = torch.zeros(in_dim, in_dim, device=grad.device, dtype=torch.float32)
                    state["P_L"] = torch.eye(out_dim, device=grad.device, dtype=torch.float32)
                    state["P_R"] = torch.eye(in_dim, device=grad.device, dtype=torch.float32)
                if "normuon" in self._components:
                    state["r"] = torch.zeros(grad.shape[0], device=grad.device, dtype=torch.float32)

            # 1. KL-Shampoo factor update from RAW gradient (every step, FP32)
            if "shampoo" in self._components:
                L = state["L"]
                R = state["R"]
                L.mul_(beta_k).add_(grad @ grad.transpose(-1, -2), alpha=(1 - beta_k))
                R.mul_(beta_k).add_(grad.transpose(-1, -2) @ grad, alpha=(1 - beta_k))

                # 2. Periodic eigendecomp -> P_L, P_R (every K steps, FP32 island)
                if self._step_count % eigen_period == 0:
                    state["P_L"] = _inv_quarter(L, delta=delta)
                    state["P_R"] = _inv_quarter(R, delta=delta)

            # 3. MONA augmented momentum (FP32) — optional
            m = state["m"]
            if "mona" in self._components:
                A_buf = state["A"]
                g_prev = state["g_prev"]
                A_buf.mul_(beta_n).add_(grad - g_prev)
                g_prev.copy_(grad)
                m.mul_(mu).add_(grad + alpha * A_buf)
            else:
                # Plain momentum
                m.mul_(mu).add_(grad)
            if self._telem_on:
                self._comp_acc["sp_n"] += 1
                self._comp_acc["mom_sq"] += float((m * m).sum())
                if "mona" in self._components:
                    self._comp_acc["monaA_sq"] += float((state["A"] * state["A"]).sum())

            # 4. Apply KL-Shampoo preconditioner (optional)
            if "shampoo" in self._components:
                P_L_h = state["P_L"].to(hot_dtype)
                P_R_h = state["P_R"].to(hot_dtype)
                m_h = m.to(hot_dtype)
                m_pre = P_L_h @ m_h @ P_R_h
            else:
                m_pre = m.to(hot_dtype)
            if self._telem_on:
                self._comp_acc["mpre_sq"] += float((m_pre.float() * m_pre.float()).sum())

            # 5. Muon NS5 spectral normalisation (optional). --fusion-split-qkv: a fused
            # attention up-projection (_nblk>1) is orthogonalised PER dim-row block, so q/k/v
            # (+ the differential-attention diff blocks) each get their own spectral treatment
            # and per-block aspect scale; blocks are written back stacked. _nblk==1 (default) is
            # whole-matrix NS5, unchanged.
            if "ns5" in self._components:
                _ns5 = newton_schulz_5_fp16_safe if use_safe_ns5 else newton_schulz_5
                if _nblk > 1:
                    _bh = m_pre.shape[0] // _nblk
                    _blocks = []
                    for _b in range(_nblk):
                        _Ub = _ns5(m_pre[_b * _bh:(_b + 1) * _bh]).float()
                        _od, _idim = _Ub.shape
                        _blocks.append(_Ub * ((max(1.0, _od / _idim)) ** 0.5))
                    U = torch.cat(_blocks, dim=0)
                else:
                    U = _ns5(m_pre).float()
                    # FP32 audit (optional; whole-matrix only — a whole-vs-block rel-error would
                    # be misleading, so it's skipped for the split path): re-run in FP32, record
                    # the relative error. Doesn't affect the actual update.
                    audit_period = group.get("fp32_audit_period", 0)
                    if (audit_period > 0 and hot_dtype != torch.float32 and
                            self._step_count > 0 and self._step_count % audit_period == 0):
                        with torch.no_grad():
                            m_pre_fp32 = m_pre.float()
                            U_fp32 = newton_schulz_5(m_pre_fp32)
                        diff = (U - U_fp32).abs()
                        denom = U_fp32.abs().clamp_min(1e-12)
                        rel = (diff / denom).flatten()
                        self._audit_stats.append({
                            "step":  self._step_count,
                            "shape": tuple(U.shape),
                            "rel_mean": float(rel.mean().detach()),
                            "rel_max":  float(rel.max().detach()),
                            "abs_max":  float(diff.max().detach()),
                        })
                    out_dim, in_dim = U.shape
                    U = U * ((max(1.0, out_dim / in_dim)) ** 0.5)  # aspect-ratio scale
            else:
                U = m_pre.float()
            if self._telem_on:
                self._comp_acc["ns5_sq"] += float((U * U).sum())

            # 6. SF-NorMuon per-neuron row scaling (optional)
            if "normuon" in self._components:
                r = state["r"]
                row_ss = (U * U).sum(dim=-1)  # (out_dim,)
                r.mul_(beta_r).add_(row_ss, alpha=(1 - beta_r))
                U = U / (r.clamp_min(1e-12).sqrt().unsqueeze(-1))
            # 6a. SNR gate (optional, 'snr'): scale by |EMA(U)|/RMS(U) per row|elem — the brake
            # NS5+NorMuon lack (see snr_gate). Uses the optimizer's own step count for bias correction.
            if "snr" in self._components:
                U, _g = snr_gate(U, state, mode=group.get("snr_mode", "row"),
                                 beta=group.get("snr_beta", 0.9), floor=group.get("snr_floor", 0.0),
                                 power=group.get("snr_power", 1.0), step=state["step"] + 1,
                                 ref=(grad if group.get("snr_source", "grad") == "grad" else None))
                if self._telem_on:
                    self._comp_acc["snr_gain"] += float(_g.mean())
                    self._comp_acc["snr_n"] += 1.0
            if self._telem_on:
                self._comp_acc["final_sq"] += float((U * U).sum())

            # 6b. Cautious masking (optional) — drop update coords that fight the
            # gradient, rescale survivors. Attacks the constant-magnitude wandering
            # of the orthogonalised (LMO) update in the flat control landscape.
            if "cautious" in self._components:
                U = apply_cautious(U, grad)
                if self._telem_on:
                    self._comp_acc["caut_kept"] += float((U != 0).to(U.dtype).mean())
                    self._comp_acc["caut_n"] += 1.0

            # 6c. Per-DiT-layer update-weight schedule (GOA TASK B): final per-layer step-size
            # modulation on the fully-shaped spectral step. No-op unless the schedule is set.
            _lm = self._layer_mult(_pname)
            if _lm != 1.0:
                U = U * _lm

            # 7. Update — Schedule-Free averaging (with WD on z_t) OR direct on p
            t = state["step"] + 1
            state["step"] = t
            if "sf" in self._components:
                z = state["z"]
                x = state["x"]
                # z_{t+1} = (1 - gamma*wd) z_t - gamma * U
                z.mul_(1 - gamma_t * wd).add_(U, alpha=-gamma_t)
                # x_{t+1} = (1 - 1/t) x_t + (1/t) z_{t+1}
                x.mul_(1 - 1.0 / t).add_(z, alpha=1.0 / t)
            elif self._hyperball:
                # HYPERBALL (arXiv 2606.16899, Alg. 1): constrain W to the hypersphere of
                # radius R = ‖W0‖_F, captured ONCE (norm of the LOADED weights on the first
                # step for this param, before applying). Weight decay is IGNORED (the norm
                # constraint replaces it). u_t == U (the finalized spectral update).
                #   u_hat   = U / (‖U‖ + eps)
                #   W_tilde = W - gamma_t * R * u_hat
                #   W       = R * W_tilde / (‖W_tilde‖ + eps)
                if "hyperball_R" not in state:
                    state["hyperball_R"] = p.data.norm()
                    # ZERO-INIT ESCAPE HATCH. R = ‖W0‖ = 0 makes both halves of the
                    # update identically zero (−γ·R·û = 0, then R·W̃/‖W̃‖ = 0), pinning
                    # the parameter at zero for the entire run. Every adapter we train is
                    # zero-init by construction — LoRA/DoRA lora_B and the Head-B
                    # cross-attn to_out — so hyperball on an adapter recipe trains
                    # NOTHING and reports it as a weak result, not as an error. Fall back
                    # to the unconstrained update for such params and say so.
                    if float(state["hyperball_R"]) <= 1e-12:
                        state["hyperball_off"] = True
                        self._hyperball_skipped = getattr(self, "_hyperball_skipped", 0) + 1
                        if self._hyperball_skipped <= 3 or self._hyperball_skipped % 100 == 0:
                            print(f"[fusion] hyperball DISABLED for a zero-init param "
                                  f"{tuple(p.shape)} (‖W0‖=0 would freeze it at zero); "
                                  f"{self._hyperball_skipped} such params so far — it "
                                  f"takes the ordinary update instead", flush=True)
                R = state["hyperball_R"]
                if state.get("hyperball_off"):
                    p.data.mul_(1 - gamma_t * wd).add_(U.to(p.dtype), alpha=-gamma_t)
                else:
                    u_hat = U / (U.norm() + 1e-12)
                    W_tilde = p.data - gamma_t * R * u_hat.to(p.dtype)
                    p.data.copy_(R * W_tilde / (W_tilde.norm() + 1e-12))
            else:
                # No SF averaging: apply WD + step directly to live weights p
                p.data.mul_(1 - gamma_t * wd).add_(U.to(p.dtype), alpha=-gamma_t)

    # ---- scalar path ----------------------------------------------------

    def _scalar_group_step(self, group, gamma_ratio, auto_mult=1.0):
        lr = group["lr"]
        beta1 = group["beta1"]
        beta2 = group["beta2"]
        eps = group["eps"]
        wd = group["weight_decay"]
        warmup = group.get("warmup_steps", 0)

        if warmup > 0 and self._step_count < warmup:
            warm = (self._step_count + 1) / warmup
        else:
            warm = 1.0

        decay = decay_factor(group.get("decay_schedule", "none"), self._step_count,
                             group.get("total_steps", 0), warmup, group.get("decay_min", 0.0),
                             group.get("decay_start_frac", 0.8))
        gamma_t = lr * gamma_ratio * auto_mult * warm * decay * float(getattr(self, "gamma_scale", 1.0))

        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad.detach().float()
            if self._telem_on:
                self._comp_acc["sc_n"] += 1
                self._comp_acc["sc_grad_sq"] += float((grad * grad).sum())
            state = self.state[p]
            if "z" not in state:
                state["z"] = p.detach().clone().float()
                state["x"] = p.detach().clone().float()
                state["m"] = torch.zeros_like(grad)
                state["v"] = torch.zeros_like(grad)
                state["step"] = 0

            t = state["step"] + 1
            state["step"] = t

            m = state["m"]
            v = state["v"]
            m.mul_(beta1).add_(grad, alpha=(1 - beta1))
            v.mul_(beta2).addcmul_(grad, grad, value=(1 - beta2))

            bias1 = 1 - beta1 ** t
            bias2 = 1 - beta2 ** t
            m_hat = m / bias1
            v_hat = v / bias2
            u = m_hat / (v_hat.sqrt() + eps)

            # Cautious masking (optional) — same rule as the spectral path: the
            # scalar group is plain ScheduleFree-AdamW, so this is literally C-AdamW.
            if "cautious" in self._components:
                u = apply_cautious(u, grad)

            if "sf" in self._components:
                z = state["z"]
                x = state["x"]
                z.mul_(1 - gamma_t * wd).add_(u, alpha=-gamma_t)
                x.mul_(1 - 1.0 / t).add_(z, alpha=1.0 / t)
            else:
                # No SF averaging: plain AdamW step on live weights p
                p.data.mul_(1 - gamma_t * wd).add_(u.to(p.dtype), alpha=-gamma_t)

    # ---- diagnostic -----------------------------------------------------

    def diagnostic_summary(self) -> dict[str, Any]:
        """Return a small dict of optimizer-level metrics for WandB logging."""
        out = {
            "fusion/step_count": self._step_count,
            "fusion/mode": 0 if self._mode == "train" else 1,
        }
        if self._gnorm_ema is not None:
            out["fusion/gnorm_ema"] = float(self._gnorm_ema.detach().cpu())
        if self._loss_ema is not None:
            out["fusion/loss_ema"] = float(self._loss_ema.detach().cpu())
        # FP32 audit roll-ups (latest sample) — useful for catching quantisation drift
        if self._audit_stats:
            # Keep only the recent N records
            if len(self._audit_stats) > self._audit_keep:
                self._audit_stats = self._audit_stats[-self._audit_keep:]
            latest = self._audit_stats[-min(32, len(self._audit_stats)):]
            rel_means = [r["rel_mean"] for r in latest]
            rel_maxs  = [r["rel_max"]  for r in latest]
            abs_maxs  = [r["abs_max"]  for r in latest]
            out["fusion/audit_rel_mean_avg"] = sum(rel_means) / len(rel_means)
            out["fusion/audit_rel_max_p99"] = max(rel_maxs)
            out["fusion/audit_abs_max_p99"] = max(abs_maxs)
        return out
