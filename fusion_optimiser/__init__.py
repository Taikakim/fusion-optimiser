"""FusionOpt — a composed PyTorch optimiser for small-to-medium transformers.

Fuses Muon (Newton-Schulz spectral normalisation), NorMuon (per-neuron row
scaling), Schedule-Free averaging with weight decay on the fast iterate
(SF-NorMuon), MONA (curvature-augmented momentum), KL-Shampoo (Kronecker
preconditioner) and ScheduleFree+ (Polyak step size).

Bifurcated routing:
    Spectral path  -> 2D weight matrices ≥ 128 × 128 (qkv, proj, MLP, adaLN)
    Scalar path    -> biases, LayerNorm, embeddings, small/odd matrices

Shared Polyak step size γ_t = γ_base · clamp(loss_ema / gnorm_ema, 0.1, 10).

Plus TimeConditioningCache for inference acceleration on adaLN-zero models.

See README.md for the recipe, the empirical case study, and porting notes.
"""

from .optimizer import FusionOpt, newton_schulz_5, newton_schulz_5_fp16_safe
from .groups import build_fusion_param_groups, summarise_groups
from .time_cache import TimeConditioningCache, get_or_build_cache, clear_cache_registry

__version__ = "0.1.0"

__all__ = [
    "FusionOpt",
    "build_fusion_param_groups",
    "summarise_groups",
    "TimeConditioningCache",
    "get_or_build_cache",
    "clear_cache_registry",
    "newton_schulz_5",
    "newton_schulz_5_fp16_safe",
]
