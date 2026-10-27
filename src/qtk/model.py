"""Loading Pythia models and applying fake quantization to their nn.Linear layers."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from qtk.quant import QuantConfig, fake_quant

DEFAULT_REVISION = "step143000"  # final Pythia checkpoint; pinned for reproducibility
_SNAPSHOT_ATTR = "_qtk_original_weights"


def resolve_model_name(name: str) -> str:
    """'70m' | 'pythia-70m' | 'EleutherAI/pythia-70m' -> 'EleutherAI/pythia-70m'."""
    n = name.strip()
    if "/" in n:
        return n
    if not n.startswith("pythia-"):
        n = f"pythia-{n}"
    return f"EleutherAI/{n}"


def short_name(name: str) -> str:
    return resolve_model_name(name).split("/")[-1]


def load_tokenizer(name: str, revision: str | None = DEFAULT_REVISION):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(resolve_model_name(name), revision=revision)


def load_model(
    name: str,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    revision: str | None = DEFAULT_REVISION,
) -> nn.Module:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        resolve_model_name(name), revision=revision, dtype=dtype
    )
    return model.to(device).eval()


def linear_modules(model: nn.Module) -> list[tuple[str, nn.Linear]]:
    return [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]


def is_head(model: nn.Module, module: nn.Module) -> bool:
    """True for the LM head (``lm_head`` in transformers 5.x, ``embed_out`` in older releases)."""
    return module is model.get_output_embeddings()


def quantized_modules(model: nn.Module, quantize_head: bool = False) -> list[tuple[str, nn.Linear]]:
    """The Linear layers that get fake-quantized: every one, minus the LM head unless ``quantize_head``."""
    return [(n, m) for n, m in linear_modules(model) if quantize_head or not is_head(model, m)]


def head_label(quantize_head: bool) -> str:
    """Value of the ``head`` column in results CSVs."""
    return "quantized" if quantize_head else "fp32"


def fp32_logits(model: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    """LM-head logits with the matmul in float32, whatever the model's dtype.

    The hidden state and the head weight are cast to float32 *before* the
    matmul. For a float32 model the casts are no-ops, so this is exactly
    ``head(hidden)``. For an FP16 model it avoids rounding the logits to FP16:
    Pythia's logits reach ~1000 in magnitude, where FP16 spacing is 0.5.
    """
    head = model.get_output_embeddings()
    bias = head.bias.float() if head.bias is not None else None
    return F.linear(hidden.float(), head.weight.float(), bias)


def _snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    """Original Linear weights (float32, CPU), captured on first use."""
    snap = getattr(model, _SNAPSHOT_ATTR, None)
    if snap is None:
        snap = {
            n: m.weight.detach().to("cpu", torch.float32, copy=True)
            for n, m in linear_modules(model)
        }
        setattr(model, _SNAPSHOT_ATTR, snap)
    return snap


@torch.no_grad()
def apply_fake_quant(model: nn.Module, cfg: QuantConfig, quantize_head: bool = False) -> int:
    """Fake-quantize nn.Linear weights in place; returns how many were quantized.

    By default the LM head keeps its original weights; ``quantize_head=True``
    quantizes every nn.Linear including the head. Always quantizes from the
    original weights, so calling this repeatedly with different configs never
    compounds error, and layers left out are reset to their originals.
    Biases and embeddings are untouched.
    """
    snap = _snapshot(model)
    restore_weights(model)
    count = 0
    for name, mod in quantized_modules(model, quantize_head):
        w = snap[name].to(mod.weight.device)
        mod.weight.copy_(fake_quant(w, cfg))
        count += 1
    return count


@torch.no_grad()
def restore_weights(model: nn.Module) -> None:
    """Put the original Linear weights back (bit-exact)."""
    snap = getattr(model, _SNAPSHOT_ATTR, None)
    if snap is None:
        return
    for name, mod in linear_modules(model):
        mod.weight.copy_(snap[name].to(mod.weight.device))
