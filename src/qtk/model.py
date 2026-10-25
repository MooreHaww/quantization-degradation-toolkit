"""Loading Pythia models and applying fake quantization to every nn.Linear."""

from __future__ import annotations

import torch
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
def apply_fake_quant(model: nn.Module, cfg: QuantConfig) -> int:
    """Fake-quantize the weight of every nn.Linear in place; returns how many were quantized.

    Always quantizes from the original weights, so calling this repeatedly with
    different configs never compounds error. Biases and embeddings are untouched.
    """
    snap = _snapshot(model)
    count = 0
    for name, mod in linear_modules(model):
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
