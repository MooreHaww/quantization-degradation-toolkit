"""Uniform weight quantizers written from scratch in PyTorch.

All quantizer arithmetic runs in float32 regardless of the input dtype; the
fake-quantized result is cast back to the input dtype at the end.

Layout convention: a weight tensor is viewed as a matrix of shape
``[rows, cols]`` with ``rows = w.shape[0]`` (output features for ``nn.Linear``)
and ``cols`` = everything else flattened (input features). Granularities:

* ``tensor``  - one scale (and zero point) for the whole tensor
* ``channel`` - one scale per row (per output channel)
* ``group``   - one scale per contiguous block of ``group_size`` columns
                within each row; the last block of a row is zero-padded if
                ``cols`` is not a multiple of ``group_size``

Schemes (``b`` = bits, ``round`` = round-half-to-even as in ``torch.round``):

* ``sym``  - ``qmax = 2^(b-1) - 1``, ``scale = max|w| / qmax``,
             ``q = clamp(round(w / scale), -qmax, qmax)``,
             ``w_hat = q * scale``. Restricted (zero-centred) range, so 2-bit
             symmetric has the three levels {-1, 0, +1}.
* ``asym`` - range widened to contain zero (``lo = min(min w, 0)``,
             ``hi = max(max w, 0)``), ``qmax = 2^b - 1``,
             ``scale = (hi - lo) / qmax``, ``zp = round(-lo / scale)``,
             ``q = clamp(round(w / scale) + zp, 0, qmax)``,
             ``w_hat = (q - zp) * scale``.

A group whose range is zero (all zeros, or a constant under ``sym``) gets
``scale = 1`` so no division by zero occurs; all-zero groups round-trip
exactly. For every element, ``|w - w_hat| <= scale / 2`` (up to float32
rounding), because round-then-clamp picks the nearest grid point and the grid
endpoints lie within ``scale / 2`` of the data range.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, NamedTuple

import torch
import torch.nn.functional as F

Scheme = Literal["sym", "asym"]
Granularity = Literal["tensor", "channel", "group"]

MIN_BITS = 2
MAX_BITS = 8


@dataclass(frozen=True)
class QuantConfig:
    bits: int
    scheme: Scheme = "asym"
    granularity: Granularity = "group"
    group_size: int | None = 128

    def __post_init__(self) -> None:
        if isinstance(self.bits, bool) or not isinstance(self.bits, int):
            raise TypeError(f"bits must be an int, got {self.bits!r}")
        if not MIN_BITS <= self.bits <= MAX_BITS:
            raise ValueError(f"bits must be in [{MIN_BITS}, {MAX_BITS}], got {self.bits}")
        if self.scheme not in ("sym", "asym"):
            raise ValueError(f"scheme must be 'sym' or 'asym', got {self.scheme!r}")
        if self.granularity not in ("tensor", "channel", "group"):
            raise ValueError(
                f"granularity must be 'tensor', 'channel' or 'group', got {self.granularity!r}"
            )
        if self.granularity == "group":
            if self.group_size is None or isinstance(self.group_size, bool):
                raise ValueError("granularity 'group' requires a group_size")
            if not isinstance(self.group_size, int) or self.group_size < 1:
                raise ValueError(f"group_size must be a positive int, got {self.group_size!r}")
        else:
            # group_size is meaningless for tensor/channel; normalise it away
            object.__setattr__(self, "group_size", None)

    @classmethod
    def parse(cls, bits: int, scheme: str, granularity: str) -> QuantConfig:
        """Build a config from CLI-style strings: 'tensor', 'channel', 'group:128' or 'g128'."""
        gran, group_size = parse_granularity(granularity)
        return cls(bits=bits, scheme=scheme, granularity=gran, group_size=group_size)  # type: ignore[arg-type]

    @property
    def granularity_label(self) -> str:
        return f"g{self.group_size}" if self.granularity == "group" else self.granularity

    @property
    def label(self) -> str:
        return f"w{self.bits}-{self.scheme}-{self.granularity_label}"


def parse_granularity(text: str) -> tuple[str, int | None]:
    """Parse 'tensor' | 'channel' | 'group:N' | 'gN' into (granularity, group_size)."""
    t = text.strip().lower()
    if t in ("tensor", "channel"):
        return t, None
    if t.startswith("group:"):
        num = t.split(":", 1)[1]
    elif t.startswith("g") and t[1:].isdigit():
        num = t[1:]
    else:
        raise ValueError(f"unrecognised granularity {text!r}; use tensor, channel or group:N")
    if not num.isdigit() or int(num) < 1:
        raise ValueError(f"invalid group size in {text!r}")
    return "group", int(num)


class QuantizedTensor(NamedTuple):
    """Integer codes plus the parameters needed to dequantize them.

    ``q``, ``scale`` and ``zero_point`` are float32 and laid out per group:
    ``q`` is ``[n_groups, group_len]``, ``scale``/``zero_point`` are
    ``[n_groups, 1]``. ``q`` holds exact integer values.
    """

    q: torch.Tensor
    scale: torch.Tensor
    zero_point: torch.Tensor
    shape: torch.Size
    pad: int


def _to_groups(w: torch.Tensor, cfg: QuantConfig) -> tuple[torch.Tensor, int]:
    x = w.detach().to(torch.float32)
    rows = x.shape[0] if x.dim() >= 2 else 1
    x = x.reshape(rows, -1)
    if cfg.granularity == "tensor":
        return x.reshape(1, -1), 0
    if cfg.granularity == "channel":
        return x, 0
    assert cfg.group_size is not None
    cols = x.shape[1]
    pad = (-cols) % cfg.group_size
    if pad:
        # zero padding never changes max|w| (sym) or a zero-containing range (asym)
        x = F.pad(x, (0, pad))
    return x.reshape(-1, cfg.group_size), pad


def _from_groups(x: torch.Tensor, shape: torch.Size, pad: int) -> torch.Tensor:
    rows = shape[0] if len(shape) >= 2 else 1
    x = x.reshape(rows, -1)
    if pad:
        x = x[:, : x.shape[1] - pad]
    return x.reshape(shape)


def quantize(w: torch.Tensor, cfg: QuantConfig) -> QuantizedTensor:
    if w.numel() == 0:
        raise ValueError("cannot quantize an empty tensor")
    if not torch.isfinite(w).all():
        raise ValueError("cannot quantize a tensor containing NaN or inf")
    x, pad = _to_groups(w, cfg)

    if cfg.scheme == "sym":
        qmax = 2 ** (cfg.bits - 1) - 1
        scale = x.abs().amax(dim=1, keepdim=True) / qmax
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        zero_point = torch.zeros_like(scale)
        q = torch.clamp(torch.round(x / scale), -qmax, qmax)
    else:
        qmax = 2**cfg.bits - 1
        lo = x.amin(dim=1, keepdim=True).clamp(max=0.0)
        hi = x.amax(dim=1, keepdim=True).clamp(min=0.0)
        scale = (hi - lo) / qmax
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        zero_point = torch.round(-lo / scale)
        q = torch.clamp(torch.round(x / scale) + zero_point, 0, qmax)

    return QuantizedTensor(q=q, scale=scale, zero_point=zero_point, shape=w.shape, pad=pad)


def dequantize(qt: QuantizedTensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    x = (qt.q - qt.zero_point) * qt.scale
    return _from_groups(x, qt.shape, qt.pad).to(dtype)


def fake_quant(w: torch.Tensor, cfg: QuantConfig) -> torch.Tensor:
    """Quantize then dequantize ``w`` (math in float32); returns ``w``'s dtype and shape."""
    return dequantize(quantize(w, cfg), dtype=w.dtype)


def reconstruction_error(w: torch.Tensor, cfg: QuantConfig) -> dict[str, float]:
    """MSE and relative Frobenius error of fake-quantizing ``w`` (computed in float64)."""
    ref = w.detach().to(torch.float64)
    err = fake_quant(w.detach().to(torch.float32), cfg).to(torch.float64) - ref
    sq_err = float(err.pow(2).sum())
    sq_norm = float(ref.pow(2).sum())
    return {
        "mse": sq_err / ref.numel(),
        "rel_err": math.sqrt(sq_err / sq_norm) if sq_norm > 0 else float("nan"),
        "sq_err": sq_err,
        "sq_norm": sq_norm,
    }
