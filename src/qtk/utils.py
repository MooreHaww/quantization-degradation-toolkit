"""Seeding, device selection, list parsing and run metadata."""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Must be set before the first cuBLAS call for deterministic GEMMs on CUDA.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def set_seed(seed: int) -> None:
    """Seed every RNG we use and ask PyTorch for deterministic kernels.

    TF32 is disabled so that "FP32" really is IEEE float32 on Ampere and newer GPUs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")


def resolve_device(device: str) -> torch.device:
    """Map 'auto' | 'cpu' | 'cuda' | 'cuda:N' to a torch.device, failing loudly if CUDA is missing."""
    d = device.strip().lower()
    if d == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if d.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but torch.cuda.is_available() is False. On an RTX 50-series "
                "(Blackwell) GPU install a CUDA 12.8+ build of PyTorch, e.g. "
                "pip install torch --index-url https://download.pytorch.org/whl/cu128"
            )
        return torch.device(d)
    if d == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unknown device {device!r}; use auto, cpu or cuda")


def device_description(device: torch.device) -> str:
    if device.type != "cuda":
        return f"cpu ({platform.processor() or platform.machine()})"
    idx = device.index if device.index is not None else torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(idx)
    return f"{torch.cuda.get_device_name(idx)} (sm_{major}{minor})"


def check_cuda_arch(device: torch.device) -> str | None:
    """Return a warning if this PyTorch build was not compiled for the GPU's architecture."""
    if device.type != "cuda":
        return None
    idx = device.index if device.index is not None else torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(idx)
    arch = f"sm_{major}{minor}"
    arch_list = torch.cuda.get_arch_list()
    if arch_list and arch not in arch_list:
        return (
            f"this PyTorch build supports {arch_list} but the GPU is {arch}; "
            "kernels may fail. Install a build that includes it (cu128 or newer for sm_120)."
        )
    return None


def parse_int_list(text: str) -> list[int]:
    """'8,6,4' -> [8, 6, 4]. Also accepts spaces or semicolons as separators."""
    parts = [p for p in text.replace(";", ",").replace(" ", ",").split(",") if p]
    try:
        return [int(p) for p in parts]
    except ValueError as exc:
        raise ValueError(f"expected a comma-separated list of integers, got {text!r}") from exc


def parse_str_list(text: str) -> list[str]:
    return [p.strip() for p in text.replace(";", ",").split(",") if p.strip()]


def git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def write_metadata(out_dir: Path, experiment: str, args: dict[str, Any], device: torch.device) -> Path:
    import transformers

    meta = {
        "experiment": experiment,
        "args": args,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device": device_description(device),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "git_sha": git_sha(),
    }
    path = out_dir / f"{experiment}_meta.json"
    path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return path
