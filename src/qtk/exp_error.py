"""Experiment 1: weight reconstruction error vs bit-width, across granularities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from qtk import model as qmodel
from qtk.quant import QuantConfig, reconstruction_error


def run(
    models: list[str],
    bits: list[int],
    schemes: list[str],
    granularities: list[str],
    device: torch.device,
    out_dir: Path,
    max_layers: int | None = None,
    revision: str | None = qmodel.DEFAULT_REVISION,
    quantize_head: bool = False,
) -> pd.DataFrame:
    """Per-layer error for the layers that get quantized (the LM head only with ``quantize_head``)."""
    rows = []
    for name in models:
        model = qmodel.load_model(name, torch.device("cpu"), revision=revision)
        layers = qmodel.quantized_modules(model, quantize_head)
        if max_layers is not None:
            layers = layers[:max_layers]
        configs = [QuantConfig.parse(b, s, g) for s in schemes for g in granularities for b in bits]
        desc = f"{qmodel.short_name(name)} recon"
        for layer_name, mod in tqdm(layers, desc=desc, ascii=True):
            w = mod.weight.detach().to(device, torch.float32)
            for cfg in configs:
                rows.append(
                    {
                        "model": qmodel.short_name(name),
                        "head": qmodel.head_label(quantize_head),
                        "layer": layer_name,
                        "shape": "x".join(map(str, w.shape)),
                        "numel": w.numel(),
                        "scheme": cfg.scheme,
                        "granularity": cfg.granularity_label,
                        "bits": cfg.bits,
                        **reconstruction_error(w, cfg),
                    }
                )
        del model
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "error.csv", index=False)
    summary = summarize(df)
    summary.to_csv(out_dir / "error_summary.csv", index=False)
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per (model, scheme, granularity, bits): mean per-layer rel_err, and the global
    relative error sqrt(sum ||W - W_hat||^2 / sum ||W||^2) pooled over all layers."""
    keys = ["model", "head", "scheme", "granularity", "bits"]
    g = df.groupby(keys, sort=False)
    out = g.agg(
        n_layers=("layer", "count"),
        mean_rel_err=("rel_err", "mean"),
        mean_mse=("mse", "mean"),
        sq_err=("sq_err", "sum"),
        sq_norm=("sq_norm", "sum"),
    ).reset_index()
    out["global_rel_err"] = (out["sq_err"] / out["sq_norm"]) ** 0.5
    return out.drop(columns=["sq_err", "sq_norm"])
