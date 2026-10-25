"""Experiment 2: WikiText-2 perplexity at FP32, FP16 and fake-quantized bit-widths."""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from qtk import data
from qtk import model as qmodel
from qtk.quant import QuantConfig


@torch.inference_mode()
def chunk_nll(model: nn.Module, chunks: torch.Tensor, batch_size: int, desc: str) -> tuple[float, int]:
    """Summed next-token NLL (float32 log-softmax) over all chunks, and the token count.

    Chunks are independent, so the first token of each chunk is not predicted.
    """
    device = next(model.parameters()).device
    total_nll, total_tokens = 0.0, 0
    for start in tqdm(range(0, len(chunks), batch_size), desc=desc, ascii=True, leave=False):
        ids = chunks[start : start + batch_size].to(device)
        logits = model(input_ids=ids).logits[:, :-1].float()
        targets = ids[:, 1:]
        nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="sum")
        total_nll += float(nll)
        total_tokens += targets.numel()
    return total_nll, total_tokens


def precisions(bits: list[int], scheme: str, granularity: str) -> Iterator[tuple[str, QuantConfig | None]]:
    yield "fp32", None
    for b in bits:
        yield f"w{b}", QuantConfig.parse(b, scheme, granularity)


def run(
    models: list[str],
    bits: list[int],
    scheme: str,
    granularity: str,
    device: torch.device,
    out_dir: Path,
    seq_len: int = 1024,
    max_chunks: int | None = None,
    batch_size: int = 4,
    fp16: bool = True,
    revision: str | None = qmodel.DEFAULT_REVISION,
) -> pd.DataFrame:
    text = data.load_wikitext_test_text()
    rows: list[dict] = []
    csv_path = out_dir / "ppl.csv"

    def record(name: str, precision: str, n_bits: int, cfg: QuantConfig | None, nll: float, n_tok: int, n_chunks: int) -> None:
        rows.append(
            {
                "model": qmodel.short_name(name),
                "precision": precision,
                "bits": n_bits,
                "scheme": cfg.scheme if cfg else "",
                "granularity": cfg.granularity_label if cfg else "",
                "n_chunks": n_chunks,
                "n_tokens": n_tok,
                "nll": nll / n_tok,
                "ppl": math.exp(nll / n_tok),
            }
        )
        # write after every row so a long run that dies keeps what it finished
        pd.DataFrame(rows).to_csv(csv_path, index=False)

    for name in models:
        tok = qmodel.load_tokenizer(name, revision=revision)
        chunks = data.tokenize_chunks(tok, text, seq_len=seq_len, max_chunks=max_chunks)
        short = qmodel.short_name(name)

        model = qmodel.load_model(name, device, dtype=torch.float32, revision=revision)
        for precision, cfg in precisions(bits, scheme, granularity):
            if cfg is None:
                qmodel.restore_weights(model)
            else:
                qmodel.apply_fake_quant(model, cfg)
            nll, n_tok = chunk_nll(model, chunks, batch_size, f"{short} {precision}")
            record(name, precision, cfg.bits if cfg else 32, cfg, nll, n_tok, len(chunks))
        del model

        if fp16 and device.type == "cuda":
            model16 = qmodel.load_model(name, device, dtype=torch.float16, revision=revision)
            nll, n_tok = chunk_nll(model16, chunks, batch_size, f"{short} fp16")
            record(name, "fp16", 16, None, nll, n_tok, len(chunks))
            del model16
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return pd.DataFrame(rows)
