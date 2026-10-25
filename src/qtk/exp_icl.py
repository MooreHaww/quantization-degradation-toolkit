"""Experiment 3: few-shot SST-2 accuracy vs number of shots, per precision.

Each prompt is k demonstrations followed by the query; the prediction is the
label word with the higher summed log-likelihood as the continuation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from qtk import data
from qtk import model as qmodel
from qtk.exp_ppl import precisions
from qtk.quant import QuantConfig

DEMO_TEMPLATE = "Review: {sentence}\nSentiment:{label}\n\n"
QUERY_TEMPLATE = "Review: {sentence}\nSentiment:"


def build_prompt(demos: list[data.Example], sentence: str) -> str:
    parts = [DEMO_TEMPLATE.format(sentence=s, label=data.LABEL_WORDS[y]) for s, y in demos]
    return "".join(parts) + QUERY_TEMPLATE.format(sentence=sentence)


@torch.inference_mode()
def score_labels(
    model: nn.Module,
    prompt_ids: list[list[int]],
    label_ids: list[list[int]],
    batch_size: int,
    max_len: int,
) -> tuple[np.ndarray, int]:
    """Summed log p(label tokens | prompt) for every prompt and label.

    Returns ``(scores [n_prompts, n_labels], n_truncated)``. Prompts that would
    exceed ``max_len`` are truncated from the left. Labels whose tokens share
    the same prefix share one forward pass (so single-token labels cost one
    pass per prompt). Only the needed positions go through the LM head.
    """
    # item = (input ids, index of the logit predicting the first label token, [(prompt, label)])
    items: list[tuple[list[int], int, list[tuple[int, int]]]] = []
    truncated: set[int] = set()
    for i, p in enumerate(prompt_ids):
        groups: dict[tuple[int, ...], list[int]] = {}
        for j, lab in enumerate(label_ids):
            groups.setdefault(tuple(lab[:-1]), []).append(j)
        for prefix, js in groups.items():
            seq = p + list(prefix)
            overflow = len(seq) - max_len
            if overflow > 0:
                seq = seq[overflow:]
                truncated.add(i)
            items.append((seq, len(seq) - len(prefix) - 1, [(i, j) for j in js]))

    device = next(model.parameters()).device
    base = model.base_model
    head = model.get_output_embeddings()
    scores = np.zeros((len(prompt_ids), len(label_ids)), dtype=np.float64)
    order = sorted(range(len(items)), key=lambda n: -len(items[n][0]))  # stable, deterministic
    for start in tqdm(range(0, len(order), batch_size), ascii=True, leave=False, desc="score"):
        batch = [items[n] for n in order[start : start + batch_size]]
        width = max(len(seq) for seq, _, _ in batch)
        ids = torch.zeros((len(batch), width), dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for b, (seq, _, _) in enumerate(batch):
            ids[b, : len(seq)] = torch.tensor(seq)
            mask[b, : len(seq)] = 1
        hidden = base(input_ids=ids.to(device), attention_mask=mask.to(device))[0]
        for b, (_, first, pairs) in enumerate(batch):
            n_pos = max(len(label_ids[j]) for _, j in pairs)
            logp = torch.log_softmax(head(hidden[b, first : first + n_pos]).float(), dim=-1)
            for i, j in pairs:
                lab = label_ids[j]
                scores[i, j] = float(logp[torch.arange(len(lab)), torch.tensor(lab)].sum())
    return scores, len(truncated)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "precision", "bits", "k"]
    out = (
        df.groupby(keys, sort=False)
        .agg(
            n_seeds=("seed", "nunique"),
            acc_mean=("accuracy", "mean"),
            acc_std=("accuracy", "std"),
            margin_mean=("mean_margin", "mean"),
        )
        .reset_index()
    )
    out["acc_std"] = out["acc_std"].fillna(0.0)
    acc0 = out[out["k"] == 0].set_index(["model", "precision"])["acc_mean"]
    out["gain"] = out["acc_mean"] - out.set_index(["model", "precision"]).index.map(acc0)
    return out


def run(
    models: list[str],
    bits: list[int],
    scheme: str,
    granularity: str,
    shots: list[int],
    seeds: list[int],
    n_eval: int,
    device: torch.device,
    out_dir: Path,
    batch_size: int = 8,
    fp16: bool = True,
    subset_seed: int = 0,
    save_preds: bool = False,
    revision: str | None = qmodel.DEFAULT_REVISION,
) -> pd.DataFrame:
    train, val = data.load_sst2()
    eval_set = data.select_eval_subset(val, n_eval, subset_seed)
    labels = np.array([y for _, y in eval_set])
    max_k = max(shots)
    demo_seqs = {s: data.sample_demo_sequences(train, len(eval_set), max_k, s) for s in seeds}
    # k = 0 does not depend on the demo seed: evaluate once, report it under every seed
    conditions = [(k, s) for k in sorted(shots) for s in (seeds[:1] if k == 0 else seeds)]

    rows: list[dict] = []
    pred_rows: list[dict] = []

    for name in models:
        short = qmodel.short_name(name)
        tok = qmodel.load_tokenizer(name, revision=revision)
        label_ids = [tok(w, add_special_tokens=False)["input_ids"] for w in data.LABEL_WORDS]
        prompts = {
            (k, s): [
                tok(build_prompt(demo_seqs[s][i][:k], sent), add_special_tokens=False)["input_ids"]
                for i, (sent, _) in enumerate(eval_set)
            ]
            for k, s in conditions
        }

        def evaluate(model: nn.Module, precision: str, n_bits: int, cfg: QuantConfig | None) -> None:
            max_len = model.config.max_position_embeddings
            for k, s in tqdm(conditions, desc=f"{short} {precision}", ascii=True):
                scores, n_trunc = score_labels(model, prompts[(k, s)], label_ids, batch_size, max_len)
                preds = scores.argmax(axis=1)
                idx = np.arange(len(labels))
                margin = scores[idx, labels] - scores[idx, 1 - labels]
                for seed in seeds if k == 0 else [s]:
                    rows.append(
                        {
                            "model": short,
                            "precision": precision,
                            "bits": n_bits,
                            "scheme": cfg.scheme if cfg else "",
                            "granularity": cfg.granularity_label if cfg else "",
                            "k": k,
                            "seed": seed,
                            "n_eval": len(labels),
                            "accuracy": float((preds == labels).mean()),
                            "mean_margin": float(margin.mean()),
                            "n_truncated": n_trunc,
                        }
                    )
                if save_preds:
                    for i in idx:
                        pred_rows.append(
                            {
                                "model": short, "precision": precision, "k": k, "seed": s,
                                "example": int(i), "label": int(labels[i]),
                                "logp_negative": scores[i, 0], "logp_positive": scores[i, 1],
                            }
                        )
            # write after every precision so a long run that dies keeps what it finished
            pd.DataFrame(rows).to_csv(out_dir / "icl.csv", index=False)

        model = qmodel.load_model(name, device, dtype=torch.float32, revision=revision)
        for precision, cfg in precisions(bits, scheme, granularity):
            if cfg is None:
                qmodel.restore_weights(model)
            else:
                qmodel.apply_fake_quant(model, cfg)
            evaluate(model, precision, cfg.bits if cfg else 32, cfg)
        del model

        if fp16 and device.type == "cuda":
            model16 = qmodel.load_model(name, device, dtype=torch.float16, revision=revision)
            evaluate(model16, "fp16", 16, None)
            del model16
        if device.type == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    summarize(df).to_csv(out_dir / "icl_summary.csv", index=False)
    if save_preds:
        pd.DataFrame(pred_rows).to_csv(out_dir / "icl_predictions.csv", index=False)
    return df
