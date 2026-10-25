"""Dataset loading: WikiText-2 (perplexity) and SST-2 (in-context learning)."""

from __future__ import annotations

import numpy as np
import torch

WIKITEXT = ("Salesforce/wikitext", "wikitext-2-raw-v1")
SST2 = "stanfordnlp/sst2"
LABEL_WORDS = (" negative", " positive")  # index = SST-2 label id
DEMO_MIN_WORDS = 5  # SST-2 train contains many 1-3 word phrases; keep demos as real reviews

Example = tuple[str, int]


def load_wikitext_test_text() -> str:
    """WikiText-2 (raw) test split joined with blank lines, as in the HF perplexity guide."""
    from datasets import load_dataset

    ds = load_dataset(*WIKITEXT, split="test")
    return "\n\n".join(ds["text"])


def tokenize_chunks(
    tokenizer, text: str, seq_len: int = 1024, max_chunks: int | None = None
) -> torch.Tensor:
    """Tokenize ``text`` once and cut it into non-overlapping ``seq_len`` chunks.

    The trailing partial chunk is dropped. ``max_chunks`` keeps the first N chunks.
    """
    ids = tokenizer(text, return_attention_mask=False, verbose=False)["input_ids"]
    n = len(ids) // seq_len
    if max_chunks is not None:
        n = min(n, max_chunks)
    if n == 0:
        raise ValueError(f"text has {len(ids)} tokens, fewer than one chunk of {seq_len}")
    return torch.tensor(ids[: n * seq_len], dtype=torch.long).view(n, seq_len)


def load_sst2() -> tuple[list[Example], list[Example]]:
    """(train, validation) as lists of (sentence, label); label 0 = negative, 1 = positive."""
    from datasets import load_dataset

    ds = load_dataset(SST2)

    def to_list(split) -> list[Example]:
        return [(s.strip(), int(y)) for s, y in zip(split["sentence"], split["label"])]

    return to_list(ds["train"]), to_list(ds["validation"])


def select_eval_subset(examples: list[Example], n: int, seed: int) -> list[Example]:
    """Label-stratified, seeded subset of ``n`` examples (all of them if ``n`` >= len)."""
    if n >= len(examples):
        return list(examples)
    rng = np.random.default_rng(seed)
    by_label = {lab: [e for e in examples if e[1] == lab] for lab in (0, 1)}
    n_neg = round(n * len(by_label[0]) / len(examples))
    picked: list[Example] = []
    for lab, k in ((0, n_neg), (1, n - n_neg)):
        idx = rng.choice(len(by_label[lab]), size=k, replace=False)
        picked += [by_label[lab][i] for i in sorted(idx)]
    order = rng.permutation(len(picked))
    return [picked[i] for i in order]


def sample_demo_sequences(
    train: list[Example], n_examples: int, max_k: int, seed: int
) -> list[list[Example]]:
    """One demo sequence of length ``max_k`` per eval example; k-shot uses its first k.

    Each sequence is built from (negative, positive) pairs in random within-pair
    order, so every even-length prefix is exactly label-balanced (k = 1 gets a
    random label). Demos are drawn without replacement within a sequence and
    depend only on (seed, example index), so every precision sees the same demos.
    """
    pools = {
        lab: [e for e in train if e[1] == lab and len(e[0].split()) >= DEMO_MIN_WORDS]
        for lab in (0, 1)
    }
    n_pairs = (max_k + 1) // 2
    for lab, pool in pools.items():
        if len(pool) < n_pairs:
            raise ValueError(f"only {len(pool)} label-{lab} demos available, need {n_pairs}")
    sequences = []
    for i in range(n_examples):
        rng = np.random.default_rng([seed, i])
        neg = rng.choice(len(pools[0]), size=n_pairs, replace=False)
        pos = rng.choice(len(pools[1]), size=n_pairs, replace=False)
        seq: list[Example] = []
        for a, b in zip(neg, pos):
            pair = [pools[0][a], pools[1][b]]
            if rng.random() < 0.5:
                pair.reverse()
            seq += pair
        sequences.append(seq[:max_k])
    return sequences
