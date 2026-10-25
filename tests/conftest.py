"""Offline fixtures: a tiny random GPT-NeoX model, a byte-level BPE tokenizer and synthetic data."""

import random

import pytest
import torch

WORDS = (
    "the a film movie story acting plot great terrible boring wonderful awful moving dull funny "
    "sharp clumsy brilliant tedious charming flat warm cold smart lazy fresh stale bright grim"
).split()


def synthetic_sentence(rng: random.Random, n_words: int = 8) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(n_words))


def synthetic_sst2(n_train: int = 200, n_val: int = 60):
    rng = random.Random(0)
    train = [(synthetic_sentence(rng, rng.randint(5, 12)), i % 2) for i in range(n_train)]
    val = [(synthetic_sentence(rng, rng.randint(5, 12)), i % 2) for i in range(n_val)]
    return train, val


def synthetic_text(n_lines: int = 400) -> str:
    rng = random.Random(1)
    return "\n\n".join(synthetic_sentence(rng, 20) for _ in range(n_lines))


def make_tokenizer():
    from tokenizers import ByteLevelBPETokenizer
    from transformers import PreTrainedTokenizerFast

    corpus = [synthetic_text(50), "Review: \nSentiment: negative positive"]
    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator(corpus, vocab_size=400, min_frequency=1, show_progress=False)
    return PreTrainedTokenizerFast(tokenizer_object=bpe._tokenizer)


def make_model(vocab_size: int = 512, dtype: torch.dtype = torch.float32, max_positions: int = 256):
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

    torch.manual_seed(0)
    config = GPTNeoXConfig(
        vocab_size=vocab_size,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=max_positions,
    )
    return GPTNeoXForCausalLM(config).to(dtype).eval()


@pytest.fixture(scope="session")
def tokenizer():
    return make_tokenizer()


@pytest.fixture
def tiny_model():
    return make_model()


@pytest.fixture
def offline(monkeypatch):
    """Route every hub/dataset download in qtk to the offline fixtures."""
    from qtk import data, model

    tok = make_tokenizer()
    monkeypatch.setattr(model, "load_tokenizer", lambda name, revision=None: tok)
    monkeypatch.setattr(
        model, "load_model", lambda name, device, dtype=torch.float32, revision=None: make_model(dtype=dtype).to(device)
    )
    monkeypatch.setattr(data, "load_wikitext_test_text", synthetic_text)
    monkeypatch.setattr(data, "load_sst2", synthetic_sst2)
    return tok
