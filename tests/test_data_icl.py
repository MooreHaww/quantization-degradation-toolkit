import numpy as np
import torch

from conftest import make_model, synthetic_sst2, synthetic_text
from qtk import data
from qtk.exp_icl import build_prompt, score_labels


def test_tokenize_chunks_shapes(tokenizer):
    text = synthetic_text()
    chunks = data.tokenize_chunks(tokenizer, text, seq_len=64)
    n_tokens = len(tokenizer(text)["input_ids"])
    assert chunks.shape == (n_tokens // 64, 64)
    assert data.tokenize_chunks(tokenizer, text, seq_len=64, max_chunks=3).shape == (3, 64)
    assert torch.equal(chunks[:3], data.tokenize_chunks(tokenizer, text, seq_len=64, max_chunks=3))


def test_eval_subset_is_stratified_and_deterministic():
    _, val = synthetic_sst2(n_val=100)
    a = data.select_eval_subset(val, 40, seed=0)
    assert a == data.select_eval_subset(val, 40, seed=0)
    assert a != data.select_eval_subset(val, 40, seed=1)
    assert len(a) == 40 and sum(y for _, y in a) == 20
    assert len(set(a)) == 40
    assert data.select_eval_subset(val, 1000, seed=0) == val


def test_demo_sequences_balanced_nested_and_deterministic():
    train, _ = synthetic_sst2()
    seqs = data.sample_demo_sequences(train, n_examples=20, max_k=16, seed=3)
    assert seqs == data.sample_demo_sequences(train, n_examples=20, max_k=16, seed=3)
    assert seqs != data.sample_demo_sequences(train, n_examples=20, max_k=16, seed=4)
    for seq in seqs:
        assert len(seq) == 16 and len(set(seq)) == 16
        for k in (2, 4, 8, 16):
            assert sum(y for _, y in seq[:k]) == k // 2
        assert all(len(s.split()) >= data.DEMO_MIN_WORDS for s, _ in seq)


def test_build_prompt_format():
    p = build_prompt([("good fun", 1), ("bad", 0)], "okay film")
    assert p == "Review: good fun\nSentiment: positive\n\nReview: bad\nSentiment: negative\n\nReview: okay film\nSentiment:"
    assert build_prompt([], "x") == "Review: x\nSentiment:"


def _reference_scores(model, prompt_ids, label_ids):
    out = np.zeros((len(prompt_ids), len(label_ids)))
    with torch.inference_mode():
        for i, p in enumerate(prompt_ids):
            for j, lab in enumerate(label_ids):
                ids = torch.tensor([p + lab])
                logp = torch.log_softmax(model(input_ids=ids).logits[0].float(), -1)
                out[i, j] = sum(float(logp[len(p) - 1 + t, tok]) for t, tok in enumerate(lab))
    return out


def test_score_labels_matches_naive_full_forward():
    model = make_model()
    rng = np.random.default_rng(0)
    prompts = [rng.integers(0, 512, size=n).tolist() for n in (5, 17, 9, 30, 2)]
    for labels in ([[7], [11]], [[7, 3], [11]], [[7, 3, 5], [7, 4]]):
        got, n_trunc = score_labels(model, prompts, labels, batch_size=2, max_len=256)
        np.testing.assert_allclose(got, _reference_scores(model, prompts, labels), rtol=1e-4, atol=1e-4)
        assert n_trunc == 0


def test_score_labels_truncates_from_left():
    model = make_model()
    long_prompt = list(range(1, 300))
    got, n_trunc = score_labels(model, [long_prompt], [[7], [11]], batch_size=1, max_len=256)
    assert n_trunc == 1
    np.testing.assert_allclose(got, _reference_scores(model, [long_prompt[-256:]], [[7], [11]]), rtol=1e-4, atol=1e-4)
