"""The LM head runs in float32: FP16 logits are no longer rounded to FP16, and FP32 /
quantized results are bit-for-bit what they were before the change."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from conftest import make_model

from qtk import model as qmodel
from qtk.exp_icl import score_labels
from qtk.exp_ppl import chunk_nll
from qtk.quant import QuantConfig

CFG = QuantConfig(bits=3, scheme="asym", granularity="group", group_size=16)
IDS = torch.randint(0, 512, (3, 40), generator=torch.Generator().manual_seed(0))


def _old_chunk_nll(model, chunks, batch_size):
    """chunk_nll as it was before the LM head was moved to float32."""
    total_nll, total_tokens = 0.0, 0
    with torch.inference_mode():
        for start in range(0, len(chunks), batch_size):
            ids = chunks[start : start + batch_size]
            logits = model(input_ids=ids).logits[:, :-1].float()
            targets = ids[:, 1:]
            nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="sum")
            total_nll += float(nll)
            total_tokens += targets.numel()
    return total_nll, total_tokens


def _variants():
    """(label, fp32 model) for the unquantized model and both head variants."""
    yield "fp32", make_model()
    for quantize_head in (False, True):
        m = make_model()
        qmodel.apply_fake_quant(m, CFG, quantize_head=quantize_head)
        yield f"quantized, head={qmodel.head_label(quantize_head)}", m


@pytest.mark.parametrize("label,model", list(_variants()), ids=lambda v: v if isinstance(v, str) else "")
def test_fp32_and_quantized_ppl_unchanged_bitwise(label, model):
    for batch_size in (1, 2):
        assert chunk_nll(model, IDS, batch_size, "") == _old_chunk_nll(model, IDS, batch_size), label


@pytest.mark.parametrize("label,model", list(_variants()), ids=lambda v: v if isinstance(v, str) else "")
def test_fp32_and_quantized_icl_scores_unchanged_bitwise(label, model, monkeypatch):
    rng = np.random.default_rng(0)
    prompts = [rng.integers(0, 512, size=n).tolist() for n in (5, 17, 9, 30)]
    labels = [[7, 3], [11]]
    new, _ = score_labels(model, prompts, labels, batch_size=2, max_len=256)
    # the old scorer applied the head module directly, then upcast
    monkeypatch.setattr(qmodel, "fp32_logits", lambda m, h: m.get_output_embeddings()(h).float())
    old, _ = score_labels(model, prompts, labels, batch_size=2, max_len=256)
    assert np.array_equal(new, old), label


def test_fp32_logits_is_the_head_itself_for_fp32_models():
    model = make_model()
    h = torch.randn(2, 5, model.config.hidden_size, generator=torch.Generator().manual_seed(1))
    assert torch.equal(qmodel.fp32_logits(model, h), model.get_output_embeddings()(h))


def test_fp16_model_gets_fp32_matmul_before_rounding():
    m16 = make_model(dtype=torch.float16)
    # large hidden states, as in Pythia, so FP16 output rounding is visible
    h16 = (torch.randn(2, 5, m16.config.hidden_size, generator=torch.Generator().manual_seed(2)) * 300).half()
    w16 = m16.get_output_embeddings().weight
    logits = qmodel.fp32_logits(m16, h16)
    assert logits.dtype == torch.float32
    expected = F.linear(h16.float(), w16.float())  # cast before the matmul, not after
    torch.testing.assert_close(logits, expected, rtol=1e-6, atol=1e-5)
    rounded = expected.half().float()
    assert not torch.equal(rounded, expected)  # FP16 storage would have lost precision
