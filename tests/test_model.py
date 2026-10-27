import pytest
import torch
from torch import nn

from qtk import model as qmodel
from qtk.quant import QuantConfig, fake_quant


def test_resolve_model_name():
    assert qmodel.resolve_model_name("70m") == "EleutherAI/pythia-70m"
    assert qmodel.resolve_model_name("pythia-410m") == "EleutherAI/pythia-410m"
    assert qmodel.resolve_model_name("EleutherAI/pythia-160m") == "EleutherAI/pythia-160m"
    assert qmodel.short_name("70m") == "pythia-70m"


def test_head_is_found_by_identity(tiny_model):
    heads = [n for n, m in qmodel.linear_modules(tiny_model) if qmodel.is_head(tiny_model, m)]
    assert heads == ["lm_head"]  # transformers 5.x name (embed_out in 4.x)
    assert [n for n, _ in qmodel.quantized_modules(tiny_model)] == [
        n for n, _ in qmodel.linear_modules(tiny_model) if n != "lm_head"
    ]
    assert len(qmodel.quantized_modules(tiny_model, quantize_head=True)) == len(qmodel.linear_modules(tiny_model))


@pytest.mark.parametrize("quantize_head", [False, True])
def test_apply_fake_quant_touches_the_right_linears_only(tiny_model, quantize_head):
    linears = qmodel.linear_modules(tiny_model)
    # 2 layers x (qkv, dense, fc_in, fc_out) + LM head
    assert len(linears) == 2 * 4 + 1
    before = {n: p.detach().clone() for n, p in tiny_model.named_parameters()}
    cfg = QuantConfig(bits=3, scheme="asym", granularity="group", group_size=16)
    n_quantized = qmodel.apply_fake_quant(tiny_model, cfg, quantize_head=quantize_head)
    assert n_quantized == len(linears) - (0 if quantize_head else 1)

    quantized = {f"{n}.weight" for n, _ in qmodel.quantized_modules(tiny_model, quantize_head)}
    assert ("lm_head.weight" in quantized) == quantize_head
    for name, p in tiny_model.named_parameters():
        if name in quantized:
            assert torch.equal(p, fake_quant(before[name], cfg)), name
            assert not torch.equal(p, before[name]), name
        else:
            assert torch.equal(p, before[name]), name  # head (by default), biases, embeddings, layer norms untouched


def test_switching_head_variant_resets_the_head(tiny_model):
    head_before = tiny_model.lm_head.weight.detach().clone()
    cfg = QuantConfig(bits=2, scheme="asym", granularity="channel")
    qmodel.apply_fake_quant(tiny_model, cfg, quantize_head=True)
    assert not torch.equal(tiny_model.lm_head.weight, head_before)
    qmodel.apply_fake_quant(tiny_model, cfg)
    assert torch.equal(tiny_model.lm_head.weight, head_before)


def test_requantizing_does_not_compound_and_restore_is_exact(tiny_model):
    original = {n: p.detach().clone() for n, p in tiny_model.named_parameters()}
    qmodel.apply_fake_quant(tiny_model, QuantConfig(bits=2, scheme="sym", granularity="tensor"), quantize_head=True)
    cfg8 = QuantConfig(bits=8, scheme="asym", granularity="channel")
    qmodel.apply_fake_quant(tiny_model, cfg8, quantize_head=True)
    for n, m in qmodel.linear_modules(tiny_model):
        assert torch.equal(m.weight, fake_quant(original[f"{n}.weight"], cfg8))
    qmodel.restore_weights(tiny_model)
    for n, p in tiny_model.named_parameters():
        assert torch.equal(p, original[n]), n


def test_restore_without_quantizing_is_noop(tiny_model):
    original = {n: p.detach().clone() for n, p in tiny_model.named_parameters()}
    qmodel.restore_weights(tiny_model)
    for n, p in tiny_model.named_parameters():
        assert torch.equal(p, original[n])


def test_two_bit_model_gives_finite_logits(tiny_model):
    ids = torch.randint(0, 512, (2, 16), generator=torch.Generator().manual_seed(0))
    for scheme in ("sym", "asym"):
        for quantize_head in (False, True):
            cfg = QuantConfig(bits=2, scheme=scheme, granularity="group", group_size=32)
            qmodel.apply_fake_quant(tiny_model, cfg, quantize_head=quantize_head)
            with torch.inference_mode():
                assert torch.isfinite(tiny_model(input_ids=ids).logits).all()


def test_fp16_model_quantized_from_fp32_math():
    from conftest import make_model

    m16 = make_model(dtype=torch.float16)
    w16 = {n: m.weight.detach().clone() for n, m in qmodel.linear_modules(m16)}
    cfg = QuantConfig(bits=4, scheme="asym", granularity="group", group_size=32)
    qmodel.apply_fake_quant(m16, cfg, quantize_head=True)
    for n, m in qmodel.linear_modules(m16):
        assert m.weight.dtype == torch.float16
        assert torch.equal(m.weight, fake_quant(w16[n].float(), cfg).half())
    assert isinstance(m16, nn.Module)
