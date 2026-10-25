import torch
from torch import nn

from qtk import model as qmodel
from qtk.quant import QuantConfig, fake_quant


def test_resolve_model_name():
    assert qmodel.resolve_model_name("70m") == "EleutherAI/pythia-70m"
    assert qmodel.resolve_model_name("pythia-410m") == "EleutherAI/pythia-410m"
    assert qmodel.resolve_model_name("EleutherAI/pythia-160m") == "EleutherAI/pythia-160m"
    assert qmodel.short_name("70m") == "pythia-70m"


def test_apply_fake_quant_touches_every_linear_and_only_linears(tiny_model):
    linears = qmodel.linear_modules(tiny_model)
    # 2 layers x (qkv, dense, fc_in, fc_out) + LM head
    assert len(linears) == 2 * 4 + 1
    before = {n: p.detach().clone() for n, p in tiny_model.named_parameters()}
    cfg = QuantConfig(bits=3, scheme="asym", granularity="group", group_size=16)
    assert qmodel.apply_fake_quant(tiny_model, cfg) == len(linears)

    linear_weights = {f"{n}.weight" for n, _ in linears}
    for name, p in tiny_model.named_parameters():
        if name in linear_weights:
            assert torch.equal(p, fake_quant(before[name], cfg)), name
            assert not torch.equal(p, before[name]), name
        else:
            assert torch.equal(p, before[name]), name  # biases, embeddings, layer norms untouched


def test_requantizing_does_not_compound_and_restore_is_exact(tiny_model):
    original = {n: p.detach().clone() for n, p in tiny_model.named_parameters()}
    qmodel.apply_fake_quant(tiny_model, QuantConfig(bits=2, scheme="sym", granularity="tensor"))
    cfg8 = QuantConfig(bits=8, scheme="asym", granularity="channel")
    qmodel.apply_fake_quant(tiny_model, cfg8)
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
        qmodel.apply_fake_quant(tiny_model, QuantConfig(bits=2, scheme=scheme, granularity="group", group_size=32))
        with torch.inference_mode():
            assert torch.isfinite(tiny_model(input_ids=ids).logits).all()


def test_fp16_model_quantized_from_fp32_math():
    from conftest import make_model

    m16 = make_model(dtype=torch.float16)
    w16 = {n: m.weight.detach().clone() for n, m in qmodel.linear_modules(m16)}
    cfg = QuantConfig(bits=4, scheme="asym", granularity="group", group_size=32)
    qmodel.apply_fake_quant(m16, cfg)
    for n, m in qmodel.linear_modules(m16):
        assert m.weight.dtype == torch.float16
        assert torch.equal(m.weight, fake_quant(w16[n].float(), cfg).half())
    assert isinstance(m16, nn.Module)
