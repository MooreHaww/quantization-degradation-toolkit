import itertools

import pytest
import torch

from qtk.quant import QuantConfig, dequantize, fake_quant, parse_granularity, quantize, reconstruction_error

BITS = range(2, 9)
SCHEMES = ("sym", "asym")
GRANS = (("tensor", None), ("channel", None), ("group", 32), ("group", 64), ("group", 128))
ALL_CONFIGS = [
    QuantConfig(bits=b, scheme=s, granularity=g, group_size=gs)
    for b, s, (g, gs) in itertools.product(BITS, SCHEMES, GRANS)
]


def weight(rows=64, cols=256, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(rows, cols, generator=gen) * 0.02


@pytest.mark.parametrize("cfg", ALL_CONFIGS, ids=lambda c: c.label)
def test_shape_dtype_and_error_bound(cfg):
    w = weight()
    qt = quantize(w, cfg)
    w_hat = fake_quant(w, cfg)
    assert w_hat.shape == w.shape and w_hat.dtype == w.dtype
    # |w - w_hat| <= scale / 2 element-wise, with scale broadcast back to weight layout
    scale = dequantize(qt._replace(q=torch.ones_like(qt.q), zero_point=torch.zeros_like(qt.zero_point)))
    assert torch.all((w - w_hat).abs() <= scale / 2 * (1 + 1e-5) + 1e-9)


@pytest.mark.parametrize("cfg", ALL_CONFIGS, ids=lambda c: c.label)
def test_codes_are_integers_in_range(cfg):
    qt = quantize(weight(), cfg)
    assert torch.equal(qt.q, torch.round(qt.q))
    if cfg.scheme == "sym":
        qmax = 2 ** (cfg.bits - 1) - 1
        assert qt.q.min() >= -qmax and qt.q.max() <= qmax
    else:
        assert qt.q.min() >= 0 and qt.q.max() <= 2**cfg.bits - 1
        assert torch.all((qt.zero_point >= 0) & (qt.zero_point <= 2**cfg.bits - 1))
    # at most 2^b distinct levels per group
    for row in qt.q:
        assert row.unique().numel() <= 2**cfg.bits


@pytest.mark.parametrize("cfg", ALL_CONFIGS, ids=lambda c: c.label)
def test_idempotent(cfg):
    once = fake_quant(weight(), cfg)
    twice = fake_quant(once, cfg)
    torch.testing.assert_close(twice, once, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("gran", GRANS, ids=str)
def test_error_decreases_with_bits(scheme, gran):
    w = weight(128, 512)
    errs = [
        reconstruction_error(w, QuantConfig(bits=b, scheme=scheme, granularity=gran[0], group_size=gran[1]))["mse"]
        for b in BITS
    ]
    assert all(a > b for a, b in zip(errs, errs[1:])), errs


@pytest.mark.parametrize("bits", BITS)
def test_finer_granularity_is_no_worse(bits):
    w = weight(128, 512)
    errs = [
        reconstruction_error(w, QuantConfig(bits=bits, scheme="asym", granularity=g, group_size=gs))["mse"]
        for g, gs in (("tensor", None), ("channel", None), ("group", 128), ("group", 32))
    ]
    assert all(a >= b * 0.999 for a, b in zip(errs, errs[1:])), errs


@pytest.mark.parametrize("cfg", ALL_CONFIGS, ids=lambda c: c.label)
def test_all_zero_tensor_round_trips_exactly(cfg):
    w = torch.zeros(8, 256)
    out = fake_quant(w, cfg)
    assert torch.equal(out, w)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_all_zero_group_inside_nonzero_tensor(scheme):
    w = weight(4, 256)
    w[1, 32:64] = 0.0  # exactly one all-zero group of 32
    out = fake_quant(w, QuantConfig(bits=4, scheme=scheme, granularity="group", group_size=32))
    assert torch.isfinite(out).all()
    assert torch.equal(out[1, 32:64], torch.zeros(32))


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("value", [0.37, -1.5])
def test_constant_groups_are_finite_and_accurate(scheme, value):
    w = torch.full((4, 64), value)
    out = fake_quant(w, QuantConfig(bits=2, scheme=scheme, granularity="group", group_size=32))
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, w, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("bits", BITS)
def test_asym_represents_zero_exactly(bits):
    w = weight(16, 128) + 0.05  # mostly positive, range still widened to include 0
    w[:, 0] = 0.0
    out = fake_quant(w, QuantConfig(bits=bits, scheme="asym", granularity="group", group_size=32))
    assert torch.equal(out[:, 0], torch.zeros(16))


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("cols", [1, 31, 100, 129])
def test_group_size_not_dividing_columns(scheme, cols):
    w = weight(8, cols)
    cfg = QuantConfig(bits=4, scheme=scheme, granularity="group", group_size=32)
    out = fake_quant(w, cfg)
    assert out.shape == w.shape
    # padded result matches quantizing each row's groups independently
    expected = torch.cat(
        [fake_quant(w[:, i : i + 32].contiguous(), QuantConfig(bits=4, scheme=scheme, granularity="channel"))
         for i in range(0, cols, 32)],
        dim=1,
    )
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_group_equals_channel_when_group_spans_row():
    w = weight(16, 128)
    g = fake_quant(w, QuantConfig(bits=3, scheme="asym", granularity="group", group_size=128))
    c = fake_quant(w, QuantConfig(bits=3, scheme="asym", granularity="channel"))
    assert torch.equal(g, c)


def test_sym_two_bit_has_three_levels():
    qt = quantize(weight(), QuantConfig(bits=2, scheme="sym", granularity="tensor"))
    assert set(qt.q.unique().tolist()) <= {-1.0, 0.0, 1.0}


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64])
def test_dtype_preserved_and_math_in_fp32(dtype):
    w32 = weight()
    w = w32.to(dtype)
    cfg = QuantConfig(bits=4, scheme="asym", granularity="group", group_size=64)
    out = fake_quant(w, cfg)
    assert out.dtype == dtype
    # same as quantizing the fp32 upcast, then casting the result
    assert torch.equal(out, fake_quant(w.float(), cfg).to(dtype))


def test_one_dim_and_3d_tensors():
    for shape in [(300,), (4, 8, 33)]:
        w = torch.randn(*shape, generator=torch.Generator().manual_seed(1))
        for g, gs in GRANS:
            out = fake_quant(w, QuantConfig(bits=4, scheme="sym", granularity=g, group_size=gs))
            assert out.shape == w.shape


@pytest.mark.parametrize("bits", [0, 1, 9, 16, 4.0, True])
def test_invalid_bits_raise(bits):
    with pytest.raises((TypeError, ValueError)):
        QuantConfig(bits=bits)


@pytest.mark.parametrize("kwargs", [
    {"scheme": "nf4"},
    {"granularity": "row"},
    {"granularity": "group", "group_size": None},
    {"granularity": "group", "group_size": 0},
    {"granularity": "group", "group_size": -32},
])
def test_invalid_config_raises(kwargs):
    with pytest.raises(ValueError):
        QuantConfig(bits=4, **kwargs)


def test_non_finite_or_empty_input_raises():
    cfg = QuantConfig(bits=4)
    with pytest.raises(ValueError):
        fake_quant(torch.tensor([[1.0, float("nan")]]), cfg)
    with pytest.raises(ValueError):
        fake_quant(torch.tensor([[1.0, float("inf")]]), cfg)
    with pytest.raises(ValueError):
        fake_quant(torch.empty(0, 4), cfg)


def test_group_size_ignored_for_tensor_and_channel():
    assert QuantConfig(bits=4, granularity="channel", group_size=64).group_size is None


@pytest.mark.parametrize("text,expected", [
    ("tensor", ("tensor", None)),
    ("channel", ("channel", None)),
    ("group:128", ("group", 128)),
    ("g64", ("group", 64)),
    (" Group:32 ", ("group", 32)),
])
def test_parse_granularity(text, expected):
    assert parse_granularity(text) == expected


@pytest.mark.parametrize("text", ["group:", "group:x", "g", "group:0", "block:32"])
def test_parse_granularity_rejects(text):
    with pytest.raises(ValueError):
        parse_granularity(text)


def test_input_not_modified():
    w = weight()
    before = w.clone()
    fake_quant(w, QuantConfig(bits=2, scheme="asym", granularity="tensor"))
    assert torch.equal(w, before)
