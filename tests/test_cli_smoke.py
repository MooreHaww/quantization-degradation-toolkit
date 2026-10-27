"""End-to-end CLI runs on the tiny offline model: every command writes its CSV, PNG and metadata."""

import json

import pandas as pd
from typer.testing import CliRunner

from qtk.cli import app

runner = CliRunner()


def invoke(*args):
    result = runner.invoke(app, [str(a) for a in args])
    assert result.exit_code == 0, result.output + repr(result.exception)
    return result


def test_error_command(offline, tmp_path):
    invoke("error", "--models", "tiny", "--bits", "2,4,8", "--granularities", "tensor,channel,group:16",
           "--device", "cpu", "--out-dir", tmp_path)
    df = pd.read_csv(tmp_path / "error.csv")
    assert {"model", "head", "layer", "scheme", "granularity", "bits", "mse", "rel_err"} <= set(df.columns)
    assert len(df) == 8 * 2 * 3 * 3  # layers (LM head excluded) x schemes x granularities x bits
    assert (df["head"] == "fp32").all() and "lm_head" not in set(df["layer"])
    summary = pd.read_csv(tmp_path / "error_summary.csv")
    assert (summary["head"] == "fp32").all()
    for _, g in summary.groupby(["scheme", "granularity"]):
        assert g.sort_values("bits")["global_rel_err"].is_monotonic_decreasing
    assert (tmp_path / "error.png").stat().st_size > 0
    meta = json.loads((tmp_path / "error_meta.json").read_text())
    assert meta["args"]["bits"] == "2,4,8" and meta["args"]["quantize_head"] is False


def test_error_command_quantize_head(offline, tmp_path):
    invoke("error", "--models", "tiny", "--bits", "4", "--schemes", "asym", "--granularities", "channel",
           "--device", "cpu", "--out-dir", tmp_path, "--quantize-head")
    df = pd.read_csv(tmp_path / "error.csv")
    assert len(df) == 9 and "lm_head" in set(df["layer"]) and (df["head"] == "quantized").all()


def test_ppl_command(offline, tmp_path):
    invoke("ppl", "--models", "tiny", "--bits", "8,2", "--granularity", "group:16", "--seq-len", "64",
           "--max-chunks", "3", "--device", "cpu", "--out-dir", tmp_path)
    df = pd.read_csv(tmp_path / "ppl.csv")
    assert list(df["precision"]) == ["fp32", "w8", "w2"]  # no fp16 row on CPU
    assert (df["head"] == "fp32").all()
    assert (df["n_chunks"] == 3).all() and (df["n_tokens"] == 3 * 63).all()
    assert (df["ppl"] > 0).all()
    assert df["delta_nll"].iloc[0] == 0
    assert (df["delta_nll"] == df["nll"] - df["nll"].iloc[0]).all()
    assert (tmp_path / "ppl.png").stat().st_size > 0


def test_ppl_command_several_granularities_and_quantized_head(offline, tmp_path):
    common = ["--models", "tiny", "--bits", "8,4", "--granularity", "channel,group:16", "--seq-len", "64",
              "--max-chunks", "2", "--device", "cpu"]
    invoke("ppl", *common, "--out-dir", tmp_path / "body")
    invoke("ppl", *common, "--out-dir", tmp_path / "head", "--quantize-head")
    body = pd.read_csv(tmp_path / "body" / "ppl.csv")
    head = pd.read_csv(tmp_path / "head" / "ppl.csv")
    assert list(body["granularity"].fillna("")) == ["", "channel", "channel", "g16", "g16"]
    assert list(head["head"]) == ["fp32"] + ["quantized"] * 4  # the fp32 baseline never quantizes the head
    assert body["nll"].iloc[0] == head["nll"].iloc[0]
    assert (body["nll"].iloc[1:].to_numpy() != head["nll"].iloc[1:].to_numpy()).all()
    assert (tmp_path / "head" / "ppl.png").stat().st_size > 0


def test_icl_command(offline, tmp_path):
    invoke("icl", "--models", "tiny", "--bits", "8,2", "--granularity", "group:16", "--shots", "0,2,4",
           "--seeds", "0,1", "--n-eval", "10", "--device", "cpu", "--out-dir", tmp_path, "--save-preds")
    df = pd.read_csv(tmp_path / "icl.csv")
    assert set(df["precision"]) == {"fp32", "w8", "w2"}
    assert (df["head"] == "fp32").all()
    assert len(df) == 3 * 3 * 2  # precisions x shots x seeds (k=0 reported under each seed)
    k0 = df[df["k"] == 0]
    assert k0.groupby("precision")["accuracy"].nunique().eq(1).all()
    summary = pd.read_csv(tmp_path / "icl_summary.csv")
    assert (summary.loc[summary["k"] == 0, "gain"] == 0).all()
    assert "head" in summary.columns
    preds = pd.read_csv(tmp_path / "icl_predictions.csv")
    assert len(preds) == 3 * (1 + 2 * 2) * 10 and "head" in preds.columns
    for name in ("icl.png", "icl_gain.png"):
        assert (tmp_path / name).stat().st_size > 0


def test_runs_are_reproducible(offline, tmp_path):
    for out in ("a", "b"):
        invoke("ppl", "--models", "tiny", "--bits", "4", "--granularity", "group:16", "--seq-len", "64",
               "--max-chunks", "2", "--device", "cpu", "--out-dir", tmp_path / out)
        invoke("icl", "--models", "tiny", "--bits", "4", "--granularity", "group:16", "--shots", "0,2",
               "--seeds", "0", "--n-eval", "8", "--device", "cpu", "--out-dir", tmp_path / out)
    for name in ("ppl.csv", "icl.csv"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


def test_bad_arguments_fail_cleanly(offline, tmp_path):
    result = runner.invoke(app, ["ppl", "--granularity", "block:3", "--out-dir", str(tmp_path)])
    assert result.exit_code != 0
    result = runner.invoke(app, ["ppl", "--scheme", "nf4", "--out-dir", str(tmp_path)])
    assert result.exit_code != 0


def test_all_quick(offline, tmp_path, monkeypatch):
    from conftest import make_model
    from qtk import model

    # quick mode uses 1024-token chunks, longer than the default tiny model's context
    monkeypatch.setattr(
        model, "load_model",
        lambda name, device, dtype=None, revision=None: make_model(max_positions=1024).to(device),
    )
    for flags, head in (([], "fp32"), (["--quantize-head"], "quantized")):
        out = tmp_path / head
        invoke("all", "--quick", "--device", "cpu", "--out-dir", out, *flags)
        for name in ("error.csv", "error.png", "ppl.csv", "ppl.png", "icl.csv", "icl.png", "icl_gain.png"):
            assert (out / name).exists(), name
        icl = pd.read_csv(out / "icl.csv")
        assert set(icl.loc[icl["precision"] != "fp32", "head"]) == {head}
