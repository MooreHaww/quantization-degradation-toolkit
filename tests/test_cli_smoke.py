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
    assert {"model", "layer", "scheme", "granularity", "bits", "mse", "rel_err"} <= set(df.columns)
    assert len(df) == 9 * 2 * 3 * 3  # layers x schemes x granularities x bits
    summary = pd.read_csv(tmp_path / "error_summary.csv")
    for _, g in summary.groupby(["scheme", "granularity"]):
        assert g.sort_values("bits")["global_rel_err"].is_monotonic_decreasing
    assert (tmp_path / "error.png").stat().st_size > 0
    assert json.loads((tmp_path / "error_meta.json").read_text())["args"]["bits"] == "2,4,8"


def test_ppl_command(offline, tmp_path):
    invoke("ppl", "--models", "tiny", "--bits", "8,2", "--granularity", "group:16", "--seq-len", "64",
           "--max-chunks", "3", "--device", "cpu", "--out-dir", tmp_path)
    df = pd.read_csv(tmp_path / "ppl.csv")
    assert list(df["precision"]) == ["fp32", "w8", "w2"]  # no fp16 row on CPU
    assert (df["n_chunks"] == 3).all() and (df["n_tokens"] == 3 * 63).all()
    assert (df["ppl"] > 0).all()
    assert (tmp_path / "ppl.png").stat().st_size > 0


def test_icl_command(offline, tmp_path):
    invoke("icl", "--models", "tiny", "--bits", "8,2", "--granularity", "group:16", "--shots", "0,2,4",
           "--seeds", "0,1", "--n-eval", "10", "--device", "cpu", "--out-dir", tmp_path, "--save-preds")
    df = pd.read_csv(tmp_path / "icl.csv")
    assert set(df["precision"]) == {"fp32", "w8", "w2"}
    assert len(df) == 3 * 3 * 2  # precisions x shots x seeds (k=0 reported under each seed)
    k0 = df[df["k"] == 0]
    assert k0.groupby("precision")["accuracy"].nunique().eq(1).all()
    summary = pd.read_csv(tmp_path / "icl_summary.csv")
    assert (summary.loc[summary["k"] == 0, "gain"] == 0).all()
    preds = pd.read_csv(tmp_path / "icl_predictions.csv")
    assert len(preds) == 3 * (1 + 2 * 2) * 10
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
    invoke("all", "--quick", "--device", "cpu", "--out-dir", tmp_path)
    for name in ("error.csv", "error.png", "ppl.csv", "ppl.png", "icl.csv", "icl.png", "icl_gain.png"):
        assert (tmp_path / name).exists(), name
