"""Command-line interface: qtk error | ppl | icl | all.

List options take comma-separated values. In PowerShell, quote them
(``--bits "8,4,2"``); an unquoted 8,4,2 is turned into an array and split.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # noisy on Windows without dev mode

import torch  # noqa: E402
import typer  # noqa: E402

from qtk import exp_error, exp_icl, exp_ppl, plotting  # noqa: E402
from qtk.model import DEFAULT_REVISION  # noqa: E402
from qtk.quant import QuantConfig  # noqa: E402
from qtk.utils import (  # noqa: E402
    check_cuda_arch,
    device_description,
    parse_int_list,
    parse_str_list,
    resolve_device,
    set_seed,
    write_metadata,
)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode=None,  # plain ASCII help: legacy PowerShell consoles garble box-drawing characters
    pretty_exceptions_enable=False,
    help="Quantization degradation toolkit.",
)

DEFAULT_MODELS = "pythia-70m,pythia-160m,pythia-410m"

Device = Annotated[str, typer.Option(help="auto, cpu or cuda.")]
OutDir = Annotated[Path, typer.Option(help="Directory for CSVs, PNGs and metadata.")]
Seed = Annotated[int, typer.Option(help="Global seed (also picks the ICL eval subset).")]
Scheme = Annotated[str, typer.Option(help="sym or asym.")]
Granularity = Annotated[str, typer.Option(help="tensor, channel or group:N.")]
Fp16 = Annotated[bool, typer.Option(help="Also evaluate an FP16 baseline (CUDA only).")]
Revision = Annotated[str, typer.Option(help="Hugging Face model revision (step143000 = final Pythia checkpoint).")]
QuantizeHead = Annotated[
    bool,
    typer.Option("--quantize-head", help="Also quantize the LM head (default: it stays at full precision)."),
]


def _setup(device: str, out_dir: Path, seed: int) -> torch.device:
    set_seed(seed)
    dev = resolve_device(device)
    out_dir.mkdir(parents=True, exist_ok=True)
    typer.echo(f"device: {device_description(dev)}")
    warning = check_cuda_arch(dev)
    if warning:
        typer.echo(f"WARNING: {warning}", err=True)
    return dev


def _check_quant_args(scheme: str, granularity: str) -> None:
    try:
        QuantConfig.parse(8, scheme, granularity)
    except (TypeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc


def _fp16_flag(fp16: bool, dev: torch.device) -> bool:
    if fp16 and dev.type != "cuda":
        typer.echo("note: skipping the FP16 baseline on CPU (half-precision CPU kernels are slow and not representative)")
        return False
    return fp16


@app.command()
def error(
    models: Annotated[str, typer.Option(help="Comma-separated model names.")] = "pythia-70m,pythia-160m",
    bits: Annotated[str, typer.Option(help="Comma-separated bit-widths.")] = "2,3,4,5,6,7,8",
    schemes: Annotated[str, typer.Option(help="Comma-separated: sym, asym.")] = "sym,asym",
    granularities: Annotated[str, typer.Option(help="Comma-separated granularities.")] = "tensor,channel,group:32,group:64,group:128",
    max_layers: Annotated[int | None, typer.Option(help="Only the first N Linear layers (quick runs).")] = None,
    device: Device = "auto",
    out_dir: OutDir = Path("results"),
    seed: Seed = 0,
    revision: Revision = DEFAULT_REVISION,
    quantize_head: QuantizeHead = False,
) -> None:
    """Experiment 1: weight reconstruction error vs bit-width and granularity."""
    args = dict(locals())
    for g in parse_str_list(granularities):
        for s in parse_str_list(schemes):
            _check_quant_args(s, g)
    dev = _setup(device, out_dir, seed)
    df = exp_error.run(
        models=parse_str_list(models), bits=parse_int_list(bits), schemes=parse_str_list(schemes),
        granularities=parse_str_list(granularities), device=dev, out_dir=out_dir, max_layers=max_layers, revision=revision,
        quantize_head=quantize_head,
    )
    summary = exp_error.summarize(df)
    plotting.plot_error(summary, out_dir / "error.png")
    write_metadata(out_dir, "error", args, dev)
    typer.echo(f"wrote {out_dir / 'error.csv'}, {out_dir / 'error_summary.csv'}, {out_dir / 'error.png'}")


@app.command()
def ppl(
    models: Annotated[str, typer.Option(help="Comma-separated model names.")] = DEFAULT_MODELS,
    bits: Annotated[str, typer.Option(help="Comma-separated bit-widths.")] = "8,6,4,3,2",
    scheme: Scheme = "asym",
    granularity: Annotated[str, typer.Option(help="tensor, channel or group:N; comma-separated for several.")] = "group:128",
    seq_len: Annotated[int, typer.Option(help="Tokens per non-overlapping chunk.")] = 1024,
    max_chunks: Annotated[int | None, typer.Option(help="Only the first N chunks (quick runs).")] = None,
    batch_size: Annotated[int, typer.Option(help="Chunks per forward pass.")] = 4,
    fp16: Fp16 = True,
    device: Device = "auto",
    out_dir: OutDir = Path("results"),
    seed: Seed = 0,
    revision: Revision = DEFAULT_REVISION,
    quantize_head: QuantizeHead = False,
) -> None:
    """Experiment 2: WikiText-2 perplexity at FP32, FP16 and each bit-width."""
    args = dict(locals())
    grans = parse_str_list(granularity)
    if not grans:
        raise typer.BadParameter("--granularity must not be empty")
    for g in grans:
        _check_quant_args(scheme, g)
    dev = _setup(device, out_dir, seed)
    fp16 = _fp16_flag(fp16, dev)
    df = exp_ppl.run(
        models=parse_str_list(models), bits=parse_int_list(bits), scheme=scheme, granularity=grans,
        device=dev, out_dir=out_dir, seq_len=seq_len, max_chunks=max_chunks, batch_size=batch_size, fp16=fp16, revision=revision,
        quantize_head=quantize_head,
    )
    plotting.plot_ppl(df, out_dir / "ppl.png")
    write_metadata(out_dir, "ppl", args, dev)
    typer.echo(f"wrote {out_dir / 'ppl.csv'}, {out_dir / 'ppl.png'}")


@app.command()
def icl(
    models: Annotated[str, typer.Option(help="Comma-separated model names.")] = DEFAULT_MODELS,
    bits: Annotated[str, typer.Option(help="Comma-separated bit-widths.")] = "8,6,4,3,2",
    scheme: Scheme = "asym",
    granularity: Granularity = "group:128",
    shots: Annotated[str, typer.Option(help="Comma-separated numbers of demonstrations.")] = "0,1,4,8,16",
    seeds: Annotated[str, typer.Option(help="Comma-separated demo-sampling seeds.")] = "0,1,2",
    n_eval: Annotated[int, typer.Option(help="SST-2 validation examples to evaluate.")] = 500,
    batch_size: Annotated[int, typer.Option(help="Prompts per forward pass.")] = 8,
    save_preds: Annotated[bool, typer.Option(help="Also write per-example log-probs.")] = False,
    fp16: Fp16 = True,
    device: Device = "auto",
    out_dir: OutDir = Path("results"),
    seed: Seed = 0,
    revision: Revision = DEFAULT_REVISION,
    quantize_head: QuantizeHead = False,
) -> None:
    """Experiment 3: few-shot SST-2 accuracy vs shots for each precision."""
    args = dict(locals())
    _check_quant_args(scheme, granularity)
    shot_list, seed_list = parse_int_list(shots), parse_int_list(seeds)
    if not shot_list or min(shot_list) < 0:
        raise typer.BadParameter("--shots must be non-negative integers")
    if not seed_list:
        raise typer.BadParameter("--seeds must not be empty")
    dev = _setup(device, out_dir, seed)
    fp16 = _fp16_flag(fp16, dev)
    df = exp_icl.run(
        models=parse_str_list(models), bits=parse_int_list(bits), scheme=scheme, granularity=granularity,
        shots=shot_list, seeds=seed_list, n_eval=n_eval, device=dev, out_dir=out_dir,
        batch_size=batch_size, fp16=fp16, subset_seed=seed, save_preds=save_preds, revision=revision,
        quantize_head=quantize_head,
    )
    summary = exp_icl.summarize(df)
    plotting.plot_icl(summary, out_dir / "icl.png")
    plotting.plot_icl_gain(summary, out_dir / "icl_gain.png", n_eval=int(df["n_eval"].iloc[0]))
    write_metadata(out_dir, "icl", args, dev)
    typer.echo(f"wrote {out_dir / 'icl.csv'}, {out_dir / 'icl_summary.csv'}, {out_dir / 'icl.png'}, {out_dir / 'icl_gain.png'}")


@app.command(name="all")
def run_all(
    quick: Annotated[bool, typer.Option(help="Tiny smoke-test version of every experiment (pythia-70m only).")] = False,
    device: Device = "auto",
    out_dir: Annotated[Path | None, typer.Option(help="Defaults to results/ (results/quick with --quick).")] = None,
    seed: Seed = 0,
    revision: Revision = DEFAULT_REVISION,
    quantize_head: QuantizeHead = False,
) -> None:
    """Run all three experiments with their defaults (or a quick smoke test)."""
    if out_dir is None:
        out_dir = Path("results/quick") if quick else Path("results")
    if quick:
        error(models="pythia-70m", bits="2,4,8", max_layers=4, device=device, out_dir=out_dir, seed=seed, revision=revision,
              schemes="sym,asym", granularities="tensor,channel,group:128", quantize_head=quantize_head)
        ppl(models="pythia-70m", bits="8,4,2", max_chunks=4, device=device, out_dir=out_dir, seed=seed, revision=revision,
            scheme="asym", granularity="group:128", seq_len=1024, batch_size=4, fp16=True, quantize_head=quantize_head)
        icl(models="pythia-70m", bits="8,2", shots="0,4", seeds="0", n_eval=32, device=device, out_dir=out_dir,
            seed=seed, revision=revision, scheme="asym", granularity="group:128", batch_size=8, save_preds=False, fp16=True,
            quantize_head=quantize_head)
    else:
        error(models="pythia-70m,pythia-160m", bits="2,3,4,5,6,7,8", max_layers=None, device=device,
              out_dir=out_dir, seed=seed, revision=revision, schemes="sym,asym", granularities="tensor,channel,group:32,group:64,group:128",
              quantize_head=quantize_head)
        ppl(models=DEFAULT_MODELS, bits="8,6,4,3,2", max_chunks=None, device=device, out_dir=out_dir, seed=seed, revision=revision,
            scheme="asym", granularity="group:128", seq_len=1024, batch_size=4, fp16=True, quantize_head=quantize_head)
        icl(models=DEFAULT_MODELS, bits="8,6,4,3,2", shots="0,1,4,8,16", seeds="0,1,2", n_eval=500,
            device=device, out_dir=out_dir, seed=seed, revision=revision, scheme="asym", granularity="group:128",
            batch_size=8, save_preds=False, fp16=True, quantize_head=quantize_head)


if __name__ == "__main__":
    app()
