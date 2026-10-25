"""PNG figures for each experiment (matplotlib, Agg backend)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Categorical slots, in fixed order (identity: models, granularities).
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
# Single-hue ordinal ramp for bit-width: fewer bits = lighter.
BITS_RAMP = ["#86b6ef", "#5598e7", "#256abf", "#184f95", "#0d366b"]
# Marker per quantized precision (fewest bits first) so lines are not told apart by shade alone.
BITS_MARKERS = ["v", "^", "D", "s", "P", "X", "*"]
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"

STYLE = {
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.edgecolor": INK_MUTED,
    "axes.labelcolor": INK,
    "axes.titlesize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "legend.frameon": False,
    "lines.linewidth": 1.6,
    "lines.markersize": 5,
}


def _precision_order(precisions: list[str]) -> list[str]:
    """fp32, fp16, then wN from most to fewest bits."""
    fp = [p for p in ("fp32", "fp16") if p in precisions]
    w = sorted((p for p in precisions if p.startswith("w")), key=lambda p: -int(p[1:]))
    return fp + w


def _precision_style(precision: str, all_w: list[str]) -> dict:
    if precision == "fp32":
        return {"color": INK, "linestyle": "-", "marker": "o"}
    if precision == "fp16":
        return {"color": INK_MUTED, "linestyle": "--", "marker": "o"}
    # ramp position by rank among the quantized precisions (fewest bits = lightest)
    ranked = sorted(all_w, key=lambda p: int(p[1:]))
    pos = ranked.index(precision)
    step = round(pos * (len(BITS_RAMP) - 1) / max(len(ranked) - 1, 1))
    return {"color": BITS_RAMP[step], "linestyle": "-", "marker": BITS_MARKERS[pos % len(BITS_MARKERS)]}


def _precision_label(p: str) -> str:
    return p.upper() if p.startswith("fp") else f"{p[1:]}-bit"


def plot_error(summary: pd.DataFrame, path: Path) -> None:
    """Global relative error vs bits; rows = models, columns = schemes, lines = granularity."""
    models = list(dict.fromkeys(summary["model"]))
    schemes = list(dict.fromkeys(summary["scheme"]))
    grans = list(dict.fromkeys(summary["granularity"]))
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(
            len(models), len(schemes), figsize=(4.2 * len(schemes), 3.2 * len(models)),
            squeeze=False, sharey=True, layout="constrained",
        )
        for r, model in enumerate(models):
            for c, scheme in enumerate(schemes):
                ax = axes[r][c]
                sub = summary[(summary["model"] == model) & (summary["scheme"] == scheme)]
                for i, gran in enumerate(grans):
                    g = sub[sub["granularity"] == gran].sort_values("bits")
                    ax.plot(g["bits"], g["global_rel_err"], marker="o",
                            color=CATEGORICAL[i % len(CATEGORICAL)], label=gran)
                ax.set_yscale("log")
                ax.set_title(f"{model} - {scheme}")
                ax.set_xticks(sorted(summary["bits"].unique()))
                ax.set_xlabel("bits")
                if c == 0:
                    ax.set_ylabel("relative error ||W - W_hat|| / ||W||")
        axes[0][-1].legend(title="granularity", loc="upper right")
        fig.suptitle("Weight reconstruction error vs bit-width (all nn.Linear weights pooled)")
        fig.savefig(path)
        plt.close(fig)


def plot_ppl(df: pd.DataFrame, path: Path) -> None:
    """Perplexity per precision (categorical x: fp32, fp16, 8 ... 2 bits), one line per model."""
    order = _precision_order(list(df["precision"].unique()))
    models = list(dict.fromkeys(df["model"]))
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 3.8), layout="constrained")
        x = range(len(order))
        for i, model in enumerate(models):
            sub = df[df["model"] == model].set_index("precision").reindex(order)
            color = CATEGORICAL[i % len(CATEGORICAL)]
            ax.plot(x, sub["ppl"], marker="o", color=color, label=model)
            last = sub["ppl"].dropna()
            if len(last):
                ax.annotate(_fmt_ppl(last.iloc[-1]), (order.index(last.index[-1]), last.iloc[-1]),
                            textcoords="offset points", xytext=(6, 0), va="center",
                            fontsize=8, color=INK_MUTED)
        ax.set_xticks(list(x), [_precision_label(p) for p in order])
        ax.set_yscale("log")
        ax.set_xlabel("weight precision")
        ax.set_ylabel("WikiText-2 perplexity (log scale)")
        gran = df.loc[df["granularity"] != "", ["scheme", "granularity"]].drop_duplicates()
        cfg = ", ".join(f"{s} {g}" for s, g in gran.itertuples(index=False))
        ax.set_title(f"Perplexity vs weight precision ({cfg})" if cfg else "Perplexity vs weight precision")
        fig.legend(title="model", loc="outside right center")
        fig.savefig(path)
        plt.close(fig)


def _fmt_ppl(v: float) -> str:
    return f"{v:,.0f}" if v >= 100 else f"{v:.1f}"


def plot_icl(summary: pd.DataFrame, path: Path) -> None:
    """Accuracy vs number of shots, one panel per model, one line per precision."""
    models = list(dict.fromkeys(summary["model"]))
    order = _precision_order(list(summary["precision"].unique()))
    all_w = [p for p in order if p.startswith("w")]
    shots = sorted(summary["k"].unique())
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.6), squeeze=False, sharey=True,
                                 layout="constrained")
        for ax, model in zip(axes[0], models):
            sub = summary[summary["model"] == model]
            for p in order:
                g = sub[sub["precision"] == p].set_index("k").reindex(shots)
                xs = range(len(shots))
                ax.errorbar(xs, g["acc_mean"], yerr=g["acc_std"], capsize=2, elinewidth=0.8,
                            label=_precision_label(p), **_precision_style(p, all_w))
            ax.axhline(0.5, color=INK_MUTED, linewidth=0.8, linestyle=":")
            ax.set_xticks(list(range(len(shots))), [str(k) for k in shots])
            ax.set_xlabel("shots (k)")
            ax.set_title(model)
        axes[0][0].set_ylabel("SST-2 accuracy (mean +/- std over demo seeds)")
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, title="weights", loc="outside right center")
        fig.suptitle("Few-shot accuracy vs shots, by weight precision (dotted = chance)")
        fig.savefig(path)
        plt.close(fig)


def plot_icl_gain(summary: pd.DataFrame, path: Path) -> None:
    """Gain from 0 to the largest k shots, per precision, one line per model."""
    k_max = summary["k"].max()
    sub = summary[summary["k"] == k_max]
    order = _precision_order(list(sub["precision"].unique()))
    models = list(dict.fromkeys(sub["model"]))
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 3.8), layout="constrained")
        x = range(len(order))
        for i, model in enumerate(models):
            g = sub[sub["model"] == model].set_index("precision").reindex(order)
            ax.errorbar(x, g["gain"], yerr=g["acc_std"], marker="o", capsize=2, elinewidth=0.8,
                        color=CATEGORICAL[i % len(CATEGORICAL)], label=model)
        ax.axhline(0.0, color=INK_MUTED, linewidth=0.8)
        ax.set_xticks(list(x), [_precision_label(p) for p in order])
        ax.set_xlabel("weight precision")
        ax.set_ylabel(f"accuracy gain, {k_max}-shot minus 0-shot")
        ax.set_title("Does lower precision shrink the gain from in-context examples?")
        fig.legend(title="model", loc="outside right center")
        fig.savefig(path)
        plt.close(fig)
