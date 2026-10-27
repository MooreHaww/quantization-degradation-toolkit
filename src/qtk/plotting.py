"""PNG figures for each experiment (matplotlib, Agg backend)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Categorical slots, in fixed order (identity: models, granularities).
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
# Single-hue ordinal ramp for bit-width: fewer bits = lighter.
BITS_RAMP = ["#86b6ef", "#5598e7", "#256abf", "#184f95", "#0d366b"]
# Marker per quantized precision (fewest bits first) so lines are not told apart by shade alone.
BITS_MARKERS = ["v", "^", "D", "s", "P", "X", "*"]
# Granularity identity: fixed slot per granularity, so a colour means the same thing in every figure.
GRAN_ORDER = ["tensor", "channel", "g32", "g64", "g128"]
GRAN_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
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


def _gran_style(gran: str, grans: list[str]) -> dict:
    """Colour and marker for a granularity: GRAN_ORDER slots first, unknown ones after."""
    order = GRAN_ORDER + [g for g in grans if g not in GRAN_ORDER]
    i = order.index(gran)
    return {"color": CATEGORICAL[i % len(CATEGORICAL)], "marker": GRAN_MARKERS[i % len(GRAN_MARKERS)]}


def _head_note(df: pd.DataFrame) -> str:
    """Which LM-head variant a results table shows, for figure titles."""
    if "head" not in df.columns:
        return ""
    return "LM head quantized" if (df["head"] == "quantized").any() else "LM head fp32 (not quantized)"


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
                for gran in grans:
                    g = sub[sub["granularity"] == gran].sort_values("bits")
                    ax.plot(g["bits"], g["global_rel_err"], label=gran, **_gran_style(gran, grans))
                ax.set_yscale("log")
                ax.set_title(f"{model} - {scheme}")
                ax.set_xticks(sorted(summary["bits"].unique()))
                ax.set_xlabel("bits")
                if c == 0:
                    ax.set_ylabel("relative error ||W - W_hat|| / ||W||")
        axes[0][-1].legend(title="granularity", loc="upper right")
        fig.suptitle(f"Weight reconstruction error vs bit-width (quantized nn.Linear weights pooled; {_head_note(summary)})")
        fig.savefig(path)
        plt.close(fig)


def plot_ppl(df: pd.DataFrame, path: Path) -> None:
    """Two panels per model: Delta NLL vs bits (log y) over perplexity vs bits (log y).

    One line per granularity. FP32 and FP16 are horizontal reference lines (FP16
    only in the perplexity panel and, if positive, the Delta NLL panel). The
    dashed grey line in the Delta NLL panel is c * 4^-b with c set so it passes
    through the median Delta NLL at the middle plotted bit-width: a slope guide,
    not a fit. Points with Delta NLL <= 0 or infinite perplexity cannot be drawn
    on a log axis and are left out (noted under the figure).
    """
    models = list(dict.fromkeys(df["model"]))
    q = df[df["precision"].str.startswith("w")]
    grans = list(dict.fromkeys(q["granularity"]))
    bits = sorted(q["bits"].unique())
    omitted = 0
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, len(models), figsize=(4.0 * len(models), 6.4), squeeze=False,
                                 sharex=True, layout="constrained")
        for c, model in enumerate(models):
            ax_d, ax_p = axes[0][c], axes[1][c]
            sub = df[df["model"] == model]
            sq = sub[sub["precision"].str.startswith("w")]
            drawn = {ax_d: 0, ax_p: 0}  # points actually plotted, so empty panels stay linear
            for gran in grans:
                g = sq[sq["granularity"] == gran].sort_values("bits")
                style = _gran_style(gran, grans)
                d = g[(g["delta_nll"] > 0) & np.isfinite(g["delta_nll"])]
                p = g[np.isfinite(g["ppl"])]
                omitted += (len(g) - len(d)) + (len(g) - len(p))
                drawn[ax_d] += len(d)
                drawn[ax_p] += len(p)
                ax_d.plot(d["bits"], d["delta_nll"], label=gran, **style)
                ax_p.plot(p["bits"], p["ppl"], label=gran, **style)
            # 4^-b slope guide, anchored at the median Delta NLL of the middle bit-width
            mid = bits[len(bits) // 2]
            at_mid = sq.loc[(sq["bits"] == mid) & (sq["delta_nll"] > 0), "delta_nll"]
            if len(bits) > 1 and len(at_mid):
                b = np.linspace(min(bits), max(bits), 50)
                ax_d.plot(b, float(at_mid.median()) * 4.0 ** (mid - b), color=INK_MUTED, linestyle="--",
                          linewidth=1.0, label="4^-b guide (not a fit)")
            for prec, ls in (("fp32", "-"), ("fp16", ":")):  # dashed is reserved for the 4^-b guide
                row = sub[sub["precision"] == prec]
                if len(row):
                    ax_p.axhline(row["ppl"].iloc[0], color=INK if prec == "fp32" else INK_MUTED, linestyle=ls,
                                 linewidth=0.9, label=f"{prec.upper()}" + (" (fp32 logits)" if prec == "fp16" else ""))
                    if prec == "fp16" and row["delta_nll"].iloc[0] > 0:
                        ax_d.axhline(row["delta_nll"].iloc[0], color=INK_MUTED, linestyle=":", linewidth=0.9,
                                     label="FP16 (fp32 logits)")
            for ax in (ax_d, ax_p):
                if drawn[ax]:
                    ax.set_yscale("log")
                ax.set_xticks(bits)
            if not drawn[ax_d]:
                ax_d.text(0.5, 0.5, "no positive Delta NLL", transform=ax_d.transAxes, ha="center",
                          color=INK_MUTED)
            ax_d.set_title(model)
            ax_p.set_xlabel("weight bits")
            if c == 0:
                ax_d.set_ylabel("Delta NLL vs FP32 (nats/token, log)")
                ax_p.set_ylabel("WikiText-2 perplexity (log)")
        axes[0][0].invert_xaxis()  # x is shared: most bits on the left, as precision drops to the right
        handles = {}
        for ax in axes.flat:
            for h, lab in zip(*ax.get_legend_handles_labels()):
                handles.setdefault(lab, h)
        fig.legend(list(handles.values()), list(handles), title="granularity / reference", loc="outside right center")
        scheme = ", ".join(q["scheme"].unique())
        fig.suptitle(f"WikiText-2: Delta NLL and perplexity vs weight bits ({scheme}; {_head_note(df)})")
        if omitted:
            fig.text(0.01, -0.01, "Points with Delta NLL <= 0 or infinite perplexity are not shown on the log axes.",
                     fontsize=7, color=INK_MUTED, ha="left", va="top")
        fig.savefig(path)
        plt.close(fig)


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
        fig.suptitle(f"Few-shot accuracy vs shots, by weight precision (dotted = chance; {_head_note(summary)})")
        fig.savefig(path)
        plt.close(fig)


def at_chance_models(summary: pd.DataFrame, n_eval: int) -> list[str]:
    """Models whose FP32 accuracy at the largest k is within 2 binomial standard errors of 0.5."""
    k_max = summary["k"].max()
    fp32 = summary[(summary["k"] == k_max) & (summary["precision"] == "fp32")]
    tol = 2 * (0.25 / n_eval) ** 0.5
    return [m for m, acc in zip(fp32["model"], fp32["acc_mean"]) if abs(acc - 0.5) < tol]


def plot_icl_gain(summary: pd.DataFrame, path: Path, n_eval: int) -> None:
    """Gain from 0 to the largest k shots, per precision, one line per model.

    Models at chance at FP32 (see ``at_chance_models``) are drawn as thin faded
    lines with a note, since their gain is not interpretable. Error bars are the
    std of the k_max accuracy over demo seeds only.
    """
    k_max = summary["k"].max()
    sub = summary[summary["k"] == k_max]
    order = _precision_order(list(sub["precision"].unique()))
    models = list(dict.fromkeys(sub["model"]))
    faded = at_chance_models(summary, n_eval)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 3.8), layout="constrained")
        x = range(len(order))
        for i, model in enumerate(models):
            g = sub[sub["model"] == model].set_index("precision").reindex(order)
            weak = model in faded
            ax.errorbar(x, g["gain"], yerr=g["acc_std"], marker="o", capsize=2, elinewidth=0.8,
                        color=CATEGORICAL[i % len(CATEGORICAL)], label=model + (" (at chance)" if weak else ""),
                        linewidth=0.8 if weak else STYLE["lines.linewidth"], alpha=0.35 if weak else 1.0,
                        markersize=3 if weak else STYLE["lines.markersize"])
        ax.axhline(0.0, color=INK_MUTED, linewidth=0.8)
        ax.set_xticks(list(x), [_precision_label(p) for p in order])
        ax.set_xlabel("weight precision")
        ax.set_ylabel(f"accuracy gain, {k_max}-shot minus 0-shot")
        ax.set_title(f"Does lower precision shrink the gain from in-context examples? ({_head_note(summary)})")
        notes = ["Error bars: std over demo seeds only."]
        if faded:
            notes.insert(0, f"{', '.join(faded)}: at chance at FP32 ({k_max}-shot), so their gain is not interpretable.")
        fig.text(0.01, -0.01, "\n".join(notes), fontsize=7, color=INK_MUTED, ha="left", va="top")
        fig.legend(title="model", loc="outside right center")
        fig.savefig(path)
        plt.close(fig)
