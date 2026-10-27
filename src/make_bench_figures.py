"""
make_bench_figures.py  (CPU)

Figures for the benchmark, the aggregation ablation, the external validation and
the R-peak sensitivity check. Everything is read from the CSV files the training
scripts write, so this can be re-run cheaply whenever a result changes.

Outputs (results/paper_figures_v2/):
  fig_backbones.png     macro-F1 per backbone with 95% confidence intervals
  fig_aggregation.png   single beat vs majority vote vs PAF, per backbone
  fig_external.png      internal, zero-shot, recalibrated and fine-tuned
  fig_rpeak.png         original against corrected R-peak detection
  fig_cost.png          macro-F1 against inference cost per patient

Usage:  py -3.12 make_bench_figures.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BENCH = "results/paf_bench"
BENCH_FIX = "results/paf_bench_fix"
EXTERNAL = "results/external"
OUT = "results/paper_figures_v2"
REFERENCE = "CNN (ours)"


def ci_bounds(df, metric):
    """Half-width of the 95% CI stored as a string column, per row."""
    lo, hi = [], []
    for s in df[metric + "_ci"]:
        a, b = str(s).strip("[]").split(",")
        lo.append(float(a))
        hi.append(float(b))
    return np.array(lo), np.array(hi)


def fig_backbones():
    path = os.path.join(BENCH, "table_models.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path).sort_values("macro_f1")
    lo, hi = ci_bounds(df, "macro_f1")
    y = np.arange(len(df))
    colors = ["crimson" if m == REFERENCE else "steelblue" for m in df.model]
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(df) + 2))
    ax.errorbar(df.macro_f1, y,
                xerr=[df.macro_f1 - lo, hi - df.macro_f1],
                fmt="none", ecolor="grey", capsize=4, lw=1.2)
    ax.scatter(df.macro_f1, y, c=colors, s=70, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(df.model)
    ax.set_xlabel("patient-level macro-F1 (mean of 5 splits, 95% CI)")
    ax.set_title("Backbones under Patient-Aggregated Fusion", fontweight="bold")
    ax.grid(axis="x", ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_backbones.png"), dpi=150)
    plt.close(fig)


def fig_aggregation():
    path = os.path.join(BENCH, "table_aggregation.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    labels = {"single": "single beat", "vote": "majority vote", "paf": "PAF (mean posterior)"}
    models = sorted(df.model.unique())
    width = 0.26
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(1.6 * len(models) + 3, 4.6))
    for i, (key, color) in enumerate(zip(["single", "vote", "paf"],
                                         ["#9ecae1", "#4292c6", "crimson"])):
        sub = df[df.aggregation == key].set_index("model").reindex(models)
        err = sub["macro_f1_sd"].values
        ax.bar(x + (i - 1) * width, sub["macro_f1"].values, width,
               yerr=err, capsize=3, label=labels[key], color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("patient-level macro-F1")
    ax.set_ylim(0, 1)
    ax.set_title("Effect of the aggregation rule", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_aggregation.png"), dpi=150)
    plt.close(fig)


def fig_external():
    path = os.path.join(EXTERNAL, "external_results.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    df = df[df.aggregation == "paf"]
    order = ["internal PTB-XL test", "external zero-shot",
             "external recalibrated", "external fine-tuned"]
    df = df.set_index("setting").reindex([o for o in order if o in set(df.setting)])
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar(x - 0.2, df.accuracy, 0.4, label="accuracy", color="#4292c6")
    ax.bar(x + 0.2, df.macro_f1, 0.4, label="macro-F1", color="crimson")
    for i, (a, f) in enumerate(zip(df.accuracy, df.macro_f1)):
        ax.text(i - 0.2, a + 0.01, "{:.2f}".format(a), ha="center", fontsize=8)
        ax.text(i + 0.2, f + 0.01, "{:.2f}".format(f), ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("external ", "") for s in df.index], rotation=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("patient-level score")
    ax.set_title("PTB-XL training, PTB Diagnostic testing", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_external.png"), dpi=150)
    plt.close(fig)


def fig_rpeak():
    a = os.path.join(BENCH, "runs.csv")
    b = os.path.join(BENCH_FIX, "runs.csv")
    if not (os.path.exists(a) and os.path.exists(b)):
        return
    da = pd.read_csv(a)
    db = pd.read_csv(b)
    da = da[(da.model == REFERENCE) & (da.aggregation == "paf")]
    db = db[(db.model == REFERENCE) & (db.aggregation == "paf")]
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.2))
    for i, metric in enumerate(["accuracy", "macro_f1"]):
        ax[i].boxplot([da[metric], db[metric]],
                      tick_labels=["detector at 300 Hz", "detector at 100 Hz"],
                      showfliers=False)
        for j, d in enumerate([da, db], start=1):
            ax[i].scatter(np.full(len(d), j), d[metric], color="crimson", zorder=3, s=25)
        ax[i].set_ylabel(metric.replace("_", "-"))
        ax[i].set_title(metric.replace("_", "-"))
    fig.suptitle("Sensitivity to R-peak detection settings", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(OUT, "fig_rpeak.png"), dpi=150)
    plt.close(fig)


def fig_cost():
    path = os.path.join(BENCH, "table_models.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for _, r in df.iterrows():
        col = "crimson" if r.model == REFERENCE else "steelblue"
        ax.scatter(r.infer_ms_per_patient, r.macro_f1, s=40 + 12 * r.params_M,
                   color=col, alpha=0.8)
        ax.annotate(r.model, (r.infer_ms_per_patient, r.macro_f1),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("inference time per patient (ms)")
    ax.set_ylabel("patient-level macro-F1")
    ax.set_title("Accuracy against inference cost\n(marker area grows with parameter count)",
                 fontweight="bold")
    ax.grid(ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_cost.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for fn in (fig_backbones, fig_aggregation, fig_external, fig_rpeak, fig_cost):
        try:
            fn()
            print("ok  {}".format(fn.__name__))
        except Exception as e:
            print("skip {}: {}".format(fn.__name__, e))
    print("figures in {}/".format(OUT))
