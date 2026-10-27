"""
stats_paf_bench.py  (CPU)

Statistical analysis of the benchmark produced by train_paf_bench.py. Answers
the three things a reviewer asks for: how large is the spread across splits,
what are the confidence intervals, and are the differences significant.

Produces (results/paf_bench/):
  table_models.csv       one row per backbone under PAF: mean +/- sd and 95% CI
  table_aggregation.csv  single beat vs majority vote vs PAF, per backbone
  tests_aggregation.csv  McNemar, PAF against each other aggregation strategy
  tests_models.csv       McNemar and Wilcoxon, PAF-CNN against each backbone
  bootstrap_headline.csv patient-level bootstrap CI for the headline model

Usage:  python stats_paf_bench.py
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy import stats

OUT = os.environ.get("PAF_OUT", "results/paf_bench")
PRED_DIR = os.path.join(OUT, "preds")
RUNS_CSV = os.path.join(OUT, "runs.csv")
REFERENCE = "CNN (ours)"
N_BOOT = 5000
BOOT_SEED = 2026


# ---------------------------------------------------------------------------
def macro_f1(y_true, y_pred):
    f1s = []
    for c in (0, 1):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        pr = tp / (tp + fp) if tp + fp else 0.0
        se = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * pr * se / (pr + se) if pr + se else 0.0)
    return float(np.mean(f1s))


def ci95_t(values):
    """Two-sided 95% CI of the mean across splits (t distribution)."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    n = len(v)
    if n < 2:
        return (np.nan, np.nan)
    half = stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n)
    return (v.mean() - half, v.mean() + half)


def mcnemar(y_true, pred_a, pred_b):
    """Exact McNemar test on paired patient-level decisions."""
    a_ok = pred_a == y_true
    b_ok = pred_b == y_true
    n01 = int(np.sum(~a_ok & b_ok))     # only B correct
    n10 = int(np.sum(a_ok & ~b_ok))     # only A correct
    n = n01 + n10
    if n == 0:
        return n10, n01, 1.0
    p = float(stats.binomtest(n10, n, 0.5).pvalue)
    return n10, n01, p


def load_preds():
    """{(model, seed): npz-as-dict} from the saved per-patient predictions."""
    out = {}
    for path in glob.glob(os.path.join(PRED_DIR, "*.npz")):
        base = os.path.basename(path)[:-4]
        model, seed = base.rsplit("_seed", 1)
        out[(model.replace("_", " "), int(seed))] = dict(np.load(path))
    return out


def bootstrap_ci(y_true, y_pred, score, n_boot=N_BOOT):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(BOOT_SEED)
    n = len(y_true)
    acc, f1, auc = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp = y_true[idx], y_pred[idx]
        if len(np.unique(yt)) < 2:
            continue
        acc.append(np.mean(yt == yp))
        f1.append(macro_f1(yt, yp))
        auc.append(roc_auc_score(yt, score[idx]))
    q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return dict(accuracy_lo=q(acc)[0], accuracy_hi=q(acc)[1],
                macro_f1_lo=q(f1)[0], macro_f1_hi=q(f1)[1],
                auroc_lo=q(auc)[0], auroc_hi=q(auc)[1])


# ---------------------------------------------------------------------------
def main():
    df = pd.read_csv(RUNS_CSV)
    preds = load_preds()
    metrics = ["accuracy", "macro_f1", "auroc", "se_MI", "sp_MI"]

    # --- per-backbone summary under PAF ------------------------------------
    rows = []
    for model, g in df[df.aggregation == "paf"].groupby("model"):
        r = dict(model=model, n_splits=len(g))
        for m in metrics:
            lo, hi = ci95_t(g[m])
            r[m] = round(g[m].mean(), 4)
            r[m + "_sd"] = round(g[m].std(ddof=1), 4)
            r[m + "_ci"] = "[{:.3f}, {:.3f}]".format(lo, hi)
        for c in ["params_M", "train_time_s", "infer_ms_per_patient",
                  "peak_gpu_mb", "model_size_mb"]:
            r[c] = round(g[c].mean(), 3)
        rows.append(r)
    models_tbl = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    models_tbl.to_csv(os.path.join(OUT, "table_models.csv"), index=False)

    # --- aggregation ablation ----------------------------------------------
    rows = []
    for (model, agg), g in df.groupby(["model", "aggregation"]):
        r = dict(model=model, aggregation=agg)
        for m in ["accuracy", "macro_f1", "auroc"]:
            lo, hi = ci95_t(g[m])
            r[m] = round(g[m].mean(), 4)
            r[m + "_sd"] = round(g[m].std(ddof=1), 4)
            r[m + "_ci"] = "[{:.3f}, {:.3f}]".format(lo, hi)
        rows.append(r)
    agg_tbl = pd.DataFrame(rows)
    agg_tbl.to_csv(os.path.join(OUT, "table_aggregation.csv"), index=False)

    # --- McNemar: PAF against the weaker aggregation strategies -------------
    rows = []
    for (model, seed), d in sorted(preds.items()):
        for other in ("single", "vote"):
            n10, n01, p = mcnemar(d["y_true"], d["pred_paf"], d["pred_" + other])
            rows.append(dict(model=model, seed=seed, comparison="paf vs " + other,
                             paf_only_correct=n10, other_only_correct=n01, p=p))
    tests_agg = pd.DataFrame(rows)
    if len(tests_agg):
        pooled = []
        for (model, comp), g in tests_agg.groupby(["model", "comparison"]):
            n10, n01 = int(g.paf_only_correct.sum()), int(g.other_only_correct.sum())
            n = n10 + n01
            p = float(stats.binomtest(n10, n, 0.5).pvalue) if n else 1.0
            pooled.append(dict(model=model, seed="pooled", comparison=comp,
                               paf_only_correct=n10, other_only_correct=n01, p=p))
        tests_agg = pd.concat([tests_agg, pd.DataFrame(pooled)], ignore_index=True)
    tests_agg.to_csv(os.path.join(OUT, "tests_aggregation.csv"), index=False)

    # --- PAF-CNN against every other backbone -------------------------------
    rows = []
    seeds = sorted({s for (_, s) in preds})
    for model in sorted({m for (m, _) in preds}):
        if model == REFERENCE:
            continue
        n10_tot = n01_tot = 0
        for seed in seeds:
            a = preds.get((REFERENCE, seed))
            b = preds.get((model, seed))
            if a is None or b is None:
                continue
            n10, n01, _ = mcnemar(a["y_true"], a["pred_paf"], b["pred_paf"])
            n10_tot += n10
            n01_tot += n01
        n = n10_tot + n01_tot
        p_mc = float(stats.binomtest(n10_tot, n, 0.5).pvalue) if n else 1.0

        ref_f1 = df[(df.model == REFERENCE) & (df.aggregation == "paf")].sort_values("seed").macro_f1.values
        oth_f1 = df[(df.model == model) & (df.aggregation == "paf")].sort_values("seed").macro_f1.values
        k = min(len(ref_f1), len(oth_f1))
        if k >= 3:
            try:
                p_w = float(stats.wilcoxon(ref_f1[:k], oth_f1[:k]).pvalue)
            except ValueError:
                p_w = float("nan")
        else:
            p_w = float("nan")
        rows.append(dict(comparison="{} vs {}".format(REFERENCE, model),
                         ref_only_correct=n10_tot, other_only_correct=n01_tot,
                         mcnemar_p=p_mc,
                         delta_macro_f1=round(float(np.mean(ref_f1[:k]) - np.mean(oth_f1[:k])), 4),
                         wilcoxon_p=p_w, n_splits=k))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "tests_models.csv"), index=False)

    # --- patient-level bootstrap for the headline configuration -------------
    rows = []
    for seed in seeds:
        d = preds.get((REFERENCE, seed))
        if d is None:
            continue
        r = bootstrap_ci(d["y_true"], d["pred_paf"], d["score_paf"])
        r.update(model=REFERENCE, seed=seed, aggregation="paf",
                 accuracy=float(np.mean(d["y_true"] == d["pred_paf"])),
                 macro_f1=macro_f1(d["y_true"], d["pred_paf"]))
        rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "bootstrap_headline.csv"), index=False)

    print("=== backbones under PAF (mean over splits) ===")
    print(models_tbl[["model", "accuracy", "accuracy_ci", "macro_f1", "macro_f1_ci",
                      "auroc", "params_M", "infer_ms_per_patient"]].to_string(index=False))
    print("\n=== aggregation ablation ===")
    print(agg_tbl.to_string(index=False))
    print("\nwrote tables and tests to {}/".format(OUT))


if __name__ == "__main__":
    main()
