"""
beats_scaling.py  (GPU / WSL)

Measures how the benefit of Patient-Aggregated Fusion depends on the number of
beats available per patient, which is the mechanism the method rests on.

Comparing two databases that happen to differ in recording length confounds the
beat count with everything else that differs between them. This script removes
the confound: a single trained model scores every beat of every test patient
once, and the aggregation is then recomputed while restricting each patient to
$k$ randomly drawn beats, for a range of $k$. Nothing else changes, so the curve
that results isolates the effect of beat count alone.

Both the internal PTB-XL test patients and the external PTB Diagnostic patients
are measured, with repeated draws at each $k$ to average out which beats happen
to be selected.

Outputs (results/beats_scaling/):
  scaling.csv        accuracy and macro-F1 per k, per dataset, per rule
  fig_scaling.png    the gain of aggregation over single-beat inference vs k

Usage (from WSL):  ./run_gpu.sh beats_scaling.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.models import load_model

from train_paf_bench import patient_split, binary_metrics, NORM_LABEL

PTBXL = os.environ.get("PAF_DATA", "data_fusion_fix")
PTBDB = "data_ptbdb"
MODEL = os.environ.get("PAF_MODEL", "results/external_fix/model_paf.keras")
OUT = "results/beats_scaling"
SEED = 45
BATCH = 256
K_VALUES = [1, 2, 3, 5, 8, 13, 20, 35, 60, 100, 200]
N_REPEATS = 20

for _g in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(_g, True)
    except Exception:
        pass


def score_beats(model, X, idx, batch=BATCH):
    """Per-beat posteriors for the given rows of a memory-mapped array."""
    out = []
    for i in range(0, len(idx), batch):
        xb = np.asarray(X[np.sort(idx[i:i + batch])], dtype=np.float32)
        out.append(model.predict(xb, verbose=0))
    return np.concatenate(out, axis=0)


def aggregate_k(proba, y_beat, groups, k, rng):
    """Patient-level decisions using at most k beats per patient."""
    patients = np.unique(groups)
    y_true = np.empty(len(patients), dtype=int)
    pred = {r: np.empty(len(patients), dtype=int) for r in ("single", "vote", "paf")}
    score = {r: np.empty(len(patients), dtype=float) for r in ("single", "vote", "paf")}
    hard = proba.argmax(axis=1)
    for i, p in enumerate(patients):
        idx = np.where(groups == p)[0]
        y_true[i] = np.bincount(y_beat[idx], minlength=2).argmax()
        take = idx if len(idx) <= k else rng.choice(idx, k, replace=False)

        j = take[rng.integers(len(take))]
        pred["single"][i] = hard[j]
        score["single"][i] = proba[j, 1]

        counts = np.bincount(hard[take], minlength=2)
        pred["vote"][i] = counts.argmax()
        score["vote"][i] = counts[1] / counts.sum()

        mean_p = proba[take].mean(axis=0)
        pred["paf"][i] = mean_p.argmax()
        score["paf"][i] = mean_p[1]
    return y_true, pred, score


def curve(name, proba, y_beat, groups, rows):
    per_patient = np.bincount(
        np.unique(groups, return_inverse=True)[1]).astype(float)
    print("{}: {} patients, {:.1f} beats per patient".format(
        name, len(per_patient), per_patient.mean()), flush=True)
    for k in K_VALUES:
        if k > per_patient.max():
            continue
        acc = {r: [] for r in ("single", "vote", "paf")}
        f1 = {r: [] for r in ("single", "vote", "paf")}
        for rep in range(N_REPEATS):
            rng = np.random.default_rng(SEED + rep)
            y_true, pred, score = aggregate_k(proba, y_beat, groups, k, rng)
            for r in acc:
                m = binary_metrics(y_true, pred[r], score[r])
                acc[r].append(m["accuracy"])
                f1[r].append(m["macro_f1"])
        for r in acc:
            rows.append(dict(dataset=name, k=k, rule=r,
                             beats_used=float(np.minimum(per_patient, k).mean()),
                             accuracy=float(np.mean(acc[r])),
                             accuracy_sd=float(np.std(acc[r], ddof=1)),
                             macro_f1=float(np.mean(f1[r])),
                             macro_f1_sd=float(np.std(f1[r], ddof=1))))
        print("  k={:>3} | single {:.4f} | vote {:.4f} | paf {:.4f}".format(
            k, np.mean(acc["single"]), np.mean(acc["vote"]), np.mean(acc["paf"])),
            flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    model = load_model(MODEL)
    rows = []

    X = np.load(PTBXL + "/all_X.npy", mmap_mode="r")
    y = (np.load(PTBXL + "/all_y.npy") != NORM_LABEL).astype(int)
    g = np.load(PTBXL + "/all_groups.npy")
    _, te = patient_split(y, g, SEED, 0.15)
    te = np.sort(te)
    curve("PTB-XL (internal)", score_beats(model, X, te), y[te], g[te], rows)

    Xe = np.load(PTBDB + "/all_X.npy", mmap_mode="r")
    ye = np.load(PTBDB + "/all_y.npy")
    ge = np.load(PTBDB + "/all_groups.npy")
    idx = np.arange(len(ye))
    curve("PTB Diagnostic (external)", score_beats(model, Xe, idx), ye, ge, rows)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "scaling.csv"), index=False)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for a, ds in zip(ax, df.dataset.unique()):
        d = df[df.dataset == ds]
        base = d[d.rule == "single"].set_index("k")
        for rule, colour, label in [("paf", "crimson", "PAF (mean posterior)"),
                                    ("vote", "steelblue", "majority vote")]:
            r = d[d.rule == rule].set_index("k")
            gain = 100 * (r.macro_f1 - base.macro_f1)
            a.plot(gain.index, gain.values, "o-", color=colour, label=label)
        a.axhline(0, ls="--", c="grey", lw=1)
        a.set_xscale("log")
        a.set_xlabel("beats per patient used")
        a.set_title(ds)
        a.grid(ls=":", alpha=0.5)
    ax[0].set_ylabel("macro-F1 gain over single beat (points)")
    ax[0].legend()
    fig.suptitle("Does aggregation help more when more beats are available?",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(OUT, "fig_scaling.png"), dpi=150)
    plt.close(fig)
    print("\nwrote {}/scaling.csv and fig_scaling.png".format(OUT))


if __name__ == "__main__":
    main()
