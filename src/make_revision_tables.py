"""
make_revision_tables.py  (CPU)

Rebuilds every per-class number and confusion matrix of the revised manuscript
from the saved per-patient predictions, so that all of them come from the same
runs as the headline tables (reviewer 2, points 2 and 9).

Inputs
  results/paf_bench_fix/preds/CNN_(ours)_seed{45..49}.npz   binary, PAF and vote
  results/attention_pool/preds/attention_seed{45..49}.npz   binary, attention
  results/territory/preds/{mean,attention}_seed{45..49}.npz four territories
Outputs (results/revision/)
  perclass.csv            per-class Se/Sp/Prec/F1 with support, pooled over seeds
  fig_cm_binary.png       pooled patient-level confusion matrix, binary, PAF
  fig_cm_territory.png    pooled patient-level confusion matrix, territory, PAF
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

OUT = "results/revision"
SEEDS = [45, 46, 47, 48, 49]
BIN = ["NORM", "MI"]
TERR = ["NORM", "Anterior", "Inferior", "Lat./Post."]


def pooled(files, pred_key):
    yt, yp = [], []
    for f in files:
        d = np.load(f)
        yt.append(d["y_true"])
        yp.append(d[pred_key])
    return np.concatenate(yt), np.concatenate(yp)


def per_class(cm, names, task, method):
    rows = []
    for i, n in enumerate(names):
        tp = cm[i, i]
        fn = cm[i].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        se = tp / (tp + fn) if tp + fn else 0.0
        sp = tn / (tn + fp) if tn + fp else 0.0
        pr = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * pr * se / (pr + se) if pr + se else 0.0
        rows.append(dict(task=task, method=method, cls=n, se=se, sp=sp, prec=pr,
                         f1=f1, support=int(cm[i].sum())))
    return rows


def plot_cm(cm, names, title, path):
    cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(1.4 * len(names) + 2, 1.2 * len(names) + 1.5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, "{}\n({:.0%})".format(cm[i, j], cmn[i, j]), ha="center",
                    va="center", fontsize=9,
                    color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontweight="bold")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []

    cnn = ["results/paf_bench_fix/preds/CNN_(ours)_seed{}.npz".format(s) for s in SEEDS]
    for method, key in (("PAF (mean)", "pred_paf"), ("Majority vote", "pred_vote"),
                        ("Single beat", "pred_single")):
        yt, yp = pooled(cnn, key)
        cm = confusion_matrix(yt, yp, labels=[0, 1])
        rows += per_class(cm, BIN, "binary", method)
        if key == "pred_paf":
            plot_cm(cm, BIN, "MI vs NORM, PAF (5 splits pooled)",
                    os.path.join(OUT, "fig_cm_binary.png"))
    att = ["results/attention_pool/preds/attention_seed{}.npz".format(s) for s in SEEDS]
    yt, yp = pooled(att, "pred")
    rows += per_class(confusion_matrix(yt, yp, labels=[0, 1]), BIN, "binary",
                      "Attention MIL")

    for method, tag in (("PAF (mean)", "mean"), ("Attention MIL", "attention")):
        files = ["results/territory/preds/{}_seed{}.npz".format(tag, s) for s in SEEDS]
        yt, yp = pooled(files, "pred")
        cm = confusion_matrix(yt, yp, labels=list(range(4)))
        rows += per_class(cm, TERR, "territory", method)
        if tag == "mean":
            plot_cm(cm, TERR, "Four territories, PAF (5 splits pooled)",
                    os.path.join(OUT, "fig_cm_territory.png"))

    df = pd.DataFrame(rows).round(4)
    df.to_csv(os.path.join(OUT, "perclass.csv"), index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
