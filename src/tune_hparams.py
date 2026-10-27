"""
tune_hparams.py  (GPU / WSL)

Documents how the PAF backbone's hyperparameters were chosen, rather than
asserting them. A random search over kernel sizes, filter width, dropout,
learning rate and batch size is scored on a patient-disjoint validation split
carved out of the training patients. The test patients are never touched here,
so the search cannot leak into the reported results.

The output table lets the paper state the search space, the budget, the selected
configuration and how much the choice actually mattered.

Outputs (results/hparam_search/):
  search.csv     one row per sampled configuration, ranked by validation macro-F1
  summary.txt    search space, budget, selected configuration, spread

Usage (from WSL):  ./run_gpu.sh tune_hparams.py
"""

import os
import gc
import json
import time
import numpy as np
import pandas as pd

import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv1D, Dense, Dropout, MaxPooling1D, Flatten
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight

from train_paf_bench import patient_split, aggregate, binary_metrics, NORM_LABEL

DATA = "data_fusion"
OUT = "results/hparam_search"
CSV = os.path.join(OUT, "search.csv")
SEED = 45
N_TRIALS = 24
EPOCHS = 40
PATIENCE = 5

SPACE = {
    "k1": [7, 9, 11, 15],
    "k2": [9, 11, 13, 17],
    "k3": [3, 5, 7, 9],
    "filters": [64, 96, 128],
    "dense": [64, 128, 256],
    "dropout": [0.1, 0.2, 0.3, 0.4],
    "lr": [3e-4, 5e-4, 1e-3, 2e-3],
    "batch": [64, 128, 256],
}

# The configuration used throughout the paper, always evaluated first so the
# search table shows where it lands.
DEFAULT = dict(k1=11, k2=13, k3=7, filters=128, dense=128,
               dropout=0.2, lr=1e-3, batch=128)

for _g in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(_g, True)
    except Exception:
        pass


def build(cfg, input_shape):
    f, d = cfg["filters"], cfg["dropout"]
    return Sequential([
        Conv1D(f, cfg["k1"], activation="relu", input_shape=input_shape),
        MaxPooling1D(), Dropout(d),
        Conv1D(f, cfg["k2"], activation="relu"), MaxPooling1D(), Dropout(d),
        Conv1D(f, cfg["k3"], activation="relu"), Flatten(),
        Dense(cfg["dense"], activation="relu"), Dropout(min(d + 0.1, 0.5)),
        Dense(2, activation="softmax"),
    ])


def main():
    os.makedirs(OUT, exist_ok=True)
    X = np.load(DATA + "/all_X.npy", mmap_mode="r")
    y = (np.load(DATA + "/all_y.npy") != NORM_LABEL).astype(int)
    groups = np.load(DATA + "/all_groups.npy")

    # Same outer split as the benchmark: the test patients stay untouched.
    tr_all, _ = patient_split(y, groups, SEED, 0.15)
    tr, va = patient_split(y[tr_all], groups[tr_all], SEED, 0.15)
    tr, va = tr_all[tr], tr_all[va]
    Xtr, ytr = np.asarray(X[tr]), y[tr]
    Xva, yva, gva = np.asarray(X[va]), y[va], groups[va]
    print("train beats {} | val beats {} | val patients {}".format(
        len(tr), len(va), len(np.unique(gva))), flush=True)

    cls = np.unique(ytr)
    cw = compute_class_weight("balanced", classes=cls, y=ytr)
    class_weight = {int(c): float(w) for c, w in zip(cls, cw)}

    rng = np.random.default_rng(SEED)
    trials = [dict(DEFAULT)]
    seen = {json.dumps(DEFAULT, sort_keys=True)}
    while len(trials) < N_TRIALS:
        cfg = {k: v[rng.integers(len(v))] for k, v in SPACE.items()}
        cfg = {k: (int(v) if k != "dropout" and k != "lr" else float(v))
               for k, v in cfg.items()}
        key = json.dumps(cfg, sort_keys=True)
        if key not in seen:
            seen.add(key)
            trials.append(cfg)

    rows = []
    if os.path.exists(CSV):
        rows = pd.read_csv(CSV).to_dict("records")
        print("resuming with {} completed trials".format(len(rows)), flush=True)
    done = {json.dumps({k: (float(r[k]) if k in ("dropout", "lr") else int(r[k]))
                    for k in SPACE}, sort_keys=True) for r in rows}

    for i, cfg in enumerate(trials):
        key = json.dumps(cfg, sort_keys=True)
        if key in done:
            continue
        tf.keras.utils.set_random_seed(SEED)
        model = build(cfg, (X.shape[1], X.shape[2]))
        model.compile(loss="categorical_crossentropy",
                      optimizer=Adam(learning_rate=cfg["lr"]), metrics=["accuracy"])
        t0 = time.time()
        model.fit(Xtr, to_categorical(ytr, 2), epochs=EPOCHS,
                  batch_size=cfg["batch"], verbose=0, class_weight=class_weight,
                  validation_data=(Xva, to_categorical(yva, 2)),
                  callbacks=[EarlyStopping(monitor="val_loss", patience=PATIENCE,
                                           restore_best_weights=True)])
        proba = model.predict(Xva, batch_size=cfg["batch"], verbose=0)
        _, yt, pred, score = aggregate(yva, proba, gva, np.random.default_rng(SEED))
        met = binary_metrics(yt, pred["paf"], score["paf"])
        row = dict(cfg)
        row.update(is_default=(cfg == DEFAULT),
                   val_accuracy=round(met["accuracy"], 4),
                   val_macro_f1=round(met["macro_f1"], 4),
                   val_auroc=round(met["auroc"], 4),
                   params_M=round(model.count_params() / 1e6, 3),
                   fit_time_s=round(time.time() - t0, 1))
        rows.append(row)
        pd.DataFrame(rows).to_csv(CSV, index=False)
        print("trial {:>2}/{} | macro-F1 {:.4f} | {}".format(
            i + 1, len(trials), met["macro_f1"], cfg), flush=True)
        del model
        gc.collect()
        K.clear_session()

    df = pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False)
    df.to_csv(CSV, index=False)
    best = df.iloc[0]
    default_row = df[df.is_default].iloc[0] if df.is_default.any() else None
    # pandas hands back numpy scalars, which json cannot encode
    best_cfg = {k: (float(best[k]) if k in ("dropout", "lr") else int(best[k]))
                for k in SPACE}

    lines = [
        "Random search over the PAF backbone, validated on held-out patients.",
        "space: " + json.dumps(SPACE),
        "budget: {} configurations, up to {} epochs each, early stopping "
        "patience {}".format(len(df), EPOCHS, PATIENCE),
        "validation macro-F1: best {:.4f}, median {:.4f}, worst {:.4f} "
        "(spread {:.4f})".format(df.val_macro_f1.max(), df.val_macro_f1.median(),
                                 df.val_macro_f1.min(),
                                 df.val_macro_f1.max() - df.val_macro_f1.min()),
        "best configuration: " + json.dumps(best_cfg),
    ]
    if default_row is not None:
        rank = int((df.val_macro_f1 > default_row.val_macro_f1).sum()) + 1
        lines.append("configuration used in the paper ranks {} of {} "
                     "(macro-F1 {:.4f}, best {:.4f})".format(
                         rank, len(df), default_row.val_macro_f1, best.val_macro_f1))
    open(os.path.join(OUT, "summary.txt"), "w").write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print("\ntop 8 configurations")
    print(df.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
