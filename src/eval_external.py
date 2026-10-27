"""
eval_external.py  (GPU / WSL)

External validation: train Patient-Aggregated Fusion on PTB-XL and test it on
the PTB Diagnostic ECG Database without retraining. The two databases come from
different hospitals, different equipment and different decades, so this is the
test of generalisation that a single-database study cannot provide.

Three settings are reported, all patient-level:
  zero-shot     the PTB-XL model applied directly to PTB Diagnostic
  recalibrated  the same model with its decision threshold refitted on a small
                calibration subset of PTB patients, testing whether the drop is
                a calibration shift rather than a representation failure
  fine-tuned    the same model with a short fine-tuning pass on the calibration
                subset, testing how much target data is needed

The calibration subset is patient-disjoint from the external test patients, and
the internal PTB-XL test result is reported alongside for reference.

Outputs (results/external/):
  model_paf.keras        the PTB-XL trained model
  external_results.csv   metrics for every setting and aggregation strategy
  external_summary.txt   readable summary

Usage (from WSL):  ./run_gpu.sh eval_external.py
"""

import os
import numpy as np
import pandas as pd

import tensorflow as tf
from keras.models import load_model
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight

from models_zoo import build_cnn_ours
from train_paf_bench import (patient_split, aggregate, binary_metrics,
                             make_dataset, NORM_LABEL)

PTBXL = os.environ.get("PAF_DATA", "data_fusion")
PTBDB = "data_ptbdb"
OUT = os.environ.get("EXT_OUT", "results/external")
MODEL_PATH = os.path.join(OUT, "model_paf.keras")
SEED = 45
EPOCHS = 50
PATIENCE = 6
BATCH = 128
CALIB_FRACTION = 0.30      # share of external patients used for calibration
FT_EPOCHS = 10
FT_LR = 1e-4

for _g in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(_g, True)
    except Exception:
        pass


def evaluate(model, X, y, groups, rng, batch=BATCH, threshold=None):
    """Patient-level metrics for all three aggregation strategies."""
    proba = model.predict(X, batch_size=batch, verbose=0)
    if threshold is not None:
        shifted = np.zeros_like(proba)
        shifted[:, 1] = (proba[:, 1] >= threshold).astype(float)
        shifted[:, 0] = 1.0 - shifted[:, 1]
        hard = shifted
    else:
        hard = proba
    _, y_true, pred, score = aggregate(y, hard, groups, rng)
    if threshold is not None:
        # keep the continuous score for AUROC, only the decision uses the threshold
        _, _, _, cont = aggregate(y, proba, groups, np.random.default_rng(SEED))
        score = cont
    out = {}
    for strategy in ("single", "vote", "paf"):
        out[strategy] = binary_metrics(y_true, pred[strategy], score[strategy])
    return out, y_true, pred, score


def best_threshold(y_true, score):
    """Threshold on the patient-level MI score that maximises macro-F1."""
    from train_paf_bench import binary_metrics as bm
    best, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        pred = (score >= t).astype(int)
        f1 = bm(y_true, pred, score)["macro_f1"]
        if f1 > best_f1:
            best, best_f1 = t, f1
    return float(best), float(best_f1)


def rows_from(setting, metrics, n_patients):
    out = []
    for strategy, m in metrics.items():
        r = dict(setting=setting, aggregation=strategy, n_patients=n_patients)
        r.update({k: round(v, 4) for k, v in m.items()})
        out.append(r)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # ---- PTB-XL: same split as the benchmark ---------------------------
    X = np.load(PTBXL + "/all_X.npy", mmap_mode="r")
    y = (np.load(PTBXL + "/all_y.npy") != NORM_LABEL).astype(int)
    g = np.load(PTBXL + "/all_groups.npy")
    tr_all, te = patient_split(y, g, SEED, 0.15)
    tr, va = patient_split(y[tr_all], g[tr_all], SEED, 0.15)
    tr, va = tr_all[tr], tr_all[va]

    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        print("loaded {}".format(MODEL_PATH), flush=True)
    else:
        tf.keras.utils.set_random_seed(SEED)
        cls = np.unique(y[tr])
        cw = compute_class_weight("balanced", classes=cls, y=y[tr])
        weights = np.zeros(len(y), dtype=np.float32)
        for c, w in zip(cls, cw):
            weights[y == c] = w
        model = build_cnn_ours((X.shape[1], X.shape[2]), 2)
        model.compile(loss="categorical_crossentropy",
                      optimizer=Adam(learning_rate=1e-3), metrics=["accuracy"])
        model.fit(make_dataset(X, tr, y, weights, BATCH, shuffle=True, seed=SEED),
                  epochs=EPOCHS,
                  validation_data=make_dataset(X, va, y, weights, BATCH), verbose=2,
                  callbacks=[EarlyStopping(monitor="val_loss", patience=PATIENCE,
                                           restore_best_weights=True)])
        model.save(MODEL_PATH)

    rows = []
    internal, _, _, _ = evaluate(model, np.asarray(X[te]), y[te], g[te], rng)
    rows += rows_from("internal PTB-XL test", internal, len(np.unique(g[te])))
    print("internal PAF: acc {:.4f} macro-F1 {:.4f}".format(
        internal["paf"]["accuracy"], internal["paf"]["macro_f1"]), flush=True)

    # ---- PTB Diagnostic --------------------------------------------------
    Xe = np.load(PTBDB + "/all_X.npy", mmap_mode="r")
    ye = np.load(PTBDB + "/all_y.npy")
    ge = np.load(PTBDB + "/all_groups.npy")
    print("external beats {} | patients {} | MI beats {:.1%}".format(
        len(ye), len(np.unique(ge)), ye.mean()), flush=True)

    calib_idx, test_idx = next(GroupShuffleSplit(
        n_splits=1, test_size=1 - CALIB_FRACTION,
        random_state=SEED).split(Xe, ye, ge))
    assert not (set(ge[calib_idx]) & set(ge[test_idx])), "patient leak"
    Xc, yc, gc_ = np.asarray(Xe[calib_idx]), ye[calib_idx], ge[calib_idx]
    Xt, yt_, gt = np.asarray(Xe[test_idx]), ye[test_idx], ge[test_idx]
    n_ext = len(np.unique(gt))

    zero, _, _, _ = evaluate(model, Xt, yt_, gt, rng)
    rows += rows_from("external zero-shot", zero, n_ext)
    print("external zero-shot PAF: acc {:.4f} macro-F1 {:.4f}".format(
        zero["paf"]["accuracy"], zero["paf"]["macro_f1"]), flush=True)

    # threshold refitted on the calibration patients only
    _, yc_true, _, sc = evaluate(model, Xc, yc, gc_, rng)
    thr, f1c = best_threshold(yc_true, sc["paf"])
    print("calibration threshold {:.2f} (calibration macro-F1 {:.4f})".format(thr, f1c),
          flush=True)
    recal, _, _, _ = evaluate(model, Xt, yt_, gt, rng, threshold=thr)
    rows += rows_from("external recalibrated", recal, n_ext)
    print("external recalibrated PAF: acc {:.4f} macro-F1 {:.4f}".format(
        recal["paf"]["accuracy"], recal["paf"]["macro_f1"]), flush=True)

    # short fine-tuning pass on the calibration patients
    ft = load_model(MODEL_PATH)
    cls = np.unique(yc)
    cw = compute_class_weight("balanced", classes=cls, y=yc)
    ft.compile(loss="categorical_crossentropy",
               optimizer=Adam(learning_rate=FT_LR), metrics=["accuracy"])
    ft.fit(Xc, to_categorical(yc, 2), epochs=FT_EPOCHS, batch_size=BATCH, verbose=2,
           class_weight={int(c): float(w) for c, w in zip(cls, cw)})
    tuned, _, _, _ = evaluate(ft, Xt, yt_, gt, rng)
    rows += rows_from("external fine-tuned", tuned, n_ext)
    print("external fine-tuned PAF: acc {:.4f} macro-F1 {:.4f}".format(
        tuned["paf"]["accuracy"], tuned["paf"]["macro_f1"]), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "external_results.csv"), index=False)
    view = df[df.aggregation == "paf"][["setting", "n_patients", "accuracy",
                                        "macro_f1", "se_MI", "sp_MI", "auroc"]]
    open(os.path.join(OUT, "external_summary.txt"), "w").write(
        "calibration threshold: {:.2f}\n\n".format(thr) + view.to_string(index=False) + "\n")
    print("\n=== PAF, patient level ===")
    print(view.to_string(index=False))
    print("\nfull table written to {}/external_results.csv".format(OUT))


if __name__ == "__main__":
    main()
