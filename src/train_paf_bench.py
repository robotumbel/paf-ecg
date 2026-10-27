"""
train_paf_bench.py  (GPU / WSL)

Unified benchmark for Patient-Aggregated Fusion (PAF) on the 12-lead PTB-XL
fusion data, MI vs NORM, strictly patient-independent.

What it produces, in one pass:
  * PAF against published high-performing backbones (InceptionTime, xresnet1d50)
    and the usual baselines (CNN-LSTM, ResNet1D, Transformer), every model
    trained and evaluated under an identical protocol, so the comparison is
    like-for-like rather than a table of numbers copied from other papers.
  * The aggregation ablation: single beat vs majority vote vs mean probability
    (PAF), for every backbone.
  * Repeated runs over N_SEEDS independent patient-disjoint splits, which gives
    the spread needed for confidence intervals and significance tests.
  * Computational cost: parameters, training time, inference time per patient,
    peak GPU memory and on-disk model size.

Per-patient predictions are written to results/paf_bench/preds/ so that
stats_paf_bench.py can run McNemar and bootstrap tests without retraining.

The run is resumable: completed (model, seed) pairs are read back from
results/paf_bench/runs.csv and skipped.

Usage (from WSL):  ./run_gpu.sh train_paf_bench.py
"""

import os
import gc
import time
import numpy as np
import pandas as pd

import tensorflow as tf
from keras import backend as K
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight

from models_zoo import BUILDERS, BATCH_OVERRIDE

# The dataset and output directory can be overridden so the same harness can be
# re-run on the corrected-R-peak copy of the data (PAF_DATA=data_fusion_fix).
DATA = os.environ.get("PAF_DATA", "data_fusion")
OUT = os.environ.get("PAF_OUT", "results/paf_bench")
PRED_DIR = os.path.join(OUT, "preds")
RUNS_CSV = os.path.join(OUT, "runs.csv")

N_SEEDS = 5
SEEDS = [45, 46, 47, 48, 49][:N_SEEDS]
if os.environ.get("PAF_SEEDS"):
    SEEDS = [int(s) for s in os.environ["PAF_SEEDS"].split(",")]
    N_SEEDS = len(SEEDS)
# PAF_SPLIT=official replaces the five random patient-disjoint splits with the
# patient-stratified folds that ship with PTB-XL (train 1-8, validation 9,
# test 10). The seed then only changes the weight initialisation.
SPLIT = os.environ.get("PAF_SPLIT", "random")
PTBXL_DB = "ptbxl/ptbxl/ptbxl_database.csv"
TEST_FRACTION = 0.15
VAL_FRACTION = 0.15
EPOCHS = 50
PATIENCE = 6
BATCH = 128
NORM_LABEL = 4          # PTB-XL class index of NORM in the 10-class encoding
CLASS_NAMES = ["NORM", "MI"]

for _g in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(_g, True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
def patient_split(y, groups, seed, fraction):
    """Patient-disjoint split, stratified on the label where possible."""
    try:
        k = max(2, round(1.0 / fraction))
        a, b = next(StratifiedGroupKFold(n_splits=k, shuffle=True,
                                         random_state=seed).split(y, y, groups))
    except Exception:
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=fraction,
                                      random_state=seed).split(y, y, groups))
    assert not (set(groups[a]) & set(groups[b])), "patient leak"
    return a, b


def make_splits(y, groups, seed):
    """Return train, validation and test beat indices for one run."""
    if SPLIT == "official":
        db = pd.read_csv(PTBXL_DB, usecols=["patient_id", "strat_fold"])
        fold_of = db.groupby("patient_id").strat_fold.first()
        fold = fold_of.reindex(groups).to_numpy()
        tr = np.where(fold <= 8)[0]
        va = np.where(fold == 9)[0]
        te = np.where(fold == 10)[0]
        assert not (set(groups[tr]) & set(groups[te])), "patient leak"
        return tr, va, te
    tr_all, te = patient_split(y, groups, seed, TEST_FRACTION)
    tr, va = patient_split(y[tr_all], groups[tr_all], seed, VAL_FRACTION)
    return tr_all[tr], tr_all[va], te


# ---------------------------------------------------------------------------
# Aggregation strategies
# ---------------------------------------------------------------------------
def make_dataset(X, idx, y=None, sample_weight=None, batch=128,
                 shuffle=False, seed=0):
    """Stream batches straight off the memory-mapped array.

    Handing Keras the whole split as one array copies several gigabytes onto the
    GPU before training starts, which does not fit once the corrected R-peak
    detection triples the beat count. A generator keeps the data on the host and
    moves one batch at a time.
    """
    rng = np.random.default_rng(seed)
    n_classes = 2

    def gen():
        order = idx.copy()
        if shuffle:
            rng.shuffle(order)
        for i in range(0, len(order), batch):
            b = np.sort(order[i:i + batch])
            xb = np.asarray(X[b], dtype=np.float32)
            if y is None:
                yield xb
            elif sample_weight is None:
                yield xb, to_categorical(y[b], n_classes)
            else:
                yield (xb, to_categorical(y[b], n_classes),
                       sample_weight[b].astype(np.float32))

    spec_x = tf.TensorSpec((None, X.shape[1], X.shape[2]), tf.float32)
    if y is None:
        sig = spec_x
    elif sample_weight is None:
        sig = (spec_x, tf.TensorSpec((None, n_classes), tf.float32))
    else:
        sig = (spec_x, tf.TensorSpec((None, n_classes), tf.float32),
               tf.TensorSpec((None,), tf.float32))
    return tf.data.Dataset.from_generator(gen, output_signature=sig).prefetch(2)


def aggregate(y_beat, proba, groups, rng):
    """Return per-patient truth plus predictions and scores for each strategy.

    single : the decision from one randomly drawn beat (no aggregation)
    vote   : majority vote over per-beat hard decisions
    paf    : arg max of the mean posterior over the patient's beats
    """
    patients = np.unique(groups)
    y_true = np.empty(len(patients), dtype=int)
    pred = {k: np.empty(len(patients), dtype=int) for k in ("single", "vote", "paf")}
    score = {k: np.empty(len(patients), dtype=float) for k in ("single", "vote", "paf")}

    hard = proba.argmax(axis=1)
    for i, p in enumerate(patients):
        idx = np.where(groups == p)[0]
        y_true[i] = np.bincount(y_beat[idx], minlength=2).argmax()

        j = idx[rng.integers(len(idx))]
        pred["single"][i] = hard[j]
        score["single"][i] = proba[j, 1]

        counts = np.bincount(hard[idx], minlength=2)
        pred["vote"][i] = counts.argmax()
        score["vote"][i] = counts[1] / counts.sum()

        mean_p = proba[idx].mean(axis=0)
        pred["paf"][i] = mean_p.argmax()
        score["paf"][i] = mean_p[1]
    return patients, y_true, pred, score


def binary_metrics(y_true, y_pred, score):
    from sklearn.metrics import confusion_matrix, roc_auc_score
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    out = {"accuracy": float(np.mean(y_true == y_pred))}
    f1s = []
    for i, name in enumerate(CLASS_NAMES):
        tp = cm[i, i]
        fn = cm[i].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        se = tp / (tp + fn) if tp + fn else 0.0
        sp = tn / (tn + fp) if tn + fp else 0.0
        pr = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * pr * se / (pr + se) if pr + se else 0.0
        f1s.append(f1)
        out["se_" + name] = se
        out["sp_" + name] = sp
        out["f1_" + name] = f1
    out["macro_f1"] = float(np.mean(f1s))
    try:
        out["auroc"] = float(roc_auc_score(y_true, score))
    except ValueError:
        out["auroc"] = float("nan")
    return out


# ---------------------------------------------------------------------------
# One training run
# ---------------------------------------------------------------------------
def run_one(name, builder, seed, X, y, groups):
    rng = np.random.default_rng(seed)
    tf.keras.utils.set_random_seed(seed)

    tr, va, te = make_splits(y, groups, seed)

    ytr, yva = y[tr], y[va]
    yte, gte = y[te], groups[te]

    classes = np.unique(ytr)
    cw = compute_class_weight("balanced", classes=classes, y=ytr)
    class_weight = {int(c): float(w) for c, w in zip(classes, cw)}
    weights = np.zeros(len(y), dtype=np.float32)
    for c, w in class_weight.items():
        weights[y == c] = w

    batch = BATCH_OVERRIDE.get(name, BATCH)
    ds_tr = make_dataset(X, tr, y, weights, batch, shuffle=True, seed=seed)
    ds_va = make_dataset(X, va, y, weights, batch)
    ds_te = make_dataset(X, te, batch=batch)
    model = builder((X.shape[1], X.shape[2]), 2)
    params = int(model.count_params())
    model.compile(loss="categorical_crossentropy",
                  optimizer=Adam(learning_rate=0.001), metrics=["accuracy"])

    try:
        tf.config.experimental.reset_memory_stats("GPU:0")
    except Exception:
        pass

    t0 = time.time()
    model.fit(ds_tr, epochs=EPOCHS, validation_data=ds_va, verbose=2,
              callbacks=[EarlyStopping(monitor="val_loss", patience=PATIENCE,
                                       restore_best_weights=True)])
    train_time = time.time() - t0

    model.predict(np.asarray(X[te[:batch]]), verbose=0)   # warm up the graph
    t0 = time.time()
    proba = model.predict(ds_te, verbose=0)
    infer_time = time.time() - t0

    # the test generator walks each batch in ascending index order, so line the
    # labels up the same way
    order = np.concatenate([np.sort(te[i:i + batch])
                            for i in range(0, len(te), batch)])
    remap = {int(v): k for k, v in enumerate(order)}
    perm = np.array([remap[int(v)] for v in te])
    proba = proba[perm]

    try:
        peak_mb = tf.config.experimental.get_memory_info("GPU:0")["peak"] / 1e6
    except Exception:
        peak_mb = float("nan")

    tmp = os.path.join(OUT, "_tmp_model.keras")
    model.save(tmp)
    size_mb = os.path.getsize(tmp) / 1e6
    os.remove(tmp)

    patients, y_true, pred, score = aggregate(yte, proba, gte, rng)
    np.savez_compressed(
        os.path.join(PRED_DIR, "{}_seed{}.npz".format(name.replace(" ", "_"), seed)),
        patients=patients, y_true=y_true,
        **{"pred_" + k: v for k, v in pred.items()},
        **{"score_" + k: v for k, v in score.items()})

    n_patients = len(patients)
    rows = []
    for strategy in ("single", "vote", "paf"):
        m = binary_metrics(y_true, pred[strategy], score[strategy])
        m.update(model=name, seed=seed, aggregation=strategy,
                 params=params, params_M=round(params / 1e6, 3),
                 train_time_s=round(train_time, 1),
                 infer_ms_per_patient=round(1000 * infer_time / n_patients, 3),
                 peak_gpu_mb=round(peak_mb, 1), model_size_mb=round(size_mb, 2),
                 n_test_patients=n_patients, n_test_beats=len(te),
                 beats_per_patient=round(len(te) / n_patients, 2))
        rows.append(m)

    del model, ds_tr, ds_va, ds_te
    gc.collect()
    K.clear_session()
    return rows


# ---------------------------------------------------------------------------
def main():
    os.makedirs(PRED_DIR, exist_ok=True)
    X = np.load(DATA + "/all_X.npy", mmap_mode="r")
    y10 = np.load(DATA + "/all_y.npy")
    groups = np.load(DATA + "/all_groups.npy")
    y = (y10 != NORM_LABEL).astype(int)
    print("beats {} | patients {} | MI beats {:.1%}".format(
        len(y), len(np.unique(groups)), y.mean()), flush=True)

    done = set()
    rows = []
    if os.path.exists(RUNS_CSV):
        old = pd.read_csv(RUNS_CSV)
        rows = old.to_dict("records")
        done = set(zip(old["model"], old["seed"]))
        print("resuming: {} rows already present".format(len(rows)), flush=True)

    only = os.environ.get("PAF_MODELS")
    wanted = [s.strip() for s in only.split(",")] if only else list(BUILDERS)
    for name in wanted:
        builder = BUILDERS[name]
        for seed in SEEDS:
            if (name, seed) in done:
                continue
            print("\n=== {} | seed {} ===".format(name, seed), flush=True)
            try:
                rows.extend(run_one(name, builder, seed, X, y, groups))
            except Exception as e:
                print("[error] {} seed {}: {}".format(name, seed, e), flush=True)
                K.clear_session()
                continue
            df = pd.DataFrame(rows)
            df.to_csv(RUNS_CSV, index=False)
            last = df[(df.model == name) & (df.seed == seed) & (df.aggregation == "paf")]
            print("{} seed {} | PAF acc {:.4f} macro-F1 {:.4f}".format(
                name, seed, last.accuracy.iloc[0], last.macro_f1.iloc[0]), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RUNS_CSV, index=False)
    summary = (df.groupby(["model", "aggregation"])[["accuracy", "macro_f1", "auroc"]]
                 .agg(["mean", "std"]).round(4))
    summary.to_csv(os.path.join(OUT, "summary_mean_std.csv"))
    print("\n=== mean +/- sd over {} splits ===".format(N_SEEDS))
    print(summary.to_string())


if __name__ == "__main__":
    main()
