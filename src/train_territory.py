"""
train_territory.py  (GPU, launched from run_territory.ps1)

Repeats the pooling comparison on the four-territory task, which is the
clinically interesting one: not merely whether an infarction is present, but
which wall of the myocardium it sits in.

Two methods are trained on identical patient-disjoint splits, so the comparison
is paired:

  mean      the compact CNN trained on individual beats, with the per-beat
            posteriors of a patient averaged into one decision (PAF)
  attention the same backbone embedded in a gated attention pooling model, so
            the weight given to each beat is learned rather than fixed

Labels collapse the ten PTB-XL classes onto NORM and the three infarct
territories that the data can support. The lateral/posterior group is very
small, and its per-class numbers should be read as an indication rather than an
estimate; the macro-F1 reported here is dominated by how well the three
well-populated classes are separated.

The trained beat-level model of the first seed is saved, because
territory_saliency.py needs a model that was actually asked to localise in order
to test whether per-lead importance follows infarct territory. The binary
detector cannot answer that question, since nothing ever asked it to.

Resumable at the level of one (method, seed) pair: finished pairs are appended
to results/territory/runs.csv and skipped on a later run.

Outputs (results/territory/):
  runs.csv               one row per method and seed
  preds/*.npz            per-patient predictions, for paired tests
  model_beat_seed45.keras  beat-level four-class model, for the saliency test
  summary.txt            mean pooling against attention, with tests
"""

import os
import gc
import time
import numpy as np
import pandas as pd

import tensorflow as tf
import keras
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, roc_auc_score
from scipy import stats

from models_zoo import build_cnn_ours
from train_paf_bench import patient_split
import train_attention_pool as att

DATA = os.environ.get("PAF_DATA", "data_fusion_fix")
OUT = os.environ.get("TERR_OUT", "results/territory")
RUNS_CSV = os.path.join(OUT, "runs.csv")
PRED_DIR = os.path.join(OUT, "preds")

SEEDS = [45, 46, 47, 48, 49]
EPOCHS = 50
PATIENCE = 6
BEAT_BATCH = 128
LR = 1e-3

# PTB-XL class order: 0 IMI, 1 ASMI, 2 ILMI, 3 LMI, 4 NORM, 5 AMI, 6 ALMI,
# 7 IPLMI, 8 IPMI, 9 PMI
TERRITORY_MAP = np.array([2, 1, 2, 3, 0, 1, 1, 2, 2, 3])
CLASS_NAMES = ["NORM", "Anterior", "Inferior", "LatPost"]
N_CLASSES = 4

for _g in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(_g, True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
def multiclass_metrics(y_true, y_pred, proba=None):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))
    out = {"accuracy": float(np.mean(y_true == y_pred))}
    f1s = []
    for i, name in enumerate(CLASS_NAMES):
        tp = cm[i, i]
        fn = cm[i].sum() - tp
        fp = cm[:, i].sum() - tp
        se = tp / (tp + fn) if tp + fn else 0.0
        pr = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * pr * se / (pr + se) if pr + se else 0.0
        f1s.append(f1)
        out["se_" + name] = se
        out["f1_" + name] = f1
        out["n_" + name] = int(cm[i].sum())
    out["macro_f1"] = float(np.mean(f1s))
    if proba is not None:
        try:
            out["auroc"] = float(roc_auc_score(y_true, proba, multi_class="ovr",
                                               average="macro"))
        except ValueError:
            out["auroc"] = float("nan")
    return out


def beat_dataset(X, idx, y, weights, batch, shuffle, seed):
    """Stream beats from the memory-mapped array, one batch at a time."""
    rng = np.random.default_rng(seed)

    def gen():
        order = idx.copy()
        if shuffle:
            rng.shuffle(order)
        for i in range(0, len(order), batch):
            b = np.sort(order[i:i + batch])
            yield (np.asarray(X[b], dtype=np.float32),
                   keras.utils.to_categorical(y[b], N_CLASSES),
                   weights[b].astype(np.float32))

    sig = (tf.TensorSpec((None, X.shape[1], X.shape[2]), tf.float32),
           tf.TensorSpec((None, N_CLASSES), tf.float32),
           tf.TensorSpec((None,), tf.float32))
    return tf.data.Dataset.from_generator(gen, output_signature=sig).prefetch(2)


def mean_pool(model, X, te, y, groups, batch=BEAT_BATCH):
    """PAF: average the per-beat posteriors of each patient."""
    proba = []
    for i in range(0, len(te), batch):
        proba.append(model.predict(np.asarray(X[np.sort(te[i:i + batch])]),
                                   verbose=0))
    proba = np.concatenate(proba, axis=0)
    order = np.concatenate([np.sort(te[i:i + batch])
                            for i in range(0, len(te), batch)])
    remap = {int(v): k for k, v in enumerate(order)}
    proba = proba[np.array([remap[int(v)] for v in te])]

    patients = np.unique(groups[te])
    y_true = np.empty(len(patients), dtype=int)
    bag = np.empty((len(patients), N_CLASSES), dtype=float)
    for i, p in enumerate(patients):
        rows = np.where(groups[te] == p)[0]
        y_true[i] = np.bincount(y[te][rows], minlength=N_CLASSES).argmax()
        bag[i] = proba[rows].mean(axis=0)
    return patients, y_true, bag


# ---------------------------------------------------------------------------
def run_mean(seed, X, y, groups):
    tf.keras.utils.set_random_seed(seed)
    tr_all, te = patient_split(y, groups, seed, 0.15)
    tr, va = patient_split(y[tr_all], groups[tr_all], seed, 0.15)
    tr, va = tr_all[tr], tr_all[va]

    counts = np.bincount(y[tr], minlength=N_CLASSES).astype(float)
    cw = counts.sum() / (N_CLASSES * np.maximum(counts, 1.0))
    weights = cw[y].astype(np.float32)

    model = build_cnn_ours((X.shape[1], X.shape[2]), N_CLASSES)
    model.compile(loss="categorical_crossentropy",
                  optimizer=Adam(learning_rate=LR), metrics=["accuracy"])
    t0 = time.time()
    model.fit(beat_dataset(X, tr, y, weights, BEAT_BATCH, True, seed),
              validation_data=beat_dataset(X, va, y, weights, BEAT_BATCH, False, seed),
              epochs=EPOCHS, verbose=2,
              callbacks=[EarlyStopping(monitor="val_loss", patience=PATIENCE,
                                       restore_best_weights=True)])
    train_time = time.time() - t0

    t0 = time.time()
    patients, y_true, bag = mean_pool(model, X, te, y, groups)
    infer_time = time.time() - t0
    pred = bag.argmax(axis=1)

    if seed == SEEDS[0]:
        model.save(os.path.join(OUT, "model_beat_seed{}.keras".format(seed)))

    np.savez_compressed(os.path.join(PRED_DIR, "mean_seed{}.npz".format(seed)),
                        patients=patients, y_true=y_true, pred=pred, proba=bag)
    row = dict(method="mean", seed=seed, params_M=round(model.count_params() / 1e6, 3),
               train_time_s=round(train_time, 1),
               infer_ms_per_patient=round(1000 * infer_time / len(patients), 3),
               n_test_patients=len(patients))
    row.update({k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in multiclass_metrics(y_true, pred, bag).items()})
    del model
    gc.collect()
    keras.backend.clear_session()
    return row


def run_attention(seed, X, y, groups):
    tf.keras.utils.set_random_seed(seed)
    tr_all, te = patient_split(y, groups, seed, 0.15)
    tr, va = patient_split(y[tr_all], groups[tr_all], seed, 0.15)
    tr, va = tr_all[tr], tr_all[va]

    bags_tr = att.patient_index(groups, tr)
    bags_va = att.patient_index(groups, va)
    bags_te = att.patient_index(groups, te)
    pats_tr, pats_va = sorted(bags_tr), sorted(bags_va)
    pats_te = np.array(sorted(bags_te))

    # Class weights come from the beat distribution, exactly as they do for the
    # mean-pooling model, so the two methods differ only in how the beats are
    # pooled. Deriving them from the bag distribution instead makes the weights
    # far more extreme, because the patient-level imbalance is an order of
    # magnitude worse than the beat-level one (roughly 128:1 against 12:1), and
    # at batch sizes of a few bags that drives the model to predict the rarest
    # class for every patient.
    counts = np.bincount(y[tr], minlength=N_CLASSES).astype(float)
    weights = counts.sum() / (N_CLASSES * np.maximum(counts, 1.0))
    print("class weights {}".format(np.round(weights, 3)), flush=True)

    model = att.attention_mil((X.shape[1], X.shape[2]), att.BAG_SIZE)
    # swap the binary head for a four-class one, keeping the pooling untouched
    z = model.layers[-2].output
    model = keras.Model(model.inputs,
                        keras.layers.Dense(N_CLASSES, activation="softmax")(z),
                        name="attention_mil_territory")
    model.compile(loss="categorical_crossentropy",
                  optimizer=Adam(learning_rate=LR), metrics=["accuracy"])

    t0 = time.time()
    model.fit(bag_dataset_nc(X, y, bags_tr, pats_tr, att.BAGS_PER_BATCH, True,
                             seed, weights),
              validation_data=bag_dataset_nc(X, y, bags_va, pats_va,
                                             att.BAGS_PER_BATCH, False, seed, weights),
              epochs=EPOCHS, verbose=2,
              callbacks=[EarlyStopping(monitor="val_loss", patience=PATIENCE,
                                       restore_best_weights=True)])
    train_time = time.time() - t0

    t0 = time.time()
    y_true, proba = predict_bags_nc(model, X, y, bags_te, pats_te, seed)
    infer_time = time.time() - t0
    pred = proba.argmax(axis=1)

    np.savez_compressed(os.path.join(PRED_DIR, "attention_seed{}.npz".format(seed)),
                        patients=pats_te, y_true=y_true, pred=pred, proba=proba)
    row = dict(method="attention", seed=seed,
               params_M=round(model.count_params() / 1e6, 3),
               train_time_s=round(train_time, 1),
               infer_ms_per_patient=round(1000 * infer_time / len(pats_te), 3),
               n_test_patients=len(pats_te))
    row.update({k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in multiclass_metrics(y_true, pred, proba).items()})
    del model
    gc.collect()
    keras.backend.clear_session()
    return row


def bag_dataset_nc(X, y, bags, patients, batch, shuffle, seed, weights):
    """The bag generator of train_attention_pool, widened to four classes."""
    rng = np.random.default_rng(seed)
    beat_shape = (X.shape[1], X.shape[2])
    K = att.BAG_SIZE

    def gen():
        order = np.array(patients)
        if shuffle:
            rng.shuffle(order)
        for i in range(0, len(order), batch):
            chunk = order[i:i + batch]
            xb = np.zeros((len(chunk), K) + beat_shape, dtype=np.float32)
            mb = np.zeros((len(chunk), K), dtype=np.float32)
            yb = np.zeros((len(chunk), N_CLASSES), dtype=np.float32)
            wb = np.ones(len(chunk), dtype=np.float32)
            for j, p in enumerate(chunk):
                rows = bags[p]
                take = rows if len(rows) <= K else rng.choice(rows, K, replace=False)
                take = np.sort(take)
                xb[j, :len(take)] = np.asarray(X[take], dtype=np.float32)
                mb[j, :len(take)] = 1.0
                label = int(np.bincount(y[rows], minlength=N_CLASSES).argmax())
                yb[j, label] = 1.0
                wb[j] = weights[label]
            yield (xb, mb), yb, wb

    sig = ((tf.TensorSpec((None, K) + beat_shape, tf.float32),
            tf.TensorSpec((None, K), tf.float32)),
           tf.TensorSpec((None, N_CLASSES), tf.float32),
           tf.TensorSpec((None,), tf.float32))
    return tf.data.Dataset.from_generator(gen, output_signature=sig).prefetch(2)


def predict_bags_nc(model, X, y, bags, patients, seed):
    rng = np.random.default_rng(seed)
    beat_shape = (X.shape[1], X.shape[2])
    K, B = att.BAG_SIZE, att.BAGS_PER_BATCH
    proba = np.zeros((len(patients), N_CLASSES), dtype=float)
    y_true = np.zeros(len(patients), dtype=int)
    for draw in range(att.EVAL_DRAWS):
        for i in range(0, len(patients), B):
            chunk = patients[i:i + B]
            xb = np.zeros((len(chunk), K) + beat_shape, dtype=np.float32)
            mb = np.zeros((len(chunk), K), dtype=np.float32)
            for j, p in enumerate(chunk):
                rows = bags[p]
                take = rows if len(rows) <= K else rng.choice(rows, K, replace=False)
                take = np.sort(take)
                xb[j, :len(take)] = np.asarray(X[take], dtype=np.float32)
                mb[j, :len(take)] = 1.0
                if draw == 0:
                    y_true[i + j] = int(np.bincount(y[rows],
                                                    minlength=N_CLASSES).argmax())
            proba[i:i + len(chunk)] += model.predict([xb, mb], verbose=0)
    return y_true, proba / att.EVAL_DRAWS


# ---------------------------------------------------------------------------
def main():
    os.makedirs(PRED_DIR, exist_ok=True)
    if os.environ.get("ATT_IN_RAM", "1") == "1":
        print("loading beats into RAM (about 4 GB) ...", flush=True)
        X = np.load(DATA + "/all_X.npy")
    else:
        X = np.load(DATA + "/all_X.npy", mmap_mode="r")
    y10 = np.load(DATA + "/all_y.npy")
    y = TERRITORY_MAP[y10]
    groups = np.load(DATA + "/all_groups.npy")
    print("beats {} | patients {} | beat class counts {}".format(
        len(y), len(np.unique(groups)), np.bincount(y, minlength=N_CLASSES)),
        flush=True)

    rows, done = [], set()
    if os.path.exists(RUNS_CSV):
        old = pd.read_csv(RUNS_CSV)
        rows = old.to_dict("records")
        done = set(zip(old["method"], old["seed"].astype(int)))
        print("resuming: {} pair(s) already done".format(len(done)), flush=True)

    for seed in SEEDS:
        for method, fn in (("mean", run_mean), ("attention", run_attention)):
            if (method, seed) in done:
                continue
            print("\n=== {} | seed {} ===".format(method, seed), flush=True)
            try:
                rows.append(fn(seed, X, y, groups))
            except Exception as e:
                print("[error] {} seed {}: {}".format(method, seed, e), flush=True)
                keras.backend.clear_session()
                continue
            pd.DataFrame(rows).to_csv(RUNS_CSV, index=False)
            r = rows[-1]
            print("{} seed {} | acc {:.4f} macro-F1 {:.4f}".format(
                method, seed, r["accuracy"], r["macro_f1"]), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RUNS_CSV, index=False)

    lines = ["four-territory task, {} seeds".format(df.seed.nunique())]
    for method, g in df.groupby("method"):
        lines.append("{:>9} | accuracy {:.4f} +/- {:.4f} | macro-F1 {:.4f} +/- {:.4f}"
                     .format(method, g.accuracy.mean(), g.accuracy.std(ddof=1),
                             g.macro_f1.mean(), g.macro_f1.std(ddof=1)))
    for name in CLASS_NAMES:
        col = "f1_" + name
        if col in df.columns:
            lines.append("  F1 {:<9} mean {:.3f} | attention {:.3f}".format(
                name, df[df.method == "mean"][col].mean(),
                df[df.method == "attention"][col].mean()))

    n10 = n01 = 0
    for seed in sorted(df.seed.unique()):
        fa = os.path.join(PRED_DIR, "attention_seed{}.npz".format(seed))
        fm = os.path.join(PRED_DIR, "mean_seed{}.npz".format(seed))
        if not (os.path.exists(fa) and os.path.exists(fm)):
            continue
        a, m = dict(np.load(fa)), dict(np.load(fm))
        order = {int(p): i for i, p in enumerate(m["patients"])}
        sel = np.array([order[int(p)] for p in a["patients"] if int(p) in order])
        if len(sel) != len(a["patients"]):
            continue
        ok_a = a["pred"] == a["y_true"]
        ok_m = m["pred"][sel] == m["y_true"][sel]
        n10 += int(np.sum(ok_a & ~ok_m))
        n01 += int(np.sum(~ok_a & ok_m))
    if n10 + n01:
        p = float(stats.binomtest(n10, n10 + n01, 0.5).pvalue)
        lines += ["", "attention against mean pooling, discordant patients pooled:",
                  "  attention only correct {} | mean only correct {} | "
                  "McNemar p = {:.4g}".format(n10, n01, p)]
    a_f1 = df[df.method == "attention"].sort_values("seed").macro_f1.values
    m_f1 = df[df.method == "mean"].sort_values("seed").macro_f1.values
    if len(a_f1) == len(m_f1) >= 3:
        lines.append("  Wilcoxon over seeds p = {:.4f}".format(
            float(stats.wilcoxon(a_f1, m_f1).pvalue)))

    open(os.path.join(OUT, "summary.txt"), "w").write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
