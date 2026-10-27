"""
train_attention_pool.py  (GPU, launched from run_attention.ps1)

Tests whether a learned pooling rule beats the fixed one.

Patient-Aggregated Fusion averages the per-beat posteriors of a patient, and the
saturation experiment shows that this fixed rule stops extracting anything after
roughly twenty beats. Attention-based multiple-instance learning replaces the
average with weights the network learns, so beats that carry more evidence count
for more. If the plateau is a property of mean pooling, attention should break
it; if the plateau is a property of the data, attention will sit on it too. Both
outcomes are informative, which is why this is worth running.

The model embeds each beat of a patient with the same compact CNN backbone used
elsewhere in the paper, computes gated attention weights over those embeddings,
and classifies the attention-weighted bag embedding. Training is therefore
end-to-end at the patient level rather than at the beat level.

Comparison is paired: the splits come from the same function and the same seeds
as train_paf_bench.py, so the test patients are identical and the saved
per-patient predictions of both methods can be compared with an exact McNemar
test without retraining anything.

Resumable. Every finished seed is appended to results/attention_pool/runs.csv
and skipped when the script is run again, so an interrupted run costs at most
one seed.

Outputs (results/attention_pool/):
  runs.csv                 one row per seed and pooling rule
  preds/attention_seedNN.npz   per-patient predictions, for paired tests
  summary.txt              attention against PAF, with McNemar
"""

import os
import gc
import time
import numpy as np
import pandas as pd

import tensorflow as tf
import keras
from keras import layers
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from scipy import stats

from train_paf_bench import make_splits, binary_metrics, NORM_LABEL

DATA = os.environ.get("PAF_DATA", "data_fusion_fix")
PAF_PREDS = os.environ.get("PAF_PREDS", "results/paf_bench_fix/preds")
OUT = os.environ.get("ATT_OUT", "results/attention_pool")
RUNS_CSV = os.path.join(OUT, "runs.csv")
PRED_DIR = os.path.join(OUT, "preds")

SEEDS = [45, 46, 47, 48, 49]
if os.environ.get("PAF_SEEDS"):
    SEEDS = [int(v) for v in os.environ["PAF_SEEDS"].split(",")]
BAG_SIZE = 20             # beats per bag: the knee of the saturation curve
BAGS_PER_BATCH = 16
EPOCHS = 50
PATIENCE = 6
LR = 1e-3
EVAL_DRAWS = 5            # random bags per test patient, averaged

for _g in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(_g, True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def beat_encoder(input_shape):
    """The compact CNN backbone, truncated before the classifier."""
    return keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(128, 11, activation="relu"),
        layers.MaxPooling1D(), layers.Dropout(0.2),
        layers.Conv1D(128, 13, activation="relu"),
        layers.MaxPooling1D(), layers.Dropout(0.2),
        layers.Conv1D(128, 7, activation="relu"),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
    ], name="beat_encoder")


def attention_mil(beat_shape, bag_size, att_dim=64):
    """Gated attention pooling over the beats of one patient."""
    beats = layers.Input(shape=(bag_size,) + beat_shape, name="beats")
    mask = layers.Input(shape=(bag_size,), name="mask")

    h = layers.TimeDistributed(beat_encoder(beat_shape))(beats)   # (B, K, 128)
    h = layers.Dropout(0.3)(h)

    v = layers.TimeDistributed(layers.Dense(att_dim, activation="tanh"))(h)
    u = layers.TimeDistributed(layers.Dense(att_dim, activation="sigmoid"))(h)
    a = layers.TimeDistributed(layers.Dense(1))(layers.Multiply()([v, u]))
    a = layers.Reshape((bag_size,))(a)

    # padded slots must not receive attention
    neg = layers.Lambda(lambda m: (1.0 - m) * -1e9,
                        output_shape=(bag_size,))(mask)
    a = layers.Add()([a, neg])
    a = layers.Softmax(axis=1, name="attention")(a)

    z = layers.Dot(axes=(1, 1))([a, h])                          # (B, 128)
    z = layers.Dropout(0.3)(z)
    out = layers.Dense(2, activation="softmax")(z)
    return keras.Model([beats, mask], out, name="attention_mil")


# ---------------------------------------------------------------------------
# Bags
# ---------------------------------------------------------------------------
def patient_index(groups, idx):
    """{patient: array of row indices} restricted to idx."""
    out = {}
    for p in np.unique(groups[idx]):
        out[p] = idx[groups[idx] == p]
    return out


def bag_dataset(X, y, bags, patients, batch, shuffle, seed, weights=None):
    """Yield (beats, mask), label, [sample weight] one batch of bags at a time."""
    rng = np.random.default_rng(seed)
    beat_shape = (X.shape[1], X.shape[2])

    def gen():
        order = np.array(patients)
        if shuffle:
            rng.shuffle(order)
        for i in range(0, len(order), batch):
            chunk = order[i:i + batch]
            xb = np.zeros((len(chunk), BAG_SIZE) + beat_shape, dtype=np.float32)
            mb = np.zeros((len(chunk), BAG_SIZE), dtype=np.float32)
            yb = np.zeros((len(chunk), 2), dtype=np.float32)
            wb = np.ones(len(chunk), dtype=np.float32)
            for j, p in enumerate(chunk):
                rows = bags[p]
                take = rows if len(rows) <= BAG_SIZE else rng.choice(
                    rows, BAG_SIZE, replace=False)
                take = np.sort(take)
                xb[j, :len(take)] = np.asarray(X[take], dtype=np.float32)
                mb[j, :len(take)] = 1.0
                label = int(np.bincount(y[rows], minlength=2).argmax())
                yb[j, label] = 1.0
                if weights is not None:
                    wb[j] = weights[label]
            if weights is None:
                yield (xb, mb), yb
            else:
                yield (xb, mb), yb, wb

    sig_x = (tf.TensorSpec((None, BAG_SIZE) + beat_shape, tf.float32),
             tf.TensorSpec((None, BAG_SIZE), tf.float32))
    sig_y = tf.TensorSpec((None, 2), tf.float32)
    sig = (sig_x, sig_y) if weights is None else (
        sig_x, sig_y, tf.TensorSpec((None,), tf.float32))
    return tf.data.Dataset.from_generator(gen, output_signature=sig).prefetch(2)


def predict_bags(model, X, y, bags, patients, seed):
    """Average the model's posterior over EVAL_DRAWS random bags per patient."""
    rng = np.random.default_rng(seed)
    beat_shape = (X.shape[1], X.shape[2])
    proba = np.zeros((len(patients), 2), dtype=float)
    y_true = np.zeros(len(patients), dtype=int)
    for draw in range(EVAL_DRAWS):
        for i in range(0, len(patients), BAGS_PER_BATCH):
            chunk = patients[i:i + BAGS_PER_BATCH]
            xb = np.zeros((len(chunk), BAG_SIZE) + beat_shape, dtype=np.float32)
            mb = np.zeros((len(chunk), BAG_SIZE), dtype=np.float32)
            for j, p in enumerate(chunk):
                rows = bags[p]
                take = rows if len(rows) <= BAG_SIZE else rng.choice(
                    rows, BAG_SIZE, replace=False)
                take = np.sort(take)
                xb[j, :len(take)] = np.asarray(X[take], dtype=np.float32)
                mb[j, :len(take)] = 1.0
                if draw == 0:
                    y_true[i + j] = int(np.bincount(y[rows], minlength=2).argmax())
            proba[i:i + len(chunk)] += model.predict([xb, mb], verbose=0)
    return y_true, proba / EVAL_DRAWS


def mcnemar(y_true, a, b):
    a_ok, b_ok = a == y_true, b == y_true
    n10 = int(np.sum(a_ok & ~b_ok))
    n01 = int(np.sum(~a_ok & b_ok))
    n = n10 + n01
    p = float(stats.binomtest(n10, n, 0.5).pvalue) if n else 1.0
    return n10, n01, p


# ---------------------------------------------------------------------------
def run_seed(seed, X, y, groups):
    tf.keras.utils.set_random_seed(seed)
    tr, va, te = make_splits(y, groups, seed)

    bags_tr = patient_index(groups, tr)
    bags_va = patient_index(groups, va)
    bags_te = patient_index(groups, te)
    pats_tr = sorted(bags_tr)
    pats_va = sorted(bags_va)
    pats_te = np.array(sorted(bags_te))

    labels_tr = np.array([int(np.bincount(y[bags_tr[p]], minlength=2).argmax())
                          for p in pats_tr])
    counts = np.bincount(labels_tr, minlength=2).astype(float)
    weights = counts.sum() / (2.0 * np.maximum(counts, 1.0))
    print("seed {} | train bags {} | val bags {} | test bags {} | "
          "class weights {}".format(seed, len(pats_tr), len(pats_va),
                                    len(pats_te), np.round(weights, 3)),
          flush=True)

    model = attention_mil((X.shape[1], X.shape[2]), BAG_SIZE)
    model.compile(loss="categorical_crossentropy",
                  optimizer=Adam(learning_rate=LR), metrics=["accuracy"])
    t0 = time.time()
    model.fit(bag_dataset(X, y, bags_tr, pats_tr, BAGS_PER_BATCH, True, seed, weights),
              validation_data=bag_dataset(X, y, bags_va, pats_va,
                                          BAGS_PER_BATCH, False, seed, weights),
              epochs=EPOCHS, verbose=2,
              callbacks=[EarlyStopping(monitor="val_loss", patience=PATIENCE,
                                       restore_best_weights=True)])
    train_time = time.time() - t0

    t0 = time.time()
    y_true, proba = predict_bags(model, X, y, bags_te, pats_te, seed)
    infer_time = time.time() - t0
    pred = proba.argmax(axis=1)
    met = binary_metrics(y_true, pred, proba[:, 1])

    os.makedirs(PRED_DIR, exist_ok=True)
    np.savez_compressed(os.path.join(PRED_DIR, "attention_seed{}.npz".format(seed)),
                        patients=pats_te, y_true=y_true, pred=pred,
                        score=proba[:, 1])

    row = dict(seed=seed, rule="attention", bag_size=BAG_SIZE,
               params=int(model.count_params()),
               params_M=round(model.count_params() / 1e6, 3),
               train_time_s=round(train_time, 1),
               infer_ms_per_patient=round(1000 * infer_time / len(pats_te), 3),
               n_test_patients=len(pats_te))
    row.update({k: round(v, 4) for k, v in met.items()})

    # paired comparison against mean-posterior pooling on the same patients
    paf_file = os.path.join(PAF_PREDS, "CNN_(ours)_seed{}.npz".format(seed))
    if os.path.exists(paf_file):
        d = dict(np.load(paf_file))
        order = {int(p): i for i, p in enumerate(d["patients"])}
        sel = np.array([order[int(p)] for p in pats_te if int(p) in order])
        keep = np.array([i for i, p in enumerate(pats_te) if int(p) in order])
        if len(sel) == len(pats_te):
            n10, n01, p = mcnemar(y_true[keep], pred[keep], d["pred_paf"][sel])
            row.update(vs_paf_attention_only=n10, vs_paf_paf_only=n01,
                       vs_paf_mcnemar_p=p,
                       paf_accuracy=round(float(np.mean(
                           d["pred_paf"][sel] == d["y_true"][sel])), 4))
    del model
    gc.collect()
    keras.backend.clear_session()
    return row


def main():
    os.makedirs(OUT, exist_ok=True)
    # Bags draw beats scattered through the array, so reading them from a
    # memory-mapped file makes every training step wait on disk. Holding the
    # array in RAM costs about 4 GB and roughly fifty seconds to load, and makes
    # a training step around a hundred times faster. Set ATT_IN_RAM=0 to fall
    # back to the memory-mapped path on a machine that cannot spare the memory.
    if os.environ.get("ATT_IN_RAM", "1") == "1":
        print("loading beats into RAM (about 4 GB) ...", flush=True)
        X = np.load(DATA + "/all_X.npy")
    else:
        X = np.load(DATA + "/all_X.npy", mmap_mode="r")
    y = (np.load(DATA + "/all_y.npy") != NORM_LABEL).astype(int)
    groups = np.load(DATA + "/all_groups.npy")
    print("beats {} | patients {} | bag size {}".format(
        len(y), len(np.unique(groups)), BAG_SIZE), flush=True)

    rows, done = [], set()
    if os.path.exists(RUNS_CSV):
        old = pd.read_csv(RUNS_CSV)
        rows = old.to_dict("records")
        done = set(old["seed"].astype(int))
        print("resuming: seeds already done {}".format(sorted(done)), flush=True)

    for seed in SEEDS:
        if seed in done:
            continue
        try:
            rows.append(run_seed(seed, X, y, groups))
        except Exception as e:
            print("[error] seed {}: {}".format(seed, e), flush=True)
            keras.backend.clear_session()
            continue
        pd.DataFrame(rows).to_csv(RUNS_CSV, index=False)
        r = rows[-1]
        print("seed {} | attention acc {:.4f} macro-F1 {:.4f} | PAF acc {}".format(
            seed, r["accuracy"], r["macro_f1"], r.get("paf_accuracy", "n/a")),
            flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RUNS_CSV, index=False)
    lines = ["attention pooling, bag size {}, {} seeds".format(BAG_SIZE, len(df)),
             "accuracy {:.4f} +/- {:.4f}".format(df.accuracy.mean(),
                                                 df.accuracy.std(ddof=1)),
             "macro-F1 {:.4f} +/- {:.4f}".format(df.macro_f1.mean(),
                                                 df.macro_f1.std(ddof=1)),
             "AUROC    {:.4f} +/- {:.4f}".format(df.auroc.mean(),
                                                 df.auroc.std(ddof=1))]
    if "vs_paf_attention_only" in df.columns:
        n10 = int(df.vs_paf_attention_only.sum())
        n01 = int(df.vs_paf_paf_only.sum())
        n = n10 + n01
        p = float(stats.binomtest(n10, n, 0.5).pvalue) if n else 1.0
        lines += ["", "against mean-posterior pooling, discordant patients pooled "
                      "over seeds:",
                  "  attention only correct {} | PAF only correct {} | "
                  "McNemar p = {:.4g}".format(n10, n01, p),
                  "  PAF accuracy {:.4f}".format(df.paf_accuracy.mean())]
    open(os.path.join(OUT, "summary.txt"), "w").write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
