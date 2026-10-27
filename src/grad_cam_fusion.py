"""
grad_cam_fusion.py  (run on GPU/WSL)

Interpretability for the 12-lead FUSION MI-vs-NORM model (the strong honest
model). Produces two figures the paper needs:

  results/gradcam_fusion/temporal_gradcam.png
     Temporal Grad-CAM (last Conv1D): which part of the beat (ST/QRS/T region)
     drives the MI vs NORM decision, overlaid on the mean MI and NORM beats.

  results/gradcam_fusion/lead_importance.png
     Per-lead saliency: mean |d(MI score)/d(input)| per lead channel -> which of
     the 12 leads the model relies on (ties to infarct territory).

Trains the binary 12-lead CNN inline from data_fusion/ (balanced classes), or
loads results/gradcam_fusion/model_binary.h5 if present.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from keras.models import Sequential, load_model
from keras.layers import Conv1D, Dense, Dropout, MaxPooling1D, Flatten
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight

DATA = "data_fusion"
OUT = "results/gradcam_fusion"
LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
REF_LEAD = 1            # lead II index, for overlaying the mean beat
MI, NORM = 1, 0


def build_cnn_12lead(input_shape):
    return Sequential([
        Conv1D(128, 11, activation="relu", input_shape=input_shape),
        MaxPooling1D(), Dropout(0.2),
        Conv1D(128, 13, activation="relu"),
        MaxPooling1D(), Dropout(0.2),
        Conv1D(128, 7, activation="relu"),
        Flatten(), Dense(128, activation="relu"), Dropout(0.3),
        Dense(2, activation="softmax"),
    ])


def get_model(Xtr, ytr):
    os.makedirs(OUT, exist_ok=True)
    path = OUT + "/model_binary.h5"
    if os.path.exists(path):
        return load_model(path)
    cls = np.unique(ytr)
    cw = compute_class_weight("balanced", classes=cls, y=ytr)
    class_weight = {int(c): float(w) for c, w in zip(cls, cw)}
    model = build_cnn_12lead((Xtr.shape[1], Xtr.shape[2]))
    model.compile(loss="categorical_crossentropy",
                  optimizer=Adam(learning_rate=0.001), metrics=["accuracy"])
    model.fit(Xtr, to_categorical(ytr, 2), epochs=50, batch_size=128,
              validation_split=0.15, verbose=2, class_weight=class_weight,
              callbacks=[EarlyStopping(monitor="val_loss", patience=6,
                                       restore_best_weights=True)])
    model.save(path)
    return model


def last_conv(model):
    return [l for l in model.layers if isinstance(l, Conv1D)][-1]


def grad_model_of(model, conv):
    # Keras-3 safe: rebuild a functional graph reusing the same layers/weights
    inp = keras.Input(shape=model.input_shape[1:])
    x, cout = inp, None
    for layer in model.layers:
        x = layer(x)
        if layer is conv:
            cout = x
    return keras.Model(inp, [cout, x])


def temporal_cam(gm, X, class_idx):
    """mean temporal Grad-CAM over a batch X (n,700,12) -> (700,) in [0,1]."""
    x = tf.convert_to_tensor(X, dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_out, preds = gm(x)
        loss = preds[:, class_idx]
    grads = tape.gradient(loss, conv_out)              # (n, T', C)
    weights = tf.reduce_mean(grads, axis=1, keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(conv_out * weights, axis=-1))  # (n, T')
    cam = tf.reduce_mean(cam, axis=0).numpy()          # (T',)
    if cam.max() > 0:
        cam = cam / cam.max()
    T = X.shape[1]
    return np.interp(np.linspace(0, len(cam), T), np.arange(len(cam)), cam)


def lead_saliency(model, X, class_idx, batch=512):
    """mean |d score / d input| per lead channel over X -> (12,)."""
    acc = np.zeros(X.shape[2])
    n = 0
    for i in range(0, len(X), batch):
        xb = tf.convert_to_tensor(X[i:i+batch], dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(xb)
            loss = model(xb)[:, class_idx]
        g = tape.gradient(loss, xb).numpy()            # (b,700,12)
        acc += np.abs(g).mean(axis=(0, 1)) * len(xb)
        n += len(xb)
    s = acc / n
    return s / s.max() if s.max() > 0 else s


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    Xtr = np.load(DATA + "/train_X.npy"); ytr0 = np.load(DATA + "/train_y.npy")
    Xte = np.load(DATA + "/test_X.npy"); yte0 = np.load(DATA + "/test_y.npy")
    ytr = (ytr0 != 4).astype(int)        # NORM(4)->0, MI->1
    yte = (yte0 != 4).astype(int)

    model = get_model(Xtr, ytr)
    conv = last_conv(model)
    gm = grad_model_of(model, conv)

    # correctly-classified test subsets
    pred = model.predict(Xte, verbose=0).argmax(1)
    mi_idx = np.where((yte == MI) & (pred == MI))[0]
    nm_idx = np.where((yte == NORM) & (pred == NORM))[0]
    print("correct MI {} | correct NORM {}".format(len(mi_idx), len(nm_idx)), flush=True)

    cam_mi = temporal_cam(gm, Xte[mi_idx][:400], MI)
    cam_nm = temporal_cam(gm, Xte[nm_idx][:400], NORM)
    beat_mi = Xte[mi_idx][:, :, REF_LEAD].mean(0)
    beat_nm = Xte[nm_idx][:, :, REF_LEAD].mean(0)

    # --- Figure 1: temporal Grad-CAM ---
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    T = len(beat_mi)
    for a, beat, cam, title, col in [
            (ax[0], beat_mi, cam_mi, "MI", "crimson"),
            (ax[1], beat_nm, cam_nm, "NORM", "navy")]:
        a.imshow(cam[None, :], aspect="auto", cmap="jet", alpha=0.55,
                 extent=[0, T, beat.min(), beat.max()], origin="lower")
        a.plot(np.arange(T), beat, color="black", lw=1.4)
        a.set_title("Mean beat (Lead II) + Grad-CAM — {}".format(title))
        a.set_ylabel("amplitude")
    ax[1].set_xlabel("sample index (R-peak-centred beat)")
    fig.suptitle("Which beat region drives the MI vs NORM decision",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT + "/temporal_gradcam.png", dpi=150); plt.close(fig)

    # --- Figure 2: per-lead saliency ---
    sal = lead_saliency(model, Xte[mi_idx][:2000], MI)
    order = np.argsort(sal)[::-1]
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = ["crimson" if i in order[:4] else "steelblue" for i in range(len(LEADS))]
    ax.bar(LEADS, sal, color=colors)
    ax.set_ylabel("normalised saliency"); ax.set_xlabel("ECG lead")
    ax.set_title("Per-lead importance for MI detection (12-lead fusion model)\n"
                 "top-4 leads highlighted", fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT + "/lead_importance.png", dpi=150); plt.close(fig)

    # save the numbers too
    np.savetxt(OUT + "/lead_saliency.csv",
               np.array([sal]), delimiter=",", header=",".join(LEADS), comments="")
    print("Top leads for MI:", [LEADS[i] for i in order[:5]],
          "saliency", [round(float(sal[i]), 3) for i in order[:5]])
    print("Figures saved to", OUT)
