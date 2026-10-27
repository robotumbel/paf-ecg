"""
xai_faithfulness.py  (GPU / WSL)

Quantitative checks on the Grad-CAM explanations of the binary PAF backbone,
added for reviewer 2, point 4 (faithfulness, stability, sanity).

All analyses use correctly classified test beats of the seed-45 patient-disjoint
split, i.e. beats of patients the model never saw.

1. Faithfulness by deletion. The time samples a saliency map ranks highest are
   removed (replaced by the beat's per-lead median) in steps of 5% up to 50%,
   and the drop in the predicted-class probability is recorded. The same is done
   for randomly chosen samples. The area over the perturbation curve (AOPC) of
   Grad-CAM is compared with random deletion by a paired Wilcoxon test. A
   faithful map removes more evidence per sample than chance.
2. Interval occlusion. Each physiological interval (PR, QRS, ST, T, late) is
   replaced in turn by a straight line between its end points on every lead and
   the probability drop is recorded. Per beat, the Spearman correlation between
   Grad-CAM mass per unit width and occlusion drop per unit width across the five
   intervals is a faithfulness correlation.
3. Sanity check (model randomisation). Grad-CAM of the trained network is
   compared with Grad-CAM of the same architecture with re-initialised weights.
   Rank correlation near zero means the map depends on what the model learned.
4. Stability. Grad-CAM is recomputed after adding white noise at 30 dB and 20 dB
   SNR and compared with the clean map (Spearman). Grad-CAM is also compared with
   integrated gradients summed over leads, as a cross-method agreement check.

Outputs (results/xai_faithfulness/): deletion_curves.csv, summary.csv,
occlusion.csv, fig_deletion.png
Usage (from WSL):  ./run_gpu.sh xai_faithfulness.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from keras.layers import Conv1D
from keras.models import load_model
from scipy import stats

import grad_cam_fusion as gcf
from gradcam_quant import per_beat_cam, WINDOWS, T
from train_paf_bench import patient_split

DATA = os.environ.get("GQ_DATA", "data_fusion_fix")
MODEL = os.environ.get("GQ_MODEL", "results/external_fix/model_paf.keras")
OUT = os.environ.get("XAI_OUT", "results/xai_faithfulness")
N_BEATS = 1000            # per class
FRACS = np.arange(0.0, 0.51, 0.05)
SEED = 45
BATCH = 256

for _g in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(_g, True)
    except Exception:
        pass


def prob(model, X, cls):
    return model.predict(X, batch_size=BATCH, verbose=0)[:, cls]


def delete(X, order, frac):
    """Replace the first frac of the ranked time samples by the per-lead median."""
    Xd = X.copy()
    k = int(round(frac * T))
    if k == 0:
        return Xd
    med = np.median(X, axis=1, keepdims=True)                     # (n, 1, 12)
    rows = np.repeat(np.arange(len(X)), k)
    cols = order[:, :k].ravel()
    Xd[rows, cols, :] = np.broadcast_to(med, X.shape)[rows, cols, :]
    return Xd


def occlude(X, a, b):
    Xo = X.copy()
    a2, b2 = max(a, 1), min(b, T - 1)
    t = np.linspace(0.0, 1.0, b2 - a2 + 2)[1:-1][None, :, None]
    left, right = X[:, a2 - 1:a2, :], X[:, b2:b2 + 1, :]
    Xo[:, a2:b2, :] = left + (right - left) * t
    return Xo


def integrated_gradients(model, X, cls, steps=32):
    out = []
    for i in range(0, len(X), 64):
        x = tf.convert_to_tensor(X[i:i + 64], tf.float32)
        base = tf.zeros_like(x)
        total = tf.zeros_like(x)
        for a in np.linspace(0.0, 1.0, steps):
            xi = base + a * (x - base)
            with tf.GradientTape() as tape:
                tape.watch(xi)
                p = model(xi)[:, cls]
            total += tape.gradient(p, xi)
        ig = ((x - base) * total / steps).numpy()
        out.append(np.abs(ig).sum(axis=2))                          # (b, 700)
    return np.concatenate(out)


def rowwise_spearman(A, B):
    return np.array([stats.spearmanr(a, b).statistic for a, b in zip(A, B)])


def add_noise(X, snr_db, rng):
    p_sig = X.var(axis=1, keepdims=True)
    p_noise = p_sig / (10 ** (snr_db / 10))
    return (X + rng.normal(size=X.shape) * np.sqrt(p_noise)).astype(np.float32)


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    X = np.load(DATA + "/all_X.npy", mmap_mode="r")
    y = (np.load(DATA + "/all_y.npy") != 4).astype(int)
    groups = np.load(DATA + "/all_groups.npy")
    _, te = patient_split(y, groups, SEED, 0.15)
    te = np.sort(te)

    model = load_model(MODEL)
    conv = [l for l in model.layers if isinstance(l, Conv1D)][-1]
    gm = gcf.grad_model_of(model, conv)

    rand_model = keras.models.clone_model(model)                   # fresh weights
    rand_model.build(model.input_shape)
    gm_rand = gcf.grad_model_of(rand_model, [l for l in rand_model.layers
                                             if isinstance(l, Conv1D)][-1])

    curves, summary, occl_rows = [], [], []
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (cls, name) in zip(axes, ((1, "MI"), (0, "NORM"))):
        cand = te[y[te] == cls]
        cand = np.sort(rng.choice(cand, size=min(4 * N_BEATS, len(cand)), replace=False))
        Xc = np.asarray(X[cand], dtype=np.float32)
        keep = prob(model, Xc, cls) > 0.5
        Xc = Xc[keep][:N_BEATS]
        p0 = prob(model, Xc, cls)
        cam = per_beat_cam(gm, Xc, cls)
        print("{}: {} correctly classified beats".format(name, len(Xc)), flush=True)

        # 1. deletion curves
        order_cam = np.argsort(-cam, axis=1)
        order_rnd = np.argsort(rng.random(cam.shape), axis=1)
        drop_cam, drop_rnd = [], []
        for f in FRACS:
            dc = p0 - prob(model, delete(Xc, order_cam, f), cls)
            dr = p0 - prob(model, delete(Xc, order_rnd, f), cls)
            drop_cam.append(dc)
            drop_rnd.append(dr)
            curves.append(dict(cls=name, frac=round(f, 2), drop_gradcam=dc.mean(),
                               drop_random=dr.mean()))
        aopc_cam = np.mean(drop_cam, axis=0)
        aopc_rnd = np.mean(drop_rnd, axis=0)
        w = stats.wilcoxon(aopc_cam, aopc_rnd)

        # 2. interval occlusion and faithfulness correlation
        widths = np.array([(b - a) / T for a, b in WINDOWS.values()])
        mass = np.stack([cam[:, a:b].sum(1) for a, b in WINDOWS.values()], 1)
        mass = mass / np.maximum(mass.sum(1, keepdims=True), 1e-12)
        drops = np.stack([p0 - prob(model, occlude(Xc, a, b), cls)
                          for a, b in WINDOWS.values()], 1)
        for j, wname in enumerate(WINDOWS):
            occl_rows.append(dict(cls=name, interval=wname, width=widths[j],
                                  gradcam_concentration=mass[:, j].mean() / widths[j],
                                  occlusion_drop=drops[:, j].mean(),
                                  occlusion_drop_per_width=drops[:, j].mean() / widths[j]))
        fc = rowwise_spearman(mass / widths, drops / widths)

        # 3. sanity check and 4. stability
        cam_rand = per_beat_cam(gm_rand, Xc, cls)
        r_rand = rowwise_spearman(cam, cam_rand)
        r_30 = rowwise_spearman(cam, per_beat_cam(gm, add_noise(Xc, 30, rng), cls))
        r_20 = rowwise_spearman(cam, per_beat_cam(gm, add_noise(Xc, 20, rng), cls))
        r_ig = rowwise_spearman(cam, integrated_gradients(model, Xc, cls))

        summary.append(dict(
            cls=name, n_beats=len(Xc),
            aopc_gradcam=aopc_cam.mean(), aopc_random=aopc_rnd.mean(),
            aopc_ratio=aopc_cam.mean() / max(aopc_rnd.mean(), 1e-12),
            aopc_wilcoxon_p=w.pvalue,
            faithfulness_corr_median=np.nanmedian(fc),
            faithfulness_corr_pos_share=np.nanmean(fc > 0),
            sanity_rand_spearman_median=np.nanmedian(r_rand),
            stability_30db_median=np.nanmedian(r_30),
            stability_20db_median=np.nanmedian(r_20),
            gradcam_vs_ig_median=np.nanmedian(r_ig)))
        print(summary[-1], flush=True)

        ax.plot(100 * FRACS, np.mean(drop_cam, 1), "o-", c="crimson", label="Grad-CAM order")
        ax.plot(100 * FRACS, np.mean(drop_rnd, 1), "s--", c="grey", label="random order")
        ax.set_xlabel("time samples removed (%)")
        ax.set_ylabel("drop in P({})".format(name))
        ax.set_title(name)
        ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_deletion.png"), dpi=200)
    plt.close(fig)
    pd.DataFrame(curves).to_csv(os.path.join(OUT, "deletion_curves.csv"), index=False)
    pd.DataFrame(occl_rows).round(4).to_csv(os.path.join(OUT, "occlusion.csv"), index=False)
    s = pd.DataFrame(summary)
    s.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    print(s.T.to_string())
    print(pd.DataFrame(occl_rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
