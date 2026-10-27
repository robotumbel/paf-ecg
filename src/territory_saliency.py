"""
territory_saliency.py  (GPU)

Repeats the per-lead saliency test on a model that was actually asked to
localise.

The binary detector studied in gradcam_quant.py showed no alignment between
per-lead importance and infarct territory, but that test was unfair: a model
that only decides MI against NORM is never rewarded for distributing its
attention by territory. The four-territory model is, so this is the setting in
which the question has a meaningful answer.

For each territory class, the input-gradient saliency of that class output is
averaged over correctly classified test beats and compared against the leads a
cardiologist reads for that wall: V1-V4 for anterior, II, III and aVF for
inferior, and I, aVL, V5 and V6 for lateral/posterior. A ratio above one means
the model concentrates on the clinically expected leads more than a uniform
distribution would.

Requires the beat-level model saved by train_territory.py.

Outputs (results/territory/):
  saliency_by_territory.csv   per-lead saliency for every class
  territory_alignment.csv     expected-lead mass against chance
  fig_territory_saliency.png  lead by class heat map
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.models import load_model

from train_paf_bench import patient_split
from train_territory import TERRITORY_MAP, CLASS_NAMES, N_CLASSES

DATA = os.environ.get("PAF_DATA", "data_fusion_fix")
OUT = os.environ.get("TERR_OUT", "results/territory")
MODEL = os.path.join(OUT, "model_beat_seed45.keras")
SEED = 45
BATCH = 256
MAX_BEATS = 3000

LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
EXPECTED = {
    "Anterior": ["V1", "V2", "V3", "V4"],
    "Inferior": ["II", "III", "AVF"],
    "LatPost": ["I", "AVL", "V5", "V6"],
}

for _g in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(_g, True)
    except Exception:
        pass


def lead_saliency(model, X, rows, class_idx, batch=BATCH):
    """Mean absolute input gradient per lead, normalised to sum to one."""
    acc = np.zeros(X.shape[2])
    n = 0
    for i in range(0, len(rows), batch):
        xb = tf.convert_to_tensor(
            np.asarray(X[np.sort(rows[i:i + batch])], dtype=np.float32))
        with tf.GradientTape() as tape:
            tape.watch(xb)
            loss = model(xb)[:, class_idx]
        g = tape.gradient(loss, xb).numpy()
        acc += np.abs(g).mean(axis=(0, 1)) * len(xb)
        n += len(xb)
    s = acc / max(n, 1)
    return s / s.sum() if s.sum() > 0 else s


def main():
    if not os.path.exists(MODEL):
        raise SystemExit("run train_territory.py first: {} is missing".format(MODEL))
    model = load_model(MODEL)

    X = np.load(DATA + "/all_X.npy", mmap_mode="r")
    y = TERRITORY_MAP[np.load(DATA + "/all_y.npy")]
    groups = np.load(DATA + "/all_groups.npy")
    _, te = patient_split(y, groups, SEED, 0.15)
    te = np.sort(te)

    proba = []
    for i in range(0, len(te), BATCH):
        proba.append(model.predict(np.asarray(X[te[i:i + BATCH]]), verbose=0))
    pred = np.concatenate(proba, axis=0).argmax(axis=1)
    print("test beats {} | correct {:.1%}".format(len(te), np.mean(pred == y[te])),
          flush=True)

    rows, align, heat, used = [], [], [], []
    for c, name in enumerate(CLASS_NAMES):
        sel = te[(y[te] == c) & (pred == c)][:MAX_BEATS]
        if len(sel) < 20:
            print("skip {} ({} correctly classified beats)".format(name, len(sel)))
            continue
        s = lead_saliency(model, X, sel, c)
        rows.append(dict(territory=name, beats=int(len(sel)),
                         **{L: round(float(v), 4) for L, v in zip(LEADS, s)}))
        heat.append(s)
        used.append(name)
        if name in EXPECTED:
            exp = EXPECTED[name]
            mass = float(sum(s[LEADS.index(L)] for L in exp))
            chance = len(exp) / len(LEADS)
            top3 = [LEADS[i] for i in np.argsort(s)[::-1][:3]]
            align.append(dict(territory=name, beats=int(len(sel)),
                              expected_leads=" ".join(exp),
                              expected_mass=round(mass, 4),
                              chance_mass=round(chance, 4),
                              ratio=round(mass / chance, 3),
                              top3_leads=" ".join(top3),
                              hit=int(any(L in exp for L in top3))))

    pd.DataFrame(rows).to_csv(os.path.join(OUT, "saliency_by_territory.csv"),
                              index=False)
    adf = pd.DataFrame(align)
    adf.to_csv(os.path.join(OUT, "territory_alignment.csv"), index=False)
    print("\n=== per-lead saliency against expected territory ===")
    if len(adf):
        print(adf.to_string(index=False))
        print("\nmean ratio {:.3f} (1.0 = chance); top-3 hits {}/{}".format(
            adf.ratio.mean(), int(adf.hit.sum()), len(adf)))

    if heat:
        H = np.array(heat)
        fig, ax = plt.subplots(figsize=(10, 0.6 * len(H) + 2.5))
        im = ax.imshow(H, aspect="auto", cmap="magma")
        ax.set_xticks(range(len(LEADS)))
        ax.set_xticklabels(LEADS)
        ax.set_yticks(range(len(used)))
        ax.set_yticklabels(used)
        for i, name in enumerate(used):
            for L in EXPECTED.get(name, []):
                ax.add_patch(plt.Rectangle((LEADS.index(L) - 0.5, i - 0.5), 1, 1,
                                           fill=False, edgecolor="cyan", lw=2))
        ax.set_title("Per-lead saliency of the four-territory model\n"
                     "(clinically expected leads outlined)", fontweight="bold")
        fig.colorbar(im, ax=ax, label="normalised saliency")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "fig_territory_saliency.png"), dpi=150)
        plt.close(fig)
    print("\nwrote results to {}/".format(OUT))


if __name__ == "__main__":
    main()
