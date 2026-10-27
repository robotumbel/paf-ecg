"""
gradcam_quant.py  (GPU / WSL)

Turns the Grad-CAM evidence from a qualitative picture into numbers, which is
what a reviewer needs before accepting an interpretability claim.

Two quantitative analyses on the 12-lead MI-vs-NORM fusion model:

1. Where in the beat does the saliency sit?
   Each beat is R-peak aligned, so the physiological intervals occupy fixed
   index ranges. For every correctly classified test beat the temporal Grad-CAM
   is normalised to unit mass and the mass falling inside each interval
   (baseline/PR, QRS, ST segment, T wave, late) is measured. A concentration
   ratio above one means the model attends to that interval more than a flat
   saliency map would. MI and NORM beats are compared with a Mann-Whitney U
   test and a rank-biserial effect size.

2. Does per-lead importance follow infarct territory?
   Input-gradient saliency per lead is computed separately for each PTB-XL MI
   subtype in the test set. If the model has learned something clinically real,
   inferior subtypes (IMI, ILMI) should weight II, III and aVF, and
   anteroseptal subtypes (ASMI, AMI) should weight V1-V4. The script reports the
   lead ranking per subtype and the mass falling in the clinically expected
   lead group against the mass expected by chance.

Outputs (results/gradcam_quant/):
  window_mass.csv         per-beat interval mass, both classes
  window_summary.csv      mean, sd, concentration ratio, test statistics
  lead_by_subtype.csv     per-lead saliency for every MI subtype
  territory_alignment.csv expected-lead-group mass vs chance, per subtype
  fig_window_mass.png     distribution of ST-T mass, MI vs NORM
  fig_lead_subtype.png    lead x subtype saliency heat map

Usage (from WSL):  ./run_gpu.sh gradcam_quant.py
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
from scipy import stats

import grad_cam_fusion as gcf

# Defaults reproduce the original analysis; the revision re-runs it on the
# corrected-R-peak data with GQ_DATA=data_fusion_fix,
# GQ_MODEL=results/external_fix/model_paf.keras, GQ_OUT=results/gradcam_quant_fix.
DATA = os.environ.get("GQ_DATA", "data_fusion")
OUT = os.environ.get("GQ_OUT", "results/gradcam_quant")
MODEL = os.environ.get("GQ_MODEL", "results/external/model_paf.keras")
LEADS = gcf.LEADS
MI, NORM = 1, 0
MAX_BEATS = 3000          # per class, for the window analysis
BATCH = 256

# ---------------------------------------------------------------------------
# Beat geometry. Segments were cut 70 samples before and 110 samples after the
# R-peak at 100 Hz (1.8 s) and resampled to 700 points, so the R-peak sits at
# index round(70 / 180 * 700) = 272 and one millisecond is 0.3889 points.
# ---------------------------------------------------------------------------
T = 700
R_IDX = int(round(70 / 180 * T))
PTS_PER_MS = T / 1800.0


def ms(x):
    return int(round(R_IDX + x * PTS_PER_MS))


WINDOWS = {
    "baseline_PR": (0, ms(-60)),
    "QRS": (ms(-60), ms(60)),
    "ST": (ms(60), ms(160)),
    "T": (ms(160), ms(400)),
    "late": (ms(400), T),
}
ST_T = ("ST", "T")

# PTB-XL 10-class order used throughout the project.
SUBTYPES = ["IMI", "ASMI", "ILMI", "LMI", "NORM", "AMI", "ALMI", "IPLMI", "IPMI", "PMI"]
# Leads a cardiologist would read for each infarct territory.
EXPECTED_LEADS = {
    "IMI": ["II", "III", "AVF"],
    "ILMI": ["II", "III", "AVF", "V5", "V6"],
    "IPLMI": ["II", "III", "AVF", "V5", "V6"],
    "IPMI": ["II", "III", "AVF", "V1", "V2"],
    "ASMI": ["V1", "V2", "V3", "V4"],
    "AMI": ["V1", "V2", "V3", "V4"],
    "ALMI": ["V3", "V4", "V5", "V6", "I", "AVL"],
    "LMI": ["I", "AVL", "V5", "V6"],
    "PMI": ["V1", "V2", "V3"],
}


# ---------------------------------------------------------------------------
def per_beat_cam(gm, X, class_idx, batch=BATCH):
    """Temporal Grad-CAM for every beat -> (n, 700), each row non-negative."""
    out = []
    for i in range(0, len(X), batch):
        x = tf.convert_to_tensor(X[i:i + batch], dtype=tf.float32)
        with tf.GradientTape() as tape:
            conv_out, preds = gm(x)
            loss = preds[:, class_idx]
        grads = tape.gradient(loss, conv_out)                 # (b, T', C)
        weights = tf.reduce_mean(grads, axis=1, keepdims=True)
        cam = tf.nn.relu(tf.reduce_sum(conv_out * weights, axis=-1)).numpy()  # (b, T')
        # upsample every row to the full beat length
        src = np.arange(cam.shape[1])
        dst = np.linspace(0, cam.shape[1], T)
        out.append(np.stack([np.interp(dst, src, row) for row in cam]))
    return np.concatenate(out, axis=0)


def window_mass(cams):
    """Normalised saliency mass inside each interval -> DataFrame (n, windows)."""
    total = cams.sum(axis=1, keepdims=True)
    total[total == 0] = 1.0
    norm = cams / total
    return pd.DataFrame({name: norm[:, a:b].sum(axis=1)
                         for name, (a, b) in WINDOWS.items()})


def lead_saliency_batch(model, X, class_idx, batch=BATCH):
    """Mean |d score / d input| per lead over X -> (12,), summing to one."""
    acc = np.zeros(X.shape[2])
    n = 0
    for i in range(0, len(X), batch):
        xb = tf.convert_to_tensor(X[i:i + batch], dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(xb)
            loss = model(xb)[:, class_idx]
        g = tape.gradient(loss, xb).numpy()
        acc += np.abs(g).mean(axis=(0, 1)) * len(xb)
        n += len(xb)
    s = acc / max(n, 1)
    return s / s.sum() if s.sum() > 0 else s


def rank_biserial(a, b):
    """Effect size for Mann-Whitney U, in [-1, 1]."""
    u = stats.mannwhitneyu(a, b, alternative="two-sided").statistic
    return 2.0 * u / (len(a) * len(b)) - 1.0


# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT, exist_ok=True)
    # Use the same seed-45 patient-disjoint split the trained model was built
    # on, so the beats analysed here were never seen during training.
    from train_paf_bench import patient_split
    X = np.load(DATA + "/all_X.npy", mmap_mode="r")
    y10 = np.load(DATA + "/all_y.npy")
    groups = np.load(DATA + "/all_groups.npy")
    ybin = (y10 != 4).astype(int)
    tr_all, te = patient_split(ybin, groups, 45, 0.15)
    Xte = np.asarray(X[te])
    yte10 = y10[te]
    yte = ybin[te]
    Xtr, ytr = None, None

    # Prefer the model trained by eval_external.py: it uses a patient-disjoint
    # validation split, so its early stopping was not selected on beats from the
    # training patients.
    honest = MODEL
    if os.path.exists(honest):
        from keras.models import load_model
        model = load_model(honest)
        print("using {}".format(honest), flush=True)
    else:
        raise SystemExit("run eval_external.py first to produce " + honest)
    gm = gcf.grad_model_of(model, [l for l in model.layers if isinstance(l, Conv1D)][-1])

    pred = model.predict(Xte, batch_size=BATCH, verbose=0).argmax(1)
    mi_idx = np.where((yte == MI) & (pred == MI))[0][:MAX_BEATS]
    nm_idx = np.where((yte == NORM) & (pred == NORM))[0][:MAX_BEATS]
    print("correctly classified: MI {} | NORM {}".format(len(mi_idx), len(nm_idx)),
          flush=True)
    print("windows (sample index): " + ", ".join(
        "{} {}".format(k, v) for k, v in WINDOWS.items()), flush=True)

    # --- 1. where in the beat -------------------------------------------
    mass_mi = window_mass(per_beat_cam(gm, Xte[mi_idx], MI))
    mass_nm = window_mass(per_beat_cam(gm, Xte[nm_idx], NORM))
    mass_mi["class"] = "MI"
    mass_nm["class"] = "NORM"
    allmass = pd.concat([mass_mi, mass_nm], ignore_index=True)
    allmass.to_csv(OUT + "/window_mass.csv", index=False)

    rows = []
    for name, (a, b) in WINDOWS.items():
        width = (b - a) / T
        mi_v, nm_v = mass_mi[name].values, mass_nm[name].values
        p = stats.mannwhitneyu(mi_v, nm_v, alternative="two-sided").pvalue
        rows.append(dict(
            window=name, width_fraction=round(width, 4),
            MI_mass=round(mi_v.mean(), 4), MI_sd=round(mi_v.std(ddof=1), 4),
            NORM_mass=round(nm_v.mean(), 4), NORM_sd=round(nm_v.std(ddof=1), 4),
            MI_concentration=round(mi_v.mean() / width, 3),
            NORM_concentration=round(nm_v.mean() / width, 3),
            mannwhitney_p=p, rank_biserial=round(rank_biserial(mi_v, nm_v), 3)))

    st_t_mi = mass_mi[list(ST_T)].sum(axis=1).values
    st_t_nm = mass_nm[list(ST_T)].sum(axis=1).values
    width = sum((WINDOWS[w][1] - WINDOWS[w][0]) for w in ST_T) / T
    rows.append(dict(
        window="ST+T", width_fraction=round(width, 4),
        MI_mass=round(st_t_mi.mean(), 4), MI_sd=round(st_t_mi.std(ddof=1), 4),
        NORM_mass=round(st_t_nm.mean(), 4), NORM_sd=round(st_t_nm.std(ddof=1), 4),
        MI_concentration=round(st_t_mi.mean() / width, 3),
        NORM_concentration=round(st_t_nm.mean() / width, 3),
        mannwhitney_p=stats.mannwhitneyu(st_t_mi, st_t_nm,
                                         alternative="two-sided").pvalue,
        rank_biserial=round(rank_biserial(st_t_mi, st_t_nm), 3)))
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT + "/window_summary.csv", index=False)
    print("\n=== saliency mass per interval ===")
    print(summary.to_string(index=False))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].boxplot([st_t_mi, st_t_nm], tick_labels=["MI", "NORM"], showfliers=False)
    ax[0].axhline(width, ls="--", c="grey", lw=1)
    ax[0].text(2.35, width, "chance", va="center", fontsize=8, color="grey")
    ax[0].set_ylabel("Grad-CAM mass in ST+T interval")
    ax[0].set_title("Saliency inside the ST-T interval")
    names = list(WINDOWS)
    xpos = np.arange(len(names))
    ax[1].bar(xpos - 0.2, [summary.loc[summary.window == n, "MI_concentration"].iloc[0]
                           for n in names], 0.4, label="MI", color="crimson")
    ax[1].bar(xpos + 0.2, [summary.loc[summary.window == n, "NORM_concentration"].iloc[0]
                           for n in names], 0.4, label="NORM", color="navy")
    ax[1].axhline(1.0, ls="--", c="grey", lw=1)
    ax[1].set_xticks(xpos)
    ax[1].set_xticklabels(names, rotation=20)
    ax[1].set_ylabel("concentration ratio (1 = flat saliency)")
    ax[1].set_title("Attention relative to interval width")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(OUT + "/fig_window_mass.png", dpi=150)
    plt.close(fig)

    # --- 2. lead importance per infarct subtype ---------------------------
    rows, align = [], []
    heat = []
    labels_used = []
    for c, sub in enumerate(SUBTYPES):
        if sub == "NORM":
            idx = np.where((yte10 == c) & (pred == NORM))[0]
            target = NORM
        else:
            idx = np.where((yte10 == c) & (pred == MI))[0]
            target = MI
        if len(idx) < 20:
            print("skip {} (only {} correctly classified beats)".format(sub, len(idx)))
            continue
        sal = lead_saliency_batch(model, Xte[idx][:MAX_BEATS], target)
        rows.append(dict(subtype=sub, n_beats=int(len(idx)),
                         **{L: round(float(s), 4) for L, s in zip(LEADS, sal)}))
        heat.append(sal)
        labels_used.append(sub)
        if sub in EXPECTED_LEADS:
            exp = EXPECTED_LEADS[sub]
            mass = float(sum(sal[LEADS.index(L)] for L in exp))
            chance = len(exp) / len(LEADS)
            top3 = [LEADS[i] for i in np.argsort(sal)[::-1][:3]]
            align.append(dict(subtype=sub, n_beats=int(len(idx)),
                              expected_leads=" ".join(exp),
                              expected_mass=round(mass, 4),
                              chance_mass=round(chance, 4),
                              ratio=round(mass / chance, 3),
                              top3_leads=" ".join(top3),
                              hit=int(any(L in exp for L in top3))))
    pd.DataFrame(rows).to_csv(OUT + "/lead_by_subtype.csv", index=False)
    align_df = pd.DataFrame(align)
    align_df.to_csv(OUT + "/territory_alignment.csv", index=False)
    print("\n=== per-lead saliency vs expected infarct territory ===")
    if len(align_df):
        print(align_df.to_string(index=False))
        print("\nsubtypes whose top-3 leads include an expected lead: {}/{}".format(
            int(align_df.hit.sum()), len(align_df)))

    if heat:
        H = np.array(heat)
        fig, ax = plt.subplots(figsize=(10, 0.5 * len(H) + 2.5))
        im = ax.imshow(H, aspect="auto", cmap="magma")
        ax.set_xticks(range(len(LEADS)))
        ax.set_xticklabels(LEADS)
        ax.set_yticks(range(len(labels_used)))
        ax.set_yticklabels(labels_used)
        ax.set_title("Per-lead input saliency by diagnostic class")
        fig.colorbar(im, ax=ax, label="normalised saliency")
        fig.tight_layout()
        fig.savefig(OUT + "/fig_lead_subtype.png", dpi=150)
        plt.close(fig)
    print("\nwrote results to {}/".format(OUT))


if __name__ == "__main__":
    main()
