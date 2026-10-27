"""
preprocess_fusion_fix.py  (CPU / Windows)

Rebuilds the 12-lead fusion dataset with the R-peak detector run at the true
sampling rate of the PTB-XL 100 Hz records.

The original pipeline called gqrs_detect with fs=300 on signals sampled at
100 Hz. The peaks it returned are real R-peaks, but the detector's internal
timing windows were three times too long, so it accepted only a fraction of the
beats in each record. This script repeats the same preprocessing with fs=100 and
writes a parallel copy of the data, which lets the paper report how sensitive
Patient-Aggregated Fusion is to the quality of R-peak detection instead of
simply assuming it does not matter.

Everything else (wavelet denoising, amplitude normalisation, the 70/110 sample
window around the R-peak, resampling to 700 points, patient grouping) is
identical to preprocess_ptbxl_all_leads.py, so the two datasets differ in one
factor only.

Outputs:
  data_fix/{LEAD}/clean.npy        denoised, normalised signals
  data_fix/II/peaks.pkl            R-peaks detected at fs = 100
  data_fusion_fix/all_X.npy        (n, 700, 12) beats
  data_fusion_fix/all_y.npy        10-class labels
  data_fusion_fix/all_groups.npy   patient id per beat

Usage:  py -3.12 preprocess_fusion_fix.py
"""

import os
import pickle
import numpy as np
import pandas as pd
from wfdb import processing

import preprocess_ptbxl_all_leads as m

FS_TRUE = 100
LEADS = m.LEADS
REF = "II"
T0, T1, MAXLEN = m.T0, m.T1, m.MAX_LENGTH
OUT_SIG = "data_fix"
OUT_FUSE = "data_fusion_fix"


def detect_peaks_true_fs(signals):
    """R-peak detection at the true sampling rate."""
    peaks = []
    search_radius = int(FS_TRUE * 60 / 90)          # shortest accepted RR at 90 bpm
    for sig in signals:
        base = np.nan_to_num(np.asarray(sig, dtype=float),
                             nan=0.0, posinf=0.0, neginf=0.0)
        if base.size == 0 or np.ptp(base) == 0:
            peaks.append(np.array([], dtype=int))
            continue
        qrs = processing.qrs.gqrs_detect(sig=base, fs=FS_TRUE)
        if len(qrs) == 0:
            peaks.append(np.array([], dtype=int))
            continue
        peaks.append(processing.peaks.correct_peaks(
            base, peak_inds=qrs, search_radius=search_radius,
            smooth_window_size=int(0.15 * FS_TRUE)))
    return peaks


def clean_lead(signals):
    out = []
    for sig in signals:
        s = np.array(m.normalize_signal(m.denoising(sig)))
        out.append(np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0))
    return out


def main():
    os.makedirs(OUT_FUSE, exist_ok=True)
    ecg = pd.read_csv(m.PTBXL_PATH + "ptbxl_database.csv", index_col="ecg_id")
    plan = m.build_record_plan(ecg)
    print("records in plan: {}".format(len(plan)), flush=True)

    labels, groups = [], []
    for rec in plan:
        labels.extend([rec["label"]] * rec["count"])
        groups.extend([rec["patient_id"]] * rec["count"])
    labels = np.asarray(labels, dtype=int)
    groups = np.asarray(groups, dtype=float)

    # R-peaks come from the rhythm lead only; every other lead is aligned to them.
    ref_dir = os.path.join(OUT_SIG, REF)
    os.makedirs(ref_dir, exist_ok=True)
    peak_path = os.path.join(ref_dir, "peaks.pkl")
    raw_ref, _, _ = m.load_lead_signals(plan, REF)
    if os.path.exists(peak_path):
        with open(peak_path, "rb") as f:
            peaks = pickle.load(f)
        print("reusing peaks from {}".format(peak_path), flush=True)
    else:
        print("detecting R-peaks on lead {} at fs={} ...".format(REF, FS_TRUE), flush=True)
        peaks = detect_peaks_true_fs(raw_ref)
        with open(peak_path, "wb") as f:
            pickle.dump(peaks, f)
    counts = np.array([len(p) for p in peaks])
    print("beats per record: mean {:.2f} | median {} | zero-peak records {}".format(
        counts.mean(), int(np.median(counts)), int((counts == 0).sum())), flush=True)
    del raw_ref

    clean = {}
    for L in LEADS:
        path = os.path.join(OUT_SIG, L, "clean.npy")
        os.makedirs(os.path.join(OUT_SIG, L), exist_ok=True)
        if os.path.exists(path):
            clean[L] = np.load(path, allow_pickle=True)
        else:
            sigs, _, _ = m.load_lead_signals(plan, L)
            arr = np.array(clean_lead(sigs), dtype=object)
            np.save(path, arr, allow_pickle=True)
            clean[L] = arr
        print("lead {} ready".format(L), flush=True)

    n = len(labels)
    assert all(len(clean[L]) == n for L in LEADS) and len(peaks) == n, "misaligned"

    # Correct R-peak detection roughly doubles the beat count, so the output is
    # written straight into a preallocated float32 array rather than a list of
    # float64 segments, which would otherwise need tens of gigabytes.
    upper = int(sum(len(p) for p in peaks))
    X = np.empty((upper, MAXLEN, len(LEADS)), dtype=np.float32)
    y = np.empty(upper, dtype=int)
    g = np.empty(upper, dtype=float)
    grid = np.linspace(0, T0 + T1, MAXLEN)
    src = np.arange(T0 + T1)

    k = 0
    for idx in range(n):
        rows = {L: np.asarray(clean[L][idx], dtype=float) for L in LEADS} \
            if len(peaks[idx]) else None
        for pk in peaks[idx]:
            ok = True
            for c, L in enumerate(LEADS):
                s = rows[L]
                if pk - T0 < 0 or pk + T1 > len(s):
                    ok = False
                    break
                X[k, :, c] = np.interp(grid, src, s[pk - T0: pk + T1])
            if not ok:
                continue
            y[k] = labels[idx]
            g[k] = groups[idx]
            k += 1
        if (idx + 1) % 2000 == 0:
            print("  {}/{} records, {} beats".format(idx + 1, n, k), flush=True)

    X, y, g = X[:k], y[:k], g[:k]
    print("fusion beats {} | patients {} | beats per patient {:.2f}".format(
        len(X), len(np.unique(g)), len(X) / len(np.unique(g))), flush=True)
    np.save(OUT_FUSE + "/all_X.npy", X)
    np.save(OUT_FUSE + "/all_y.npy", y)
    np.save(OUT_FUSE + "/all_groups.npy", g)
    print("saved to {}/".format(OUT_FUSE))


if __name__ == "__main__":
    main()
