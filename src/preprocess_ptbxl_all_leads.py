"""
preprocess_ptbxl_all_leads.py

Loops the PTB-XL preprocessing pipeline from
read_data_ptbxl_12lead_fix_11112023.ipynb over ALL 12 leads, so you don't have
to set sig_name = "I", "II", ... by hand and re-run the notebook 12 times.

For every lead it reproduces the SAME outputs the notebook produced for Lead I:
  data/rev/{L}/lead {L} signal.pickle , labels.csv          (cell 7)
  data/{L}/lead {L} peak.pickle , clean signal.pickle       (cell 11)
  data/{L}/lead {L} train|test data.csv , labels.csv        (cell 14  DWT + seg, beat-level)
  data/{L}/lead {L} ... _no_dwt.csv                         (cell 16  no DWT + seg, beat-level)
  data/{L}/lead {L} ... _dwt_no_seg.pickle                  (cell 17  DWT, no seg, beat-level)
  data/{L}/lead {L} ... _patient_level.csv                  (cell 19  DWT + seg, patient-level)
  data/{L}/lead {L} ... _patient_level_dwt_no_seg.pickle    (cell 20  DWT, no seg, patient-level)
  data/{L}/lead {L} ... _patient_level_no_dwt.csv           (cell 21  no DWT + seg, patient-level)

INTEGRITY FIX (see README note printed at runtime):
  The notebook grouped "patient-level" data by ecg_id (one row per RECORDING).
  GROUP_BY_PATIENT = True groups by the real ptbxl patient_id instead, so no
  patient leaks across train/test. Set it to False to bit-reproduce the
  notebook's original (recording-level) behaviour.

NOTE on the 10x oversampling (kept as-is to match the notebook): duplicating
rare MI classes BEFORE the beat-level split inflates the beat-level numbers.
That is fine here -- the beat-level result is intentionally the "optimistic"
baseline you compare against. The patient-level path collapses duplicates by
averaging, so it is unaffected.
"""

import os
import ast
import pickle
import numpy as np
import pandas as pd

import wfdb
import pywt
from wfdb import processing
from wfdb.processing import normalize_bound
from statsmodels.robust import mad
from sklearn.model_selection import train_test_split
from collections import defaultdict

# ---------------------------------------------------------------------------
# Config (matches the notebook)
# ---------------------------------------------------------------------------
PTBXL_PATH = os.path.join(os.environ.get("PTBXL_DIR", "ptbxl/ptbxl"), "")
LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
FS = 300
T0, T1 = 70, 110
MAX_LENGTH = 700
SEED = 45
TEST_SIZE = 0.15
GROUP_BY_PATIENT = True   # True = correct patient-level; False = notebook's ecg_id behaviour

dict_label = {"IMI": 0, "ASMI": 1, "ILMI": 2, "LMI": 3, "NORM": 4,
              "AMI": 5, "ALMI": 6, "IPLMI": 7, "IPMI": 8, "PMI": 9}
LISTPENYAKIT = ["IMI", "ASMI", "ILMI", "LMI", "NORM", "AMI", "ALMI", "IPLMI", "IPMI", "PMI"]
OVERSAMPLE = {"AMI", "ALMI", "IPLMI", "IPMI", "PMI", "LMI"}   # appended 10x (cell 6)


# ---------------------------------------------------------------------------
# cells 8/9: denoising + normalization (unchanged)
# ---------------------------------------------------------------------------
def denoising(signal):
    w = pywt.Wavelet("sym5")
    coeffs, a = [], signal
    for _ in range(8):
        a, d = pywt.dwt(a, w)
        coeffs.append(d)
    coeffs.append(np.array([0] * len(a)))
    sigma = mad(coeffs[0])
    uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
    new_coeffs = [pywt.threshold(d, value=uthresh, mode="soft") for d in coeffs]
    new_coeffs.reverse()
    return pywt.waverec(new_coeffs, w)


def normalize_signal(signal):
    return normalize_bound(signal, lb=0, ub=1)


# ---------------------------------------------------------------------------
# cell 6 (lead-independent part): decide which records are kept, their label,
# oversample count, and patient id. Done ONCE for all leads.
# ---------------------------------------------------------------------------
def build_record_plan(ecg_data):
    """Return list of dicts: {ecg_id, filename, label, count, patient_id}."""
    plan = []
    for i in ecg_data.index:
        codes_str = ecg_data.scp_codes[i]
        try:
            codes = ast.literal_eval(codes_str)            # {'NORM': 100.0, ...}
        except Exception:
            continue
        patient_id = ecg_data.patient_id[i]
        filename = ecg_data.filename_lr[i]

        if "MI" in codes_str:
            present = [(k, v) for k, v in codes.items() if k in LISTPENYAKIT]
            if len(present) != 1:
                continue
            k, v = present[0]
            if int(float(v)) > 40 and k in dict_label:
                count = 10 if k in OVERSAMPLE else 1
                plan.append(dict(ecg_id=i, filename=filename,
                                 label=dict_label[k], count=count,
                                 patient_id=patient_id))
        elif "NORM" in codes_str:
            first_k = next(iter(codes))                    # cell 6 checks the FIRST entry
            first_v = codes[first_k]
            if first_k == "NORM" and int(float(first_v)) == 100:
                plan.append(dict(ecg_id=i, filename=filename,
                                 label=dict_label["NORM"], count=1,
                                 patient_id=patient_id))
    return plan


# ---------------------------------------------------------------------------
# Build per-lead arrays (signals + aligned labels/groups), honouring oversample.
# group id = patient_id (fix) or ecg_id (notebook), per GROUP_BY_PATIENT.
# ---------------------------------------------------------------------------
def load_lead_signals(plan, sig_name):
    signals, labels, groups = [], [], []
    cache_idx = None
    for rec in plan:
        record = wfdb.rdrecord(PTBXL_PATH + rec["filename"]).__dict__
        if cache_idx is None:
            cache_idx = record["sig_name"].index(sig_name)
        sig = normalize_bound(record["p_signal"][:, cache_idx], lb=0, ub=1)
        # flat/constant leads normalize to NaN (0/0) -> sanitize to a finite array
        sig = np.nan_to_num(np.asarray(sig, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        gid = rec["patient_id"] if GROUP_BY_PATIENT else rec["ecg_id"]
        for _ in range(rec["count"]):
            signals.append(sig)
            labels.append(rec["label"])
            groups.append(gid)
    return signals, np.array(labels), np.array(groups)


# ---------------------------------------------------------------------------
# cell 11 / cell 15: denoise (or not) + R-peak detect
# ---------------------------------------------------------------------------
def detect_peaks(signal_list, denoise=True):
    clean_signals, r_peaks = [], []
    search_radius = int(FS * 60 / 90)        # min_bpm=10, max_bpm=90
    for signal in signal_list:
        if denoise:
            base = np.array(normalize_signal(denoising(signal)))
        else:
            base = np.array(normalize_signal(signal))
        base = np.nan_to_num(base, nan=0.0, posinf=0.0, neginf=0.0)
        # degenerate (flat / non-finite) lead: gqrs_detect crashes on NaN -> no peaks
        if signal.size == 0 or not np.all(np.isfinite(signal)) or np.ptp(signal) == 0:
            r_peaks.append(np.array([], dtype=int))
            clean_signals.append(base)
            continue
        qrs = processing.qrs.gqrs_detect(sig=signal, fs=FS)
        if len(qrs) == 0:
            r_peaks.append(qrs)
            clean_signals.append(base)
            continue
        corrected = processing.peaks.correct_peaks(
            base, peak_inds=qrs, search_radius=search_radius, smooth_window_size=150)
        r_peaks.append(corrected)
        clean_signals.append(base)
    return clean_signals, r_peaks


# ---------------------------------------------------------------------------
# cell 14/16: segment around R-peaks, resample to MAX_LENGTH, carry group ids
# ---------------------------------------------------------------------------
def segment(clean_signals, r_peaks, labels, groups):
    seg_X, seg_y, seg_g = [], [], []
    for idx, signal in enumerate(clean_signals):
        if len(r_peaks[idx]) == 0:
            continue
        for peak in r_peaks[idx]:
            if (peak - T0) < 0 or (peak + T1) > len(signal):
                continue
            seg = signal[peak - T0: peak + T1]
            x = np.arange(0, len(seg))
            new_x = np.linspace(0, len(seg), MAX_LENGTH)
            seg_X.append(np.interp(new_x, x, seg.reshape(-1)))
            seg_y.append(labels[idx])
            seg_g.append(groups[idx])
    return np.array(seg_X), np.array(seg_y), np.array(seg_g)


def patient_aggregate(X, y, g):
    """Average all rows sharing a group id -> one row per group (cell 19/21)."""
    bucket = defaultdict(list)
    lab = {}
    for xi, yi, gi in zip(X, y, g):
        bucket[gi].append(xi)
        lab[gi] = yi
    pX, py = [], []
    for gid, rows in bucket.items():
        pX.append(np.mean(rows, axis=0))
        py.append(lab[gid])
    return np.array(pX), np.array(py)


def split(X, y):
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)


def save_csv(L, suffix, Xtr, Xte, ytr, yte):
    d = "data/{}".format(L)
    os.makedirs(d, exist_ok=True)
    np.savetxt("{}/lead {} train data{}.csv".format(d, L, suffix), Xtr)
    np.savetxt("{}/lead {} test data{}.csv".format(d, L, suffix), Xte)
    np.savetxt("{}/lead {} train labels{}.csv".format(d, L, suffix), ytr)
    np.savetxt("{}/lead {} test labels{}.csv".format(d, L, suffix), yte)


def save_pickle(L, suffix, Xtr, Xte, ytr, yte):
    d = "data/{}".format(L)
    os.makedirs(d, exist_ok=True)
    with open("{}/lead {} train data{}.pickle".format(d, L, suffix), "wb") as f:
        pickle.dump(Xtr, f)
    with open("{}/lead {} test data{}.pickle".format(d, L, suffix), "wb") as f:
        pickle.dump(Xte, f)
    np.savetxt("{}/lead {} train labels{}.csv".format(d, L, suffix), ytr)
    np.savetxt("{}/lead {} test labels{}.csv".format(d, L, suffix), yte)


# ---------------------------------------------------------------------------
# Process one lead end-to-end (cells 7-21)
# ---------------------------------------------------------------------------
def process_lead(plan, L):
    signals, labels, groups = load_lead_signals(plan, L)

    rev = "data/rev/{}".format(L)
    os.makedirs(rev, exist_ok=True)
    with open("{}/lead {} signal.pickle".format(rev, L), "wb") as f:
        pickle.dump(np.array(signals, dtype=object), f)
    np.savetxt("{}/lead {} labels.csv".format(rev, L), labels)

    # --- DWT + segmentation -------------------------------------------------
    clean, peaks = detect_peaks(signals, denoise=True)
    d = "data/{}".format(L)
    os.makedirs(d, exist_ok=True)
    with open("{}/lead {} peak.pickle".format(d, L), "wb") as f:
        pickle.dump(peaks, f)
    with open("{}/lead {} clean signal.pickle".format(d, L), "wb") as f:
        pickle.dump(clean, f)

    Xs, ys, gs = segment(clean, peaks, labels, groups)
    save_csv(L, "", *split(Xs, ys))                                  # beat-level
    pX, py = patient_aggregate(Xs, ys, gs)
    save_csv(L, "_patient_level", *split(pX, py))                    # patient-level

    # --- no DWT + segmentation ---------------------------------------------
    rclean, rpeaks = detect_peaks(signals, denoise=False)
    Xr, yr, gr = segment(rclean, rpeaks, labels, groups)
    save_csv(L, "_no_dwt", *split(Xr, yr))                           # beat-level
    pXr, pyr = patient_aggregate(Xr, yr, gr)
    save_csv(L, "_patient_level_no_dwt", *split(pXr, pyr))           # patient-level

    # --- DWT, no segmentation ----------------------------------------------
    dwt_ns, dwt_ns_y, dwt_ns_g = [], [], []
    for i, signal in enumerate(signals):
        s = np.array(normalize_signal(denoising(signal)))
        s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
        if len(s) > len(signal):
            s = s[:len(signal)]
        dwt_ns.append(s); dwt_ns_y.append(labels[i]); dwt_ns_g.append(groups[i])
    # pad/truncate to a common length so np arrays are rectangular
    common = min(len(s) for s in dwt_ns)
    dwt_ns = np.array([s[:common] for s in dwt_ns])
    dwt_ns_y = np.array(dwt_ns_y); dwt_ns_g = np.array(dwt_ns_g)
    save_pickle(L, "_dwt_no_seg", *split(dwt_ns, dwt_ns_y))          # beat/signal-level
    pXn, pyn = patient_aggregate(dwt_ns, dwt_ns_y, dwt_ns_g)
    save_pickle(L, "_patient_level_dwt_no_seg", *split(pXn, pyn))    # patient-level

    return len(Xs), len(pX)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("GROUP_BY_PATIENT =", GROUP_BY_PATIENT,
          "(True = real patient-level; False = notebook's ecg_id behaviour)")
    ecg_data = pd.read_csv(PTBXL_PATH + "ptbxl_database.csv", index_col="ecg_id")

    plan = build_record_plan(ecg_data)
    n_rec = len(plan)
    n_patients = len({r["patient_id"] for r in plan})
    print("Records kept (with oversampling): {} | unique patients: {}"
          .format(n_rec, n_patients))

    for L in LEADS:
        n_beats, n_groups = process_lead(plan, L)
        print("lead {:>3} | beat segments: {:>6} | patient rows: {:>5} | done"
              .format(L, n_beats, n_groups), flush=True)

    print("\nAll 12 leads done. Train each protocol by pointing the training "
          "script at the matching suffix (\"\", _patient_level, _no_dwt, ...).")
