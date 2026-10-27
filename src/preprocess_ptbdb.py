"""
preprocess_ptbdb.py  (CPU / Windows)

Builds the external validation set from the PTB Diagnostic ECG Database, so the
model trained on PTB-XL can be tested on data from a different hospital,
recorded on different equipment two decades earlier.

The database stores 15 signals at 1000 Hz; only the standard 12 leads are used
and they are decimated to the 100 Hz of the PTB-XL training data. From there the
pipeline is exactly the one used for training: wavelet denoising, amplitude
normalisation, R-peak detection on lead II, a 70/110 sample window around each
peak resampled to 700 points, and the twelve leads stacked into one beat.

Labels come from the record headers. "Reason for admission" gives the binary
target (myocardial infarction against healthy control; every other diagnosis is
discarded) and the infarction localisation field gives a territory label, which
makes it possible to test the territory model externally as well.

Outputs (data_ptbdb/):
  all_X.npy        (n, 700, 12) beats
  all_y.npy        1 = MI, 0 = healthy control
  all_terr.npy     0 NORM, 1 Anterior, 2 Inferior, 3 Lateral/Posterior, -1 unknown
  all_groups.npy   patient number per beat
  records.csv      one row per record: patient, diagnosis, localisation, beats

Usage:  py -3.12 preprocess_ptbdb.py
"""

import os
import re
import glob
import numpy as np
import pandas as pd
import wfdb
from wfdb import processing
from wfdb.processing import normalize_bound
from scipy.signal import resample_poly

import preprocess_ptbxl_all_leads as m

ROOT = "ptbdb"
OUT = "data_ptbdb"
FS_SRC, FS_DST = 1000, 100
T0, T1, MAXLEN = m.T0, m.T1, m.MAX_LENGTH
LEADS_PTB = ["i", "ii", "iii", "avr", "avl", "avf",
             "v1", "v2", "v3", "v4", "v5", "v6"]
REF = "ii"

# Localisation strings as they appear in the PTB headers, mapped onto the same
# three territories used for PTB-XL. Several are truncated or hyphenated
# inconsistently in the source files, so every observed spelling is listed.
TERRITORY = {
    "anterior": 1, "anteroseptal": 1, "antero-septal": 1,
    "anterolateral": 1, "antero-lateral": 1,
    "antero-septo-lateral": 1, "anteroseptolateral": 1,
    "inferior": 2, "inferolateral": 2, "infero-lateral": 2, "infero-latera": 2,
    "inferoposterior": 2, "infero-posterior": 2,
    "inferoposterolateral": 2, "infero-postero-lateral": 2,
    "infero-poster-lateral": 2, "inferoposterlateral": 2,
    "lateral": 3, "posterior": 3, "posterolateral": 3, "postero-lateral": 3,
}


def parse_header(path):
    """Diagnosis and infarction localisation from the header comments."""
    reason, loc = None, None
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if not line.startswith("#"):
                continue
            low = line.lower()
            if "reason for admission" in low:
                reason = low.split(":", 1)[1].strip()
            elif "acute infarction" in low and "localization" in low:
                loc = low.split(":", 1)[1].strip()
            elif loc in (None, "no", "n/a") and "former infarction" in low \
                    and "localization" in low:
                v = low.split(":", 1)[1].strip()
                if v not in ("no", "n/a", ""):
                    loc = v
    return reason, loc


def territory_of(loc):
    if not loc:
        return -1
    key = re.sub(r"[^a-z]", "", loc)
    for name, code in TERRITORY.items():
        if re.sub(r"[^a-z]", "", name) == key:
            return code
    for name, code in TERRITORY.items():          # tolerate truncated strings
        base = re.sub(r"[^a-z]", "", name)
        if key and (base.startswith(key) or key.startswith(base)):
            return code
    return -1


def beats_from_record(path):
    """Return (n, 700, 12) beats for one record, or None if unusable."""
    # The header declares 15 signals: the 12 standard leads live in the .dat
    # file and the three Frank leads in a separate .xyz file that is not needed
    # here, so only the first twelve channels are requested.
    rec = wfdb.rdrecord(path, channels=list(range(12)))
    names = [s.lower() for s in rec.sig_name]
    if not all(L in names for L in LEADS_PTB):
        return None
    sig = rec.p_signal
    chans = []
    for L in LEADS_PTB:
        s = sig[:, names.index(L)]
        s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
        s = resample_poly(s, FS_DST, FS_SRC)
        s = normalize_bound(s, lb=0, ub=1)
        s = np.nan_to_num(np.asarray(s, dtype=float),
                          nan=0.0, posinf=0.0, neginf=0.0)
        s = np.array(m.normalize_signal(m.denoising(s)))
        chans.append(np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0))
    lengths = min(len(c) for c in chans)
    chans = [c[:lengths] for c in chans]

    ref = chans[LEADS_PTB.index(REF)]
    if np.ptp(ref) == 0:
        return None
    qrs = processing.qrs.gqrs_detect(sig=ref, fs=FS_DST)
    if len(qrs) == 0:
        return None
    peaks = processing.peaks.correct_peaks(
        ref, peak_inds=qrs, search_radius=int(FS_DST * 60 / 90),
        smooth_window_size=int(0.15 * FS_DST))

    out = []
    for pk in peaks:
        if pk - T0 < 0 or pk + T1 > lengths:
            continue
        seg = [np.interp(np.linspace(0, T0 + T1, MAXLEN), np.arange(T0 + T1),
                         c[pk - T0: pk + T1]) for c in chans]
        out.append(np.stack(seg, axis=-1))
    return np.asarray(out, dtype=np.float32) if out else None


def main():
    os.makedirs(OUT, exist_ok=True)
    headers = sorted(glob.glob(os.path.join(ROOT, "patient*", "*.hea")))
    print("headers found: {}".format(len(headers)), flush=True)

    X, y, terr, groups, rows = [], [], [], [], []
    for i, hea in enumerate(headers):
        reason, loc = parse_header(hea)
        if reason is None:
            continue
        if "myocardial infarction" in reason:
            label = 1
            t = territory_of(loc)
        elif "healthy control" in reason:
            label = 0
            t = 0
        else:
            continue                                    # other cardiac diagnoses
        base = hea[:-4]
        if not os.path.exists(base + ".dat"):
            continue                                    # signal file not retrieved
        patient = int(re.search(r"patient(\d+)", hea).group(1))
        try:
            beats = beats_from_record(base)
        except Exception as e:
            print("[skip] {}: {}".format(base, e), flush=True)
            continue
        n = 0 if beats is None else len(beats)
        rows.append(dict(record=os.path.basename(base), patient=patient,
                         diagnosis="MI" if label else "HC",
                         localization=loc or "", territory=t, beats=n))
        if n:
            X.append(beats)
            y.extend([label] * n)
            terr.extend([t] * n)
            groups.extend([patient] * n)
        if (i + 1) % 50 == 0:
            print("  {}/{} headers, {} beats so far".format(
                i + 1, len(headers), len(y)), flush=True)

    X = np.concatenate(X, axis=0)
    y = np.asarray(y, dtype=int)
    terr = np.asarray(terr, dtype=int)
    groups = np.asarray(groups, dtype=float)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "records.csv"), index=False)

    np.save(OUT + "/all_X.npy", X)
    np.save(OUT + "/all_y.npy", y)
    np.save(OUT + "/all_terr.npy", terr)
    np.save(OUT + "/all_groups.npy", groups)

    pats = pd.DataFrame(dict(patient=groups, y=y)).groupby("patient").y.max()
    print("\nbeats {} | patients {} | MI patients {} | control patients {}".format(
        len(X), len(pats), int((pats == 1).sum()), int((pats == 0).sum())))
    print("beats per patient: {:.1f}".format(len(X) / len(pats)))
    print("territory beats:", {int(k): int(v) for k, v in
                               zip(*np.unique(terr, return_counts=True))})
    print("saved to {}/".format(OUT))


if __name__ == "__main__":
    main()
