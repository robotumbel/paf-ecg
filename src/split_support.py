"""
Per-class patient support of every split used in the paper (reviewer 2, point 9).

Recomputes the patient-disjoint splits exactly as train_paf_bench.py and
train_territory.py draw them (StratifiedGroupKFold, seeds 45-49, test and
validation fraction 0.15) and the official PTB-XL folds (train 1-8, validation
9, test 10), and counts patients and beats per class in each partition. A
patient's label is the majority label of its beats, as in the aggregation step.

Runs on the CPU; TensorFlow is not needed.
Outputs: results/split_support/split_support.csv and, per task and split,
patients_{task}_{split}.csv listing every patient_id with its partition.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit

DATA = os.environ.get("PAF_DATA", "data_fusion_fix")
OUT = "results/split_support"
SEEDS = [45, 46, 47, 48, 49]
FRACTION = 0.15
NORM_LABEL = 4
TERRITORY_MAP = np.array([2, 1, 2, 3, 0, 1, 1, 2, 2, 3])
TASKS = {"binary": ["NORM", "MI"],
         "territory": ["NORM", "Anterior", "Inferior", "LatPost"]}


def patient_split(y, groups, seed, fraction):
    # identical to train_paf_bench.patient_split
    try:
        k = max(2, round(1.0 / fraction))
        a, b = next(StratifiedGroupKFold(n_splits=k, shuffle=True,
                                         random_state=seed).split(y, y, groups))
    except Exception:
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=fraction,
                                      random_state=seed).split(y, y, groups))
    assert not (set(groups[a]) & set(groups[b])), "patient leak"
    return a, b


def random_splits(y, groups, seed):
    tr_all, te = patient_split(y, groups, seed, FRACTION)
    tr, va = patient_split(y[tr_all], groups[tr_all], seed, FRACTION)
    return tr_all[tr], tr_all[va], te


def official_splits(groups):
    db = pd.read_csv("ptbxl/ptbxl/ptbxl_database.csv",
                     usecols=["patient_id", "strat_fold"])
    fold = db.groupby("patient_id").strat_fold.first().reindex(groups).to_numpy()
    return (np.where(fold <= 8)[0], np.where(fold == 9)[0],
            np.where(fold == 10)[0])


def count(y, groups, idx, names):
    pats = pd.DataFrame({"p": groups[idx], "y": y[idx]})
    lab = pats.groupby("p").y.agg(lambda v: np.bincount(v, minlength=len(names)).argmax())
    row = {}
    for c, n in enumerate(names):
        row["patients_" + n] = int((lab == c).sum())
        row["beats_" + n] = int((y[idx] == c).sum())
    row["patients_total"] = int(len(lab))
    row["beats_total"] = int(len(idx))
    return row


def main():
    os.makedirs(OUT, exist_ok=True)
    y10 = np.load(DATA + "/all_y.npy").astype(int)
    groups = np.load(DATA + "/all_groups.npy")
    labels = {"binary": (y10 != NORM_LABEL).astype(int),
              "territory": TERRITORY_MAP[y10]}

    rows = []
    for task, names in TASKS.items():
        y = labels[task]
        plans = [("seed{}".format(s), random_splits(y, groups, s)) for s in SEEDS]
        plans.append(("official", official_splits(groups)))
        for split, parts in plans:
            ids = pd.concat([pd.DataFrame({"patient_id": np.unique(groups[idx]).astype(int),
                                           "partition": part})
                             for part, idx in zip(("train", "val", "test"), parts)])
            ids.to_csv(os.path.join(OUT, "patients_{}_{}.csv".format(task, split)),
                       index=False)
            for part, idx in zip(("train", "val", "test"), parts):
                r = dict(task=task, split=split, partition=part)
                r.update(count(y, groups, idx, names))
                rows.append(r)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "split_support.csv"), index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(df.drop(columns=[c for c in df.columns if c.startswith("beats_")
                               and c != "beats_total"]).to_string(index=False))


if __name__ == "__main__":
    main()
