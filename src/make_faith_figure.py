"""
make_faith_figure.py  (CPU)

Figure for the Grad-CAM faithfulness checks, from the CSVs written by
xai_faithfulness.py: deletion curves (left, middle) and interval occlusion
(right).
Output: results/xai_faithfulness/fig_faithfulness.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "results/xai_faithfulness"
curves = pd.read_csv(D + "/deletion_curves.csv")
occl = pd.read_csv(D + "/occlusion.csv")

fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
for a, cls in zip(ax[:2], ("MI", "NORM")):
    c = curves[curves.cls == cls]
    a.plot(100 * c.frac, c.drop_gradcam, "o-", c="crimson", label="Grad-CAM order")
    a.plot(100 * c.frac, c.drop_random, "s--", c="grey", label="random order")
    a.set_xlabel("time samples deleted (%)")
    a.set_ylabel("drop in P({})".format(cls))
    a.set_title("Deletion, {} beats".format(cls))
    a.legend(fontsize=8)

names = ["baseline_PR", "QRS", "ST", "T", "late"]
labels = ["PR/base", "QRS", "ST", "T", "late"]
x = np.arange(len(names))
for off, cls, col in ((-0.2, "MI", "crimson"), (0.2, "NORM", "navy")):
    o = occl[occl.cls == cls].set_index("interval").loc[names]
    ax[2].bar(x + off, o.occlusion_drop_per_width, 0.4, color=col, label=cls)
ax[2].axhline(0, c="k", lw=0.8)
ax[2].set_xticks(x)
ax[2].set_xticklabels(labels)
ax[2].set_ylabel("probability drop per unit width")
ax[2].set_title("Interval occlusion")
ax[2].legend(fontsize=8)

fig.tight_layout()
fig.savefig(D + "/fig_faithfulness.png", dpi=200)
