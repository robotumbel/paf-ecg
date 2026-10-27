"""
make_arch_figure.py
Complete end-to-end methodology/architecture diagram of the proposed PAF method:
PTB-XL -> preprocessing (normalize, DWT denoise, R-peak, segment+align)
-> patient-disjoint split -> CNN backbone (per beat, with layer expansion)
-> PAF aggregation -> patient decision.
Pure matplotlib. Output: results/paper_figures/fig_architecture.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

OUT = "results/paper_figures"
os.makedirs(OUT, exist_ok=True)

GREEN = "#cdeac0"; ORANGE = "#fbe0c3"; BLUE = "#cbdff5"; TEAL = "#c4e7e6"
GREY = "#e6e6e6"; EDGE = "#3a3a3a"

fig, ax = plt.subplots(figsize=(17, 5.4))

def box(x, y, w, h, text, fc, fs=11, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.5, edgecolor=EDGE, facecolor=fc))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal", color="#101010")
    return (x, y, w, h)

def arrow(x1, y1, x2, y2, color=EDGE, lw=1.8):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
        mutation_scale=20, lw=lw, color=color))

# ---------- top row: full pipeline ----------
yT, hT = 3.3, 1.4
top = []
x = 0.2
defs = [
    ("PTB-XL\n12-lead ECG", GREEN, 2.0, True),
    ("Preprocessing\n• Normalize\n• DWT denoising (sym5)\n• R-peak detection\n• Segment + 12-lead align\n$\\rightarrow$ 700$\\times$12 beats", GREY, 3.3, False),
    ("Patient-disjoint\nsplit\n(by patient ID)", BLUE, 2.2, True),
    ("CNN backbone\n(per beat)", ORANGE, 2.2, True),
    ("PAF pooling\nmean posterior / vote\nor gated attention\n(bag of 20 beats)", TEAL, 2.4, True),
    ("Patient decision\n(MI / NORM)", GREEN, 2.0, True),
]
for t, c, w, b in defs:
    fs = 9.5 if "Preprocessing" in t else 11
    top.append(box(x, yT, w, hT, t, c, fs=fs, bold=b))
    x += w + 0.55
for i in range(len(top) - 1):
    arrow(top[i][0] + top[i][2], yT + hT / 2, top[i + 1][0], yT + hT / 2)

# ---------- expansion of the CNN backbone ----------
cnn = top[3]
cnn_cx = cnn[0] + cnn[2] / 2
yB, hB, wB, gapB = 0.55, 1.15, 1.85, 0.35
sub_defs = [
    ("Conv1D-128\nk=11\nMaxPool", ORANGE),
    ("Conv1D-128\nk=13\nMaxPool", ORANGE),
    ("Conv1D-128\nk=7", ORANGE),
    ("Flatten\nDense-128\nSoftmax", BLUE),
]
total_w = len(sub_defs) * wB + (len(sub_defs) - 1) * gapB
xB = cnn_cx - total_w / 2
subs = []
xx = xB
for t, c in sub_defs:
    subs.append(box(xx, yB, wB, hB, t, c, fs=9.5))
    xx += wB + gapB
for i in range(len(subs) - 1):
    arrow(subs[i][0] + subs[i][2], yB + hB / 2, subs[i + 1][0], yB + hB / 2, lw=1.5)
# dashed frame + label
ax.add_patch(Rectangle((xB - 0.25, yB - 0.3), total_w + 0.5, hB + 0.6,
    fill=False, linestyle=(0, (6, 4)), linewidth=1.5, edgecolor="#4a4a4a"))
ax.text(cnn_cx, yB + hB + 0.45, "CNN backbone (applied per beat)",
        ha="center", fontsize=10.5, fontweight="bold", color="#2a2a2a")
# dotted "zoom" lines from the CNN box to the expansion frame
ax.plot([cnn[0], xB - 0.25], [cnn[1], yB + hB + 0.3], ls=":", color="#888888", lw=1.2)
ax.plot([cnn[0] + cnn[2], xB + total_w + 0.25], [cnn[1], yB + hB + 0.3],
        ls=":", color="#888888", lw=1.2)

ax.set_xlim(0, top[-1][0] + top[-1][2] + 0.3)
ax.set_ylim(0.1, 5.2)
ax.axis("off")
fig.tight_layout()
fig.savefig(OUT + "/fig_architecture.png", dpi=200, bbox_inches="tight")
print("saved", OUT + "/fig_architecture.png")
