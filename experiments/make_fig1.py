"""Fig. 1: system pipeline for the streaming-Fiedler tracker + detector.

Single-column LNCS figure (vector PDF + preview PNG).
Flow: edge stream -> shifted-Laplacian tracker (gated refinement / exact
restart) -> (lambda_2, v_2) + health -> bus -> three calibrated channels ->
fused alarm. Feedback: restart policy.
"""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper" / "figures"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})

W, H = 6.8, 4.7
fig = plt.figure(figsize=(W, H))
fig.patch.set_facecolor("white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")

C_HEAD = "#1a3550"
C_OK = "#27ae60"
C_BAD = "#e67e22"
C_LAW = "#c0392b"
C_BOX = "#f4f7fa"
C_EDGE = "#b9c6d2"


def box(x, y, w, h, fc=C_BOX, ec=C_EDGE, lw=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                fc=fc, ec=ec, lw=lw))


def arrow(x0, y0, x1, y1, color=C_HEAD, lw=1.3, style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                 mutation_scale=11, lw=lw, color=color))


def line(x0, y0, x1, y1, color=C_HEAD, lw=1.1):
    ax.plot([x0, x1], [y0, y1], color=color, lw=lw, solid_capstyle="round")


FS_T = 8.2   # box title
FS_B = 7.2   # body text

# ---------------- row 1: pipeline ----------------
y1, h1 = 3.02, 1.58

# A: edge stream
ax.text(0.95, y1 + h1 + 0.10, "online", fontsize=7, color="#777",
        ha="center", style="italic")
box(0.15, y1, 1.6, h1)
ax.text(0.95, y1 + h1 - 0.26, "Edge stream", fontsize=FS_T, weight="bold",
        color=C_HEAD, ha="center")
ax.text(0.95, y1 + h1 - 0.66, "edge add / remove\nbatches, node\narrivals",
        fontsize=FS_B, color="#333", ha="center", va="top")

# B: tracker
box(2.15, y1, 2.5, h1, fc="#eef3f8")
ax.text(3.40, y1 + h1 - 0.26, "Shifted-Laplacian tracker",
        fontsize=FS_T, weight="bold", color=C_HEAD, ha="center")
ax.text(3.40, y1 + h1 - 0.52, "iteration on $T = cI - L_t$",
        fontsize=FS_B, color="#333", ha="center", va="top")
ax.text(3.40, y1 + h1 - 0.76, "gap-relative gating",
        fontsize=6.8, color="#555", ha="center", style="italic")
# inner branch chips
box(2.30, y1 + 0.16, 1.10, 0.40, fc="white", ec=C_OK)
ax.text(2.85, y1 + 0.36, "gated\nrefinement", fontsize=6.2, color=C_OK,
        ha="center", va="center")
box(3.52, y1 + 0.16, 1.00, 0.40, fc="white", ec=C_BAD)
ax.text(4.02, y1 + 0.36, "exact\nrestart", fontsize=6.2, color=C_BAD,
        ha="center", va="center")

# C: output pair + health
box(5.05, y1, 1.62, h1)
ax.text(5.86, y1 + h1 - 0.26, "$(\\hat\\lambda_2, \\hat v_2)$\n+ health",
        fontsize=FS_T, weight="bold", color=C_HEAD, ha="center", va="top")
ax.text(5.86, y1 + 0.52, "residual · est. gap\nrotation estimate",
        fontsize=FS_B, color="#333", ha="center", va="top")

arrow(1.79, y1 + h1 / 2, 2.11, y1 + h1 / 2)
arrow(4.69, y1 + h1 / 2, 5.01, y1 + h1 / 2)

# feedback: restart policy C -> B
fb_y = y1 - 0.28
line(5.86, y1, 5.86, fb_y)
line(5.86, fb_y, 3.40, fb_y)
arrow(3.40, fb_y, 3.40, y1)
ax.text(4.63, fb_y + 0.055, "calibrated restart policy", fontsize=6.6,
        color=C_LAW, ha="center")

# ---------------- bus from C to channels ----------------
bus_y = 2.42
ch_y, ch_h = 1.10, 1.05
centers = [1.12, 3.42, 5.70]
line(5.86, fb_y, 5.86, bus_y)
line(centers[0], bus_y, 5.86, bus_y)
for cx in centers:
    arrow(cx, bus_y, cx, ch_y + ch_h)

# ---------------- row 2: three channels ----------------
ch_w = 1.95
titles = [r"$\lambda_2$-z  (value)", "vec-D  (shape)", "spike  (rate)"]
bodies = [
    "windowed $z$-score of\nthe Fiedler value\n\nlevel shifts",
    "eigenvector drift\n$D_t = 1 - |\\langle v_t, v_{t-1}\\rangle|$\n\nshape reorganizations",
    "raw per-sample\ndrift angle $\\theta_t$\n\nabrupt rotations",
]
cols = [C_OK, C_HEAD, C_BAD]
for cx, tt, bb, cc in zip(centers, titles, bodies, cols):
    box(cx - ch_w / 2, ch_y, ch_w, ch_h, fc="white", ec=cc, lw=1.1)
    ax.text(cx, ch_y + ch_h - 0.22, tt, fontsize=FS_T, weight="bold",
            color=cc, ha="center")
    ax.text(cx, ch_y + ch_h - 0.50, bb, fontsize=6.6, color="#333",
            ha="center", va="top", linespacing=1.35)

ax.text(2.27, bus_y + 0.13, "calibrated thresholds (zero-FP)",
        fontsize=6.4, color="#666", ha="center", style="italic")

# ---------------- row 3: fused alarm ----------------
al_y, al_h = 0.08, 0.62
box(2.47, al_y, 1.9, al_h, fc="#eafaf1", ec=C_OK, lw=1.2)
ax.text(3.42, al_y + al_h / 2, "fused alarm /\nchange-point call",
        fontsize=7.4, weight="bold", color=C_OK, ha="center", va="center")
for cx in centers:
    arrow(cx, ch_y, cx if abs(cx - 3.42) < 0.1 else (2.47 if cx < 3.42 else 4.37),
          al_y + al_h, lw=1.0)

fig.savefig(OUT / "fig1_pipeline.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig1_pipeline.png", dpi=220, bbox_inches="tight")
print("saved fig1")
