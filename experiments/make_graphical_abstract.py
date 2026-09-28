"""Graphical abstract for the streaming-Fiedler paper (single wide figure).

Layout (left -> right):
  [P1] Problem & system : edge stream -> detection-grade tracker -> 3 channels
  [P2] The law          : MDR square-root stiffness law + mini k-band plot
  [P3] Results          : detection counts (two regimes) + G-REST comparison
Footer: takeaway sentence.
"""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper" / "figures"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})

W, H = 13.4, 6.0
fig = plt.figure(figsize=(W, H))
fig.patch.set_facecolor("white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")

C_HEAD = "#1a3550"
C_LAW = "#c0392b"
C_OK = "#27ae60"
C_BAD = "#e67e22"
C_BOX = "#f4f7fa"
C_EDGE = "#b9c6d2"


def box(x, y, w, h, title, title_color=C_HEAD, fc=C_BOX):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                fc=fc, ec=C_EDGE, lw=1.0))
    ax.text(x + 0.18, y + h - 0.34, title, fontsize=10.5, weight="bold",
            color=title_color, va="top")


def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=14, lw=1.4, color=C_HEAD))


# ---------------- header ----------------
ax.text(0.35, H - 0.32,
        "How Trackable Is the Fiedler Vector?",
        fontsize=17, weight="bold", color=C_HEAD, va="top")
ax.text(0.35, H - 0.78,
        "Coupling streaming eigenpair maintenance with change-point accessibility  "
        "(ECML-PKDD 2027 submission)",
        fontsize=9.5, color="#4a5a68", va="top")

# ---------------- P1: problem & system ----------------
x0, y0, bw, bh = 0.35, 0.62, 3.9, 4.35
box(x0, y0, bw, bh, "Detection-grade spectral tracking")
ax.text(x0 + 0.18, y0 + bh - 0.82,
        "Q: how small a rotation of $v_2$ can a\nstreaming tracker follow at all?",
        fontsize=9, color="#333", va="top")
# mini pipeline: three boxes with arrows, caption underneath
py = y0 + 2.15
for bx, lab in [(x0 + 0.18, "edge\nstream"), (x0 + 1.32, "tracker\n0.5–1.3 ms"),
                (x0 + 2.46, "$(\\lambda_2, v_2)$\n+ health")]:
    ax.add_patch(FancyBboxPatch((bx, py), 0.88, 0.6, boxstyle="round,pad=0.03",
                                fc="white", ec=C_EDGE))
    ax.text(bx + 0.44, py + 0.3, lab, fontsize=7, ha="center", va="center",
            color=C_HEAD)
arrow(x0 + 1.08, py + 0.3, x0 + 1.3, py + 0.3)
arrow(x0 + 2.22, py + 0.3, x0 + 2.44, py + 0.3)
ax.text(x0 + 1.85, py - 0.42,
        "3 channels:  $\\lambda_2$-z  ·  vec-D  ·  spike",
        fontsize=8, ha="center", color="#333")
ax.text(x0 + 0.18, y0 + 1.28,
        "• gap-relative gating locks refinement at\n   one step → blind to rotation\n"
        "• calibrated restarts → machine-precision\n   recovery after every large rotation",
        fontsize=8, color="#333", va="top")

# ---------------- P2: the law ----------------
x1 = 4.62
box(x1, y0, 4.15, bh, "The accessibility law", title_color=C_LAW)
ax.text(x1 + 0.18, y0 + bh - 0.82,
        r"MDR:  $\theta^{*} \approx k\sqrt{c-\lambda_2}$" + "\n"
        r"$k \in [6.9{\times}10^{-5},\,3.1{\times}10^{-4}]$, 43 configs" + "\n"
        "stiffness, not the spectral gap",
        fontsize=9.5, color=C_LAW, va="top")
# mini plot
px, py2, pw, ph = x1 + 0.5, y0 + 0.98, 3.25, 1.55
axp = fig.add_axes([px / W, py2 / H, pw / W, ph / H])
xs = np.logspace(-0.4, 1.6, 40)
for k in (6.9e-5, 3.1e-4):
    axp.plot(xs, k * xs, ls="--", color="#999", lw=1)
rng = np.random.default_rng(0)
sx = np.logspace(0.0, 1.35, 18)
axp.plot(sx, 1.5e-4 * sx * np.exp(rng.normal(0, 0.22, sx.size)), "o",
         ms=4, mfc="none", mec=C_HEAD)
axp.set_xscale("log"); axp.set_yscale("log")
axp.set_ylabel(r"$\theta^{*}$ (rad)", fontsize=7.5)
axp.tick_params(labelsize=6.5)
for s in ("top", "right"):
    axp.spines[s].set_visible(False)
ax.text(x1 + 2.1, y0 + 0.42,
        "tolerance-invariant (1e-4→1e-10)\nmoves as $(rtr/0.03)^{0.75}$",
        fontsize=7.5, color="#555", ha="center")

# ---------------- P3: results ----------------
x2 = 9.14
box(x2, y0, 3.9, bh, "Results", title_color=C_OK)
axb = fig.add_axes([(x2 + 0.3) / W, (y0 + 2.35) / H, 3.3 / W, 1.5 / H])
labels = ["vec-D", r"$\lambda_2$-z", "LAD"]
email = [8, 5, 3]
cmsg = [5, 0, 2]
xp = np.arange(3)
axb.bar(xp - 0.18, email, 0.34, color=C_OK, label="email-Eu (connected)")
axb.bar(xp + 0.18, cmsg, 0.34, color=C_BAD, label="CollegeMsg (quasi-disc.)")
for i, (e, c) in enumerate(zip(email, cmsg)):
    axb.text(i - 0.18, e + 0.15, f"{e}/8", ha="center", fontsize=7)
    axb.text(i + 0.18, c + 0.15, f"{c}/8", ha="center", fontsize=7)
axb.set_xticks(xp, labels, fontsize=7.5)
axb.set_ylim(0, 9.4)
axb.set_ylabel("CP det. / 8", fontsize=7)
axb.tick_params(labelsize=6.5)
axb.legend(fontsize=5.8, frameon=False, loc="upper right")
for s in ("top", "right"):
    axb.spines[s].set_visible(False)
ax.text(x2 + 0.3, y0 + 2.08, "injected change points, zero-FP calibration",
        fontsize=7.5, color="#555")
ax.text(x2 + 0.18, y0 + 1.72,
        "vs. G-REST (strongest embedding-grade tracker):",
        fontsize=8.5, weight="bold", color=C_HEAD, va="top")
ax.text(x2 + 0.18, y0 + 1.36,
        "per-step capture  WarmStart 0.998–1.00  vs.  G-REST 0.1–6.5%\n"
        "final drift            ~1e-16            vs.  7.9e-4\n"
        "cost                     3.5× slower, but detection-grade fidelity",
        fontsize=8, color="#333", va="top",
        bbox=dict(boxstyle="round,pad=0.35", fc="#fdf3ec", ec=C_BAD, lw=0.8))

# ---------------- footer takeaway ----------------
ax.text(W / 2, 0.3,
        "Takeaway:  the trackability of the Fiedler vector is governed by a square-root stiffness law — "
        "and detection needs detection-grade fidelity, not just speed.",
        fontsize=9.5, ha="center", color=C_HEAD, style="italic")

fig.savefig(OUT / "graphical_abstract.pdf", bbox_inches="tight")
fig.savefig(OUT / "graphical_abstract.png", dpi=220, bbox_inches="tight")
print("saved graphical abstract")
