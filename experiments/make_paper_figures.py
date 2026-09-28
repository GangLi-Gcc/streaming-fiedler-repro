"""Paper figure generation: data-driven figures from runs/ into paper/figures/.

Fig 2  att vs delta by hub stiffness (synth_mdr.json, P0h)
Fig 3  theta* vs sqrt(c-lam2) with constant-k band (p0k_kband.json, P0k)
Fig 4  email-Eu real stream: lambda2 (exact vs tracked) and tracker-trajectory
        vec-D around the real CP (day~388)  [trackq_email-eu.npz, P0o]
Fig 5  G-REST vs warmstart streaming comparison (grest_compare.json, P0p)
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 9,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "lines.linewidth": 1.2, "axes.spines.top": False, "axes.spines.right": False,
})
HUB_STYLES = {0.0: ("o-", "hub $w{=}0$"), 10.0: ("s--", "hub $w{=}10$"), 30.0: ("^:", "hub $w{=}30$")}


def fig2():
    recs = json.load(open(RUNS / "synth_mdr.json", encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for w_hub in (0.0, 10.0, 30.0):
        g = [r for r in recs if r["w_hub"] == w_hub]
        deltas = sorted({r["delta"] for r in g})
        att = [np.nanmedian([r["att"] for r in g if r["delta"] == d]) for d in deltas]
        sty, lab = HUB_STYLES[w_hub]
        ax.plot(deltas, att, sty, ms=3.5, label=lab)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.axhline(1.0, color="gray", lw=0.7, ls="-")
    ax.text(1.1e1, 1.15, "att = 1 (faithful)", fontsize=7, color="gray", ha="right")
    ax.set_xlabel(r"injected edge weight $\delta$")
    ax.set_ylabel(r"attenuation  att $= d_{\rm tr}/d_{\rm ex}$")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_att_stiffness.pdf")
    plt.close(fig)


def fig3():
    rows = json.load(open(RUNS / "p0k_kband.json", encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    xs = np.logspace(-0.4, 1.8, 60)
    for k in (6.9e-5, 3.1e-4):
        ax.plot(xs, k * xs, color="gray", lw=0.8, ls="--")
    ax.text(4.5e1, 6.9e-5 * 4.5e1 * 0.45, r"$\theta^{*}=k\sqrt{c-\lambda_2}$", fontsize=7.5,
            color="gray", rotation=21)
    ax.text(1.1, 3.1e-4 * 1.1 * 1.6, r"$k=3.1{\times}10^{-4}$", fontsize=7, color="gray", rotation=21)
    ax.text(1.1, 6.9e-5 * 1.1 * 0.55, r"$k=6.9{\times}10^{-5}$", fontsize=7, color="gray", rotation=21)
    mk = {0.0: "o", 10.0: "s", 30.0: "^"}
    lab = {0.0: "$w{=}0$", 10.0: "$w{=}10$", 30.0: "$w{=}30$"}
    pnear_vals = sorted({r["p_near"] for r in rows})
    pnear_col = {p: plt.cm.Greys(0.35 + 0.6 * i / max(1, len(pnear_vals) - 1))
                 for i, p in enumerate(pnear_vals)}
    seen = set()
    for r in rows:
        h = r["w_hub"]
        ax.plot(r["sqrt_stiff"], r["theta_star"], mk[h], ms=5.5, mfc="none",
                mec=pnear_col[r["p_near"]],
                label=lab[h] if h not in seen else None)
        seen.add(h)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"stiffness  $\sqrt{c-\lambda_2}$")
    ax.set_ylabel(r"MDR  $\theta^{*}$ (rad)")
    ax.legend(frameon=False, loc="lower right", title="hub weight", title_fontsize=7)
    ax.set_title(r"$p_{\rm near}\in\{0.005..0.3\}$ shaded light$\to$dark", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_kband.pdf")
    plt.close(fig)


def fig4():
    z = np.load(RUNS / "trackq_email-eu-tA.npz")
    lam_ex, lam_trk, v_trk, ts = z["lam_ex"], z["lam_trk"], z["v_trk"], z["ts"]
    day = (ts - ts[0]) / 86400.0
    v = np.asarray(v_trk, dtype=float)
    v = v / np.linalg.norm(v, axis=0, keepdims=True)
    for t in range(1, v.shape[1]):
        if np.dot(v[:, t], v[:, t - 1]) < 0:
            v[:, t] = -v[:, t]
    W = 10
    dvec = 1.0 - np.abs(np.einsum("ij,ij->j", v[:, W:], v[:, :-W]))
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 3.0), sharex=True)
    axes[0].plot(day, lam_ex, lw=0.9, color="C0", alpha=0.45, label="exact")
    axes[0].plot(day, lam_trk, lw=0.9, color="C0", label="tracked")
    axes[0].set_ylabel(r"$\lambda_2$")
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    axes[1].plot(day[W:], dvec, lw=0.8, color="C1")
    axes[1].set_yscale("log")
    axes[1].set_ylabel(r"vec-D  $1-\langle v_t,v_{t-10}\rangle$")
    axes[1].set_xlabel("day")
    for ax in axes:
        ax.axvline(388, color="red", lw=0.8, ls="--")
    y1 = axes[1].get_ylim()
    axes[1].annotate("real CP (day 388)", xy=(388, y1[1] * 0.4),
                     xytext=(60, y1[1] * 1e-4), fontsize=7, color="red",
                     arrowprops=dict(arrowstyle="->", color="red", lw=0.7))
    fig.tight_layout()
    fig.savefig(OUT / "fig4_real_stream.pdf")
    plt.close(fig)


def fig5():
    recs = json.load(open(RUNS / "grest_compare.json", encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5))
    # (a) per-step capture att vs delta (w_hub=0, median over seeds)
    ax = axes[0]
    for name, sty, col in [("warmstart", "o-", "C0"), ("grest_K3", "s--", "C1")]:
        g = [r for r in recs if r["w_hub"] == 0.0]
        deltas = sorted({s[0] for r in g for s in r[f"steps_{name}"]})
        att = []
        for d in deltas:
            vals = [s[3] for r in g for s in r[f"steps_{name}"] if s[0] == d and s[1] > 1e-9]
            att.append(np.nanmedian(vals) if vals else np.nan)
        ax.plot(deltas, att, sty, ms=3.5, color=col,
                label="WarmStart" if name == "warmstart" else "G-REST ($K{=}3$)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"injected edge weight $\delta$")
    ax.set_ylabel(r"per-step capture  att")
    ax.legend(frameon=False, loc="lower left")
    ax.set_title("(a) streaming capture ratio", fontsize=8)
    # (b) final drift bars
    ax = axes[1]
    names = ["warmstart", "grest_K3", "grest_K5"]
    labs = ["WarmStart", "G-REST $K{=}3$", "G-REST $K{=}5$"]
    whs = [0.0, 10.0, 30.0]
    xpos = np.arange(len(whs))
    width = 0.25
    for i, (name, lab) in enumerate(zip(names, labs)):
        vals = [np.nanmedian([r[f"drift_{name}"] for r in recs if r["w_hub"] == w]) for w in whs]
        ax.bar(xpos + (i - 1) * width, vals, width * 0.9, label=lab)
    ax.set_yscale("log")
    ax.set_ylim(1e-16, 1e-2)
    ax.set_xticks(xpos, ["$w{=}%g$" % w for w in whs])
    ax.set_ylabel(r"final drift  $1-|\langle\hat v_2,v_2\rangle|$")
    ax.set_title("(b) accumulated drift after 11 steps", fontsize=8)
    ax.legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_grest.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig2(); print("fig2 done", flush=True)
    fig3(); print("fig3 done", flush=True)
    fig4(); print("fig4 done", flush=True)
    fig5(); print("fig5 done", flush=True)
