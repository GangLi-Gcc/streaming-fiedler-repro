#!/usr/bin/env python3
"""Generate fig6_scale.pdf: k stability across n=500/1000/2000."""
import json, numpy as np, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
d = json.load(open(ROOT / "runs" / "scale_experiment.json"))

# Use only w=10 and w=30 (hub nodes), which are stable
valid = [r for r in d if r["k"]==r["k"] and r["k"]>0 and r["w_hub"]>0]

fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
colors = {10.0: "#2196F3", 30.0: "#FF5722"}
markers = {10.0: "o", 30.0: "s"}
labels = {10.0: r"$w=10$ (moderate stiffness)", 30.0: r"$w=30$ (high stiffness)"}

for ax_idx, (ax, what) in enumerate(zip(axes, ["k_by_n", "corr_by_n"])):
    if what == "k_by_n":
        for w in [10.0, 30.0]:
            ns, meds, q25s, q75s = [], [], [], []
            for n_per in [250, 500, 1000]:
                rows = [r for r in valid if r["n_per"]==n_per and r["w_hub"]==w]
                ks = np.array([r["k"] for r in rows])
                ns.append(2*n_per)
                meds.append(np.median(ks))
                q25s.append(np.percentile(ks, 25))
                q75s.append(np.percentile(ks, 75))
            ax.plot(ns, meds, marker=markers[w], color=colors[w], label=labels[w],
                    linewidth=1.5, markersize=6)
            ax.fill_between(ns, q25s, q75s, alpha=0.15, color=colors[w])
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks([500, 1000, 2000])
        ax.set_xticklabels(["500", "1000", "2000"])
        ax.set_xlabel("Graph size $n$")
        ax.set_ylabel(r"$k = \theta^*/\sqrt{s}$")
        ax.set_title("(a) Proportionality constant $k$ vs. scale")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
    else:
        # Spearman corr of log(theta*) vs log(sqrt_stiff) within each scale
        from scipy.stats import spearmanr
        ns, rhos = [], []
        for n_per in [250, 500, 1000]:
            rows = [r for r in valid if r["n_per"]==n_per]
            if rows:
                ts = np.log([r["theta_star"] for r in rows])
                ss = np.log([r["sqrt_stiff"] for r in rows])
                rho, _ = spearmanr(ss, ts)
                ns.append(2*n_per); rhos.append(rho)
        ax.bar(range(len(ns)), rhos, color="#607D8B", alpha=0.8)
        ax.set_xticks(range(len(ns)))
        ax.set_xticklabels([str(n) for n in ns])
        ax.set_xlabel("Graph size $n$")
        ax.set_ylabel(r"Spearman $\rho$")
        ax.set_ylim(0, 1)
        ax.set_title(r"(b) Rank correlation of $\theta^*$ vs $\sqrt{s}$")
        ax.axhline(0.7, color="gray", linestyle="--", linewidth=0.8, label="$\\rho=0.7$")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

plt.tight_layout(pad=1.0)
out = ROOT / "paper-nc" / "figures" / "fig6_scale.pdf"
plt.savefig(out, bbox_inches="tight", dpi=150)
print(f"saved: {out}")
