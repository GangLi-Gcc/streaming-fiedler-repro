#!/usr/bin/env python3
"""Regenerate fig3_kband.pdf from the Round-24 corrected MTR definition.

MTR (theta_i) = theta at the FIRST delta-grid point where att>=0.5, per
(p_near, w_hub, seed). Seeds whose att is already >=0.5 at the smallest grid
point are left-censored (no in-grid bracket) and are excluded; only the nine
(p_near, w_hub) settings with >=1 bracketed seed are plotted. Configuration
summary = median theta_i over bracketed seeds. gap = lambda3 - lambda2 at the
smallest delta (a config property). Regression is base-10 log-log.
"""
import json, numpy as np, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
recs = json.load(open(ROOT / "rebuild" / "results" / "mdr_configs.json", encoding="utf-8"))
DELTAS = sorted({r["delta"] for r in recs})

cfg = defaultdict(list)
for r in recs:
    cfg[(r["p_near"], r["w_hub"], r["seed"])].append(r)

# per-seed theta_i (exclude left-censored seeds)
theta_i = {}
for key, lst in cfg.items():
    lst = sorted(lst, key=lambda r: r["delta"])
    if lst[0]["att"] >= 0.5:
        continue  # left-censored: att>=0.5 already at smallest rotation
    theta_i[key] = next(r["theta"] for r in lst if r["att"] >= 0.5)

# config property gap (lambda3 - lambda2) at smallest delta
gap = {}
for (pn, wh) in {(k[0], k[1]) for k in theta_i}:
    gap[(pn, wh)] = float(np.median(
        [r["gap"] for r in recs if r["p_near"] == pn and r["w_hub"] == wh
         and r["delta"] == DELTAS[0]]))

configs = sorted({(k[0], k[1]) for k in theta_i})
med = {c: float(np.median([theta_i[k] for k in theta_i if (k[0], k[1]) == c]))
       for c in configs}

thetas = np.array([med[c] for c in configs])
gaps = np.array([gap[c] for c in configs])

# regression log10(theta) ~ a*log10(gap) + b
lx, ly = np.log10(gaps), np.log10(thetas)
A = np.vstack([lx, np.ones_like(lx)]).T
a, b = np.linalg.lstsq(A, ly, rcond=None)[0]
pred = A @ [a, b]
r2 = 1 - float(((ly - pred) ** 2).sum()) / float(((ly - ly.mean()) ** 2).sum())
print(f"n bracketed configs = {len(configs)}")
print(f"regression: a={a:.4f} (expect 0.278), b={b:.4f} (expect -2.165), R2={r2:.4f} (expect 0.92)")

ks = thetas / gaps ** a
k_lo, k_hi = float(ks.min()), float(ks.max())
print(f"band k = MTR/gap^{a:.3f}: [{k_lo:.4e}, {k_hi:.4e}], ratio {k_hi/k_lo:.3f}")

# ---- two-predictor fit (stiffness + gap), bootstrap CIs, per-config table ----
stiff_cfg = {}
for (pn, wh) in {(k[0], k[1]) for k in theta_i}:
    stiff_cfg[(pn, wh)] = float(np.median(
        [r["stiff"] for r in recs if r["p_near"] == pn and r["w_hub"] == wh
         and r["delta"] == DELTAS[0]]))
stiffs = np.array([stiff_cfg[c] for c in configs])
L2 = np.vstack([np.log10(stiffs), lx, np.ones_like(lx)]).T
c2 = np.linalg.lstsq(L2, ly, rcond=None)[0]
r2_2 = 1 - float(((ly - L2 @ c2) ** 2).sum()) / float(((ly - ly.mean()) ** 2).sum())
print(f"two-predictor: log10(MTR) = {c2[0]:.4f}*log10(stiff) + {c2[1]:.4f}*log10(gap) + {c2[2]:.4f}, R2={r2_2:.4f}")

rng = np.random.default_rng(0)
nboot = 10000
def boot_ci(X):
    out = []
    for _ in range(nboot):
        idx = rng.choice(len(ly), len(ly), replace=True)
        out.append(np.linalg.lstsq(X[idx], ly[idx], rcond=None)[0])
    out = np.array(out)
    return [(float(np.percentile(out[:, j], 2.5)), float(np.percentile(out[:, j], 97.5)))
            for j in range(out.shape[1])]
A1 = np.vstack([lx, np.ones_like(lx)]).T
ci1 = boot_ci(A1)
print(f"gap-only gap coeff CI95: [{ci1[0][0]:.3f}, {ci1[0][1]:.3f}]")
ci2 = boot_ci(L2)
print(f"two-pred stiff coeff CI95: [{ci2[0][0]:.3f}, {ci2[0][1]:.3f}] | gap coeff CI95: [{ci2[1][0]:.3f}, {ci2[1][1]:.3f}]")

print("nine-config summary (p_near, w, gap, stiff, n_br, med MTR, k):")
for c in configs:
    nbr = sum(1 for k in theta_i if (k[0], k[1]) == c)
    print(f"  {c[0]:.3f} {c[1]:.1f} {gap[c]:.6e} {stiff_cfg[c]:.4f} {nbr} {med[c]:.6e} {med[c]/gap[c]**a:.6e}")

# ---- plot: x = gap^a, y = theta_i ----
fig, ax = plt.subplots(figsize=(3.4, 2.6))
xs = np.logspace(np.log10((gaps ** a).min()) - 0.15, np.log10((gaps ** a).max()) + 0.15, 60)
for k in (k_lo, k_hi):
    ax.plot(xs, k * xs, color="gray", lw=0.8, ls="--")

pnear_vals = sorted({c[0] for c in configs})
pnear_col = {p: plt.cm.Greys(0.35 + 0.6 * i / max(1, len(pnear_vals) - 1))
             for i, p in enumerate(pnear_vals)}
mk = {0.0: "o", 10.0: "s", 30.0: "^"}
lab = {0.0: "$w{=}0$", 10.0: "$w{=}10$", 30.0: "$w{=}30$"}
seen = set()
for c in configs:
    h = c[1]
    ax.plot(gap[c] ** a, med[c], mk[h], ms=5.5, mfc="none", mec=pnear_col[c[0]],
            label=lab[h] if h not in seen else None)
    seen.add(h)

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"$\Delta^{0.28}$  (spectral gap)")
ax.set_ylabel(r"MDR  $\theta^{*}$ (rad)")
ax.legend(frameon=False, loc="lower right", title="hub weight", title_fontsize=7)
ax.set_title(r"$p_{\rm near}\in\{0.005..0.3\}$ shaded light$\to$dark", fontsize=7)
fig.tight_layout()
out = ROOT / "figures" / "fig3_kband.pdf"
fig.savefig(out, bbox_inches="tight", dpi=150)
plt.close(fig)
print(f"saved: {out}")
