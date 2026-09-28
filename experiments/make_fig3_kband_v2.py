#!/usr/bin/env python3
"""Regenerate fig3_kband.pdf from the Round-24 corrected MTR definition.

MTR (theta_i) = theta at the FIRST delta-grid point where att>=0.5, per
(p_near, w_hub, seed). Seeds whose att is already >=0.5 at the smallest grid
point are left-censored (no in-grid bracket) and are excluded; only the nine
(p_near, w_hub) settings with >=1 bracketed seed are plotted.

Aggregation (Round 27): every configuration summary is a median over the SAME
bracketed (uncensored) seed subset --- the per-seed first-crossing theta_i
(MTR), and the per-seed values of gap = lambda3 - lambda2, stiffness
s = c - lambda2, and the tracker-estimated gap gap_est = lambda3^tracked - q,
all read at the smallest delta. Regression is base-10 log-log. Bootstrap 95%
intervals are seed-level: resample the bracketed seeds of each configuration
with replacement (preserving per-seed (MTR, gap, stiff, gap_est) tuples),
recompute all nine medians, refit OLS; 10^4 draws, rng = default_rng(0).
"""
import json, numpy as np, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
recs = json.load(open(ROOT / "rebuild" / "results" / "mdr_configs.json", encoding="utf-8"))
aud = json.load(open(ROOT / "rebuild" / "results" / "gate_audit.json", encoding="utf-8"))
gmap = {(a["p_near"], a["w_hub"], a["seed"], a["delta"]): a["gap_est"] for a in aud}
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

configs = sorted({(k[0], k[1]) for k in theta_i})

# per-seed config properties at the smallest delta (bracketed seeds only)
def prop(key, field):
    for r in cfg[key]:
        if r["delta"] == DELTAS[0]:
            return r[field]

def gest(key):
    return gmap[(key[0], key[1], key[2], DELTAS[0])]

def cfg_med(fn):
    return {c: float(np.median([fn(k) for k in theta_i if (k[0], k[1]) == c]))
            for c in configs}

gap = cfg_med(lambda k: prop(k, "gap"))
stiff = cfg_med(lambda k: prop(k, "stiff"))
gest_m = cfg_med(gest)
med = cfg_med(lambda k: theta_i[k])
nbr = {c: sum(1 for k in theta_i if (k[0], k[1]) == c) for c in configs}

# ---- archive the 43 per-seed bootstrap inputs (tab:seedinputs / mdr_seed_inputs.json) ----
_seed_rows = []
for key in sorted(theta_i):
    pn, wh, sd = key
    _seed_rows.append({
        "p_near": pn, "w_hub": wh, "seed": sd,
        "theta_i": theta_i[key],
        "gap_i": prop(key, "gap"),
        "stiff_i": prop(key, "stiff"),
    })
_seed_out = {
    "note": "Seed-level bootstrap inputs: the 43 bracketed seeds, each with its "
            "per-seed first in-grid att>=0.5 crossing angle theta_i, and the "
            "per-seed gap and stiffness read at the smallest delta. The nine "
            "configuration medians of tab:mtrsum and the seed-level bootstrap of "
            "eq:mdr/eq:mdr2 (resample these seeds within each setting with "
            "replacement, preserving (theta_i,gap_i,stiff_i) tuples, recompute the "
            "nine medians, refit OLS; 10^4 draws, 2.5/97.5 percentiles, "
            "default_rng(0)) are computed from exactly these tuples.",
    "n_bracketed_seeds": len(_seed_rows),
    "rows": _seed_rows,
}
_seed_path = ROOT / "rebuild" / "results" / "mdr_seed_inputs.json"
json.dump(_seed_out, open(_seed_path, "w", encoding="utf-8"), indent=1)
print(f"wrote {_seed_path} with {len(_seed_rows)} rows")

def ols(x):
    lx, ly = np.log10(np.array([x[c] for c in configs])), np.log10(np.array([med[c] for c in configs]))
    A = np.vstack([lx, np.ones_like(lx)]).T
    a, b = np.linalg.lstsq(A, ly, rcond=None)[0]
    r2 = 1 - float(((ly - A @ [a, b]) ** 2).sum()) / float(((ly - ly.mean()) ** 2).sum())
    return a, b, r2

def ols2(x1, x2):
    lx1 = np.log10(np.array([x1[c] for c in configs]))
    lx2 = np.log10(np.array([x2[c] for c in configs]))
    ly = np.log10(np.array([med[c] for c in configs]))
    A = np.vstack([lx1, lx2, np.ones_like(lx1)]).T
    c = np.linalg.lstsq(A, ly, rcond=None)[0]
    r2 = 1 - float(((ly - A @ c) ** 2).sum()) / float(((ly - ly.mean()) ** 2).sum())
    return c, r2

a, b, r2 = ols(gap)
a_est, b_est, r2_est = ols(gest_m)
c2, r2_2 = ols2(stiff, gap)
print(f"n bracketed configs = {len(configs)}, n bracketed seeds = {sum(nbr.values())}")
print(f"gap-only (exact gap):  a={a:.6f} b={b:.6f} R2={r2:.6f}")
print(f"gap-only (gap_est):    a={a_est:.6f} b={b_est:.6f} R2={r2_est:.6f}")
print(f"two-pred (stiff+gap):  c_stiff={c2[0]:.6f} c_gap={c2[1]:.6f} intercept={c2[2]:.6f} R2={r2_2:.6f}")

# ---- seed-level bootstrap (resample bracketed seeds per config, tuple-preserving) ----
rng = np.random.default_rng(0)
nboot = 10000
bracketed = {c: [k for k in theta_i if (k[0], k[1]) == c] for c in configs}

def draw_medians():
    mb, gb, sb, eb = {}, {}, {}, {}
    for c in configs:
        keys = bracketed[c]
        idx = rng.integers(0, len(keys), len(keys))
        res = [keys[i] for i in idx]
        mb[c] = float(np.median([theta_i[k] for k in res]))
        gb[c] = float(np.median([prop(k, "gap") for k in res]))
        sb[c] = float(np.median([prop(k, "stiff") for k in res]))
        eb[c] = float(np.median([gest(k) for k in res]))
    return mb, gb, sb, eb

def boot():
    out = []
    for _ in range(nboot):
        mb, gb, sb, eb = draw_medians()
        aa = ols_on(gb, mb)[0]
        cc = ols2_on(sb, gb, mb)
        out.append([aa, cc[0], cc[1]])
    return np.array(out)

def ols_on(x, y):
    lx, ly = np.log10(np.array([x[c] for c in configs])), np.log10(np.array([y[c] for c in configs]))
    A = np.vstack([lx, np.ones_like(lx)]).T
    return np.linalg.lstsq(A, ly, rcond=None)[0]

def ols2_on(x1, x2, y):
    lx1 = np.log10(np.array([x1[c] for c in configs]))
    lx2 = np.log10(np.array([x2[c] for c in configs]))
    ly = np.log10(np.array([y[c] for c in configs]))
    A = np.vstack([lx1, lx2, np.ones_like(lx1)]).T
    return np.linalg.lstsq(A, ly, rcond=None)[0]

B = boot()
ci = lambda j: (float(np.percentile(B[:, j], 2.5)), float(np.percentile(B[:, j], 97.5)))
print(f"gap-only gap coeff CI95 (seed-level):   [{ci(0)[0]:.3f}, {ci(0)[1]:.3f}]")
print(f"two-pred stiff coeff CI95 (seed-level): [{ci(1)[0]:.3f}, {ci(1)[1]:.3f}]")
print(f"two-pred gap coeff CI95 (seed-level):   [{ci(2)[0]:.3f}, {ci(2)[1]:.3f}]")

ks = np.array([med[c] / gap[c] ** a for c in configs])
ke = np.array([med[c] / gest_m[c] ** a_est for c in configs])
print(f"band k (exact gap) = MTR/gap^{a:.4f}:  [{ks.min():.6e}, {ks.max():.6e}], ratio {ks.max()/ks.min():.3f}")
print(f"band k (gap_est)   = MTR/gap_est^{a_est:.4f}: [{ke.min():.6e}, {ke.max():.6e}], ratio {ke.max()/ke.min():.3f}")
print(f"max rel diff gap_est vs exact gap at smallest delta: {max(abs(gest_m[c]-gap[c])/gap[c] for c in configs):.2e}")

print("\nnine-config summary (p_near, w, gap, gap_est, stiff, n_br, MTR, k_exact, k_est):")
for c in configs:
    print(f"  {c[0]:.6g} {c[1]:.6g} {gap[c]:.8g} {gest_m[c]:.8g} {stiff[c]:.8g} {nbr[c]} "
          f"{med[c]:.8g} {med[c]/gap[c]**a:.8g} {med[c]/gest_m[c]**a_est:.8g}")

# ---- plot: x = gap^a, y = theta_i ----
fig, ax = plt.subplots(figsize=(3.4, 2.6))
xs = np.logspace(np.log10((np.array([gap[c] for c in configs]) ** a).min()) - 0.15,
                 np.log10((np.array([gap[c] for c in configs]) ** a).max()) + 0.15, 60)
for k in (ks.min(), ks.max()):
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
