#!/usr/bin/env python3
"""Model comparison for the MDR scaling claim (review R1, CRITICAL-2).

Question: does sqrt(stiffness) explain log(theta*) better than textbook
alternatives (gap, s, gap/s, sigma_w, and combinations)?

Protocol (pre-specified, no post-hoc selection):
  - Response: y = log(theta_star), 43 configurations.
  - Candidate models: OLS on log-transformed predictors, intercept included.
  - In-sample: R^2, adjusted R^2, AIC.
  - Out-of-sample: leave-one-config-group-out (group = (p_near, w_hub), so all
    5 seeds of a configuration are held out together -> 9 groups), report
    median |log10 ratio| prediction error.
  - Bootstrap CI (2000 resamples, stratified by group) for the coefficient on
    the chosen predictor and for the k-band.
Outputs runs/mdr_model_comparison.json
"""
import json, itertools, pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
rows = json.loads((ROOT / "runs" / "mdr_gate_theory.json").read_text())

# --- assemble design matrix -------------------------------------------------
recs = []
for r in rows:
    s, gap, sw, ts = r["stiff"], r["gap"], r["sigma_w"], r["theta_star"]
    if not (s > 0 and gap > 0 and sw > 0 and ts > 0):
        continue
    recs.append(dict(
        group=(r["p_near"], r["w_hub"]), seed=r["seed"],
        y=np.log(ts),
        log_s=np.log(s), log_gap=np.log(gap), log_sw=np.log(sw),
        log_gap_over_s=np.log(gap / s), log_sqrt_s=0.5 * np.log(s),
        log_gap_over_sqrt_s=np.log(gap / np.sqrt(s)),
    ))
n = len(recs)
groups = sorted({r["group"] for r in recs})
print(f"n = {n} configurations, {len(groups)} groups (p_near, w_hub)")

PREDICTORS = ["log_s", "log_gap", "log_sw", "log_gap_over_s", "log_gap_over_sqrt_s"]

MODELS = {
    "intercept_only":        [],
    "sqrt_s  [paper claim]": ["log_s"],          # coeff free; paper asserts 0.5
    "gap":                   ["log_gap"],
    "sigma_w":               ["log_sw"],
    "gap_over_s":            ["log_gap_over_s"],
    "gap_over_sqrt_s":       ["log_gap_over_sqrt_s"],
    "gap + s":               ["log_gap", "log_s"],
    "sigma_w + s":           ["log_sw", "log_s"],
    "gap + sigma_w":         ["log_gap", "log_sw"],
    "gap + sigma_w + s":     ["log_gap", "log_sw", "log_s"],
}


def design(rs, cols):
    X = np.ones((len(rs), 1))
    if cols:
        X = np.hstack([X, np.array([[r[c] for c in cols] for r in rs])])
    return X


def fit(rs, cols):
    X, y = design(rs, cols), np.array([r["y"] for r in rs])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    rss = float(resid @ resid)
    tss = float(((y - y.mean()) ** 2).sum())
    p = X.shape[1]
    r2 = 1 - rss / tss if tss > 0 else float("nan")
    adj = 1 - (1 - r2) * (len(rs) - 1) / (len(rs) - p) if len(rs) > p else float("nan")
    aic = len(rs) * np.log(rss / len(rs)) + 2 * p
    return beta, r2, adj, float(aic), rss


def logo_error(cols):
    """Leave-one-group-out: hold out all seeds of one configuration."""
    errs = []
    for g in groups:
        tr = [r for r in recs if r["group"] != g]
        te = [r for r in recs if r["group"] == g]
        if len(tr) <= len(cols) + 1:
            continue
        beta, *_ = fit(tr, cols)
        pred = design(te, cols) @ beta
        errs += list(np.abs(np.array([r["y"] for r in te]) - pred) / np.log(10))
    return float(np.median(errs)), float(np.percentile(errs, 90))


results = {}
for name, cols in MODELS.items():
    beta, r2, adj, aic, rss = fit(recs, cols)
    med, p90 = logo_error(cols)
    results[name] = dict(
        n_params=len(cols) + 1, coeffs=[float(b) for b in beta],
        cols=cols, r2=r2, adj_r2=adj, aic=aic,
        logo_median_abs_log10_err=med, logo_p90_abs_log10_err=p90,
    )

print(f"\n{'model':24s} {'p':>2s} {'R2':>7s} {'adjR2':>7s} {'AIC':>8s} "
      f"{'LOGO med':>9s} {'LOGO p90':>9s}")
print("-" * 76)
for name in sorted(results, key=lambda m: results[m]["aic"]):
    r = results[name]
    print(f"{name:24s} {r['n_params']:2d} {r['r2']:7.3f} {r['adj_r2']:7.3f} "
          f"{r['aic']:8.1f} {r['logo_median_abs_log10_err']:9.3f} "
          f"{r['logo_p90_abs_log10_err']:9.3f}")

# --- is the exponent on s actually 0.5? ------------------------------------
beta_s, *_ = fit(recs, ["log_s"])
rng = np.random.default_rng(0)
boot_exp, boot_k_lo, boot_k_hi = [], [], []
by_group = {g: [r for r in recs if r["group"] == g] for g in groups}
for _ in range(2000):
    samp = []
    for g in groups:                       # stratified by configuration
        pool = by_group[g]
        idx = rng.integers(0, len(pool), len(pool))
        samp += [pool[i] for i in idx]
    b, *_ = fit(samp, ["log_s"])
    boot_exp.append(b[1])
    ks = [np.exp(r["y"]) / np.sqrt(np.exp(r["log_s"])) for r in samp]
    boot_k_lo.append(min(ks)); boot_k_hi.append(max(ks))

exp_ci = (float(np.percentile(boot_exp, 2.5)), float(np.percentile(boot_exp, 97.5)))
print(f"\nfitted exponent on s: {beta_s[1]:.3f}  95% CI {exp_ci[0]:.3f}..{exp_ci[1]:.3f}"
      f"   (paper asserts 0.5)")
print(f"  -> 0.5 inside CI: {exp_ci[0] <= 0.5 <= exp_ci[1]}")

k_vals = np.array([np.exp(r["y"]) / np.sqrt(np.exp(r["log_s"])) for r in recs])
print(f"k over all {n} configs: min {k_vals.min():.3g} max {k_vals.max():.3g} "
      f"median {np.median(k_vals):.3g} band {k_vals.max()/k_vals.min():.2f}x")

out = dict(
    n=n, n_groups=len(groups), models=results,
    s_exponent=dict(point=float(beta_s[1]), ci95=exp_ci,
                    half_inside_ci=bool(exp_ci[0] <= 0.5 <= exp_ci[1])),
    k_band=dict(min=float(k_vals.min()), max=float(k_vals.max()),
                median=float(np.median(k_vals)),
                ratio=float(k_vals.max() / k_vals.min())),
)
(ROOT / "runs" / "mdr_model_comparison.json").write_text(json.dumps(out, indent=1))
print(f"\n--> runs/mdr_model_comparison.json")
