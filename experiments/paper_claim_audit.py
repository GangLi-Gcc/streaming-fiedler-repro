"""Paper claim audit: reconcile every quantitative claim in the paper
(sections 0, 4, 5) against the raw run artifacts in runs/.

Outputs a printed PASS/CHECK/FAIL table and saves runs/paper_claim_audit.json.
"""
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"

results = []  # (id, claim, computed, verdict)


def add(cid, claim, computed, ok, note=""):
    results.append({"id": cid, "claim": claim, "computed": computed,
                    "verdict": "PASS" if ok else "FAIL", "note": note})


def check(cid, claim, computed, lo, hi, note=""):
    ok = lo <= computed <= hi
    add(cid, claim, computed, ok, note)


# ---------------- A. MDR k-band (p0k_kband.json) ----------------
kb = json.load(open(RUNS / "p0k_kband.json", encoding="utf-8"))
ks = np.array([r["k"] for r in kb])
cfgs = sorted({(r["p_near"], r["w_hub"]) for r in kb})
# canonical protocol (P0k): per config, median of per-seed theta_star /
# median sqrt_stiff  (== median of per-seed k up to tiny stiff variation)
cfg_med = []
for c in cfgs:
    rs = [r for r in kb if (r["p_near"], r["w_hub"]) == c]
    cfg_med.append(float(np.median([r["theta_star"] for r in rs])
                         / np.median([r["sqrt_stiff"] for r in rs])))
cfg_med = np.array(cfg_med)
add("A1", "43 configurations", len(kb), len(kb) == 43)
check("A2", "k band over config medians [6.9e-5, 3.1e-4]",
      float(cfg_med.min()), 6.9e-5 * 0.97, 6.9e-5 * 1.03,
      f"exact min={cfg_med.min():.3e}")
check("A3", "k band max 3.1e-4", float(cfg_med.max()), 3.1e-4 * 0.97,
      3.1e-4 * 1.03, f"exact max={cfg_med.max():.3e}")
check("A4", "central value 1.5e-4", float(np.median(cfg_med)),
      1.5e-4 * 0.95, 1.5e-4 * 1.05, f"median of config medians={np.median(cfg_med):.3e}")
add("A5", "per-seed spread [3.8e-5, 7.1e-4] (not quoted in paper)",
    f"[{ks.min():.2e}, {ks.max():.2e}]",
    3.8e-5 * 0.9 <= ks.min() <= 3.8e-5 * 1.1 and 7.1e-4 * 0.9 <= ks.max() <= 7.1e-4 * 1.1,
    "paper now discloses ~2x per-seed scatter around config medians")
check("A6", "per-seed scatter claim '~2x around config median'",
      float(ks.max() / cfg_med.max()), 1.5, 2.5,
      "worst-case seed vs its config band edge")

# ---------------- B. tolerance plane (synth_tol_dep.json) ----------------
td = json.load(open(RUNS / "synth_tol_dep.json", encoding="utf-8"))


def mdr_k(rows):
    """per config: theta_star = min theta with att>=0.5; k = theta/sqrt(stiff)."""
    out = []
    groups = defaultdict(list)
    for r in rows:
        groups[(r["p_near"], r["w_hub"], r["seed"])].append(r)
    for g, rs in groups.items():
        good = [r for r in rs if r["att"] >= 0.5 and r["theta"] > 0]
        if not good:
            continue
        th = min(r["theta"] for r in good)
        stiff = rs[0]["c_minus_lam2"]
        out.append(th / math.sqrt(stiff))
    return np.array(out)


def mdr_geo(rows):
    """canonical P0k protocol: theta_star = geometric midpoint of the
    transition interval [last att<0.5, first att>=0.5]; k = theta/sqrt(stiff)."""
    out = []
    groups = defaultdict(list)
    for r in rows:
        groups[(r["p_near"], r["w_hub"], r["seed"])].append(r)
    for g, rs in groups.items():
        rs = sorted(rs, key=lambda r: r["theta"])
        lo = hi = None
        for r in rs:
            if r["att"] < 0.5:
                lo = r["theta"]
            elif r["att"] >= 0.5 and r["theta"] > 0:
                hi = r["theta"]
                break
        if lo and hi:
            out.append(math.sqrt(lo * hi) / math.sqrt(rs[0]["c_minus_lam2"]))
    return np.array(out)


# B1: refinement tolerance invariance (rtr = 0.03)
base = None
for at in [1e-4, 1e-6, 1e-8, 1e-10]:
    rows = [r for r in td if r["abs_tol"] == at and r["rtr"] == 0.03]
    ks_at = mdr_geo(rows)
    med = float(np.median(ks_at)) if len(ks_at) else float("nan")
    if base is None:
        base = med
    add(f"B1@{at:g}", f"median k at abs_tol={at:g} (rtr=0.03) ~ const",
        f"{med:.3e} (n={len(ks_at)})",
        (not math.isnan(med)) and 0.5 <= med / base <= 2.0,
        f"ratio to 1e-4 value = {med/base:.3f}" if base else "")

# B2: rtr power law medians
k_by_rtr = {}
for rtr in (0.01, 0.03, 0.1):
    rows = [r for r in td if r["rtr"] == rtr and r["abs_tol"] in (1e-6, 1e-8)]
    kk = mdr_geo(rows)
    k_by_rtr[rtr] = float(np.median(kk))
    add(f"B2@{rtr}", f"median k at rtr={rtr}", f"{k_by_rtr[rtr]:.3e}",
        True, "paper quotes 7.5e-5 / 1.9e-4 / 4.2e-4")
exp_fit = math.log(k_by_rtr[0.1] / k_by_rtr[0.01]) / math.log(10)
check("B3", "power-law exponent 0.75", exp_fit, 0.75 * 0.95, 0.75 * 1.05,
      f"fit from medians over rtr in [0.01,0.1]")

# ---------------- C. G-REST comparison (grest_compare.json) ----------------
gc = json.load(open(RUNS / "grest_compare.json", encoding="utf-8"))
for w in (0.0, 10.0, 30.0):
    rs = [r for r in gc if r["w_hub"] == w]
    ws_att = float(np.median([r["att_med_warmstart"] for r in rs]))
    ws_drift = float(np.median([abs(r["drift_warmstart"]) for r in rs]))
    ws_ms = float(np.median([r["tstep_warmstart"] for r in rs])) * 1000
    g3_att = float(np.median([r["att_med_grest_K3"] for r in rs]))
    g5_att = float(np.median([r["att_med_grest_K5"] for r in rs]))
    g3_drift = float(np.median([abs(r["drift_grest_K3"]) for r in rs]))
    g3_ms = float(np.median([r["tstep_grest_K3"] for r in rs])) * 1000
    g5_ms = float(np.median([r["tstep_grest_K5"] for r in rs])) * 1000
    nres = [r["n_restart_warmstart"] for r in rs]
    print(f"C w_hub={w}: WS att={ws_att:.3f} drift={ws_drift:.1e} "
          f"{ws_ms:.2f}ms restarts={min(nres)}-{max(nres)} | "
          f"G3 att={g3_att:.3f} drift={g3_drift:.1e} {g3_ms:.2f}ms | "
          f"G5 att={g5_att:.3f} {g5_ms:.2f}ms")
g_all = [r["att_med_grest_K3"] for r in gc] + [r["att_med_grest_K5"] for r in gc]
g3w = [r["att_med_grest_K3"] for r in gc if r["w_hub"] == 0.0]
g5w = [r["att_med_grest_K5"] for r in gc if r["w_hub"] == 0.0]
hub_med = [float(np.median([r[f"att_med_grest_{k}"] for r in gc if r["w_hub"] == w]))
           for w in (0.0, 10.0, 30.0) for k in ("K3", "K5")]
add("C1", "G-REST per-cell median capture 0.1-6.5%",
    f"table medians {min(hub_med):.4f}-{max(hub_med):.4f}",
    0.0005 <= min(hub_med) <= 0.002 and 0.05 <= max(hub_med) <= 0.08)
add("C2", "per-seed spread 0.3%-86% disclosed in paper",
    f"[{min(g_all):.4f}, {max(g_all):.4f}]",
    min(g_all) < 0.005 and max(g_all) > 0.8)
ws_drift_all = [abs(r["drift_warmstart"]) for r in gc]
add("C3", "single WS under-restart 1.4e-4 disclosed",
    f"max |drift| over 15 runs = {max(ws_drift_all):.2e}",
    max(ws_drift_all) > 1e-4)

# ---------------- D. real-stream detection counts ----------------
vd = json.load(open(RUNS / "vec_drift_real.json", encoding="utf-8"))
lad = json.load(open(RUNS / "lad_baseline.json", encoding="utf-8"))
ladc = json.load(open(RUNS / "lad_baseline-college-msg.json", encoding="utf-8"))

vd_rows = vd["rows"]
vecD_hits = [r for r in vd_rows if r["vec_D"][0]]
lam_hits = [r for r in vd_rows if r["lam"][0]]
add("D1", "email vec-D 8/8", f"{len(vecD_hits)}/8", len(vecD_hits) == 8)
add("D2", "email lam-z 5/8", f"{len(lam_hits)}/8", len(lam_hits) == 5)
d_vec = [r["vec_D"][2] for r in vecD_hits]
add("D3", "email vec-D delays 5-26", f"{min(d_vec)}-{max(d_vec)}",
    min(d_vec) == 5 and max(d_vec) == 26)
lam_miss = {r["run"] for r in vd_rows if not r["lam"][0]}
add("D4", "vec-D covers all lam misses",
    str(sorted(r["run"] for r in vecD_hits if r["run"] in lam_miss)),
    all(any(r["run"] == m and r["vec_D"][0] for r in vd_rows) for m in lam_miss),
    f"lam misses={sorted(lam_miss)}")
lad_rows = lad["rows"]
lad_hits = [r for r in lad_rows if r["LAD"][0]]
add("D5", "email LAD 3/8", f"{len(lad_hits)}/8", len(lad_hits) == 3)
# email run-name map: LAD log names -> vec_drift log names
NAME_MAP = {"inj015a": "frac0.15a", "inj015b": "frac0.15b",
            "inj030a": "frac0.30a", "inj030b": "frac0.30b",
            "inj050a": "frac0.50a", "inj050b": "frac0.50b",
            "sc06a": "scale0.6a", "sc06b": "scale0.6b"}
vd_by_run = {r["run"]: r for r in vd_rows}
lad_hits_mapped = [NAME_MAP[r["run"]] for r in lad_hits]
add("D6", "LAD detections subset of vec-D",
    str(lad_hits_mapped),
    all(vd_by_run[m]["vec_D"][0] for m in lad_hits_mapped))
d_lad = [r["LAD"][2] for r in lad_hits]
add("D7", "LAD delays 5-22", f"{min(d_lad)}-{max(d_lad)}",
    min(d_lad) == 5 and max(d_lad) == 22)

c_vecD = [r for r in ladc["rows"] if r["vecD"][0]]
c_lad = [r for r in ladc["rows"] if r["LAD"][0]]
c_lam = [r for r in ladc["rows"] if r["lam"][0]]
add("D8", "cmsg vec-D 5/8", f"{len(c_vecD)}/8", len(c_vecD) == 5)
add("D9", "cmsg LAD 2/8", f"{len(c_lad)}/8", len(c_lad) == 2)
add("D10", "cmsg lam 0/8", f"{len(c_lam)}/8", len(c_lam) == 0)
fp050b = [r for r in ladc["rows"] if r["run"] == "cmsg-inj050b"][0]
add("D11", "cmsg inj050b vec-D 10 FPs", fp050b["vecD"][1],
    fp050b["vecD"][1] == 10)

# D12: lambda-z note direction — which family does lam MISS on email?
lam_miss_fam = sorted({("scale" if "sc" in m else "ins") for m in lam_miss})
add("D12", "paper note says lam 'misses rescale-type CPs'",
    f"lam actually misses: {lam_miss_fam}; catches both sc06 runs",
    lam_miss_fam == ["ins"],
    "TABLE NOTE INVERTED — lam misses insertion-type, catches rescale")

# ---------------- E. day-388 real change point ----------------
r388 = lad["day388_rank"]
add("E1", "lam ranks day388 127th",
    f"{r388['lam']['rank_of_day388']}/{r388['lam']['series_len']}",
    r388["lam"]["rank_of_day388"] == 127)
add("E2", "vecD rank (paper: 'ranks it first')",
    f"rank {r388['vecD']['rank_of_day388']}/{r388['vecD']['series_len']}",
    r388["vecD"]["rank_of_day388"] <= 1,
    f"rank={r388['vecD']['rank_of_day388']} (0-based) -> 'first' is an overstatement; P0i says 次强")
add("E3", "LAD does not fire at day388",
    f"rank {r388['LAD']['rank_of_day388']}/{r388['LAD']['series_len']} = control max; thr = max*(1+1e-6)",
    r388["LAD"]["rank_of_day388"] == 0,
    "consistent: rank 0 = control-stream maximum, threshold is max+ulp -> no fire")

# ---------------- F. streams setup from npz ----------------
def npz_keys(p):
    z = np.load(p)
    return {k: z[k] for k in z.files}, z.files

try:
    d, keys = npz_keys(RUNS / "real_email-eu_lam.npz")
    lam = np.asarray(d[[k for k in keys if "lam" in k.lower() or k.startswith("lam")][0]]).ravel()
    lam = lam[np.isfinite(lam)]
    ss = lam[300:] if len(lam) > 300 else lam
    add("F1", "email-Eu steady-state lambda2 in [6.3,13]",
        f"all=[{lam.min():.2f},{lam.max():.2f}] s300+=[{ss.min():.2f},{ss.max():.2f}] n={len(lam)}",
        6.0 <= ss.min() <= 6.6 and 12.5 <= ss.max() <= 13.5)
except Exception as e:
    add("F1", "email-Eu lambda2 range", f"ERROR {e}", False)

try:
    d, keys = npz_keys(RUNS / "real_college-msg_lam.npz")
    lam = np.asarray(d[[k for k in keys if k.startswith("lam")][0]]).ravel()
    lam = lam[np.isfinite(lam)]
    frac0 = float(np.mean(lam < 1e-6))
    add("F2", "cmsg 98% samples lambda2~0", f"frac(lam<1e-6)={frac0:.3f}",
        0.95 <= frac0 <= 0.995, f"n={len(lam)}; threshold choice matters")
except Exception as e:
    add("F2", "cmsg lambda2~0 fraction", f"ERROR {e}", False)

# ---------------- G. derivation subsection (sec:derivation) ----------------
try:
    gt = json.load(open(RUNS / "mdr_gate_theory.json", encoding="utf-8"))
    rats = np.array([r["ratio"] for r in gt if np.isfinite(r["ratio"])])
    add("G1", "gate theory: median ratio 0.84",
        f"med={np.median(rats):.3f} n={len(rats)}",
        abs(np.median(rats) - 0.84) <= 0.02)
    add("G2", "gate theory: range [0.54,1.77]",
        f"[{np.min(rats):.2f},{np.max(rats):.2f}]",
        0.45 <= np.min(rats) <= 0.60 and 1.6 <= np.max(rats) <= 1.9)
except Exception as e:
    add("G1", "gate theory ratios", f"ERROR {e}", False)

try:
    pr = json.load(open(RUNS / "mdr_mechanism_probe.json", encoding="utf-8"))
    ref = [r for r in pr if r["branch"] == 0 and r["d_ex"] > 1e-12
           and np.isfinite(r.get("sm_theta_bar_w_over_theta2", np.nan))]
    rem = np.array([r["att"] for r in ref])
    pred = np.array([r["sm_theta_bar_w_over_theta2"] ** (2 * r["k_used"])
                     for r in ref])
    rho = float(np.corrcoef(rem, pred)[0, 1])
    add("G3", "refine capture tracks 1-kappa^2k (corr 0.975)",
        f"r={rho:.4f} n={len(ref)}", abs(rho - 0.975) <= 0.01)
except Exception as e:
    add("G3", "refine capture correlation", f"ERROR {e}", False)

try:
    ec = json.load(open(RUNS / "mdr_exponent_check.json", encoding="utf-8"))
    both = [r for r in ec if np.isfinite(r["meas_slope"])
            and np.isfinite(r["pred_slope"])]
    nf = [r for r in both if r["meas_slope"] > 0.5]
    m = float(np.median([r["meas_slope"] for r in nf]))
    p = float(np.median([r["elasticity_p"] for r in nf
                         if np.isfinite(r["elasticity_p"])]))
    add("G4", "local exponent median 0.89",
        f"med={m:.3f} n={len(nf)}", abs(m - 0.89) <= 0.02)
    add("G5", "sigma_w elasticity ~ 0.15",
        f"med={p:.3f}", abs(p - 0.15) <= 0.03)
except Exception as e:
    add("G4", "exponent reconciliation", f"ERROR {e}", False)

# ---------------- H. SCPD replay (scpd_baseline.json) ----------------
try:
    sb = json.load(open(RUNS / "scpd_baseline.json", encoding="utf-8"))
    for ds, expect in (("email-eu", 4), ("college-msg", 0)):
        rows = [r for r in sb if r["dataset"] == ds and r["tag"] != ds
                and not r["tag"].endswith("college-msg")]
        det = sum(1 for r in rows if r["detected"])
        add(f"H1-{ds}", f"SCPD detects {expect}/8 on {ds}",
            f"{det}/{len(rows)}", det == expect and len(rows) == 8)
except Exception as e:
    add("H1", "SCPD replay counts", f"ERROR {e}", False)

# ---------------- print ----------------
print(f"\n{'id':8s} {'verdict':6s} claim -> computed")
for r in results:
    print(f"{r['id']:8s} {r['verdict']:6s} {r['claim'][:60]:60s} -> {str(r['computed'])[:60]}")
    if r["note"]:
        print(f"{'':8s}        note: {r['note']}")

fails = [r for r in results if r["verdict"] == "FAIL"]
print(f"\n{len(results)} checks, {len(fails)} FAIL")
out = RUNS / "paper_claim_audit.json"
json.dump(results, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("saved", out)
