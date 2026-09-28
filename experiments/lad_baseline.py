"""P0i: LAD (Huang et al., KDD 2019) baseline vs lambda2-z vs vec-D.

Replays the IDENTICAL stream builder as real_data_noise.py (email-Eu core-300,
tau=2000 steps, macro=10, prune 0.02, injection at global sample 400 with the
same seeds as the saved P0 runs) and stores per-sample:
  lam  : lambda_2 (shift-invert eigsh, same as BaselineFull)
  v2   : Fiedler vector
  spec : top-6 smallest Laplacian eigenvalues  == LAD's signature vector
All three detection channels are then computed on the SAME replayed batch:
  LAD  : dual sliding windows (s=5, l=10), normal behavior = 1st principal
         component direction of window signatures, Z = cosine distance,
         Z* = positive increase of Z (LAD's final score)
  lam  : window-z on lambda_2(t)          (P0b channel)
  vec-D: window-z on cumulative drift D=1-|<v_t,v_{t-10}>|  (P0c channel)
Calibration: strict zero-alarm budget on the replayed control segment +
1e-6 relative margin (same protocol as P0b/P0c/P0d), so the comparison is
threshold-fair. Evaluation: det if any alarm in [cp, cp+40], fp = alarms
before cp, delay = first alarm - cp.

Prediction checked here: LAD's cosine-distance score is scale-invariant in
signature space, so uniform weight-scaling CPs (sc06a/b) should be invisible
to LAD, while lambda2-z sees them. Deletion CPs (inj*) change the eigenvalue
RAY direction -> LAD should fire.

Outputs: ../runs/real_{tag}_spec.npz (lam, v2, spec, ts) per run,
../runs/lad_baseline.json (calibration, per-run table, channel totals).
"""
from __future__ import annotations

import gzip
import json
import os
import sys
import time
import warnings
from collections import Counter

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TOL = 40
L_CUM = 10
TOPK = 6
# Per-dataset layout. Injection point mirrors the email-Eu layout (100
# calibration samples before CP, >=50 after): email-Eu G=400 of 466;
# college-msg G=120 of 177.
DATASETS = {
    "email-eu": {
        "path": "../data/email-Eu-core-temporal.txt.gz",
        "tau_steps": 2000, "macro": 10, "prune_w": 0.02, "core_k": 300,
        "stationary0": 300, "inject_at": 400, "saved_control": "email-eu",
        "runs": {
            "email-eu": None,
            "inj015a": (0.15, None, 1), "inj015b": (0.15, None, 2),
            "inj030a": (0.30, None, 1), "inj030b": (0.30, None, 2),
            "inj050a": (0.50, None, 1), "inj050b": (0.50, None, 2),
            "sc06a": (0.30, 0.6, 1), "sc06b": (0.30, 0.6, 2),
        },
    },
    "college-msg": {
        "path": "../data/CollegeMsg.txt.gz",
        "tau_steps": 1500, "macro": 5, "prune_w": 0.02, "core_k": 500,
        "stationary0": 20, "inject_at": 120, "saved_control": "college-msg",
        "runs": {
            "college-msg": None,
            "cmsg-inj015a": (0.15, None, 1), "cmsg-inj015b": (0.15, None, 2),
            "cmsg-inj030a": (0.30, None, 1), "cmsg-inj030b": (0.30, None, 2),
            "cmsg-inj050a": (0.50, None, 1), "cmsg-inj050b": (0.50, None, 2),
            "cmsg-sc06a": (0.30, 0.6, 1), "cmsg-sc06b": (0.30, 0.6, 2),
        },
    },
}
DATASET = "email-eu"
CFG = DATASETS[DATASET]
STATIONARY0 = CFG["stationary0"]
INJECT_AT = CFG["inject_at"]
INJECT_SEG_IDX = INJECT_AT - STATIONARY0
RUNS = CFG["runs"]


def set_dataset(name):
    global DATASET, CFG, STATIONARY0, INJECT_AT, INJECT_SEG_IDX, RUNS
    DATASET = name
    CFG = DATASETS[name]
    STATIONARY0 = CFG["stationary0"]
    INJECT_AT = CFG["inject_at"]
    INJECT_SEG_IDX = INJECT_AT - STATIONARY0
    RUNS = CFG["runs"]


def load_edges():
    edges = []
    with gzip.open(CFG["path"], "rt") as f:
        for line in f:
            p = line.split()
            if len(p) < 3:
                continue
            u, v, ts = int(p[0]), int(p[1]), int(p[2])
            if u != v:
                edges.append((u, v, ts))
    edges.sort(key=lambda e: e[2])
    act = Counter()
    for u, v, _ in edges:
        act[u] += 1
        act[v] += 1
    core = {x for x, _ in act.most_common(CFG["core_k"])}
    edges = [(u, v, ts) for (u, v, ts) in edges if u in core and v in core]
    return edges, sorted(core)


def snapshot_eigs(Lcsr, k=TOPK):
    """Same numerics as trackers.fiedler_exact: negative shift keeps the
    factorization safe on (nearly) disconnected snapshots."""
    n = Lcsr.shape[0]
    sigma = -1e-3
    Ls = (Lcsr - sigma * sp.eye(n, format="csr")).tocsc()
    lu = spla.splu(Ls)
    OPinv = spla.LinearOperator((n, n), matvec=lu.solve)
    vals, vecs = spla.eigsh(Lcsr, k=k, sigma=sigma, which="LM", OPinv=OPinv,
                            maxiter=2000, tol=1e-8)
    order = np.argsort(np.clip(vals, 0.0, None))
    vals = vals[order]
    v2 = np.asarray(vecs[:, order[1]]).ravel()
    return vals, v2


def replay(tag, inject):
    """Bit-faithful re-run of real_data_noise.main() stream construction.

    inject = None or (frac, scale, seed); injection fires at steps == 400.
    """
    edges, nodes = load_edges()
    remap = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)
    tau_edges = CFG["tau_steps"] * CFG["macro"]
    sample_every = CFG["macro"] * 50
    gamma = float(np.exp(-1.0 / tau_edges))
    prune_w = CFG["prune_w"]
    frac, scale, seed = (inject + (None,))[:3] if inject else (None, None, None)

    lam_v, spec_v, v2_v, ts_v = [], [], [], []
    pair_w = {}
    steps = 0
    t_start = time.time()
    batch = []
    for idx, (u, v, ts) in enumerate(edges):
        a, b = (remap[u], remap[v]) if remap[u] < remap[v] else (remap[v], remap[u])
        batch.append((a, b))
        if (idx + 1) % sample_every != 0:
            continue
        for k_ in list(pair_w.keys()):
            w = pair_w[k_] * gamma
            if w < prune_w:
                del pair_w[k_]
            else:
                pair_w[k_] = w
        for (a, b) in batch:
            pair_w[(a, b)] = pair_w.get((a, b), 0.0) + 1.0
        batch = []
        if inject is not None and steps == INJECT_AT:
            if scale is not None:
                for kk in list(pair_w.keys()):
                    pair_w[kk] *= scale
            else:
                rng = np.random.default_rng(seed)
                keys = list(pair_w.keys())
                kdel = int(round(frac * len(keys)))
                for kk in rng.choice(len(keys), kdel, replace=False):
                    del pair_w[keys[kk]]
        rows, cols, vals = [], [], []
        deg = {}
        for (a, b), w in pair_w.items():
            deg[a] = deg.get(a, 0.0) + w
            deg[b] = deg.get(b, 0.0) + w
            rows += [a, b]
            cols += [b, a]
            vals += [-w, -w]
        for a, d in deg.items():
            rows.append(a)
            cols.append(a)
            vals.append(d)
        Lcsr = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
        vals6, v2 = snapshot_eigs(Lcsr)
        spec_v.append(vals6)
        lam_v.append(vals6[1])          # lambda_2 = 2nd smallest eigenvalue
        v2_v.append(v2.astype(np.float32))
        ts_v.append(ts)
        steps += 1
    print(f"  replay[{tag}] {steps} samples in {time.time()-t_start:.0f}s", flush=True)
    return (np.asarray(lam_v), np.asarray(spec_v),
            np.asarray(v2_v), np.asarray(ts_v))   # v2: (samples, n)


# ---------------- LAD scoring pipeline (Huang et al., KDD 2019) ----------------

def lad_zstar(spec, s=5, l=10):
    """Return Z* score series (nan where undefined)."""
    T, k = spec.shape
    Z = np.full(T, np.nan)

    def normal_direction(W):
        Wc = W - W.mean(axis=0)
        _, _, Vt = np.linalg.svd(Wc, full_matrices=False)
        return Vt[0]

    for t in range(T):
        scores = []
        for w in (s, l):
            if t - w < 0:
                continue
            p = normal_direction(spec[t - w:t])
            x = spec[t]
            cos = float(np.dot(x, p) / (np.linalg.norm(x) * np.linalg.norm(p) + 1e-300))
            scores.append(1.0 - abs(cos))
        if scores:
            Z[t] = max(scores) if len(scores) == 2 else scores[0]
    Zstar = np.full(T, np.nan)
    for t in range(1, T):
        if np.isfinite(Z[t]) and np.isfinite(Z[t - 1]):
            Zstar[t] = max(Z[t] - Z[t - 1], 0.0)
    return Zstar


def align(v):
    out = np.array(v, dtype=float, copy=True)
    for t in range(1, len(out)):
        if np.dot(out[t], out[t - 1]) < 0:
            out[t] = -out[t]
    return out


def drift_series(v):
    va = align(v)
    D = np.full(len(v), np.nan)
    for t in range(L_CUM, len(v)):
        D[t] = 1.0 - min(1.0, abs(float(np.dot(va[t], va[t - L_CUM]))))
    return D


def window_z_alarms(x, thr, win_base=30, win_recent=10, mad_window=40,
                    conn=None):
    """Sliding window-z (P0b/P0c protocol). Windows touching a non-finite
    boundary value are SKIPPED (P0c equivalent: compact finite prefix first)
    -- partial windows inflate the MAD and distort calibration.
    conn: optional bool mask (same length as x); samples where conn is False
    are ABSTAINED (connectivity-gated variant: lambda_2 ~ 0 snapshots have
    degenerate eigenspaces and eigsh noise amplifies through the sig floor)."""
    alarms, zmax = [], 0.0
    xf = np.asarray(x, dtype=float)
    for t in range(win_base + win_recent, len(xf)):
        lo = t - win_base - win_recent
        if not np.all(np.isfinite(xf[lo:t])):
            continue
        if conn is not None and not np.all(conn[lo:t]):
            continue
        base = np.median(xf[lo: t - win_recent])
        recent = np.median(xf[t - win_recent: t])
        inc = np.diff(xf[max(0, t - mad_window):t])
        sig = 1.4826 * np.median(np.abs(inc - np.median(inc))) + 1e-12
        z = (recent - base) / (sig * np.sqrt(win_recent))
        zmax = max(zmax, abs(z))
        if abs(z) > thr:
            alarms.append(t)
    return alarms, zmax


def evaluate(alarms, cp=INJECT_SEG_IDX, tol=TOL):
    """Detection + STRICT false-alarm accounting.

    Every alarm outside the match window [cp, cp+tol] is a false alarm --
    including alarms that fire after cp+tol.  (An earlier version counted only
    alarms with a < cp, which silently discarded late alarms and reported
    fp = 0 for runs that actually raised many alarms.)
    """
    inside = [a for a in alarms if 0 <= a - cp <= tol]
    det = bool(inside)
    fp = len(alarms) - len(inside)
    delay = min(inside, default=None)
    if delay is not None:
        delay -= cp
    return det, fp, delay


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS), default="email-eu")
    cli = ap.parse_args()
    set_dataset(cli.dataset)
    control_tag = next(t for t, s in RUNS.items() if s is None)
    out_json = f"../runs/lad_baseline{'' if DATASET == 'email-eu' else '-' + DATASET}.json"

    data = {}
    for tag, inject in RUNS.items():
        cache = f"../runs/real_{tag}_spec.npz"
        if os.path.exists(cache):
            d = np.load(cache)
            data[tag] = (np.asarray(d["lam"], dtype=float),
                         np.asarray(d["spec"], dtype=float),
                         np.asarray(d["v2"], dtype=float).T,   # -> (samples, n)
                         np.asarray(d["ts"], dtype=float))
        else:
            lam, spec, v2, ts = replay(tag, inject)
            data[tag] = (lam, spec, v2, ts)
            np.savez_compressed(cache, lam=lam, spec=spec, v2=v2.T, ts=ts)

    # ---- replay fidelity check: control lam vs saved P0 run ----
    saved = np.load(f"../runs/real_{CFG['saved_control']}_lam.npz")["lam"]
    rel = np.abs(data[control_tag][0] - saved) / np.maximum(np.abs(saved), 1e-9)
    rel = np.where(np.isfinite(rel), rel, 0.0)   # both-zero samples: abs diff 0
    print(f"replay fidelity vs saved P0 control lam: max_rel_diff={rel.max():.2e} "
          f"mean_rel_diff={rel.mean():.2e}", flush=True)

    # ---- channels ----
    ch = {}
    for tag, (lam, spec, v2, ts) in data.items():
        seg = slice(STATIONARY0, None)
        zstar = lad_zstar(spec[seg])
        D = drift_series(v2[seg])
        ch[tag] = {
            "LAD": zstar, "lam": lam[seg], "vecD": D, "ts": ts[seg] / 86400.0,
        }

    # ---- zero-alarm-budget calibration on control (segment coords) ----
    conn_ctrl = ch[control_tag]["lam"] > 1e-6
    cal, cal_gated = {}, {}
    for name in ("LAD", "lam", "vecD"):
        x = ch[control_tag][name]
        if name != "LAD":
            _, zmax = window_z_alarms(x, 1e9)
            cal[name] = zmax * (1 + 1e-6)
            _, zmax_g = window_z_alarms(x, 1e9, conn=conn_ctrl)
            cal_gated[name] = zmax_g * (1 + 1e-6)
        else:
            _, zmax = _zmax_series(x)
            cal[name] = zmax * (1 + 1e-6)
            cal_gated[name] = cal[name]      # LAD cosine score has no sig floor
    print(f"control calibration (zero-alarm + ulp): {cal}")
    print(f"control calibration gated: {cal_gated}")
    print(f"control connected fraction: {conn_ctrl.mean():.3f}")

    # ---- real candidate CP day~388 (email-Eu only): channel ranks ----
    rank_info = {}
    if DATASET == "email-eu":
        i38 = int(np.argmin(np.abs(ch[control_tag]["ts"] - 388)))
        for name in ("LAD", "lam", "vecD"):
            x = ch[control_tag][name]
            xf = np.where(np.isfinite(x), x, 0.0)
            order = np.argsort(xf)[::-1]
            rank_info[name] = {"day388_seg_idx": i38,
                               "rank_of_day388": int(np.where(order == i38)[0][0]),
                               "series_len": int(np.isfinite(x).sum())}

    # ---- per-run evaluation (raw + connectivity-gated) ----
    rows, rows_gated = [], []
    for tag in RUNS:
        if tag == control_tag:
            continue
        row, row_g = {"run": tag}, {"run": tag}
        conn = ch[tag]["lam"] > 1e-6
        for name in ("LAD", "lam", "vecD"):
            if name != "LAD":
                alarms, _ = window_z_alarms(ch[tag][name], cal[name])
                alarms_g, _ = window_z_alarms(ch[tag][name], cal_gated[name],
                                              conn=conn)
            else:
                alarms, _ = _lad_alarms(ch[tag][name], cal[name])
                alarms_g, _ = _lad_alarms(ch[tag][name], cal_gated[name])
            d, f, dl = evaluate(alarms)
            row[name] = [bool(d), int(f), (int(dl) if dl is not None else None)]
            row[f"{name}_nalarms"] = len(alarms)
            dg, fg, dlg = evaluate(alarms_g)
            row_g[name] = [bool(dg), int(fg), (int(dlg) if dlg is not None else None)]
            row_g[f"{name}_nalarms"] = len(alarms_g)
        rows.append(row)
        rows_gated.append(row_g)
        print(f"[{tag}] raw  LAD:{row['LAD'][0]}/{row['LAD'][1]} "
              f"lam:{row['lam'][0]}/{row['lam'][1]} vecD:{row['vecD'][0]}/{row['vecD'][1]}"
              f" | gated LAD:{row_g['LAD'][0]}/{row_g['LAD'][1]} "
              f"lam:{row_g['lam'][0]}/{row_g['lam'][1]} "
              f"vecD:{row_g['vecD'][0]}/{row_g['vecD'][1]} "
              f"(det/fp, delay vecD={row['vecD'][2]}/{row_g['vecD'][2]})")

    totals = {name: sum(r[name][0] for r in rows) for name in ("LAD", "lam", "vecD")}
    totals_g = {name: sum(r[name][0] for r in rows_gated)
                for name in ("LAD", "lam", "vecD")}
    print(f"totals raw: {totals}/{len(rows)}  gated: {totals_g}/{len(rows_gated)}")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"dataset": DATASET, "calibration": cal,
                   "calibration_gated": cal_gated,
                   "control_connected_frac": float(conn_ctrl.mean()),
                   "day388_rank": rank_info,
                   "rows": rows, "rows_gated": rows_gated,
                   "totals": totals, "totals_gated": totals_g,
                   "tol": TOL, "inject_seg_idx": INJECT_SEG_IDX}, f, indent=2)


def _zmax_series(x):
    xf = np.where(np.isfinite(np.asarray(x, dtype=float)), x, 0.0)
    return None, float(np.max(xf))


def _lad_alarms(zstar, thr):
    alarms = []
    for t, v in enumerate(zstar):
        if np.isfinite(v) and v > thr:
            alarms.append(t)
    return alarms, 0.0


if __name__ == "__main__":
    main()
