"""rebuild/core.py — clean single source of truth for the streaming-Fiedler
experiments.

This module re-implements, from the ground up, the pieces every experiment in
the paper depends on, fixing the defects found in the original codebase:

  1. Strict false-alarm accounting.  `evaluate()` counts EVERY alarm outside
     the match window [cp, cp+tol] as a false alarm, including late alarms.
     (The original lad_baseline.py / vec_drift.py counted only alarms with
     index < cp, silently discarding late alarms and reporting fp=0 for
     streams that actually raised many alarms.)

  2. Correct tracker state.  WarmStartTracker exposes an explicit `lam2`
     attribute alongside `v2` and `lam3`, and a `seed(v2, lam2, lam3)` method,
     so callers can reset all three consistently.  (The original had no `lam2`
     attribute; a hasattr(trk,'lam2') guard in a downstream experiment never
     fired, leaving stale `lam3` state that corrupted the restart gate.)

  3. One evaluation protocol.  Every channel uses the same match window,
     calibration rule (control-stream max + ulp margin), and strict accounting.

Layout mirrors the paper:
  - stream: SNAP edge stream -> decaying-memory weighted snapshots
  - trackers: WarmStartTracker (detection-grade), GRESTTracker (embedding-grade)
  - channels: LAD, lambda2-z, vec-D (window-z), spike (max-type)
  - evaluation: strict precision/recall/FAR/delay + Wilson recall CI
"""
from __future__ import annotations

import gzip
import os
import time
from collections import Counter

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# ---------------------------------------------------------------------------
# exact Fiedler eigensolver
# ---------------------------------------------------------------------------
def fiedler_exact(L: sp.csr_matrix, k: int = 3):
    """Smallest eigenpairs of the graph Laplacian via shift-invert Arnoldi.

    sigma = -1e-3 keeps L - sigma*I positive definite, so the sparse LU is safe
    and clustered near-zero eigenvalues are resolved (plain 'SA' misses them).
    Returns (vals ascending, vecs); vals[1], vecs[:,1] are the Fiedler pair.
    """
    n = L.shape[0]
    sigma = -1e-3
    Ls = (L - sigma * sp.eye(n, format="csr")).tocsc()
    lu = spla.splu(Ls)
    op = spla.LinearOperator((n, n), matvec=lu.solve)
    vals, vecs = spla.eigsh(L, k=k, sigma=sigma, which="LM", OPinv=op,
                            return_eigenvectors=True, maxiter=2000, tol=1e-8)
    order = np.argsort(np.clip(vals, 0.0, None))
    return np.asarray(vals)[order], np.asarray(vecs)[:, order]


def largest_eigs(L, k=6):
    """k largest eigenvalues of the graph Laplacian (descending), for LAD's
    published top-singular-value signature.  No shift (which='LA'); for a PSD
    Laplacian these are its largest eigenvalues / top singular values."""
    vals = spla.eigsh(L, k=k, which="LA", return_eigenvectors=False, tol=1e-8)
    return np.sort(vals)[::-1]


def align_cos(v_new, v_old):
    return abs(float(np.dot(v_new, v_old)))


# ---------------------------------------------------------------------------
# trackers
# ---------------------------------------------------------------------------
class WarmStartTracker:
    """Detection-grade Fiedler tracker: power-refine warm start, exact restart
    when the residual of the old pair on the new Laplacian exceeds a gate.

    State is (v2, lam2, lam3); all three are set together by `seed`.
    """

    def __init__(self, restart_tol_ratio=0.03, max_refine=30,
                 refine_abs_tol=1e-6, gate_floor=0.0, verify_floor=1e-9):
        self.restart_tol_ratio = restart_tol_ratio
        self.max_refine = max_refine
        self.refine_abs_tol = refine_abs_tol
        self.gate_floor = gate_floor
        self.verify_floor = verify_floor
        self.v2 = None
        self.lam2 = None
        self.lam3 = None
        self.n_restart = 0
        self.n_refine = 0
        self.n_refine_steps = 0
        self.time_per_event = []
        self.diag = []

    def seed(self, v2, lam2, lam3):
        """Set the full state at once (used to reset to an exact pair)."""
        self.v2 = np.asarray(v2, dtype=float).copy()
        self.lam2 = float(lam2)
        self.lam3 = float(lam3)

    def _init(self, L):
        vals, vecs = fiedler_exact(L)
        self.seed(vecs[:, 1], vals[1], vals[2] if len(vals) > 2 else vals[1] * 2 + 1e-3)
        return self.lam2

    def _power_refine(self, L, v, iters, tol_res):
        n = L.shape[0]
        ones = np.ones(n) / np.sqrt(n)
        dmax = float(L.diagonal().max()) if L.nnz else 1.0
        c = 2.0 * dmax + 1.0
        lam = float(v @ (L @ v))
        k_used = 0
        for _ in range(iters):
            w = c * v - (L @ v)
            w -= ones * (ones @ w)
            nv = np.linalg.norm(w)
            if nv < 1e-12:
                break
            v = w / nv
            self.n_refine_steps += 1
            k_used += 1
            lam = float(v @ (L @ v))
            if np.linalg.norm(L @ v - lam * v) < tol_res:
                break
        v -= ones * (ones @ v)
        v /= np.linalg.norm(v) + 1e-300
        return float(v @ (L @ v)), v, k_used

    def update(self, L):
        t0 = time.perf_counter()
        if self.v2 is None:
            lam2 = self._init(L)
            self.n_restart += 1
            self.time_per_event.append(time.perf_counter() - t0)
            return lam2
        v = self.v2
        lam = float(v @ (L @ v))
        res = np.linalg.norm(L @ v - lam * v)
        gap_est = max(self.lam3 - lam, 1e-9)
        gate = self.restart_tol_ratio * max(gap_est, self.gate_floor)
        dmax = float(L.diagonal().max()) if L.nnz else 1.0

        if res <= gate:
            tol_res = (self.refine_abs_tol if self.refine_abs_tol is not None
                       else self.restart_tol_ratio * gap_est)
            lam2, v_new, k_used = self._power_refine(L, v, self.max_refine, tol_res)
            # update lam3 from the residual direction (2nd Ritz estimate)
            r2 = L @ v_new - (v_new @ (L @ v_new)) * v_new
            nr = np.linalg.norm(r2)
            if nr > 1e-12:
                u = r2 / nr
                self.lam3 = float(max(u @ (L @ u), lam2 + 1e-6))
            res3 = np.linalg.norm(L @ v_new - lam2 * v_new)
            if res3 > 0.5 * max(self.lam3 - lam2, self.verify_floor):
                vals, vecs = fiedler_exact(L)
                self.seed(vecs[:, 1], vals[1],
                          vals[2] if len(vals) > 2 else vals[1] * 2 + 1e-3)
                self.n_restart += 1
                self.diag.append((1, res, gap_est, k_used, res3, self.lam2, dmax))
            else:
                self.v2 = v_new
                self.lam2 = lam2
                self.n_refine += 1
                self.diag.append((0, res, gap_est, k_used, res3, self.lam2, dmax))
        else:
            vals, vecs = fiedler_exact(L)
            self.seed(vecs[:, 1], vals[1],
                      vals[2] if len(vals) > 2 else vals[1] * 2 + 1e-3)
            self.n_restart += 1
            self.diag.append((2, res, gap_est, 0, np.nan, self.lam2, dmax))
        self.time_per_event.append(time.perf_counter() - t0)
        return self.lam2


class GRESTTracker:
    """Embedding-grade Rayleigh-Ritz projection tracker (G-REST, fixed node set).

    restart_policy:
      'none'     - no restart (accumulates drift)
      'periodic' - exact restart every `restart_every` steps
      'residual' - exact restart when warm-start residual > restart_tol * gap_est
    """

    def __init__(self, K=5, restart_policy="none", restart_every=4,
                 restart_tol=0.03):
        self.K = K
        self.restart_policy = restart_policy
        self.restart_every = restart_every
        self.restart_tol = restart_tol
        self.X = None
        self.Lam = None
        self.c = None
        self.v2 = None
        self.lam2 = None
        self.gap_est = None
        self.n_restart = 0
        self.n_refine = 0
        self._step = 0
        self.time_per_event = []

    def _shift_c(self, L):
        return 2.0 * float(L.diagonal().max()) + 1.0

    def seed(self, v2, lam2, lam3):
        self.v2 = np.asarray(v2, dtype=float).copy()
        self.lam2 = float(lam2)
        self.gap_est = float(lam3 - lam2) if lam3 > lam2 else 0.05

    def _init(self, L):
        vals, vecs = fiedler_exact(L, k=self.K)
        self.c = self._shift_c(L)
        self.X = np.asarray(vecs)
        self.Lam = self.c - np.asarray(vals)
        self.L_prev = L.tocsr().copy()
        self.seed(vecs[:, 1], vals[1], vals[2] if len(vals) > 2 else vals[1] + 0.05)
        self.n_restart += 1

    def update(self, L):
        t0 = time.perf_counter()
        L = L.tocsr()
        self._step += 1
        if self.X is None:
            self._init(L)
            self.time_per_event.append(time.perf_counter() - t0)
            return self.lam2

        if self.restart_policy == "periodic" and self._step % self.restart_every == 0:
            self._init(L)
            self.time_per_event.append(time.perf_counter() - t0)
            return self.lam2
        if self.restart_policy == "residual":
            res = np.linalg.norm(L @ self.v2 - self.lam2 * self.v2)
            gate = self.restart_tol * max(self.gap_est, 1e-9)
            if res > gate:
                self._init(L)
                self.time_per_event.append(time.perf_counter() - t0)
                return self.lam2

        c_new = self._shift_c(L)
        dC = c_new - self.c
        dL = (L - self.L_prev) if hasattr(self, "L_prev") else 0.0 * L
        self.c = c_new
        self.L_prev = L.tocsr().copy()
        X = self.X
        K = self.K
        dTX = dC * X - dL @ X
        W = dTX - X @ (X.T @ dTX)
        import scipy.linalg as sla
        Z, _ = sla.qr(np.hstack([X, W]), mode="economic")
        D = Z.shape[1]
        ZtX = Z.T @ X
        B = (ZtX * self.Lam) @ ZtX.T + dC * np.eye(D) - np.asarray(Z.T @ dL @ Z)
        theta, F = sla.eigh(B)
        idx = np.argsort(theta)[::-1][:K]
        self.X = Z @ F[:, idx]
        self.Lam = theta[idx]
        self.v2 = np.asarray(self.X[:, 1])
        self.lam2 = float(self.c - self.Lam[1])
        if K >= 3:
            self.gap_est = float(self.Lam[1] - self.Lam[2])
        self.n_refine += 1
        self.time_per_event.append(time.perf_counter() - t0)
        return self.lam2


# ---------------------------------------------------------------------------
# edge-stream -> decaying-memory weighted snapshots
# ---------------------------------------------------------------------------
_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

DATASETS = {
    "email-eu": {
        "path": os.path.join(_DATA_DIR, "email-Eu-core-temporal.txt.gz"),
        "tau_steps": 2000, "macro": 10, "prune_w": 0.02, "core_k": 300,
        "stationary0": 300, "inject_at": 400,
        "runs": {
            "inj015a": ("del", 0.15, 1), "inj015b": ("del", 0.15, 2),
            "inj030a": ("del", 0.30, 1), "inj030b": ("del", 0.30, 2),
            "inj050a": ("del", 0.50, 1), "inj050b": ("del", 0.50, 2),
            "sc06a": ("scale", 0.6, 1), "sc06b": ("scale", 0.6, 2),
        },
    },
    "college-msg": {
        "path": os.path.join(_DATA_DIR, "CollegeMsg.txt.gz"),
        "tau_steps": 1500, "macro": 5, "prune_w": 0.02, "core_k": 500,
        "stationary0": 20, "inject_at": 120,
        "runs": {
            "cmsg-inj015a": ("del", 0.15, 1), "cmsg-inj015b": ("del", 0.15, 2),
            "cmsg-inj030a": ("del", 0.30, 1), "cmsg-inj030b": ("del", 0.30, 2),
            "cmsg-inj050a": ("del", 0.50, 1), "cmsg-inj050b": ("del", 0.50, 2),
            "cmsg-sc06a": ("scale", 0.6, 1), "cmsg-sc06b": ("scale", 0.6, 2),
        },
    },
}


def load_edges(dataset):
    cfg = DATASETS[dataset]
    edges = []
    with gzip.open(cfg["path"], "rt") as f:
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
    core = {x for x, _ in act.most_common(cfg["core_k"])}
    edges = [(u, v, ts) for (u, v, ts) in edges if u in core and v in core]
    remap = {v: i for i, v in enumerate(sorted(core))}
    return [(min(remap[u], remap[v]), max(remap[u], remap[v]), ts)
            for (u, v, ts) in edges], len(core)


def build_stream(dataset, inject=None, max_samples=None):
    """Replay the edge stream into decaying-memory snapshots.

    inject: None (control) or ("del", frac, seed) / ("scale", factor, seed).
    Returns list of (L_csr, ts) snapshots.
    """
    cfg = DATASETS[dataset]
    edges, n = load_edges(dataset)
    tau_edges = cfg["tau_steps"] * cfg["macro"]
    sample_every = cfg["macro"] * 50
    gamma = float(np.exp(-1.0 / tau_edges))
    prune_w = cfg["prune_w"]

    snapshots = []
    pair_w = {}
    batch = []
    steps = 0
    for idx, (u, v, ts) in enumerate(edges):
        batch.append((u, v))
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

        if inject is not None and steps == cfg["inject_at"]:
            kind = inject[0]
            if kind == "scale":
                for kk in list(pair_w.keys()):
                    pair_w[kk] *= inject[1]
            elif kind == "del":
                rng = np.random.default_rng(inject[2])
                keys = list(pair_w.keys())
                kdel = int(round(inject[1] * len(keys)))
                for kk in rng.choice(len(keys), kdel, replace=False):
                    del pair_w[keys[kk]]

        rows, cols, vals = [], [], []
        deg = {}
        for (a, b), w in pair_w.items():
            deg[a] = deg.get(a, 0.0) + w
            deg[b] = deg.get(b, 0.0) + w
            rows += [a, b]; cols += [b, a]; vals += [-w, -w]
        for a, d in deg.items():
            rows.append(a); cols.append(a); vals.append(d)
        L = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
        snapshots.append((L, ts))
        steps += 1
        if max_samples is not None and steps >= max_samples:
            break
    return snapshots, n


# ---------------------------------------------------------------------------
# detection channels
# ---------------------------------------------------------------------------
def align_sign(v):
    out = np.asarray(v, dtype=float).copy()
    for t in range(1, len(out)):
        if np.dot(out[t], out[t - 1]) < 0:
            out[t] = -out[t]
    return out


def drift_series(v, L_cum=10):
    """Cumulative drift D_t = 1 - |<v_t, v_{t-L}>| (sign-aligned)."""
    va = align_sign(v)
    D = np.full(len(v), np.nan)
    for t in range(L_cum, len(v)):
        D[t] = 1.0 - min(1.0, abs(float(np.dot(va[t], va[t - L_cum]))))
    return D


def consecutive_drift(v):
    """Per-sample drift d_t = 1 - <v_t, v_{t-1}> (sign-aligned)."""
    va = align_sign(v)
    d = np.zeros(len(v))
    for t in range(1, len(v)):
        d[t] = 1.0 - min(1.0, max(-1.0, float(np.dot(va[t], va[t - 1]))))
    return d


def window_z(x, win_base=30, win_recent=10, mad_window=40, sig_floor=0.0):
    """Return (z-score series, zmax). Windows touching non-finite values are skipped.

    sig_floor: a lower bound on the dispersion estimate sig.  In the steady
    state of a nearly-static graph, the raw MAD of diff(x) collapses toward
    machine precision (e.g. ~5e-8 for the vec-D drift), so z = (recent-base)/sig
    becomes noise-divided-by-noise and its value is sensitive to eigensolver
    floating-point precision.  A physical floor makes z stable.
    """
    z = np.full(len(x), np.nan)
    zmax = 0.0
    xf = np.asarray(x, dtype=float)
    for t in range(win_base + win_recent, len(xf)):
        lo = t - win_base - win_recent
        if not np.all(np.isfinite(xf[lo:t])):
            continue
        base = np.median(xf[lo:t - win_recent])
        recent = np.median(xf[t - win_recent:t])
        inc = np.diff(xf[max(0, t - mad_window):t])
        sig = max(1.4826 * np.median(np.abs(inc - np.median(inc))), sig_floor)
        z[t] = (recent - base) / (sig * np.sqrt(win_recent))
        zmax = max(zmax, abs(z[t]))
    return z, zmax


def alarms_from_z(z, thr):
    return [t for t in range(len(z)) if np.isfinite(z[t]) and abs(z[t]) > thr]


def dos_signature(L):
    """SCPD structural signature: 50-bin L1-normalized histogram of the
    normalized Laplacian spectrum (exact eigvalsh; an exact-spectrum
    implementation of the structural channel, not a KPM estimate).

    L_sym = I - D^{-1/2} A D^{-1/2} with zero-degree rows left as identity;
    eigenvalues lie in [0, 2].  This is the SCPD signature fed to the same
    dual-window lad_zstar scorer as LAD (hyperparameters identical -> isolates
    the signature)."""
    n = L.shape[0]
    d = np.asarray(L.diagonal()).copy()
    dinv = np.where(d > 0, 1.0 / np.sqrt(np.maximum(d, 1e-300)), 0.0)
    Dm = sp.diags(dinv)
    A = (sp.diags(d) - L).tocsr()
    Lsym = sp.eye(n, format="csr") - Dm @ A @ Dm
    vals = np.linalg.eigvalsh(Lsym.toarray())
    hist, _ = np.histogram(vals, bins=50, range=(0.0, 2.0))
    h = hist.astype(float)
    return h / (h.sum() + 1e-300)


def lad_zstar(spec, s=5, l=10):
    """LAD Z* score series, faithful to the published LAD implementation
    (Huang et al., KDD'20, ``Anomaly_Detection.py``):

    (1) each snapshot signature (row) is L2-normalized to unit norm;
    (2) for each window w in {short=s, long=l} the typical direction p_w(t) is
        the leading LEFT singular vector of the transposed UNcentered context
        matrix [x_{t-w},...,x_{t-1}]^T (equivalently the top right-singular
        direction of the past w normalized signatures) -- NO column centering;
    (3) the window score is the cosine distance
        z_w(t) = 1 - |<x_t, p_w(t)>| / (||x_t|| ||p_w(t)||) for the normalized
        current snapshot x_t;
    (4) each window score is DIFFERENCED first (dz_w(t) = z_w(t)-z_w(t-1)),
        then the two differenced series are max-combined:
        Z*(t) = max(dz_short(t), dz_long(t)).

    A single warm-up of `l` snapshots applies (both windows need `l` preceding
    snapshots), so Z* is first defined at index l+1."""
    T, k = spec.shape
    X = spec.copy()
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.where(nrm < 1e-300, 1.0, nrm)

    Zs = np.full(T, np.nan)
    Zl = np.full(T, np.nan)
    for t in range(l, T):
        for w, Z in ((s, Zs), (l, Zl)):
            W = X[t - w:t]                          # (w, k) normalized snapshots
            U, _, _ = np.linalg.svd(W.T, full_matrices=False)   # (k, w)
            p = U[:, 0]                             # leading left singular vector
            x = X[t]
            cos = abs(float(np.dot(x, p)
                            / (np.linalg.norm(x) * np.linalg.norm(p) + 1e-300)))
            Z[t] = 1.0 - cos

    Zstar = np.full(T, np.nan)
    for t in range(l + 1, T):
        if (np.isfinite(Zs[t]) and np.isfinite(Zs[t - 1])
                and np.isfinite(Zl[t]) and np.isfinite(Zl[t - 1])):
            Zstar[t] = max(Zs[t] - Zs[t - 1], Zl[t] - Zl[t - 1])
    return Zstar


# ---------------------------------------------------------------------------
# strict evaluation
# ---------------------------------------------------------------------------
def evaluate(alarms, cp, tol=40):
    """STRICT accounting: every alarm outside [cp, cp+tol] is a false alarm."""
    inside = [a for a in alarms if 0 <= a - cp <= tol]
    det = bool(inside)
    fp = len(alarms) - len(inside)
    delay = (min(inside) - cp) if inside else None
    return det, fp, delay


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def aggregate_detection(run_results, n_samples, tol=40, cp=100):
    """Pool per-run (det, fp, delay) into recall/CI/precision/FAR/delay."""
    n_cp = len(run_results)
    tp = sum(1 for det, _, _ in run_results if det)
    fp = sum(f for _, f, _ in run_results)
    delays = [d for _, _, d in run_results if d is not None]
    eligible = (n_samples - (tol + 1)) * n_cp
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / n_cp
    lo, hi = wilson_ci(tp, n_cp)
    far1k = 1000 * fp / eligible if eligible else 0.0
    return {"tp": tp, "n_cp": n_cp, "recall": rec, "recall_ci95": [lo, hi],
            "fp": fp, "precision": prec, "far_per_1000": far1k,
            "delay_min": min(delays) if delays else None,
            "delay_max": max(delays) if delays else None}
