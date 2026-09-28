"""Fiedler pair trackers.

BaselineFull : exact recompute per snapshot via scipy eigsh (cost reference).
WarmStartTracker : incremental maintenance of (lambda2, v2):
  - apply low-rank edge update to the Laplacian
  - run a few power-iteration / Lanczos steps warm-started from previous v2
  - full restart when residual or eigen-gap condition breaks
"""
from __future__ import annotations

import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def fiedler_exact(L: sp.csr_matrix, k=3):
    """Exact-ish small-eigenpair computation (baseline; also used for restart).

    Shift-invert with a small NEGATIVE sigma: L - sigma*I is positive definite
    (L is PSD with a zero eigenvalue), so the splu factorization is safe, and
    shift-invert accurately resolves clustered near-zero eigenvalues where
    plain 'SA' Arnoldi misses multiplicity.
    """
    n = L.shape[0]
    sigma = -1e-3
    Ls = (L - sigma * sp.eye(n, format="csr")).tocsc()
    lu = spla.splu(Ls)
    OPinv = spla.LinearOperator((n, n), matvec=lu.solve)
    vals, vecs = spla.eigsh(L, k=k, sigma=sigma, which="LM", OPinv=OPinv,
                            return_eigenvectors=True, maxiter=2000, tol=1e-8)
    order = np.argsort(vals)
    vals, vecs = vals[order], vecs[:, order]
    return vals, vecs  # vals[1] ~ lambda2 (first is ~0)


class BaselineFull:
    def __init__(self):
        self.name = "baseline_full"
        self.time_per_event = []

    def update(self, i, j, sign, L):
        t0 = time.perf_counter()
        vals, vecs = fiedler_exact(L)
        self.v2 = vecs[:, 1]
        dt = time.perf_counter() - t0
        self.time_per_event.append(dt)
        return vals[1]


class WarmStartTracker:
    """Incremental Fiedler maintenance (research prototype, v1).

    Strategy:
      - keep previous (lam2, v2, lam3)
      - after edge update, estimate Rayleigh quotient of old v2
      - if residual small relative to current estimate of gap: accept few
        power-iteration refinements; else exact restart
    """
    def __init__(self, restart_tol_ratio=0.1, max_refine=30, refine_abs_tol=None,
                 gate_floor=0.0, verify_floor=1e-9):
        self.name = "warmstart_v1"
        self.restart_tol_ratio = restart_tol_ratio
        self.max_refine = max_refine
        # if set, refinement stops at this ABSOLUTE residual (P0f: gap-relative
        # tolerance locks k=1 -- one power step per sample, tracker blind to
        # rotation); None keeps the legacy gap-relative behavior
        self.refine_abs_tol = refine_abs_tol
        # P0o: absolute floors for the disconnected regime (gap_est ~ 0 would
        # otherwise force an exact restart every sample, re-importing the
        # per-snapshot eigsh noise). gate = rtr * max(gap_est, gate_floor);
        # refine result is trusted unless res3 > 0.5*max(gap, verify_floor).
        # Defaults reproduce the legacy behavior exactly.
        self.gate_floor = gate_floor
        self.verify_floor = verify_floor
        self.v2 = None
        self.lam3 = None
        self.n_restart = 0
        self.n_refine = 0
        self.time_per_event = []
        self.n_refine_steps = 0
        # per-sample diagnostics: (branch, res_in, gap_est, k_used, res_out,
        #                         lam2, dmax); branch 0=refine-accept
        # 1=refine-fallback-exact 2=exact-restart(gate fail)
        self.diag = []

    def _init(self, L):
        vals, vecs = fiedler_exact(L)
        self.v2 = vecs[:, 1]
        self.lam3 = vals[2] if len(vals) > 2 else vals[1] * 2 + 1e-3
        return vals[1]

    def _power_refine(self, L, v, iters, tol_res=1e-3):
        """Power iteration on (cI - L), c > lambda_n, to pull the smallest
        NONTRIVIAL eigenpair. The trivial eigenvector 1/sqrt(n) (eigenvalue 0)
        is deflated every step. Using the current Rayleigh quotient as shift
        would amplify the top of the spectrum instead -- hence a uniform
        upper bound c = 2*d_max + 1 (since lambda_n <= 2 d_max).
        """
        n = L.shape[0]
        ones = np.ones(n) / np.sqrt(n)
        dmax = float(L.diagonal().max()) if L.nnz else 1.0
        c = 2.0 * dmax + 1.0
        lam = float(v @ (L @ v))
        iters_used = 0
        for _ in range(iters):
            w = c * v - (L @ v)
            w -= ones * (ones @ w)
            nv = np.linalg.norm(w)
            if nv < 1e-12:
                break
            v = w / nv
            self.n_refine_steps += 1
            iters_used += 1
            lam = float(v @ (L @ v))
            r = L @ v - lam * v
            if np.linalg.norm(r) < tol_res:
                break
        v -= ones * (ones @ v)
        v /= np.linalg.norm(v) + 1e-300
        return float(v @ (L @ v)), v, iters_used

    def update(self, i, j, sign, L):
        t0 = time.perf_counter()
        if self.v2 is None:
            lam2 = self._init(L)
            self.n_restart += 1
            self.time_per_event.append(time.perf_counter() - t0)
            return lam2
        v = self.v2
        dmax = float(L.diagonal().max()) if L.nnz else 1.0
        # Rayleigh quotient + residual norm
        w = L @ v
        lam = float(v @ w)
        r = w - lam * v
        res = np.linalg.norm(r)
        gap_est = max(self.lam3 - lam, 1e-9)
        gate = self.restart_tol_ratio * max(gap_est, self.gate_floor)
        if res <= gate:
            tol_res = self.refine_abs_tol if self.refine_abs_tol is not None \
                else self.restart_tol_ratio * gap_est
            lam2, v_new, k_used = self._power_refine(L, v, self.max_refine,
                                                     tol_res=tol_res)
            # estimate next eigenvalue from residual direction (2nd Ritz)
            w2 = L @ v_new
            r2 = w2 - (v_new @ w2) * v_new
            nr = np.linalg.norm(r2)
            if nr > 1e-12:
                u = r2 / nr
                self.lam3 = float(max(u @ (L @ u), lam2 + 1e-6))
            # verify refined pair; fall back to exact if refinement failed
            w3 = L @ v_new
            lam2 = float(v_new @ w3)
            res3 = np.linalg.norm(w3 - lam2 * v_new)
            if res3 > 0.5 * max(self.lam3 - lam2, self.verify_floor):
                vals, vecs = fiedler_exact(L)
                lam2 = vals[1]
                v_new = vecs[:, 1]
                self.lam3 = vals[2] if len(vals) > 2 else lam2 * 2 + 1e-3
                self.n_restart += 1
                self.diag.append((1, res, gap_est, k_used, res3, lam2, dmax))
            else:
                self.n_refine += 1
                self.diag.append((0, res, gap_est, k_used, res3, lam2, dmax))
            self.v2 = v_new
        else:
            vals, vecs = fiedler_exact(L)
            lam2 = vals[1]
            self.v2 = vecs[:, 1]
            self.lam3 = vals[2] if len(vals) > 2 else lam2 * 2 + 1e-3
            self.n_restart += 1
            self.diag.append((2, res, gap_est, 0, np.nan, lam2, dmax))
        self.time_per_event.append(time.perf_counter() - t0)
        return lam2
