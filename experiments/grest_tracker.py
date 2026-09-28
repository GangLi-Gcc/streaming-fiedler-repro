"""G-REST tracker (Eini, Karaaslanli, Kalantzis, Traganitis; arXiv:2603.19439).

Faithful re-implementation of Algorithm 2 ("Graph Rayleigh-Ritz Eigenspace
Tracking") for the fixed-node-set regime of our P0h experiment:

  - no node additions  ->  Delta_2 = 0, so the projection subspace degenerates
    to  Z = span{ X_K, (I - X_K X_K^T) dT X_K }   (G-REST_2 / Residual-Modes
    subspace with optimal RR coefficients; the paper's Prop. 4 shows Delta_2
    carries the new-node information only).
  - Laplacian end of the spectrum is reached via the paper's own shift
    recipe (Sec. 4.2): track the LEADING eigenpairs of the shifted Laplacian
      T = c*I - L,  c = 2*d_max + 1,
    whose leading eigenpairs are the trailing ones of L; the 2nd leading pair
    of T is exactly the Fiedler pair (lambda2, v2) of L.  The shift is NOT
    cosmetic for RR: with the naive choice M = -L the tracked end sits at the
    spectral edge 0, where the extra projection directions (I-P)dM X create
    spurious near-zero Ritz values that crowd out the true Fiedler Ritz pair
    in the top-K selection (verified numerically during bring-up).  c = 2d_max+1
    moves the target end to ~2d_max, away from the pollution.

Interface matches trackers.WarmStartTracker: update(i, j, sign, L) with the
NEW Laplacian; keeps self.v2 and timing.
"""
from __future__ import annotations

import time
import numpy as np
import scipy.sparse as sp
import scipy.linalg as sla

from trackers import fiedler_exact


def _shift_c(L: sp.csr_matrix) -> float:
    return 2.0 * float(L.diagonal().max()) + 1.0


class GRESTTracker:
    """Rayleigh-Ritz subspace-projection tracker (Alg. 2 of arXiv:2603.19439).

    restart_policy controls when an exact Fiedler recompute is triggered:
      'none'     – no restart (original G-REST, accumulates drift)
      'periodic' – exact restart every `restart_every` steps
      'residual' – exact restart when warm-start residual > restart_tol * gap_est
                   (same gate as WarmStartTracker; this is the "fair comparison" variant)
    """

    def __init__(self, K: int = 3, restart_policy: str = "none",
                 restart_every: int = 4, restart_tol: float = 0.03):
        tag = {"none": "", "periodic": f"_p{restart_every}", "residual": "_rtrig"}
        self.name = f"grest_K{K}{tag.get(restart_policy,'')}"
        self.K = K
        self.restart_policy = restart_policy
        self.restart_every = restart_every
        self.restart_tol = restart_tol
        self.X = None
        self.Lam = None
        self.c = None
        self.L_prev = None
        self.v2 = None
        self.lam2 = None       # tracked lambda2 estimate (for residual gate)
        self.gap_est = None    # estimated spectral gap (lam3 - lam2)
        self.n_restart = 0
        self.n_refine = 0
        self._step = 0
        self.time_per_event = []

    def _seed(self, L0: sp.csr_matrix):
        vals, vecs = fiedler_exact(L0, k=self.K)
        self.c = _shift_c(L0)
        self.X = np.asarray(vecs)
        self.Lam = self.c - np.asarray(vals)
        self.L_prev = L0.copy()
        self.v2 = np.asarray(self.X[:, 1])
        self.lam2 = float(vals[1])
        self.gap_est = float(vals[2] - vals[1]) if self.K >= 3 else 0.05
        self.n_restart += 1

    def _exact_restart(self, L: sp.csr_matrix):
        self._seed(L)

    def _residual(self, L: sp.csr_matrix) -> float:
        """Residual of current v2 estimate on new Laplacian."""
        r = L @ self.v2 - self.lam2 * self.v2
        return float(np.linalg.norm(r))

    def update(self, i, j, sign, L):
        t0 = time.perf_counter()
        L = L.tocsr()
        self._step += 1

        if self.X is None:
            self._seed(L)
            self.time_per_event.append(time.perf_counter() - t0)
            return float(self.lam2)

        # --- check restart conditions BEFORE the RR update ---
        do_restart = False
        if self.restart_policy == "periodic":
            do_restart = (self._step % self.restart_every == 0)
        elif self.restart_policy == "residual":
            res = self._residual(L)
            gate_floor = 1e-9
            gate = self.restart_tol * max(self.gap_est, gate_floor)
            do_restart = (res > gate)

        if do_restart:
            self._exact_restart(L)
            self.n_restart += 1
            self.time_per_event.append(time.perf_counter() - t0)
            return float(self.lam2)

        # --- standard G-REST Rayleigh-Ritz update ---
        c_new = _shift_c(L)
        dC = c_new - self.c
        dL = (L - self.L_prev).astype(np.float64)
        self.c = c_new
        self.L_prev = L.copy()
        X = self.X
        K = self.K
        dTX = dC * X - dL @ X
        W = dTX - X @ (X.T @ dTX)
        Z, _ = sla.qr(np.hstack([X, W]), mode="economic")
        D = Z.shape[1]
        ZtX = Z.T @ X
        B = (ZtX * self.Lam) @ ZtX.T
        B = B + dC * np.eye(D) - np.asarray(Z.T @ dL @ Z)
        theta, F = sla.eigh(B)
        idx = np.argsort(theta)[::-1][:K]
        theta = theta[idx]
        self.X = Z @ F[:, idx]
        self.Lam = theta
        self.v2 = np.asarray(self.X[:, 1])
        self.lam2 = float(self.c - theta[1])
        if self.K >= 3:
            self.gap_est = float(theta[1] - theta[2])  # descending: gap = 2nd - 3rd
        self.n_refine += 1
        self.time_per_event.append(time.perf_counter() - t0)
        return float(self.lam2)
