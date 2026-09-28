"""Dynamic graph with planted structural change points (edge-stream events).

Model: two-community DCSBM-like graph; edge events are insertions/deletions
drawn from block probabilities that JUMP at planted change points:
  - "pin_drop"  : within-community prob p_in jumps down (community weakening)
  - "split"     : one community splits into two (three blocks after tau)
  - "bridge_cut": a small bridge edge set is removed at tau
"""
from __future__ import annotations

import numpy as np


class DynamicSBM:
    def __init__(self, n=2000, p_in=0.02, p_out=0.002, n_blocks=2,
                 degree_corr=True, theta_gamma=1.0, ensure_connected=True,
                 seed=0):
        rng = np.random.default_rng(seed)
        self.n = n
        self.n_blocks = n_blocks
        self.labels = rng.integers(0, n_blocks, n)
        self.degree_corr = degree_corr
        if degree_corr:
            theta = rng.power(theta_gamma, n)  # theta ~ power-law-ish degrees
            self.theta = theta * n / theta.sum()
        else:
            self.theta = np.ones(n)
        self.rng = rng
        # keep each block supercritical: mean within-degree >> log(block size)
        if ensure_connected:
            bs = max(n // n_blocks, 2)
            p_in_min = 5.0 * np.log(bs) / bs
            p_in = max(p_in, p_in_min)
        self.p_in, self.p_out = p_in, p_out
        self.A = self._sample_adjacency()
        self._fix_isolates()
        self.changepoints = []          # list of dict(t=event_index, kind=..., ...)
        self.events_since_cp = 0
        self._pending = None

    def _fix_isolates(self, max_tries=100):
        """Isolated vertices make lambda_2 = 0 and poison spectral tracking.

        Rewire each isolate to a random same-block vertex until minimum degree >= 1.
        """
        for _ in range(max_tries):
            deg = self.A.sum(1)
            isol = np.where(deg == 0)[0]
            if len(isol) == 0:
                return
            for v in isol:
                block = self.labels[v]
                cand = np.where(self.labels == block)[0]
                cand = cand[cand != v]
                u = self.rng.choice(cand)
                self.A[v, u] = self.A[u, v] = 1

    # ---- sampling ----
    def _block_probs(self):
        """Return func(i,j)->p under current regime."""
        p_in, p_out = self.p_in, self.p_out
        lab = self.labels
        def p(i, j):
            return p_in if lab[i] == lab[j] else p_out
        return p

    def _sample_adjacency(self):
        n, th = self.n, self.theta
        lab = self.labels
        same = lab[:, None] == lab[None, :]
        P = np.where(same, self.p_in, self.p_out) * np.outer(th, th)
        np.fill_diagonal(P, 0.0)
        A = (self.rng.random((n, n)) < P).astype(float)
        A = np.triu(A, 1)
        A = A + A.T
        return A

    # ---- event stream ----
    def next_event(self):
        """Generate one edge insertion/deletion event from current regime."""
        n = self.n
        p = self._block_probs()
        # sample candidate pair by rejection on current probs
        while True:
            i, j = self.rng.integers(0, n, 2)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            pe = p(i, j) * min(self.theta[i] * self.theta[j], 1.0)
            if self.rng.random() < 0.5:  # proposal: flip a uniform existing/non edge
                if self.A[i, j] == 0 and self.rng.random() < max(pe * n, 1e-6):
                    self.A[i, j] = self.A[j, i] = 1
                    return (i, j, +1)
                if self.A[i, j] == 1 and self.rng.random() < 0.02:
                    self.A[i, j] = self.A[j, i] = 0
                    self._fix_isolates()
                    return (i, j, -1)
            else:
                if self.A[i, j] == 1 and self.rng.random() < 0.05:
                    self.A[i, j] = self.A[j, i] = 0
                    self._fix_isolates()
                    return (i, j, -1)
                if self.A[i, j] == 0 and self.rng.random() < max(pe * n * 0.5, 1e-6):
                    self.A[i, j] = self.A[j, i] = 1
                    return (i, j, +1)

    # ---- planted change points ----
    def trigger(self, t, kind, **kw):
        if kind == "pin_drop":
            self.p_in *= kw.get("factor", 0.5)
        elif kind == "split":
            # split block 0 into two halves -> 3 blocks
            mask = self.labels == 0
            half = self.rng.random(self.n) < 0.5
            self.labels = self.labels.copy()
            self.labels[mask & half] = self.n_blocks
            self.n_blocks += 1
        elif kind == "bridge_cut":
            # delete all cross edges among a random 200-vertex subset pair
            sub = self.rng.choice(self.n, min(200, self.n), replace=False)
            for i in sub[: len(sub) // 2]:
                for j in sub[len(sub) // 2:]:
                    self.A[i, j] = self.A[j, i] = 0
            self._fix_isolates()
        elif kind == "pout_jump":
            self.p_out *= kw.get("factor", 5.0)
        elif kind == "pin_drop_batch":
            # batch-remove a fraction of within-block edges; magnitude is
            # (1 - factor) by construction. NOTE: unlike regime-shift
            # pin_drop, this CP is instantaneous, and its size actually
            # couples to the event stream (the regime version was found to
            # be proposal-saturated: pe*n >> 1, so p_in had no effect).
            frac = 1.0 - kw.get("factor", 0.5)
            same = self.labels[:, None] == self.labels[None, :]
            W = self.A * same
            edges = np.argwhere(np.triu(W, 1) > 0)
            if len(edges):
                k = int(round(frac * len(edges)))
                sel = self.rng.choice(len(edges), k, replace=False)
                for e in sel:
                    i, j = edges[e]
                    self.A[i, j] = self.A[j, i] = 0
            self._fix_isolates()
        self.changepoints.append({"t": t, "kind": kind})
