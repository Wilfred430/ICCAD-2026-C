"""Portfolio-of-angles sequence-pair + LP legalizer (2026-07-14).

`probe_lp_vs_electro_legalize.py` found that a single sequence-pair
derivation (diagonal x+y / x-y ranking) is often infeasible once a
preplaced block is pinned, because the rank was read off a noisy,
not-yet-legal analytical-placement snapshot. A full MILP fix (letting the
solver choose consistent relations) was tried in `milp_legalize.py` and
found NOT to scale -- it fails to find any feasible solution within 20s for
n >= 51, and scipy's `milp` has no warm-start support to help.

This is the pragmatic middle path: LP solves are cheap (milliseconds), so
instead of committing to ONE rank derivation, try SEVERAL (rotate the
diagonal projection angle used for the two seq-pair permutations) and keep
whichever succeeds with the best (lowest W+H) result -- exactly the
"portfolio racing, cost-aware selection, never worse" pattern used
throughout this project's other passes. Falls back to signalling failure
(caller should use the existing legalizer) if NONE of the tried angles are
feasible.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog


def _relations_at_angle(x, y, w, h, is_pre, theta):
    """Sequence-pair via two ranks: rank1 from projecting centroids onto
    direction theta, rank2 from the perpendicular direction (theta+90deg).
    theta=45deg reproduces the original x+y / x-y diagonal derivation."""
    n = len(w)
    cx = x + w / 2.0
    cy = y + h / 2.0
    ct, st = np.cos(theta), np.sin(theta)
    proj1 = cx * ct + cy * st
    proj2 = cx * (-st) + cy * ct
    r1 = np.empty(n, dtype=int); r1[np.argsort(proj1)] = np.arange(n)
    r2 = np.empty(n, dtype=int); r2[np.argsort(proj2)] = np.arange(n)

    def exact_relation(i, j, eps=1e-6):
        if x[i] + w[i] <= x[j] + eps:
            return 'i_left_j'
        if x[j] + w[j] <= x[i] + eps:
            return 'j_left_i'
        if y[i] + h[i] <= y[j] + eps:
            return 'i_below_j'
        if y[j] + h[j] <= y[i] + eps:
            return 'j_below_i'
        return None

    rels = []
    for i in range(n):
        for j in range(i + 1, n):
            if is_pre[i] or is_pre[j]:
                kind = exact_relation(i, j)
                if kind is None:
                    continue
            elif r1[i] < r1[j] and r2[i] < r2[j]:
                kind = 'i_left_j'
            elif r1[i] > r1[j] and r2[i] > r2[j]:
                kind = 'j_left_i'
            elif r1[i] < r1[j] and r2[i] > r2[j]:
                kind = 'i_below_j'
            else:
                kind = 'j_below_i'
            rels.append((i, j, kind))
    return rels


def _solve_lp(w, h, rels, is_pre, x, y):
    n = len(w)
    nvar = 2 * n + 2
    IX = lambda i: i
    IY = lambda i: n + i
    IW, IH = 2 * n, 2 * n + 1
    c = np.zeros(nvar); c[IW] = 1.0; c[IH] = 1.0

    A_ub, b_ub = [], []
    def add(row, rhs):
        A_ub.append(row); b_ub.append(rhs)

    for (i, j, kind) in rels:
        row = np.zeros(nvar)
        if kind == 'i_left_j':
            row[IX(i)] = 1.0; row[IX(j)] = -1.0; add(row, -w[i])
        elif kind == 'j_left_i':
            row[IX(j)] = 1.0; row[IX(i)] = -1.0; add(row, -w[j])
        elif kind == 'i_below_j':
            row[IY(i)] = 1.0; row[IY(j)] = -1.0; add(row, -h[i])
        else:
            row[IY(j)] = 1.0; row[IY(i)] = -1.0; add(row, -h[j])

    for i in range(n):
        row = np.zeros(nvar); row[IX(i)] = 1.0; row[IW] = -1.0; add(row, -w[i])
        row = np.zeros(nvar); row[IY(i)] = 1.0; row[IH] = -1.0; add(row, -h[i])

    bounds = []
    for i in range(n):
        bounds.append((float(x[i]), float(x[i])) if is_pre[i] else (0, None))
    for i in range(n):
        bounds.append((float(y[i]), float(y[i])) if is_pre[i] else (0, None))
    bounds += [(0, None), (0, None)]

    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                   bounds=bounds, method='highs')
    if not res.success:
        return None
    return res.x[:n], res.x[n:2 * n], res.x[2 * n], res.x[2 * n + 1]


def portfolio_lp_legalize(x, y, w, h, is_pre, n_angles: int = 12):
    """Try n_angles evenly-spaced diagonal angles in [0, 90) degrees, keep
    the feasible result with smallest W+H. Returns (x, y, n_succeeded) or
    (None, None, 0) if every angle was infeasible."""
    best = None
    best_score = None
    n_ok = 0
    for k in range(n_angles):
        theta = (np.pi / 2) * (k / n_angles) + (np.pi / 4) / n_angles  # avoid exact axis-aligned ties
        rels = _relations_at_angle(x, y, w, h, is_pre, theta)
        sol = _solve_lp(w, h, rels, is_pre, x, y)
        if sol is None:
            continue
        n_ok += 1
        xn, yn, W, H = sol
        score = W + H
        if best_score is None or score < best_score:
            best_score = score
            best = (xn, yn)
    if best is None:
        return None, None, n_ok
    return best[0], best[1], n_ok
