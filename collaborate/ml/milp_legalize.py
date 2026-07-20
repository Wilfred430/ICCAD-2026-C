"""MILP-based legalizer: the principled fix for the sequence-pair+LP
infeasibility problem found in probe_lp_vs_electro_legalize.py (2026-07-14).

Background: `probe_lp_legalize.py` showed sequence-pair+LP has a much lower
density ceiling (~1.15x) than B*-tree+contour (~1.40x) when the topology is
derived from GT. But `probe_lp_vs_electro_legalize.py` found that PRE-fixing
a topology (via diagonal-rank OR exact-position readout) before solving is
only guaranteed realizable when ALL blocks are free -- pinning a preplaced
block to its exact coordinates can make a pre-committed topology infeasible
even with just ONE anchor, because the free blocks' rank/readout was derived
from a noisy, not-yet-legal analytical-placement snapshot that may be
logically inconsistent once the anchor can no longer "shift" to absorb it.

The fix: don't pre-commit to a relation per pair. Let the solver CHOOSE,
via the classical disjunctive (big-M) MILP formulation for 2D non-overlap:
for every pair (i, j), introduce 4 binaries e_ij in {left, right, below,
above} with "at least one true" -- exactly the well-established formulation
used in facility-layout / VLSI placement legality MILPs. This is guaranteed
feasible whenever ANY legal placement respecting the pins exists (no
pre-committed relation to contradict), at the cost of solving an NP-hard
MILP instead of an LP -- practical only with a time budget and a portfolio
fallback to the existing legalizer if it doesn't finish in time.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix


def milp_legalize(x, y, w, h, is_pre, time_limit: float = 15.0, mip_gap: float = 0.05):
    """Returns (x_new, y_new, status) where status is 'optimal', 'feasible'
    (time limit hit but a feasible incumbent was found), or 'failed'
    (no feasible solution found in time -- caller should fall back).

    x, y, w, h: float arrays, current (possibly overlapping) positions/sizes.
    is_pre: bool array, True for blocks whose (x, y) must stay exactly fixed.
    """
    n = len(w)
    t0 = time.time()

    # M = big-enough constant per pair so the "inactive" 3 of 4 disjuncts are
    # never binding. Current bbox extent (plus a margin) is a safe, tight-ish
    # bound given we're legalizing an already-roughly-placed layout.
    M_x = float((x + w).max() - x.min()) + float(w.max()) + 1.0
    M_y = float((y + h).max() - y.min()) + float(h.max()) + 1.0

    # Variable layout: [x_0..x_{n-1}, y_0..y_{n-1}, W, H, e^L_pairs, e^R_pairs, e^B_pairs, e^A_pairs]
    n_pairs = n * (n - 1) // 2
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    NV_CONT = 2 * n + 2
    IX = lambda i: i
    IY = lambda i: n + i
    IW, IH = 2 * n, 2 * n + 1
    # binary blocks, 4 per pair, in pair order
    IE = lambda k, r: NV_CONT + 4 * k + r  # r in {0:left(i,j), 1:left(j,i), 2:below(i,j), 3:below(j,i)}
    nvar = NV_CONT + 4 * n_pairs

    c = np.zeros(nvar)
    c[IW] = 1.0
    c[IH] = 1.0

    integrality = np.zeros(nvar)
    integrality[NV_CONT:] = 1  # binaries

    lb = np.full(nvar, -np.inf)
    ub = np.full(nvar, np.inf)
    lb[:2 * n] = 0.0
    lb[IW] = 0.0; lb[IH] = 0.0
    lb[NV_CONT:] = 0.0
    ub[NV_CONT:] = 1.0
    for i in range(n):
        if is_pre[i]:
            lb[IX(i)] = ub[IX(i)] = float(x[i])
            lb[IY(i)] = ub[IY(i)] = float(y[i])

    rows = []
    rhs_lo = []
    rhs_hi = []

    def add_le(coeffs: dict, rhs):
        rows.append(coeffs)
        rhs_lo.append(-np.inf)
        rhs_hi.append(rhs)

    def add_ge(coeffs: dict, rhs):
        rows.append(coeffs)
        rhs_lo.append(rhs)
        rhs_hi.append(np.inf)

    for k, (i, j) in enumerate(pairs):
        # x_i + w_i - x_j <= M_x*(1 - e0)  ->  x_i - x_j + M_x*e0 <= M_x - w_i
        add_le({IX(i): 1.0, IX(j): -1.0, IE(k, 0): M_x}, M_x - w[i])
        # x_j + w_j - x_i <= M_x*(1 - e1)
        add_le({IX(j): 1.0, IX(i): -1.0, IE(k, 1): M_x}, M_x - w[j])
        # y_i + h_i - y_j <= M_y*(1 - e2)
        add_le({IY(i): 1.0, IY(j): -1.0, IE(k, 2): M_y}, M_y - h[i])
        # y_j + h_j - y_i <= M_y*(1 - e3)
        add_le({IY(j): 1.0, IY(i): -1.0, IE(k, 3): M_y}, M_y - h[j])
        # at least one relation active
        add_ge({IE(k, 0): 1.0, IE(k, 1): 1.0, IE(k, 2): 1.0, IE(k, 3): 1.0}, 1.0)

    for i in range(n):
        add_le({IX(i): 1.0, IW: -1.0}, -w[i])
        add_le({IY(i): 1.0, IH: -1.0}, -h[i])

    A = lil_matrix((len(rows), nvar))
    for r, coeffs in enumerate(rows):
        for col, val in coeffs.items():
            A[r, col] = val
    A = A.tocsr()
    constr = LinearConstraint(A, np.array(rhs_lo), np.array(rhs_hi))
    bounds = Bounds(lb, ub)

    res = milp(c, constraints=[constr], bounds=bounds, integrality=integrality,
               options={"time_limit": time_limit, "mip_rel_gap": mip_gap, "disp": False})

    elapsed = time.time() - t0
    if res.x is None:
        return None, None, "failed", elapsed
    status = "optimal" if res.status == 0 else "feasible"
    xn = res.x[:n]
    yn = res.x[n:2 * n]
    return xn, yn, status, elapsed
