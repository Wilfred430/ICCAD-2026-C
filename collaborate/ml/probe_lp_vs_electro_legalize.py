"""Compare pop's push/evict legalizer vs a sequence-pair + LP legalizer on
REAL (non-GT) analytical placement output (2026-07-14).

`probe_lp_legalize.py` showed LP-legalization has a much lower density
ceiling than B*-tree+contour when given an independently-derived topology
FROM GT coordinates. This script asks the harder, more realistic question:
does that advantage survive when the input isn't GT at all, but pop's own
electro `analytical_place.place()` output (dense but overlapping, with real
optimization noise/error)? Replaces only the legalize() step -- everything
upstream (the analytical optimization) is untouched, so this isolates the
legalizer's own contribution.

    python -m ml.probe_lp_vs_electro_legalize --cases 0,20,40,60,80,99
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, r"C:\Users\wende\AppData\Local\Temp\electro_probe\electro")

from analytical_place import place as electro_place
from legalize import legalize as electro_legalize, remove_overlap

from .run_pipeline import load_case_raw, build_dims_and_hints
from .contest_cost import evaluate as evaluate_cost


def build_electro_inputs(blocks, b2b, p2b, pins_pos, geometry):
    n = blocks.shape[0]
    dims, is_preplaced, preplaced_xy = build_dims_and_hints(blocks, geometry)
    area_targets = blocks[:, 0].float()
    constraints = blocks[:, 1:6].float()
    target_positions = torch.full((n, 4), -1.0)
    for i in range(n):
        if is_preplaced.get(i, False):
            px, py = preplaced_xy[i]
            w, h = dims[i]
            target_positions[i] = torch.tensor([px, py, w, h])
        elif blocks[i, 1] > 0.5:  # is_fixed
            w, h = dims[i]
            target_positions[i, 2] = w
            target_positions[i, 3] = h
    return area_targets, constraints, target_positions, dims, is_preplaced


def lp_legalize(x, y, w, h, is_pre):
    """Sequence-pair (diagonal-sort) topology extraction + LP tightest-bbox,
    same method validated in probe_lp_legalize.py. Preplaced blocks are
    pinned (equality bounds) since their position is a hard constraint.

    Any pair involving a preplaced block uses an EXACT readout of the
    relation instead of the diagonal-rank proxy: a preplaced block's
    position is fixed and known exactly, so there's no reason to guess it
    from a coarse rank, and using the coarse rank risks deriving a relation
    that directly contradicts the pin (infeasible LP -- seen in practice on
    real analytical-placement output, which is why this fallback exists;
    the GT-only validation in probe_lp_legalize.py never hit this because
    GT already only contains mutually-consistent positions by construction).
    """
    from scipy.optimize import linprog

    n = len(w)
    cx = x + w / 2.0
    cy = y + h / 2.0
    g1 = np.argsort(cx + cy)
    g2 = np.argsort(cx - cy)
    r1 = np.empty(n, dtype=int); r1[g1] = np.arange(n)
    r2 = np.empty(n, dtype=int); r2[g2] = np.arange(n)

    nvar = 2 * n + 2
    IX = lambda i: i
    IY = lambda i: n + i
    IW, IH = 2 * n, 2 * n + 1
    c = np.zeros(nvar); c[IW] = 1.0; c[IH] = 1.0

    def exact_relation(i, j, eps=1e-6):
        if x[i] + w[i] <= x[j] + eps:
            return 'i_left_j'
        if x[j] + w[j] <= x[i] + eps:
            return 'j_left_i'
        if y[i] + h[i] <= y[j] + eps:
            return 'i_below_j'
        if y[j] + h[j] <= y[i] + eps:
            return 'j_below_i'
        return None  # overlapping already -- can't happen for 2 preplaced (would be infeasible input)

    A_ub, b_ub = [], []
    def add(row, rhs):
        A_ub.append(row); b_ub.append(rhs)

    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(nvar)
            if is_pre[i] or is_pre[j]:
                kind = exact_relation(i, j)
                if kind is None:
                    continue  # shouldn't happen; skip rather than crash
            elif r1[i] < r1[j] and r2[i] < r2[j]:
                kind = 'i_left_j'
            elif r1[i] > r1[j] and r2[i] > r2[j]:
                kind = 'j_left_i'
            elif r1[i] < r1[j] and r2[i] > r2[j]:
                kind = 'i_below_j'
            else:
                kind = 'j_below_i'

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
        if is_pre[i]:
            bounds.append((float(x[i]), float(x[i])))
        else:
            bounds.append((0, None))
    for i in range(n):
        if is_pre[i]:
            bounds.append((float(y[i]), float(y[i])))
        else:
            bounds.append((0, None))
    bounds += [(0, None), (0, None)]

    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                   bounds=bounds, method='highs')
    if not res.success:
        raise RuntimeError(f"LP failed: {res.message}")
    return res.x[:n], res.x[n:2 * n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default="d:/ICCAD-2026-C/ICCAD-C-FloorSet-official/LiteTensorDataTest")
    ap.add_argument("--cases", default="0,20,40,60,80,99")
    args = ap.parse_args()
    case_ids = [int(c) for c in args.cases.split(",")]

    for ci in case_ids:
        blocks, b2b, p2b, pins_pos, metrics, geometry, cfg_name = load_case_raw(args.val, ci)
        n = blocks.shape[0]
        baseline_area = float(metrics[0])
        baseline_hpwl = float(metrics[6]) + float(metrics[7])

        area_targets, constraints, target_positions, dims, is_preplaced = \
            build_electro_inputs(blocks, b2b, p2b, pins_pos, geometry)

        positions, _ = electro_place(
            n, area_targets, b2b, p2b, pins_pos, constraints, target_positions,
            iters=600, lr=0.02, device="cpu", seed=0,
        )
        x = np.array([p[0] for p in positions], dtype=float)
        y = np.array([p[1] for p in positions], dtype=float)
        w = np.array([p[2] for p in positions], dtype=float)
        h = np.array([p[3] for p in positions], dtype=float)
        is_pre = np.array([is_preplaced.get(i, False) for i in range(n)], dtype=bool)

        pre_area = w[~is_pre].sum() if (~is_pre).any() else 0.0  # unused, just sanity
        raw_bbox = (x + w).max() * (y + h).max()

        # Path A: pop's own push/evict legalizer.
        xa, ya = electro_legalize(x.copy(), y.copy(), w, h, is_pre)
        xa, ya = remove_overlap(xa, ya, w, h, is_pre)
        dims_dict = {i: (float(w[i]), float(h[i])) for i in range(n)}
        xa_d = {i: float(xa[i]) for i in range(n)}
        ya_d = {i: float(ya[i]) for i in range(n)}
        cc_a = evaluate_cost(xa_d, ya_d, dims_dict, blocks, b2b, p2b, pins_pos,
                              baseline_area, baseline_hpwl)

        # Path B: sequence-pair + LP legalizer.
        try:
            xb, yb = lp_legalize(x.copy(), y.copy(), w, h, is_pre)
            xb_d = {i: float(xb[i]) for i in range(n)}
            yb_d = {i: float(yb[i]) for i in range(n)}
            cc_b = evaluate_cost(xb_d, yb_d, dims_dict, blocks, b2b, p2b, pins_pos,
                                  baseline_area, baseline_hpwl)
            lp_status = f"area_gap={cc_b.area_gap*100:+.1f}% feas={cc_b.feasible} cost={cc_b.cost:.3f}"
        except RuntimeError as e:
            lp_status = f"INFEASIBLE ({e})"

        print(f"case={cfg_name} n={n:>3} raw_analytical_bbox_area={raw_bbox:.1f}  "
              f"| pop_legalize: area_gap={cc_a.area_gap*100:+.1f}% feas={cc_a.feasible} cost={cc_a.cost:.3f}  "
              f"| LP_legalize: {lp_status}")


if __name__ == "__main__":
    main()
