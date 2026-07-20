"""M3-probe-style validation of an LP-based legalizer (2026-07-14).

Replicates pop's `probe_m3_tree.py` methodology (see AI-deep-search or
upstream/temp:electro/NEXT_STEPS.md sec 5.3): take GT (fp_sol) positions,
reduce them to a topology, rebuild geometry from ONLY that topology, and
measure the area ratio vs GT. Pop's version used a B*-tree + contour packer
and found area ratio 1.403 (a structural density ceiling -- see M3 probe
findings, independently confirmed this session via probe_m3_tree.py results
and our own B*-tree line's area_gap).

This script tests a DIFFERENT topology-to-geometry representation: instead
of a B*-tree (which forces packer.cpp's left/bottom contour-only monotonic
growth), extract the exact pairwise separating relation each block pair
already has in GT (every pair of non-overlapping axis-aligned rectangles is
separated by at least one of: i-left-of-j, j-left-of-i, i-below-j,
j-below-i -- true by definition of non-overlap, no heuristic guessing
needed), then solve a linear program that finds the TIGHTEST bounding box
(min width+height) subject to keeping every pair's GT-derived separating
relation. GT itself is trivially a feasible point of this LP (it already
satisfies its own separating relations), so if the LP's optimal bbox is
close to GT's own bbox, the representation is NOT capping density the way
B*-tree/contour does; if it's similarly bloated, the ceiling is more
fundamental than just "which packer."

    python -m ml.probe_lp_legalize --cases 0,50,200,500,900 --limit 10
"""

from __future__ import annotations

import argparse
import random

import torch
import numpy as np
from scipy.optimize import linprog


def load_fp_sol_case(path: str, ci: int):
    """Returns (w, h, x, y) each [N] float arrays for one TRAIN-format case."""
    t = torch.load(path, weights_only=False)
    fp_sol = t[5][ci]  # [N, 4] = (w, h, x, y)
    w = fp_sol[:, 0].double().numpy()
    h = fp_sol[:, 1].double().numpy()
    x = fp_sol[:, 2].double().numpy()
    y = fp_sol[:, 3].double().numpy()
    return w, h, x, y


def derive_relations_exact(w, h, x, y, eps=1e-6):
    """(Tautological -- kept only for a sanity check.) For every pair (i, j),
    read off which of the 4 axis-aligned separating relations GT already
    satisfies. Since GT trivially satisfies its OWN exact per-pair relation,
    handing the LP GT's own answer key and asking it to match GT is circular
    -- of course it reproduces GT. Not a real test of the representation's
    power. See `derive_relations_seqpair` for the real (independent) test."""
    n = len(w)
    rels = []
    for i in range(n):
        for j in range(i + 1, n):
            if x[i] + w[i] <= x[j] + eps:
                rels.append((i, j, 'i_left_j'))
            elif x[j] + w[j] <= x[i] + eps:
                rels.append((i, j, 'j_left_i'))
            elif y[i] + h[i] <= y[j] + eps:
                rels.append((i, j, 'i_below_j'))
            elif y[j] + h[j] <= y[i] + eps:
                rels.append((i, j, 'j_below_i'))
            else:
                raise RuntimeError(f"pair ({i},{j}) overlaps in GT -- data bug")
    return rels


def derive_relations_seqpair(w, h, x, y):
    """Classical sequence-pair extraction (Murata et al. 1995): derive TWO
    permutations from a coarse per-block scalar key (diagonal sums/diffs of
    the GT centroid), not from each pair's exact GT relation -- this is the
    fair, non-circular analogue of pop's 'greedy nearest-slot B*-tree
    extraction from GT' (an independent RECONSTRUCTION of topology from
    coordinates, not a readout of the exact answer). Any two permutations
    are guaranteed realizable (no cycles), so the resulting LP is always
    feasible. Returns a list of (i, j, kind)."""
    n = len(w)
    cx = x + w / 2.0
    cy = y + h / 2.0
    gamma1 = np.argsort(cx + cy)          # rank of each block along +diag
    gamma2 = np.argsort(cx - cy)          # rank of each block along -diag
    rank1 = np.empty(n, dtype=int); rank1[gamma1] = np.arange(n)
    rank2 = np.empty(n, dtype=int); rank2[gamma2] = np.arange(n)

    rels = []
    for i in range(n):
        for j in range(i + 1, n):
            if rank1[i] < rank1[j] and rank2[i] < rank2[j]:
                rels.append((i, j, 'i_left_j'))
            elif rank1[i] > rank1[j] and rank2[i] > rank2[j]:
                rels.append((i, j, 'j_left_i'))
            elif rank1[i] < rank1[j] and rank2[i] > rank2[j]:
                rels.append((i, j, 'i_below_j'))
            else:
                rels.append((i, j, 'j_below_i'))
    return rels


def solve_tightest_bbox(w, h, rels, x0, y0):
    """LP: minimize (W + H) subject to every GT-derived separating relation
    and 0 <= x_i, 0 <= y_i (blocks can't go negative; W, H are free upper
    bounds derived from the solution itself). Variables: x_0..x_{n-1},
    y_0..y_{n-1}, W, H. Returns (x, y, W, H)."""
    n = len(w)
    nvar = 2 * n + 2
    IX = lambda i: i
    IY = lambda i: n + i
    IW = 2 * n
    IH = 2 * n + 1

    c = np.zeros(nvar)
    c[IW] = 1.0
    c[IH] = 1.0

    A_ub = []
    b_ub = []

    def add(row, rhs):
        A_ub.append(row)
        b_ub.append(rhs)

    for (i, j, kind) in rels:
        row = np.zeros(nvar)
        if kind == 'i_left_j':
            row[IX(i)] = 1.0; row[IX(j)] = -1.0
            add(row, -w[i])
        elif kind == 'j_left_i':
            row[IX(j)] = 1.0; row[IX(i)] = -1.0
            add(row, -w[j])
        elif kind == 'i_below_j':
            row[IY(i)] = 1.0; row[IY(j)] = -1.0
            add(row, -h[i])
        elif kind == 'j_below_i':
            row[IY(j)] = 1.0; row[IY(i)] = -1.0
            add(row, -h[j])

    for i in range(n):
        row = np.zeros(nvar); row[IX(i)] = 1.0; row[IW] = -1.0
        add(row, -w[i])
        row = np.zeros(nvar); row[IY(i)] = 1.0; row[IH] = -1.0
        add(row, -h[i])

    bounds = [(0, None)] * (2 * n) + [(0, None), (0, None)]

    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                   bounds=bounds, method='highs')
    if not res.success:
        raise RuntimeError(f"LP infeasible/failed: {res.message}")
    x = res.x[:n]
    y = res.x[n:2 * n]
    W = res.x[IW]
    H = res.x[IH]
    return x, y, W, H


def check_overlap_free(w, h, x, y, eps=1e-4):
    n = len(w)
    for i in range(n):
        for j in range(i + 1, n):
            if (x[i] < x[j] + w[j] - eps and x[j] < x[i] + w[i] - eps and
                    y[i] < y[j] + h[j] - eps and y[j] < y[i] + h[i] - eps):
                return False
    return True


ROOT = "d:/ICCAD-2026-C/ICCAD-C-FloorSet-official/floorset_lite/worker_0"
SIZE_DIVERSE_FILES = [
    (f"{ROOT}/layouts_5040.th", 0),   # n=21
    (f"{ROOT}/layouts_1792.th", 0),   # n=24
    (f"{ROOT}/layouts_784.th", 0),    # n=35
    (f"{ROOT}/layouts_9744.th", 0),   # n=40
    (f"{ROOT}/layouts_4928.th", 0),   # n=52
    (f"{ROOT}/layouts_3472.th", 0),   # n=60
    (f"{ROOT}/layouts_8736.th", 0),   # n=73
    (f"{ROOT}/layouts_0.th", 0),      # n=75
    (f"{ROOT}/layouts_4144.th", 0),   # n=84
    (f"{ROOT}/layouts_336.th", 0),    # n=91
    (f"{ROOT}/layouts_2128.th", 0),   # n=100
    (f"{ROOT}/layouts_1120.th", 0),   # n=108
    (f"{ROOT}/layouts_1904.th", 0),   # n=116
    (f"{ROOT}/layouts_2016.th", 0),   # n=118
    (f"{ROOT}/layouts_8624.th", 0),   # n=119
    (f"{ROOT}/layouts_3360.th", 0),   # n=120
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=None)
    ap.add_argument("--n-cases", type=int, default=11)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size-diverse", action="store_true",
                     help="use a fixed n=21..120 spread across files instead of random sampling one file")
    args = ap.parse_args()

    if args.size_diverse:
        pairs = SIZE_DIVERSE_FILES
    else:
        f = args.file or f"{ROOT}/layouts_0.th"
        t = torch.load(f, weights_only=False)
        total_cases = t[5].shape[0]
        print(f"[probe] file has {total_cases} cases")
        rng = random.Random(args.seed)
        case_ids = sorted(rng.sample(range(total_cases), min(args.n_cases, total_cases)))
        pairs = [(f, ci) for ci in case_ids]

    ratios_exact = []
    ratios_sp = []
    for (fpath, ci) in pairs:
        w, h, x, y = load_fp_sol_case(fpath, ci)
        n = len(w)
        gt_W = float((x + w).max())
        gt_H = float((y + h).max())
        gt_area = gt_W * gt_H

        rels_exact = derive_relations_exact(w, h, x, y)
        _, _, LWe, LHe = solve_tightest_bbox(w, h, rels_exact, x, y)
        ratio_exact = (LWe * LHe) / gt_area
        ratios_exact.append(ratio_exact)

        rels_sp = derive_relations_seqpair(w, h, x, y)
        lx, ly, LW, LH = solve_tightest_bbox(w, h, rels_sp, x, y)
        ok = check_overlap_free(w, h, lx, ly)
        ratio_sp = (LW * LH) / gt_area
        ratios_sp.append(ratio_sp)

        print(f"  case={ci:>6} n={n:>3} GT_bbox=({gt_W:.1f}x{gt_H:.1f})={gt_area:.1f}  "
              f"exact_areaR={ratio_exact:.4f} (sanity, expect ~1.0)  "
              f"seqpair_areaR={ratio_sp:.4f}  overlap_free={ok}")

    print(f"\n[probe] mean exact_areaR   = {sum(ratios_exact)/len(ratios_exact):.4f}  "
          f"(tautological sanity check -- should be ~1.0)")
    print(f"[probe] mean seqpair_areaR = {sum(ratios_sp)/len(ratios_sp):.4f}  "
          f"(REAL test: independent topology extraction, like pop's M3 probe;  "
          f"pop's B*-tree+contour got 1.403 on an analogous test)")


if __name__ == "__main__":
    main()
