"""Per-case constraint/violation report for pop's electro+S1 pipeline
(2026-07-14) -- the current lowest-Cost line (~2.72 vs the generative
B*-tree line's 3.3185). Same two-sheet Excel format as `ml/case_report.py`
(Per-Case + Summary), so the two lines can be compared side by side.

Requires pop's electro/ code available on disk (see WINNING_STRATEGY.md /
8_Winning_Strategy_and_Roadmap.md §8.7 for how it was pulled via
`git worktree add ../ICCAD-2026-C-pop-temp upstream/temp`, or point
--electro-dir at any checkout that has electro_optimizer.py).

    python -m ml.case_report_electro --electro-dir "C:\\Users\\wende\\AppData\\Local\\Temp\\electro_probe\\electro" --out case_report_electro.xlsx
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time

from .run_pipeline import load_case_raw, build_dims_and_hints
from .contest_cost import evaluate as evaluate_cost
from .case_report import write_report


def build_electro_inputs(blocks, b2b, p2b, pins_pos, geometry):
    import torch
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
        elif blocks[i, 1] > 0.5:
            w, h = dims[i]
            target_positions[i, 2] = w
            target_positions[i, 3] = h
    return area_targets, constraints, target_positions, dims, is_preplaced


def load_optimizer(electro_dir: str):
    sys.path.insert(0, electro_dir)
    sys.path.insert(0, "d:/ICCAD-2026-C/ICCAD-C-FloorSet-official")
    sys.path.insert(0, "d:/ICCAD-2026-C/ICCAD-C-FloorSet-official/iccad2026contest")
    spec = importlib.util.spec_from_file_location(
        "electro_optimizer_module", f"{electro_dir}/electro_optimizer.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.MyOptimizer()


def run_pipeline_for_report(electro_dir, val, limit):
    opt = load_optimizer(electro_dir)
    records = []
    for case_idx in range(min(limit, 100)):
        t0 = time.time()
        blocks, b2b, p2b, pins_pos, metrics, geometry, cfg_name = load_case_raw(val, case_idx)
        n = blocks.shape[0]
        baseline_area = float(metrics[0])
        baseline_hpwl = float(metrics[6]) + float(metrics[7])

        area_targets, constraints, target_positions, dims, is_preplaced = \
            build_electro_inputs(blocks, b2b, p2b, pins_pos, geometry)

        result = opt.solve(n, area_targets, b2b, p2b, pins_pos, constraints, target_positions)
        runtime_s = time.time() - t0

        dims_dict = {i: (float(result[i][2]), float(result[i][3])) for i in range(n)}
        xd = {i: float(result[i][0]) for i in range(n)}
        yd = {i: float(result[i][1]) for i in range(n)}
        cc = evaluate_cost(xd, yd, dims_dict, blocks, b2b, p2b, pins_pos,
                           baseline_area, baseline_hpwl)

        records.append({
            "case": cfg_name, "n": n, "runtime_s": runtime_s, "feasible": cc.feasible,
            "overlap_violation": cc.overlap_violation, "area_violation": cc.area_violation,
            "fixed_violation": cc.fixed_violation, "preplaced_violation": cc.preplaced_violation,
            "v_grouping": cc.v_grouping, "v_mib": cc.v_mib, "v_boundary": cc.v_boundary,
            "n_soft": cc.n_soft, "v_relative": cc.v_relative,
            "hpwl_int": cc.hpwl_int, "hpwl_ext": cc.hpwl_ext, "hpwl_gap_pct": cc.hpwl_gap * 100.0,
            "area_gap_pct": cc.area_gap * 100.0, "cost": cc.cost,
        })
        print(f"  {cfg_name}: n={n:3d} feasible={cc.feasible} cost={cc.cost:.3f} "
              f"Vgrp={cc.v_grouping} Vmib={cc.v_mib} Vbnd={cc.v_boundary} "
              f"runtime={runtime_s:.2f}s")
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--electro-dir", required=True,
                     help="path to a checkout containing electro_optimizer.py")
    ap.add_argument("--val", default="d:/ICCAD-2026-C/ICCAD-C-FloorSet-official/LiteTensorDataTest")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--out", default="case_report_electro.xlsx")
    args = ap.parse_args()

    print(f"[case_report_electro] electro_dir={args.electro_dir}")
    records = run_pipeline_for_report(args.electro_dir, args.val, args.limit)
    write_report(records, args.out)


if __name__ == "__main__":
    main()
