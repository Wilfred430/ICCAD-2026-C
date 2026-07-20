"""Diagnose WHERE the soft-constraint violations (V_rel) come from, per case.

    python -m ml.diag_vrel --samples 4 --limit 100

For each validation case it samples K topologies, packs with the full repair
pipeline, and reports the best-cost pack's breakdown of V_grouping / V_mib /
V_boundary and N_soft. The goal (feasible + V_rel = 0) needs this to know
which constraint type to attack first.
"""

from __future__ import annotations

import argparse
import math

import torch

from .data import case_to_features
from .model_tree import TreeGenerator
from .pack_tree import build_lc_rc, pack_btree
from .contest_cost import evaluate as evaluate_cost
from .run_pipeline import load_case_raw, build_dims_and_hints
from .eval_full import dims_with_aspect, ASPECT_RATIOS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="ml/weights/tree_v1.pt")
    ap.add_argument("--val", default="d:/ICCAD-2026-C/ICCAD-C-FloorSet-official/LiteTensorDataTest")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=100)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    gen = torch.Generator(device=args.device).manual_seed(args.seed)

    ckpt = torch.load(args.weights, map_location=args.device, weights_only=False)
    cfg = ckpt["config"]
    model = TreeGenerator(hidden_dim=cfg["hidden_dim"], n_ctx_layers=cfg["n_ctx_layers"],
                           n_dec_layers=cfg["n_dec_layers"], n_heads=cfg["n_heads"],
                           max_blocks=cfg["max_blocks"]).to(args.device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    max_n, max_t = cfg["max_blocks"], cfg["max_terms"]

    # aggregate counters
    cases_with_vrel = 0
    sum_vgrp = sum_vmib = sum_vbnd = sum_nsoft = 0
    cases_any_grp = cases_any_mib = cases_any_bnd = 0
    worst = []  # (v_relative, cfg_name, vg, vm, vb, nsoft)

    for case_idx in range(min(args.limit, 100)):
        blocks, b2b, p2b, pins_pos, metrics, geometry, cfg_name = load_case_raw(args.val, case_idx)
        n = blocks.shape[0]
        baseline_area = float(metrics[0])
        baseline_hpwl = float(metrics[6]) + float(metrics[7])

        feat = case_to_features(blocks, b2b, p2b, geometry)
        bf = torch.zeros((1, max_n, feat.shape[1])); bf[0, :n] = feat
        bm = torch.zeros((1, max_n), dtype=torch.bool); bm[0, :n] = True
        t_use = min(pins_pos.shape[0], max_t)
        tm_t = torch.zeros((1, max_t, 2)); tm_t[0, :t_use] = pins_pos[:t_use]
        tmm = torch.zeros((1, max_t), dtype=torch.bool); tmm[0, :t_use] = True

        base_dims, is_pre, pre_xy = build_dims_and_hints(blocks, geometry)
        cid = {i: int(blocks[i, 4]) for i in range(n)}
        bcode = {i: int(blocks[i, 5]) for i in range(n)}

        best = None
        for _ in range(args.samples):
            out = model.generate(bf.to(args.device), bm.to(args.device),
                                  tm_t.to(args.device), tmm.to(args.device),
                                  n_blocks=n, temperature=1.0, sample=True, generator=gen)
            root = int(out["gen_order"][0])
            lc, rc = build_lc_rc(root, out["parent_id"], out["direction"], n,
                                 gen_order=out["gen_order"].tolist())
            for r in ASPECT_RATIOS:
                dims = dims_with_aspect(blocks, base_dims, r)
                pack = pack_btree(root, lc, rc, dims, is_pre, pre_xy,
                                  baseline_area=baseline_area, cluster_id=cid, boundary_code=bcode)
                cc = evaluate_cost(pack.x, pack.y, dims, blocks, b2b, p2b, pins_pos,
                                   baseline_area, baseline_hpwl)
                if best is None or cc.cost < best.cost:
                    best = cc

        vg, vm, vb, ns, vr = (best.v_grouping, best.v_mib, best.v_boundary,
                              best.n_soft, best.v_relative)
        sum_vgrp += vg; sum_vmib += vm; sum_vbnd += vb; sum_nsoft += ns
        if vr > 1e-9:
            cases_with_vrel += 1
        if vg > 0: cases_any_grp += 1
        if vm > 0: cases_any_mib += 1
        if vb > 0: cases_any_bnd += 1
        worst.append((vr, cfg_name, vg, vm, vb, ns))

    worst.sort(reverse=True)
    print("\n" + "=" * 70)
    print(f"V_rel diagnosis over {min(args.limit,100)} cases (best-cost pack per case)")
    print("=" * 70)
    print(f"cases with ANY V_rel > 0 : {cases_with_vrel}/{min(args.limit,100)}")
    print(f"  of which grouping viol : {cases_any_grp}")
    print(f"  of which MIB viol      : {cases_any_mib}")
    print(f"  of which boundary viol : {cases_any_bnd}")
    print(f"total V_grouping={sum_vgrp}  V_mib={sum_vmib}  V_boundary={sum_vbnd}  "
          f"(sum N_soft={sum_nsoft})")
    print("\nworst 12 cases by V_rel:")
    print(f"{'case':>12} {'V_rel':>7} {'Vgrp':>5} {'Vmib':>5} {'Vbnd':>5} {'Nsoft':>6}")
    for vr, name, vg, vm, vb, ns in worst[:12]:
        print(f"{name:>12} {vr:>7.3f} {vg:>5} {vm:>5} {vb:>5} {ns:>6}")


if __name__ == "__main__":
    main()
