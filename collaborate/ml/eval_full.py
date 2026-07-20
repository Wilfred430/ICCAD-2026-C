"""Full 100-case validation harness for the generative B*-tree pipeline,
with an A/B comparison of soft-block shape optimization.

    python -m ml.eval_full --weights ml/weights/tree_v1.pt --samples 4

What it measures
----------------
Runs the trained TreeGenerator over ALL 100 validation cases (config_21 ..
config_120), and for each case compares two packing strategies on the SAME
sampled topologies:

  * BEFORE  : soft blocks are placeholder squares (w = h = sqrt(area)) -- the
              current pipeline behaviour.
  * AFTER   : soft blocks get a per-case global aspect-ratio sweep (pick the
              single r that minimizes contest Cost, re-packing for each r).
              The sweep set includes r = 1.0 (the square), so AFTER can never
              score worse than BEFORE on a given topology -- the delta is
              purely the value of the added shape freedom.

Both use the REAL contest cost formula (ml/contest_cost.py, now with the
max(0,.) gap clamp matching the official evaluator) and each case's own
baseline metrics. RuntimeFactor is fixed at 1.0 (neutral: R = 1.0) so the
comparison isolates quality + violations, not runtime -- the real RT is a
cross-team, per-case moving target we cannot compute offline (see CLAUDE.md
gotcha #7 and the note in WINNING_STRATEGY.md §7).

Total Score uses the spec's e^(n/12) weighting (NOT the plain e^n that the
local iccad2026_evaluate.py happens to implement -- the spec PDF is
authoritative). We also print the unweighted mean and feasibility rate.
"""

from __future__ import annotations

import argparse
import math
import time

import torch

from .data import case_to_features
from .model_tree import TreeGenerator
from .pack_tree import build_lc_rc, pack_btree
from .contest_cost import evaluate as evaluate_cost
from .run_pipeline import load_case_raw, build_dims_and_hints


# Aspect ratios swept for soft blocks. MUST include 1.0 (the square) so that
# the shape-optimized pack is never worse than the placeholder-square pack.
ASPECT_RATIOS = [0.5, 0.7, 0.85, 1.0, 1.25, 1.5, 2.0]


def dims_with_aspect(blocks, base_dims, r: float):
    """Copy base_dims (which has correct locked dims for fixed/preplaced) and
    reshape every SOFT block to aspect ratio r, preserving its target area
    exactly (w*h = area, so the 1% area tolerance is satisfied by construction).

    MIB consistency (by construction, drives V_mib to 0): all members of a MIB
    group (col 3 = mib_id) must share ONE (w,h). If the group contains a
    fixed/preplaced member, its locked shape is the shared shape and every soft
    member adopts it (this was the V_mib>0 bug -- soft members were being swept
    to a different aspect than their group's fixed anchor). An all-soft group
    already shares a shape (same area + same r -> identical dims), but we still
    pin them to the group's first member for robustness against float drift.
    """
    dims = dict(base_dims)
    n = blocks.shape[0]

    # First pass: soft blocks get the swept aspect.
    for i in range(n):
        is_fixed = bool(blocks[i, 1] > 0.5)
        is_pre = bool(blocks[i, 2] > 0.5)
        if is_fixed or is_pre:
            continue
        area = max(float(blocks[i, 0]), 1.0)
        dims[i] = (math.sqrt(area * r), math.sqrt(area / r))

    # Second pass: enforce one shared (w,h) per MIB group.
    mib_groups = {}
    for i in range(n):
        mid = int(blocks[i, 3])
        if mid > 0:
            mib_groups.setdefault(mid, []).append(i)
    for members in mib_groups.values():
        if len(members) <= 1:
            continue
        anchor = None
        for i in members:  # prefer a fixed/preplaced member as the shared shape
            if bool(blocks[i, 1] > 0.5) or bool(blocks[i, 2] > 0.5):
                anchor = i
                break
        if anchor is None:
            anchor = members[0]
        shared = dims[anchor]
        for i in members:
            if bool(blocks[i, 1] > 0.5) or bool(blocks[i, 2] > 0.5):
                continue  # keep locked members as-is (they equal the anchor)
            dims[i] = shared
    return dims


def best_over_aspects(root, lc, rc, blocks, base_dims, is_preplaced, preplaced_xy,
                      b2b, p2b, pins_pos, baseline_area, baseline_hpwl, aspects,
                      cluster_id=None, boundary_code=None):
    """Pack the given topology at each aspect ratio, return the lowest-Cost one.

    (A "tight vs full repair" portfolio was tested 2026-07-09 and reverted:
    the area-tight mode almost never beat full repair on cost -- even at
    area_gap +168%, using aggressive boundary to cut V_rel is cost-optimal
    because exp(2*V_rel) dominates. So it only doubled runtime for ~0.06%.
    The area blow-up is correctly priced; escaping it needs by-construction
    placement, not a cheaper repair mode.)"""
    best_cc = None
    best_pack = None
    for r in aspects:
        dims = dims_with_aspect(blocks, base_dims, r)
        # push_past on/off portfolio: ON guarantees RIGHT/TOP boundary contact
        # but grows the bbox (area); OFF keeps area tight but leaves those
        # boundary blocks violated. Cheaper wins per (topology, aspect) -- lets
        # each case trade area vs V_rel however the real Cost prefers.
        for pp in (True, False):
            pack = pack_btree(root, lc, rc, dims, is_preplaced, preplaced_xy,
                              baseline_area=baseline_area, cluster_id=cluster_id,
                              boundary_code=boundary_code, boundary_push_past=pp)
            cc = evaluate_cost(pack.x, pack.y, dims, blocks, b2b, p2b, pins_pos,
                                baseline_area, baseline_hpwl)
            if best_cc is None or cc.cost < best_cc.cost:
                best_cc, best_pack = cc, pack
    return best_pack, best_cc


def hpwl_nudge(pack, dims, blocks, b2b, p2b, pins_pos, rounds: int = 2):
    """Post-pack wirelength refinement (attacks the now-dominant hpwl_gap term).

    After constraint repairs, the aggressive boundary pass has dragged blocks
    far from their connected neighbours, inflating HPWL. This slides each FREE
    block (not preplaced / boundary / cluster -- those are constraint-pinned)
    toward the weighted centroid of its b2b/p2b neighbours, but ONLY to an
    overlap-free slot that does NOT grow the bbox. Strictly non-worsening on
    bbox area; reduces HPWL. Mutates pack.x / pack.y in place.
    """
    n = blocks.shape[0]
    x, y = pack.x, pack.y
    bbox_w, bbox_h = pack.bbox_w, pack.bbox_h

    free = {}
    for i in range(n):
        free[i] = not (blocks[i, 1] > 0.5 or blocks[i, 2] > 0.5
                       or int(blocks[i, 4]) > 0 or int(blocks[i, 5]) > 0)

    # Build weighted neighbour lists (fixed neighbour centroids; we move one block).
    nbr = {i: [] for i in range(n)}  # list of (nx, ny, weight)
    cx = {i: x[i] + dims[i][0] / 2 for i in range(n)}
    cy = {i: y[i] + dims[i][1] / 2 for i in range(n)}
    for row in b2b.tolist():
        a, b, w = int(row[0]), int(row[1]), row[2]
        if 0 <= a < n and 0 <= b < n and w > 0:
            nbr[a].append((b, w)); nbr[b].append((a, w))
    pin_nbr = {i: [] for i in range(n)}
    for row in p2b.tolist():
        pin, b, w = int(row[0]), int(row[1]), row[2]
        if 0 <= b < n and w > 0:
            pin_nbr[b].append((float(pins_pos[pin][0]), float(pins_pos[pin][1]), w))

    def overlaps(i, nx, ny):
        w, h = dims[i]
        for j in range(n):
            if j == i:
                continue
            if (nx + 1e-9 < x[j] + dims[j][0] and x[j] + 1e-9 < nx + w and
                    ny + 1e-9 < y[j] + dims[j][1] and y[j] + 1e-9 < ny + h):
                return True
        return False

    def contrib(i, ccx, ccy):
        s = 0.0
        for (j, w) in nbr[i]:
            s += w * (abs(ccx - cx[j]) + abs(ccy - cy[j]))
        for (px, py, w) in pin_nbr[i]:
            s += w * (abs(ccx - px) + abs(ccy - py))
        return s

    for _ in range(rounds):
        moved = False
        for i in range(n):
            if not free[i] or (not nbr[i] and not pin_nbr[i]):
                continue
            w, h = dims[i]
            # weighted-mean target for block i's centroid
            tw = sum(ww for _, ww in nbr[i]) + sum(ww for *_, ww in pin_nbr[i])
            if tw <= 0:
                continue
            tx = (sum(ww * cx[j] for j, ww in nbr[i])
                  + sum(ww * px for px, _, ww in pin_nbr[i])) / tw
            ty = (sum(ww * cy[j] for j, ww in nbr[i])
                  + sum(ww * py for _, py, ww in pin_nbr[i])) / tw
            # candidate lower-left positions: ideal + nearest block corners to ideal
            cands = [(tx - w / 2, ty - h / 2)]
            for j in range(n):
                if j == i:
                    continue
                cands.append((x[j], y[j]))
                cands.append((x[j] + dims[j][0], y[j]))
                cands.append((x[j], y[j] + dims[j][1]))
            cur = contrib(i, cx[i], cy[i])
            best_pos, best_c = (x[i], y[i]), cur
            # only consider candidates reasonably near the ideal (cap work)
            cands.sort(key=lambda p: abs(p[0] + w / 2 - tx) + abs(p[1] + h / 2 - ty))
            for nx, ny in cands[:24]:
                if nx < -1e-9 or ny < -1e-9:
                    continue
                if nx + w > bbox_w + 1e-9 or ny + h > bbox_h + 1e-9:
                    continue  # never grow the bbox
                c = contrib(i, nx + w / 2, ny + h / 2)
                if c < best_c - 1e-9 and not overlaps(i, nx, ny):
                    best_c, best_pos = c, (nx, ny)
            if best_pos != (x[i], y[i]):
                x[i], y[i] = best_pos
                cx[i], cy[i] = x[i] + w / 2, y[i] + h / 2
                moved = True
        if not moved:
            break


def hpwl_nudge_clusters(pack, dims, blocks, b2b, p2b, pins_pos, rounds: int = 2):
    """Rigid-body HPWL nudge for whole grouping clusters (2026-07-09).

    `hpwl_nudge` (above) explicitly excludes every cluster member
    (`blocks[i, 4] > 0`) since moving one member alone could break
    V_group. But a RIGID slide -- every member of a cluster shifted by the
    SAME delta -- never changes relative positions, so V_group is
    invariant regardless of whether the cluster happens to be connected
    (by-construction-collapsed or still post-hoc-repaired). This closes
    that gap: cluster members currently get ZERO benefit from HPWL
    nudging. Slides the whole cluster toward the weighted centroid of its
    members' EXTERNAL b2b/p2b connections only (an internal, cluster-to-
    cluster net doesn't change under a rigid shift, so it contributes
    nothing to the target and is excluded). Only accepts a move that
    doesn't grow the bbox and doesn't overlap anything outside the
    cluster. Mutates pack.x / pack.y in place.
    """
    n = blocks.shape[0]
    x, y = pack.x, pack.y
    bbox_w, bbox_h = pack.bbox_w, pack.bbox_h

    groups = {}
    for i in range(n):
        cid = int(blocks[i, 4])
        if cid > 0:
            groups.setdefault(cid, []).append(i)
    if not groups:
        return

    cx = {i: x[i] + dims[i][0] / 2 for i in range(n)}
    cy = {i: y[i] + dims[i][1] / 2 for i in range(n)}
    b2b_rows = [(int(r[0]), int(r[1]), r[2]) for r in b2b.tolist() if r[2] > 0]
    p2b_rows = [(int(r[0]), int(r[1]), r[2]) for r in p2b.tolist() if r[2] > 0]

    for _ in range(rounds):
        moved_any = False
        for members in groups.values():
            mset = set(members)
            wsum = tx_sum = ty_sum = 0.0
            for a, b, w in b2b_rows:
                a_in, b_in = a in mset, b in mset
                if a_in and not b_in:
                    tx_sum += cx[b] * w; ty_sum += cy[b] * w; wsum += w
                elif b_in and not a_in:
                    tx_sum += cx[a] * w; ty_sum += cy[a] * w; wsum += w
            for pin, b, w in p2b_rows:
                if b in mset:
                    tx_sum += float(pins_pos[pin][0]) * w
                    ty_sum += float(pins_pos[pin][1]) * w
                    wsum += w
            if wsum <= 0:
                continue
            target_cx, target_cy = tx_sum / wsum, ty_sum / wsum

            min_x = min(x[i] for i in members)
            max_x = max(x[i] + dims[i][0] for i in members)
            min_y = min(y[i] for i in members)
            max_y = max(y[i] + dims[i][1] for i in members)
            cur_cx, cur_cy = (min_x + max_x) / 2, (min_y + max_y) / 2
            ideal_dx, ideal_dy = target_cx - cur_cx, target_cy - cur_cy
            if abs(ideal_dx) < 1e-9 and abs(ideal_dy) < 1e-9:
                continue

            def fits(ddx, ddy):
                if min_x + ddx < -1e-9 or min_y + ddy < -1e-9:
                    return False
                if max_x + ddx > bbox_w + 1e-9 or max_y + ddy > bbox_h + 1e-9:
                    return False
                for i in members:
                    nx_, ny_ = x[i] + ddx, y[i] + ddy
                    wi, hi = dims[i]
                    for j in range(n):
                        if j in mset:
                            continue
                        if (nx_ + 1e-9 < x[j] + dims[j][0] and x[j] + 1e-9 < nx_ + wi and
                                ny_ + 1e-9 < y[j] + dims[j][1] and y[j] + 1e-9 < ny_ + hi):
                            return False
                return True

            best_dx = best_dy = 0.0
            for frac in (1.0, 0.5, 0.25, 0.125):
                ddx, ddy = ideal_dx * frac, ideal_dy * frac
                if fits(ddx, ddy):
                    best_dx, best_dy = ddx, ddy
                    break
            if best_dx != 0.0 or best_dy != 0.0:
                for i in members:
                    x[i] += best_dx
                    y[i] += best_dy
                    cx[i] += best_dx
                    cy[i] += best_dy
                moved_any = True
        if not moved_any:
            break


def total_score(costs, block_counts, tau: float = 12.0):
    """Spec weighting: Total = sum(cost_i * e^(n_i/tau)) / sum(e^(n_j/tau))."""
    ws = [math.exp(n / tau) for n in block_counts]
    return sum(c * w for c, w in zip(costs, ws)) / sum(ws)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="ml/weights/tree_v1.pt")
    ap.add_argument("--val", default="d:/ICCAD-2026-C/ICCAD-C-FloorSet-official/LiteTensorDataTest")
    ap.add_argument("--samples", type=int, default=4, help="topologies sampled per case")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=100, help="number of cases (debug: <100)")
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
    print(f"[eval] loaded {args.weights}  device={args.device}  samples={args.samples}")
    print(f"[eval] aspect sweep = {ASPECT_RATIOS}")

    rows = []  # (n, cost_before, feas_before, cost_after, feas_after, area_gap_before, area_gap_after)
    t_start = time.time()

    for case_idx in range(min(args.limit, 100)):
        blocks, b2b, p2b, pins_pos, metrics, geometry, cfg_name = load_case_raw(args.val, case_idx)
        n = blocks.shape[0]
        baseline_area = float(metrics[0])
        baseline_hpwl = float(metrics[6]) + float(metrics[7])

        feat = case_to_features(blocks, b2b, p2b, geometry)
        blocks_feat = torch.zeros((1, max_n, feat.shape[1]))
        blocks_feat[0, :n] = feat
        blocks_mask = torch.zeros((1, max_n), dtype=torch.bool)
        blocks_mask[0, :n] = True
        t_use = min(pins_pos.shape[0], max_t)
        terms = torch.zeros((1, max_t, 2))
        terms[0, :t_use] = pins_pos[:t_use]
        terms_mask = torch.zeros((1, max_t), dtype=torch.bool)
        terms_mask[0, :t_use] = True

        base_dims, is_preplaced, preplaced_xy = build_dims_and_hints(blocks, geometry)
        cluster_id = {i: int(blocks[i, 4]) for i in range(n)}
        boundary_code = {i: int(blocks[i, 5]) for i in range(n)}

        best_before = None   # lowest-cost square pack over the K topologies
        best_after = None    # lowest-cost shape-opt pack over the K topologies
        best_after_pack = None
        best_after_dims = None

        for _ in range(args.samples):
            out = model.generate(blocks_feat.to(args.device), blocks_mask.to(args.device),
                                  terms.to(args.device), terms_mask.to(args.device),
                                  n_blocks=n, temperature=1.0, sample=True, generator=gen)
            root = int(out["gen_order"][0])
            lc, rc = build_lc_rc(root, out["parent_id"], out["direction"], n,
                                 gen_order=out["gen_order"].tolist())

            # BEFORE: square only (r = 1.0)
            sq_dims = dims_with_aspect(blocks, base_dims, 1.0)
            sq_pack = pack_btree(root, lc, rc, sq_dims, is_preplaced, preplaced_xy,
                                 baseline_area=baseline_area, cluster_id=cluster_id,
                                 boundary_code=boundary_code)
            sq_cc = evaluate_cost(sq_pack.x, sq_pack.y, sq_dims, blocks, b2b, p2b, pins_pos,
                                  baseline_area, baseline_hpwl)
            if best_before is None or sq_cc.cost < best_before.cost:
                best_before = sq_cc

            # AFTER: best over the aspect sweep (keep the pack + its dims so we
            # can apply the HPWL nudge to the winner).
            opt_pack, opt_cc = best_over_aspects(root, lc, rc, blocks, base_dims, is_preplaced,
                                          preplaced_xy, b2b, p2b, pins_pos,
                                          baseline_area, baseline_hpwl, ASPECT_RATIOS,
                                          cluster_id=cluster_id, boundary_code=boundary_code)
            if best_after is None or opt_cc.cost < best_after.cost:
                best_after = opt_cc
                best_after_pack = opt_pack
                best_after_dims = {i: (opt_pack.w[i], opt_pack.h[i]) for i in range(n)}

        # HPWL nudge on the winning pack (attacks the dominant hpwl_gap; only
        # accepted if it actually lowers Cost). Free blocks first, then
        # whole clusters as rigid bodies (hpwl_nudge itself excludes every
        # cluster member, so they'd otherwise get zero HPWL benefit).
        if best_after_pack is not None:
            hpwl_nudge(best_after_pack, best_after_dims, blocks, b2b, p2b, pins_pos)
            hpwl_nudge_clusters(best_after_pack, best_after_dims, blocks, b2b, p2b, pins_pos)
            nudged = evaluate_cost(best_after_pack.x, best_after_pack.y, best_after_dims,
                                   blocks, b2b, p2b, pins_pos, baseline_area, baseline_hpwl)
            if nudged.cost < best_after.cost:
                best_after = nudged

        rows.append((n, best_before.cost, best_before.feasible,
                     best_after.cost, best_after.feasible,
                     best_before.area_gap, best_after.area_gap))
        if case_idx % 10 == 0 or n >= 118:
            print(f"  {cfg_name}: n={n:3d}  before Cost={best_before.cost:6.3f} "
                  f"(area_gap {best_before.area_gap:+.1%})  ->  after Cost={best_after.cost:6.3f} "
                  f"(area_gap {best_after.area_gap:+.1%})", flush=True)

    dt = time.time() - t_start
    ns = [r[0] for r in rows]
    cost_before = [r[1] for r in rows]
    cost_after = [r[3] for r in rows]
    feas_before = sum(1 for r in rows if r[2])
    feas_after = sum(1 for r in rows if r[4])

    tot_before = total_score(cost_before, ns)
    tot_after = total_score(cost_after, ns)
    mean_before = sum(cost_before) / len(cost_before)
    mean_after = sum(cost_after) / len(cost_after)

    print("\n" + "=" * 66)
    print(f"[eval] {len(rows)} cases in {dt:.1f}s  ({dt/len(rows):.2f}s/case)")
    print("=" * 66)
    print(f"{'metric':<32}{'BEFORE (square)':>16}{'AFTER (shape-opt)':>18}")
    print(f"{'feasible cases':<32}{feas_before:>13}/{len(rows)}{feas_after:>15}/{len(rows)}")
    print(f"{'mean Cost (unweighted)':<32}{mean_before:>16.4f}{mean_after:>18.4f}")
    print(f"{'Total Score  e^(n/12) weighted':<32}{tot_before:>16.4f}{tot_after:>18.4f}")
    print("=" * 66)
    dImp = tot_before - tot_after
    pct = 100.0 * dImp / tot_before if tot_before > 0 else 0.0
    print(f"[eval] Total Score improvement: {tot_before:.4f} -> {tot_after:.4f}  "
          f"(down {dImp:.4f}, {pct:.1f}% lower is better)")
    print(f"[eval] mean area_gap: before {sum(r[5] for r in rows)/len(rows):+.1%}  "
          f"after {sum(r[6] for r in rows)/len(rows):+.1%}")
    print("[eval] NOTE: RT fixed at 1.0 (R=1.0) -- this isolates quality+violation, "
          "not runtime. Shape sweep always includes r=1.0 so AFTER <= BEFORE by construction.")


if __name__ == "__main__":
    main()
