// cost.cpp -- Cost evaluator following the v9 (2026-03-25) contest spec.
//
// HPWL_int (centroid-to-centroid Manhattan, *not* bbox half-perimeter -- this
// is the v9 change!):
//     HPWL_int = sum_{i<j} W^{int}_{ij} * (|cx_i - cx_j| + |cy_i - cy_j|)
//
// HPWL_ext (block centroid to terminal):
//     HPWL_ext = sum_i sum_j W^{ext}_{ij} * (|cx_i - x_{tj}| + |cy_i - y_{tj}|)
//
// Contest cost (eq. 2):
//     Cost = (1 + 0.5 * (HPWL_gap + Area_gap)) * exp(2.0 * V_rel)
//            * max(0.7, RuntimeFactor^0.3)
// for feasible solutions, else M = 10.
//
#include "cost.hpp"
#include <algorithm>
#include <cmath>
#include <unordered_set>

namespace fp {

namespace {

// Two blocks are considered to "share an edge" (for grouping) if they touch
// along a non-zero-length segment with no gap.  That means one block's right
// edge equals the other's left edge (and y-ranges overlap with non-zero
// length), or analogously top/bottom.  A small tolerance accounts for floats.
constexpr Real TOUCH_EPS = 1e-7;

inline bool touches(Real ax, Real ay, Real aw, Real ah,
                    Real bx, Real by, Real bw, Real bh) {
    // share a vertical edge?
    if (std::abs((ax + aw) - bx) < TOUCH_EPS || std::abs((bx + bw) - ax) < TOUCH_EPS) {
        Real ylo = std::max(ay, by);
        Real yhi = std::min(ay + ah, by + bh);
        if (yhi - ylo > TOUCH_EPS) return true;
    }
    // share a horizontal edge?
    if (std::abs((ay + ah) - by) < TOUCH_EPS || std::abs((by + bh) - ay) < TOUCH_EPS) {
        Real xlo = std::max(ax, bx);
        Real xhi = std::min(ax + aw, bx + bw);
        if (xhi - xlo > TOUCH_EPS) return true;
    }
    return false;
}

inline bool overlaps_strict(Real ax, Real ay, Real aw, Real ah,
                            Real bx, Real by, Real bw, Real bh) {
    return ax + TOUCH_EPS < bx + bw && bx + TOUCH_EPS < ax + aw
        && ay + TOUCH_EPS < by + bh && by + TOUCH_EPS < ay + ah;
}

// Connected components in a small graph: blocks {ids}, edges = touching pairs.
int count_components(const std::vector<int>& ids,
                     const FloorplanInstance& inst, const BTree& t) {
    int n = static_cast<int>(ids.size());
    if (n <= 1) return n;
    std::vector<int> par(n);
    for (int i = 0; i < n; ++i) par[i] = i;
    auto find = [&](int x){ while (par[x] != x){ par[x] = par[par[x]]; x = par[x]; } return x; };
    auto uni  = [&](int a, int b){ a = find(a); b = find(b); if (a != b) par[a] = b; };
    for (int i = 0; i < n; ++i) for (int j = i + 1; j < n; ++j) {
        int A = ids[i], B = ids[j];
        if (touches(t.x[A], t.y[A], t.w[A], t.h[A],
                    t.x[B], t.y[B], t.w[B], t.h[B])) uni(i, j);
    }
    int comps = 0;
    for (int i = 0; i < n; ++i) if (find(i) == i) ++comps;
    return comps;
}

inline bool boundary_ok(BoundaryEdge e, Real x, Real y, Real w, Real h,
                        Real Wbb, Real Hbb) {
    // A block "touches the bounding-box edge" if its corresponding
    // x/y matches the bbox edge within tolerance.
    bool L = std::abs(x) < TOUCH_EPS;
    bool B = std::abs(y) < TOUCH_EPS;
    bool R = std::abs((x + w) - Wbb) < TOUCH_EPS;
    bool T = std::abs((y + h) - Hbb) < TOUCH_EPS;
    switch (e) {
        case E_LEFT:   return L;
        case E_RIGHT:  return R;
        case E_BOTTOM: return B;
        case E_TOP:    return T;
        case C_BL:     return L && B;
        case C_BR:     return R && B;
        case C_TL:     return L && T;
        case C_TR:     return R && T;
        default:       return true;
    }
}

} // anonymous

Costs Evaluator::evaluate(const FloorplanInstance& inst, const BTree& tree,
                          const PackResult& pr) const {
    Costs c;
    const int n = inst.n_blocks;

    // ---- HPWL ----
    Real hpwl_int = 0.0;
    for (const auto& net : inst.b2b_nets) {
        Real cxa = tree.x[net.a] + tree.w[net.a] * 0.5;
        Real cya = tree.y[net.a] + tree.h[net.a] * 0.5;
        Real cxb = tree.x[net.b] + tree.w[net.b] * 0.5;
        Real cyb = tree.y[net.b] + tree.h[net.b] * 0.5;
        hpwl_int += net.w * (std::abs(cxa - cxb) + std::abs(cya - cyb));
    }
    Real hpwl_ext = 0.0;
    for (const auto& net : inst.p2b_nets) {
        const Terminal& t = inst.terminals[net.a];
        Real cxb = tree.x[net.b] + tree.w[net.b] * 0.5;
        Real cyb = tree.y[net.b] + tree.h[net.b] * 0.5;
        hpwl_ext += net.w * (std::abs(cxb - t.x) + std::abs(cyb - t.y));
    }
    c.hpwl_int   = hpwl_int;
    c.hpwl_ext   = hpwl_ext;
    c.hpwl_total = hpwl_int + hpwl_ext;

    // ---- Area ----
    c.bbox_w   = pr.bbox_w;
    c.bbox_h   = pr.bbox_h;
    c.area_bbox = pr.bbox_area;

    // ---- Gaps (for the contest cost) ----
    if (inst.baseline_hpwl > 0) {
        c.hpwl_gap = (c.hpwl_total - inst.baseline_hpwl) / inst.baseline_hpwl;
    }
    if (inst.baseline_area > 0) {
        c.area_gap = (c.area_bbox - inst.baseline_area) / inst.baseline_area;
    }

    // ---- Soft constraints ----
    int Nsoft = 0;
    int Vgrp = 0, Vmib = 0, Vbnd = 0;

    // Grouping
    for (const auto& g : inst.grouping_groups) {
        if ((int)g.size() <= 1) continue;
        int comps = count_components(g, inst, tree);
        Vgrp += std::max(0, comps - 1);
        Nsoft += static_cast<int>(g.size()) - 1;
    }
    // MIB
    for (const auto& g : inst.mib_groups) {
        if ((int)g.size() <= 1) continue;
        // Count distinct (w,h) pairs (rounded).
        std::vector<std::pair<long long, long long>> shapes;
        shapes.reserve(g.size());
        for (int b : g) {
            long long iw = static_cast<long long>(std::llround(tree.w[b] * 1e6));
            long long ih = static_cast<long long>(std::llround(tree.h[b] * 1e6));
            shapes.emplace_back(iw, ih);
        }
        std::sort(shapes.begin(), shapes.end());
        shapes.erase(std::unique(shapes.begin(), shapes.end()), shapes.end());
        Vmib += std::max<int>(0, (int)shapes.size() - 1);
        Nsoft += static_cast<int>(g.size()) - 1;
    }
    // Boundary
    for (int i = 0; i < n; ++i) {
        if (inst.blocks[i].bedge == E_NONE) continue;
        ++Nsoft;
        if (!boundary_ok(inst.blocks[i].bedge, tree.x[i], tree.y[i],
                         tree.w[i], tree.h[i], pr.bbox_w, pr.bbox_h)) {
            ++Vbnd;
        }
    }
    c.v_grouping  = Vgrp;
    c.v_mib       = Vmib;
    c.v_boundary  = Vbnd;
    c.n_soft_norm = Nsoft;
    c.v_relative  = (Nsoft > 0)
                  ? std::min(1.0, (Real)(Vgrp + Vmib + Vbnd) / (Real)Nsoft)
                  : 0.0;

    // ---- Hard constraints ----
    bool ov, av, fv, pv;
    bool ok = check_hard_constraints(inst, tree, ov, av, fv, pv);
    c.overlap_violation   = ov;
    c.area_violation      = av;
    c.fixed_violation     = fv;
    c.preplaced_violation = pv;
    c.feasible = ok;

    return c;
}

Real Evaluator::sa_cost(const Costs& c, const SAWeights& W,
                        const FloorplanInstance& inst) const {
    // Normalise area & HPWL by baselines so weights are comparable.
    Real abase = (inst.baseline_area > 0) ? inst.baseline_area : 1.0;
    Real hbase = (inst.baseline_hpwl > 0) ? inst.baseline_hpwl : 1.0;

    Real cost = W.w_area * (c.area_bbox / abase)
              + W.w_hpwl * (c.hpwl_total / hbase)
              + W.w_group * (Real)c.v_grouping
              + W.w_mib   * (Real)c.v_mib
              + W.w_bound * (Real)c.v_boundary;

    // --- SDB-AP: Adaptive Fixed-Outline Penalty ---
    if (inst.outline_w > 0 && inst.outline_h > 0) {
        Real target_R = inst.outline_h / inst.outline_w;
        Real current_R = c.bbox_h / std::max(c.bbox_w, 1e-9);
        
        // 1. Aspect Ratio Penalty (Forces square-ish global shape)
        Real ar_penalty = std::pow(current_R - target_R, 2.0);
        
        // 2. Linear Overflow Penalty (Continuous gradient to feasibility)
        Real overflow_x = std::max(0.0, c.bbox_w - inst.outline_w);
        Real overflow_y = std::max(0.0, c.bbox_h - inst.outline_h);
        Real overflow_norm = (overflow_x / inst.outline_w) + (overflow_y / inst.outline_h);
        
        // Apply penalties for shape violation, scaled by the dynamic w_outline
        cost += W.w_outline * (ar_penalty + 2.0 * overflow_norm);
    }

    if (c.overlap_violation)   cost += W.w_overlap;
    if (c.area_violation)      cost += W.w_softarea;
    if (c.fixed_violation)     cost += W.w_overlap;
    if (c.preplaced_violation) cost += W.w_overlap;
    return cost;
}

Real Evaluator::contest_cost(const Costs& c, Real runtime_factor) const {
    if (!c.feasible) return 10.0;
    constexpr Real alpha = 0.5, beta = 2.0, gamma = 0.3;
    Real q = 1.0 + alpha * (c.hpwl_gap + c.area_gap);
    Real p = std::exp(beta * c.v_relative);
    Real rf = std::pow(std::max(runtime_factor, 1e-9), gamma);
    rf = std::max(0.7, rf);
    return q * p * rf;
}

bool check_hard_constraints(const FloorplanInstance& inst, const BTree& tree,
                            bool& overlap_v, bool& area_v,
                            bool& fixed_v, bool& preplaced_v, Real area_tol) {
    overlap_v = area_v = fixed_v = preplaced_v = false;
    const int n = inst.n_blocks;

    // 1) area tolerance for soft blocks
    for (int i = 0; i < n; ++i) {
        const Block& b = inst.blocks[i];
        if (b.is_fixed || b.is_preplaced) continue;
        Real a = tree.w[i] * tree.h[i];
        if (b.area_target > 0 &&
            std::abs(a - b.area_target) / b.area_target > area_tol + 1e-12) {
            area_v = true;
            break;
        }
    }
    // 2) fixed-shape immutability
    for (int i = 0; i < n; ++i) {
        const Block& b = inst.blocks[i];
        if (!b.is_fixed) continue;
        if (std::abs(tree.w[i] - b.w_input) > 1e-6 ||
            std::abs(tree.h[i] - b.h_input) > 1e-6) {
            fixed_v = true;
            break;
        }
    }
    // 3) preplaced immutability
    for (int i = 0; i < n; ++i) {
        const Block& b = inst.blocks[i];
        if (!b.is_preplaced) continue;
        if (std::abs(tree.w[i] - b.w_input) > 1e-6 ||
            std::abs(tree.h[i] - b.h_input) > 1e-6 ||
            std::abs(tree.x[i] - b.x_input) > 1e-6 ||
            std::abs(tree.y[i] - b.y_input) > 1e-6) {
            preplaced_v = true;
            break;
        }
    }
    // 4) overlap (full pairwise scan, n is small)
    for (int i = 0; i < n && !overlap_v; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (overlaps_strict(tree.x[i], tree.y[i], tree.w[i], tree.h[i],
                                tree.x[j], tree.y[j], tree.w[j], tree.h[j])) {
                overlap_v = true; break;
            }
        }
    }
    return !(overlap_v || area_v || fixed_v || preplaced_v);
}

} // namespace fp
