// cost.hpp -- Cost evaluation.
//
// We separate two notions of "cost":
//   1. SA cost (sa_cost):  smooth, weighted sum used to drive simulated
//      annealing.  Penalises area, HPWL, soft-constraint violations, and
//      hard-constraint violations (with high weights so SA is steered
//      toward feasibility).
//   2. Contest cost (final_cost): exactly the formula in the v9 spec.
//      Used only for reporting / picking the best of a parallel run.
//
#pragma once
#include "types.hpp"
#include "btree.hpp"
#include "packer.hpp"

namespace fp {

struct Costs {
    // Raw quantities
    Real hpwl_int = 0.0;
    Real hpwl_ext = 0.0;
    Real hpwl_total = 0.0;
    Real area_bbox = 0.0;
    Real bbox_w = 0.0, bbox_h = 0.0;

    // Soft-constraint counts
    int v_grouping  = 0;
    int v_mib       = 0;
    int v_boundary  = 0;
    int n_soft_norm = 0;       // Nsoft from the spec (denominator)
    Real v_relative = 0.0;     // (V_g + V_m + V_b) / Nsoft

    // Hard-constraint flags
    bool overlap_violation = false;
    bool area_violation    = false;   // any soft block whose w*h drifts > 1%
    bool fixed_violation   = false;   // a fixed block has wrong dims (shouldn't happen if we obey)
    bool preplaced_violation = false;
    bool feasible = true;

    // Gap-based metrics (vs. baseline)
    Real hpwl_gap = 0.0;
    Real area_gap = 0.0;
};

// Weights used by the SA cost.  Tuned empirically; expose as struct so the
// caller can ramp them across SA stages.
struct SAWeights {
    Real w_area     = 1.0;     // bounding-box area (normalised by baseline)
    Real w_hpwl     = 1.0;     // total HPWL (normalised by baseline)
    Real w_overlap  = 1000.0;  // huge -- overlap is hard
    Real w_softarea = 1000.0;  // huge -- soft-block area-tolerance is hard
    Real w_group    = 5.0;     // grouping
    Real w_mib      = 5.0;     // MIB
    Real w_bound    = 5.0;     // boundary
    Real w_outline  = 5.0;     // SDB-AP: fixed-outline and aspect-ratio penalty
};

class Evaluator {
public:
    // Evaluate a packed BTree.  Fills in `out` with all the components.
    // Does NOT touch the tree.
    Costs evaluate(const FloorplanInstance& inst, const BTree& tree, const PackResult& pr) const;

    // SA-friendly scalar cost.
    Real sa_cost(const Costs& c, const SAWeights& W, const FloorplanInstance& inst) const;

    // Contest cost (Eq. 2 of the v9 spec).  runtime_factor defaults to 1.0
    // (no runtime info available during the run).
    Real contest_cost(const Costs& c, Real runtime_factor = 1.0) const;
};

// Hard-constraint check that scans every pair of blocks for overlap and every
// soft block for area-tolerance.  Used to decide feasibility for the contest
// cost; the per-pack overlap_free flag in PackResult only catches anchored
// overlaps, so we run this whole check at the end.
bool check_hard_constraints(const FloorplanInstance& inst, const BTree& tree,
                            bool& overlap_v, bool& area_v,
                            bool& fixed_v, bool& preplaced_v,
                            Real area_tol = 0.01);

} // namespace fp
