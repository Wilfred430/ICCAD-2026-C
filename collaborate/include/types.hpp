// types.hpp -- Core data structures for the FloorSet-Lite floorplanner.
//
// All geometry uses double precision. Block ids and terminal ids are 0-based
// integers. The whole instance is owned by a single FloorplanInstance struct;
// per-search (per-thread) state lives in BTree (see btree.hpp).
//
// v9 (2026-03-25) constraint summary:
//   HARD : (1) soft-block area within 1%   (2) overlap-free
//          (3) fixed-shape immutability     (4) preplaced immutability
//   SOFT : grouping, MIB, boundary  (penalty only)
//
#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace fp {

using Real = double;
constexpr int NO_NODE = -1;
constexpr Real REAL_INF = std::numeric_limits<Real>::infinity();

// Boundary spec. The contest input gives one slot per block;
// E_NONE means the block has no boundary constraint.
enum BoundaryEdge : int8_t {
    E_NONE   = -1,
    E_LEFT   = 0,
    E_RIGHT  = 1,
    E_BOTTOM = 2,
    E_TOP    = 3,
    C_BL     = 4,   // bottom-left corner   (touch left AND bottom)
    C_BR     = 5,   // bottom-right corner  (touch right AND bottom)
    C_TL     = 6,   // top-left corner      (touch left AND top)
    C_TR     = 7,   // top-right corner     (touch right AND top)
};

// Block / partition.
struct Block {
    int  id            = -1;
    Real area_target   = 0.0;       // ignored for fixed and preplaced

    // Current realised geometry (filled in by the packer).
    Real x = 0.0, y = 0.0;
    Real w = 0.0, h = 0.0;

    // ---- Constraint flags ----
    bool is_fixed     = false;      // hard: w,h immutable
    bool is_preplaced = false;      // hard: x,y,w,h immutable
    int  mib_group    = -1;         // soft : -1 if not in any MIB group
    int  group_id     = -1;         // soft : grouping (cluster)
    BoundaryEdge bedge = E_NONE;    // soft : boundary

    // Original / target geometry (for fixed and preplaced).
    Real w_input = 0.0, h_input = 0.0;
    Real x_input = 0.0, y_input = 0.0;

    // Allowed dimension search range for soft blocks.  By default we use a
    // fairly permissive aspect-ratio band; the SA aspect-ratio move samples
    // in this band while preserving the 1 % area tolerance.
    Real ar_min = 0.25;             // h/w lower bound
    Real ar_max = 4.00;             // h/w upper bound

    // Convenience.
    inline Real cx() const { return x + w * 0.5; }
    inline Real cy() const { return y + h * 0.5; }
    inline bool dims_locked()  const { return is_fixed || is_preplaced; }
    inline bool place_locked() const { return is_preplaced; }
};

// External terminal / pin (treated as a fixed point).
struct Terminal {
    int  id = -1;
    Real x  = 0.0;
    Real y  = 0.0;
};

// Pair-wise weighted net.
//   For b2b nets, .a and .b are block ids.
//   For p2b nets, .a is a terminal id and .b is a block id.
struct Net {
    int  a = -1;
    int  b = -1;
    Real w = 0.0;
};

// One instance of the problem.
struct FloorplanInstance {
    int n_blocks    = 0;
    int n_terminals = 0;

    std::vector<Block>    blocks;
    std::vector<Terminal> terminals;

    // Connectivity, stored as edge lists (much smaller than a dense matrix
    // and avoids carrying around k*k zeros).  Both lists are deduplicated:
    // every undirected edge appears once.
    std::vector<Net> b2b_nets;
    std::vector<Net> p2b_nets;

    // groups[i] = list of block ids that form grouping group i.
    // mib_groups[i] = list of block ids that share dimensions in MIB group i.
    std::vector<std::vector<int>> grouping_groups;
    std::vector<std::vector<int>> mib_groups;

    // Baselines come from the dataset; we use them to compute the gap-based
    // metrics from the contest cost.  If unknown the evaluator falls back to
    // raw HPWL/area.
    Real baseline_hpwl = 0.0;
    Real baseline_area = 0.0;

    // Convenience -- canvas is unbounded in this contest formulation; we
    // track the baseline outline only for SA cost shaping.
    Real outline_w = 0.0;
    Real outline_h = 0.0;

    // ---- ML warm-start (optional) -------------------------------------------
    // Populated only when the input file carries a WARM_POSITIONS section
    // (emitted by my_optimizer_ml.py).  (warm_cx, warm_cy) is the ML predictor's
    // guessed center for each block id and (warm_w, warm_h) its guessed shape;
    // warm_priority is an optional suggested insertion order (block ids).  When
    // has_warm is false these stay empty and the solver behaves exactly as
    // before -- the warm start is a pure add-on with no effect on legacy inputs.
    bool has_warm = false;
    std::vector<Real> warm_cx, warm_cy, warm_w, warm_h;
    std::vector<int>  warm_priority;
};

} // namespace fp
