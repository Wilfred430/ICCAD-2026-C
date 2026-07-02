// packer.cpp -- Contour-based packing.
//
// The contour is stored as a sorted vector of (x_start, height) entries.
// Segment i covers [seg[i].x, seg[i+1].x) (and the last segment has +inf
// extent).  This is simpler than the doubly-linked list of the original
// B*-tree paper and runs comfortably for n ≤ 200.
//
#include "packer.hpp"
#include <algorithm>
#include <vector>
#include <stack>
#include <cmath>

namespace fp {

namespace {

struct Seg { Real x; Real h; };  // segment starting at x with height h

// Query the maximum contour height in the half-open interval [xl, xr).
inline Real max_height_in(const std::vector<Seg>& C, Real xl, Real xr) {
    if (C.empty()) return 0.0;
    Real best = 0.0;
    // first segment with x >= xl is found via lower_bound
    auto it = std::lower_bound(C.begin(), C.end(), xl,
                               [](const Seg& s, Real v){ return s.x < v; });
    // need to start one before, because the segment containing xl starts earlier
    if (it != C.begin()) --it;
    for (; it != C.end() && it->x < xr; ++it) {
        if (it->h > best) best = it->h;
    }
    return best;
}

// Update the contour after placing a block at [xl, xr) with top y = top.
// Replaces the portion of the contour in [xl, xr) with a single segment of
// height `top`, preserving the height of segments outside this range.
void update_contour(std::vector<Seg>& C, Real xl, Real xr, Real top) {
    // height immediately to the right of xr (so we can re-introduce it after we splice)
    Real h_after = 0.0;
    {
        auto it = std::lower_bound(C.begin(), C.end(), xr,
                                   [](const Seg& s, Real v){ return s.x < v; });
        if (it != C.begin()) {
            auto prev = it; --prev;
            h_after = prev->h;
        }
    }
    // remove every segment with x in [xl, xr)
    auto lo = std::lower_bound(C.begin(), C.end(), xl,
                               [](const Seg& s, Real v){ return s.x < v; });
    auto hi = std::lower_bound(C.begin(), C.end(), xr,
                               [](const Seg& s, Real v){ return s.x < v; });
    C.erase(lo, hi);
    // insert (xl, top) and (xr, h_after)
    auto pos = std::lower_bound(C.begin(), C.end(), xl,
                                [](const Seg& s, Real v){ return s.x < v; });
    pos = C.insert(pos, Seg{xl, top});
    pos = std::lower_bound(C.begin(), C.end(), xr,
                           [](const Seg& s, Real v){ return s.x < v; });
    if (pos == C.end() || pos->x > xr) {
        C.insert(pos, Seg{xr, h_after});
    }
    // collapse runs of equal height
    std::vector<Seg> out;
    out.reserve(C.size());
    for (const auto& s : C) {
        if (!out.empty() && std::abs(out.back().h - s.h) < 1e-12) continue;
        out.push_back(s);
    }
    C.swap(out);
}

// Check whether two axis-aligned rectangles overlap (with non-zero area).
inline bool rect_overlap(Real ax, Real ay, Real aw, Real ah,
                         Real bx, Real by, Real bw, Real bh) {
    return ax < bx + bw && bx < ax + aw && ay < by + bh && by < ay + ah;
}

} // namespace

// Slide every non-preplaced block as far left and down as it will go
// without overlapping its neighbours or going below (0,0).  This is the
// standard "B*-tree post-pack compaction" — it does NOT change the topology,
// only the realised (x,y), so SA invariants are preserved (the next move
// will repack from scratch).
//
// Why we compact ALL non-preplaced blocks (including boundary-pinned ones):
//   An earlier version skipped E_TOP/E_RIGHT blocks here in the hope of
//   preserving the boundary constraint.  Result: those blocks stayed at
//   whatever (high) y the contour packer placed them at, forcing a tall
//   bbox with everything else compacted below.  Net: huge dead space, the
//   bbox aspect went the WRONG way.
//   Now we compact everyone.  If a boundary constraint is temporarily
//   violated, SA's cost function (w_bound term) tells SA to use a
//   FixBoundary move to put the block back at the edge.  This converges
//   to both compact AND feasible far more often than the skip-approach.
//
//   preplaced is still skipped because its (x,y) is HARD-locked: any
//   deviation = hard-constraint violation = contest_cost = 10.
static void compact_left_down(const FloorplanInstance& inst, BTree& tree) {
    const int n = (int)tree.nodes.size();

    auto compact_axis_x = [&](int i) {
        if (inst.blocks[i].is_preplaced) return;
        Real best_x = 0.0;
        Real ay = tree.y[i], ah = tree.h[i];
        for (int j = 0; j < n; ++j) {
            if (j == i) continue;
            Real by = tree.y[j], bh = tree.h[j];
            // y overlap (strict)?
            if (ay + 1e-9 >= by + bh) continue;
            if (by + 1e-9 >= ay + ah) continue;
            // j's right edge as a candidate
            Real candidate = tree.x[j] + tree.w[j];
            if (candidate <= tree.x[i] + 1e-9 && candidate > best_x) best_x = candidate;
        }
        // Don't move it further than it currently is (only shrink, never grow).
        if (best_x < tree.x[i] - 1e-9) tree.x[i] = best_x;
    };
    auto compact_axis_y = [&](int i) {
        if (inst.blocks[i].is_preplaced) return;
        Real best_y = 0.0;
        Real ax = tree.x[i], aw = tree.w[i];
        for (int j = 0; j < n; ++j) {
            if (j == i) continue;
            Real bx = tree.x[j], bw = tree.w[j];
            if (ax + 1e-9 >= bx + bw) continue;
            if (bx + 1e-9 >= ax + aw) continue;
            Real candidate = tree.y[j] + tree.h[j];
            if (candidate <= tree.y[i] + 1e-9 && candidate > best_y) best_y = candidate;
        }
        if (best_y < tree.y[i] - 1e-9) tree.y[i] = best_y;
    };

    // Alternate (sort-by-y, compact-down) and (sort-by-x, compact-left).
    // Order matters because compacting one block can free up space for the
    // next.  Earlier version was a fixed 3 passes -- on case 056 that was
    // not enough to reach fixpoint, leaving visible gaps (e.g. dead space
    // below block 15 in the right-side visualisation).  Now we loop until
    // no block moves on a full y+x cycle, with a safety cap of 12 cycles.
    std::vector<int> idx(n);
    for (int i = 0; i < n; ++i) idx[i] = i;
    for (int pass = 0; pass < 12; ++pass) {
        bool changed = false;
        std::sort(idx.begin(), idx.end(), [&](int a, int b){ return tree.y[a] < tree.y[b]; });
        for (int i : idx) {
            Real old_y = tree.y[i];
            compact_axis_y(i);
            if (std::abs(tree.y[i] - old_y) > 1e-9) changed = true;
        }
        std::sort(idx.begin(), idx.end(), [&](int a, int b){ return tree.x[a] < tree.x[b]; });
        for (int i : idx) {
            Real old_x = tree.x[i];
            compact_axis_x(i);
            if (std::abs(tree.x[i] - old_x) > 1e-9) changed = true;
        }
        if (!changed) break;        // fixpoint reached
    }
}

// ---------------------------------------------------------------------------
// bbox_balance_pass -- aspect-ratio repair after compaction.
//
// Motivation (case 056 / 055 "tall pole" pathology):
//   The contour packer + left/down compaction produces overlap-free layouts,
//   but the bbox aspect ratio is whatever the tree topology dictates.  When
//   the tree has a deep right-spine, the floorplan ends up tall & thin
//   (135×270 for case 056, ideal ~150×175).  This wastes silicon area, blows
//   external HPWL by sending blocks far from the terminal extent, and forces
//   the floorplan past the IO-pin row (the "skewer" pin-violation visual).
//
// What we do:
//   After compaction, if the bbox is significantly imbalanced (AR > 1.15 or
//   AR < 1/1.15), repeatedly find the worst "spike" block (the unconstrained
//   movable block whose extreme edge defines the long dim) and slide it to a
//   position with smaller long-dim extension, capped to not extend the short
//   dim past its current value.  Cheap O(n²) per pass.
//
// Why this preserves invariants:
//   The packer remains deterministic given (topology, dims, instance) -- the
//   relocation is a deterministic greedy step on the post-compaction state.
//   It only moves NON-preplaced, NON-boundary-constrained blocks (so it
//   never breaks hard constraints, and never disturbs a boundary block that
//   SA may have already placed correctly).
//   The next move's pack() rebuilds the contour from scratch, so SA's
//   per-iter "fresh pack" model is unchanged -- balance_pass simply gives
//   SA a better-quality cost signal per topology to evaluate.
//
//   Non-overlap is preserved by can_place().
// ---------------------------------------------------------------------------
static bool relocate_one_spike(const FloorplanInstance& inst, BTree& tree,
                               bool tall, Real long_dim, Real short_dim) {
    const int n = (int)tree.nodes.size();

    // Find the worst spike: the movable, unconstrained block whose extreme
    // edge in the long-dim is closest to bbox_long.  Multiple blocks may
    // share the long-dim edge; we pick whichever we find first (the result
    // is still deterministic because the loop order is fixed).
    int spike = -1;
    Real spike_ext = -1;
    for (int i = 0; i < n; ++i) {
        if (inst.blocks[i].is_preplaced) continue;
        if (inst.blocks[i].is_fixed)     continue;
        if (inst.blocks[i].bedge != E_NONE) continue;       // boundary-pinned: don't disturb
        Real ext = tall ? (tree.y[i] + tree.h[i]) : (tree.x[i] + tree.w[i]);
        if (ext > spike_ext) { spike_ext = ext; spike = i; }
    }
    if (spike < 0) return false;
    if (spike_ext < long_dim - 1e-6) return false;          // not actually at the extreme

    const int i = spike;
    const Real cw = tree.w[i], ch = tree.h[i];
    const Real cur_long = tall ? (tree.y[i] + ch) : (tree.x[i] + cw);

    // Candidate short-dim coordinates: 0 and the trailing-edge of every other
    // block (these are the "shelves" we could line our block up against).
    std::vector<Real> short_coords; short_coords.reserve(n + 1);
    short_coords.push_back(0.0);
    for (int j = 0; j < n; ++j) {
        if (j == i) continue;
        short_coords.push_back(tall ? tree.x[j] + tree.w[j]
                                    : tree.y[j] + tree.h[j]);
    }
    std::sort(short_coords.begin(), short_coords.end());
    short_coords.erase(std::unique(short_coords.begin(), short_coords.end(),
                                   [](Real a, Real b){ return std::abs(a - b) < 1e-9; }),
                       short_coords.end());

    Real best_long  = cur_long;
    Real best_short = tall ? tree.x[i] : tree.y[i];

    const Real self_short_extent = tall ? cw : ch;
    const Real self_long_extent  = tall ? ch : cw;

    for (Real sc : short_coords) {
        if (sc < -1e-9) continue;
        if (sc + self_short_extent > short_dim + 1e-6) continue;   // would extend short dim

        // Find the lowest long-dim coordinate where (sc, lc, w, h) doesn't
        // overlap any other block in the short-dim band [sc, sc+self_short).
        Real lc = 0;
        for (int j = 0; j < n; ++j) {
            if (j == i) continue;
            Real j_short_lo = tall ? tree.x[j] : tree.y[j];
            Real j_short_hi = j_short_lo + (tall ? tree.w[j] : tree.h[j]);
            if (j_short_hi <= sc + 1e-9) continue;
            if (j_short_lo >= sc + self_short_extent - 1e-9) continue;
            Real j_long_hi = (tall ? tree.y[j] : tree.x[j])
                           + (tall ? tree.h[j] : tree.w[j]);
            if (j_long_hi > lc) lc = j_long_hi;
        }
        Real candidate_long_end = lc + self_long_extent;
        if (candidate_long_end < best_long - 1e-6) {
            best_long  = candidate_long_end;
            best_short = sc;
        }
    }

    if (best_long >= cur_long - 1e-6) return false;
    if (tall) { tree.x[i] = best_short; tree.y[i] = best_long - ch; }
    else      { tree.y[i] = best_short; tree.x[i] = best_long - cw; }
    return true;
}

// ---------------------------------------------------------------------------
// holes_fill_pass -- general L-shaped whitespace closer (v3, 2026-05-21).
//
// Motivation:
//   compact_left_down is purely axis-aligned: it can pull a block left, or
//   down, but never diagonally.  bbox_balance_pass relocates spike blocks
//   only.  So a block stuck in an "L-hole" -- where moving JUST left or
//   JUST down hits a neighbour, but moving both at once into a corner is
//   free -- never gets compacted.  Visible symptom in case 056 v2: empty
//   strips above blocks 24 / 19 / 16 / 63 that no axis sweep can reach.
//
// What this does:
//   For every movable block (not preplaced / fixed / boundary / cluster
//   member), search all candidate (x, y) positions formed by the corners
//   of OTHER blocks' rectangles.  Pick the position that minimises
//   max(x+w, y+h) -- i.e. tucks the block as far toward the origin as
//   possible without overlapping anything.
//
// Why skip cluster members:
//   Grouping requires touching at least one other cluster member.  A
//   pure min-extent search ignores that constraint, so moving cluster
//   members would routinely break V_grouping and explode contest cost.
//   Boundary members already constrained.  Preplaced/fixed: invariants.
//
// Cost:
//   Worst case O(n² candidates × n overlap × n movable) = O(n^4), but
//   the early-`break` on sorted candidates collapses it in practice.
//   For n=120 and ~30 movable spikes, real cost is a few ms per pack.
//   Triggered only after balance + compact, so common-case healthy
//   packings see it at most twice (the post-pack and after-balance
//   compactions cover them already).
// ---------------------------------------------------------------------------
static bool relocate_to_min_corner(const FloorplanInstance& inst,
                                   BTree& tree, int i) {
    (void)inst;        // reserved for future constraint-aware position gating
    const int n = (int)tree.nodes.size();
    const Real cw = tree.w[i], ch = tree.h[i];
    const Real cur_max = std::max(tree.x[i] + cw, tree.y[i] + ch);
    const Real cur_sum = (tree.x[i] + cw) + (tree.y[i] + ch);

    // Already at origin?  Nothing to do.
    if (cur_max < 1e-6) return false;

    // Candidate coords from other blocks' trailing edges (and 0).
    std::vector<Real> xs; xs.reserve(n + 1);
    std::vector<Real> ys; ys.reserve(n + 1);
    xs.push_back(0.0); ys.push_back(0.0);
    for (int j = 0; j < n; ++j) {
        if (j == i) continue;
        xs.push_back(tree.x[j] + tree.w[j]);
        ys.push_back(tree.y[j] + tree.h[j]);
    }
    std::sort(xs.begin(), xs.end());
    xs.erase(std::unique(xs.begin(), xs.end(),
                        [](Real a, Real b){ return std::abs(a - b) < 1e-9; }),
             xs.end());
    std::sort(ys.begin(), ys.end());
    ys.erase(std::unique(ys.begin(), ys.end(),
                        [](Real a, Real b){ return std::abs(a - b) < 1e-9; }),
             ys.end());

    auto overlaps_anything = [&](Real cx, Real cy) -> bool {
        for (int j = 0; j < n; ++j) {
            if (j == i) continue;
            if (cx + cw <= tree.x[j] + 1e-9) continue;
            if (tree.x[j] + tree.w[j] <= cx + 1e-9) continue;
            if (cy + ch <= tree.y[j] + 1e-9) continue;
            if (tree.y[j] + tree.h[j] <= cy + 1e-9) continue;
            return true;
        }
        return false;
    };

    Real best_x = tree.x[i], best_y = tree.y[i];
    Real best_max = cur_max;
    Real best_sum = cur_sum;

    // xs and ys are sorted ascending.  Once cx + cw > best_max the rest of
    // the loop can only get worse, so we `break` -- this is what keeps the
    // worst-case O(n^4) practical.
    for (Real cx : xs) {
        if (cx < -1e-9) continue;
        if (cx + cw > best_max + 1e-6) break;
        for (Real cy : ys) {
            if (cy < -1e-9) continue;
            if (cy + ch > best_max + 1e-6) break;
            Real new_max = std::max(cx + cw, cy + ch);
            Real new_sum = (cx + cw) + (cy + ch);
            // Strict improvement: smaller max, OR equal-max-smaller-sum.
            if (new_max > best_max + 1e-9) continue;
            if (new_max > best_max - 1e-9 && new_sum >= best_sum - 1e-9) continue;
            if (overlaps_anything(cx, cy)) continue;
            best_max = new_max;
            best_sum = new_sum;
            best_x = cx;
            best_y = cy;
        }
    }

    if (std::abs(best_x - tree.x[i]) < 1e-9 &&
        std::abs(best_y - tree.y[i]) < 1e-9) return false;
    tree.x[i] = best_x;
    tree.y[i] = best_y;
    return true;
}

static void holes_fill_pass(const FloorplanInstance& inst, BTree& tree) {
    const int n = (int)tree.nodes.size();
    if (n < 2) return;

    // Collect movable, constraint-free blocks.  Cluster members excluded
    // (see header comment for why).
    std::vector<int> order;
    order.reserve(n);
    for (int i = 0; i < n; ++i) {
        if (inst.blocks[i].is_preplaced) continue;
        if (inst.blocks[i].bedge != E_NONE) continue;
        if (inst.blocks[i].group_id >= 0) continue;
        order.push_back(i);
    }
    if (order.empty()) return;

    // Process farthest-from-origin first so spikes get first pick of new
    // slots (this is what compresses bbox the fastest).
    std::sort(order.begin(), order.end(), [&](int a, int b){
        Real ka = std::max(tree.x[a] + tree.w[a], tree.y[a] + tree.h[a]);
        Real kb = std::max(tree.x[b] + tree.w[b], tree.y[b] + tree.h[b]);
        return ka > kb;
    });

    // Two passes max -- once a block has been relocated, neighbours' best
    // slots may shift, so a second sweep catches the cascade.  Past 2
    // we're in deminishing-returns territory.
    for (int pass = 0; pass < 2; ++pass) {
        bool any_moved = false;
        for (int i : order) {
            if (relocate_to_min_corner(inst, tree, i)) any_moved = true;
        }
        if (!any_moved) break;
    }
}

static void bbox_balance_pass(const FloorplanInstance& inst, BTree& tree) {
    const int n = (int)tree.nodes.size();
    if (n < 2) return;

    auto compute_bbox = [&](Real& bw, Real& bh) {
        bw = 0; bh = 0;
        for (int i = 0; i < n; ++i) {
            bw = std::max(bw, tree.x[i] + tree.w[i]);
            bh = std::max(bh, tree.y[i] + tree.h[i]);
        }
    };

    // Target square side -- the ideal short-dim that takes the floorplan
    // to a near-square outline of equivalent area.  Without this cap,
    // v2 balance_pass exhibited "short-dim creep": each spike relocation
    // used the CURRENT (just-grown) short_dim as its limit, so case 056
    // bbox_w drifted from 150 → 170 → 200 across passes while bbox_h
    // shrunk only slowly.  The net was a wider-but-still-tall floorplan
    // (case-056 v2 image: ~170×275 instead of v1's 150×270 -- both bad).
    //
    // Anchoring the cap at sqrt(baseline_area) gives a hard ceiling: the
    // floorplan may grow toward the target square but never past it.
    // When baseline_area is unset, geometric mean of the initial bbox is
    // the safe fallback (it's the area-preserving square).
    Real init_w, init_h;
    compute_bbox(init_w, init_h);
    const Real target_side =
        (inst.baseline_area > 0)
        ? std::sqrt(inst.baseline_area)
        : std::sqrt(std::max(init_w * init_h, Real(1.0)));

    // Hard cap on relocations.  Earlier version capped at 4, which was
    // empirically nowhere near enough for case 056 (77 blocks, ~15 spikes
    // needing relocation to bring bbox from 270 down to ~180).  We now
    // allow up to min(n, 60) -- each spike costs O(n²) overlap-search, so
    // total worst case is O(n³) ~= 1.7M ops at n=120 (well under one ms
    // even on a slow machine).  Most packs have healthy AR and early-exit
    // immediately so SA cost-per-iter is unaffected on good topologies.
    const int max_passes = std::min(n, 60);
    int no_improve = 0;
    for (int pass = 0; pass < max_passes; ++pass) {
        Real bbox_w, bbox_h;
        compute_bbox(bbox_w, bbox_h);
        if (bbox_w <= 0 || bbox_h <= 0) return;
        Real ar = bbox_h / bbox_w;
        // Tighter exit threshold (was 1.15).  We push harder toward a square
        // outline so SA's cost evaluator sees a genuinely-balanced layout
        // every time, not a half-fixed tall pole.
        if (ar < 1.10 && ar > 1.0 / 1.10) return;

        bool tall = (ar > 1.0);
        Real long_dim  = tall ? bbox_h : bbox_w;
        Real short_dim = tall ? bbox_w : bbox_h;
        // Allow short_dim to grow toward target_side, but never past it.
        // If short_dim is already past target (e.g. AR very tilted in the
        // other direction during a transient), use the current value so
        // we don't shrink retroactively.
        Real short_dim_cap = std::max(short_dim, target_side);

        if (!relocate_one_spike(inst, tree, tall, long_dim, short_dim_cap)) {
            // No spike found a better slot.  Two consecutive failures means
            // we're stuck in a local optimum -- give up.
            if (++no_improve >= 2) return;
        } else {
            no_improve = 0;
        }
    }
}

// ---------------------------------------------------------------------------
// grouping_repair_pass -- deterministic grouping (cluster) fixer.
//
// Grouping requires every member of a cluster to form ONE connected component
// (each member touching at least one sibling along a shared edge).  Like the
// boundary constraint this is already PENALISED (w_group in sa_cost, V_grouping
// in the contest cost), but only the stochastic 5% FixGrouping move actually
// repairs it -- and compaction keeps scattering members every pass.  So dense
// cases finish with leftover grouping violations even though the cost "knows"
// about them.  This pass deterministically reattaches ISOLATED members (those
// touching no sibling) by sliding each flush against a sibling whenever a free
// adjacent slot exists.  Conservative on purpose:
//   * only moves members that touch NO sibling (minimal disturbance)
//   * never moves preplaced or boundary-constrained members (boundary pass
//     owns those; keeps the two repairs from fighting)
// Reduces the component count -> lowers V_grouping -> shrinks exp(2*V_rel).
static void grouping_repair_pass(const FloorplanInstance& inst, BTree& tree) {
    const int n = (int)tree.nodes.size();
    if (inst.grouping_groups.empty()) return;

    auto touches = [&](int a, int b) -> bool {
        Real ax = tree.x[a], ay = tree.y[a], aw = tree.w[a], ah = tree.h[a];
        Real bx = tree.x[b], by = tree.y[b], bw = tree.w[b], bh = tree.h[b];
        if (std::abs((ax + aw) - bx) < 1e-7 || std::abs((bx + bw) - ax) < 1e-7) {
            Real ylo = std::max(ay, by), yhi = std::min(ay + ah, by + bh);
            if (yhi - ylo > 1e-7) return true;            // shared vertical edge
        }
        if (std::abs((ay + ah) - by) < 1e-7 || std::abs((by + bh) - ay) < 1e-7) {
            Real xlo = std::max(ax, bx), xhi = std::min(ax + aw, bx + bw);
            if (xhi - xlo > 1e-7) return true;            // shared horizontal edge
        }
        return false;
    };
    auto cell_free = [&](int i, Real nx, Real ny) -> bool {
        Real w = tree.w[i], h = tree.h[i];
        for (int j = 0; j < n; ++j) {
            if (j == i) continue;
            if (rect_overlap(nx, ny, w, h, tree.x[j], tree.y[j], tree.w[j], tree.h[j]))
                return false;
        }
        return true;
    };
    auto movable = [&](int i) -> bool {
        return !inst.blocks[i].is_preplaced && inst.blocks[i].bedge == E_NONE;
    };

    for (const auto& g : inst.grouping_groups) {
        if ((int)g.size() <= 1) continue;
        for (int pass = 0; pass < 2; ++pass) {
            bool moved = false;
            for (int s : g) {
                if (!movable(s)) continue;
                bool connected = false;
                for (int t : g) if (t != s && touches(s, t)) { connected = true; break; }
                if (connected) continue;                  // already attached
                bool placed = false;
                for (int t : g) {
                    if (t == s) continue;
                    const Real tx = tree.x[t], ty = tree.y[t], tw = tree.w[t], th = tree.h[t];
                    const Real sw = tree.w[s], sh = tree.h[s];
                    const Real cand[4][2] = {
                        { tx + tw, ty      },             // right of t
                        { tx - sw, ty      },             // left  of t
                        { tx,      ty + th },             // above t
                        { tx,      ty - sh },             // below t
                    };
                    for (int k = 0; k < 4; ++k) {
                        Real nx = cand[k][0], ny = cand[k][1];
                        if (nx < -1e-9 || ny < -1e-9) continue;
                        if (!cell_free(s, nx, ny)) continue;
                        tree.x[s] = nx; tree.y[s] = ny;
                        placed = true; moved = true;
                        break;
                    }
                    if (placed) break;
                }
            }
            if (!moved) break;
        }
    }
}

// ---------------------------------------------------------------------------
// boundary_repair_pass -- deterministic soft-boundary fixer.
//
// Boundary blocks must touch a specific bbox edge (L/R/T/B) or corner.  The
// contour packer + left/down compaction pull every block toward the origin on
// every pack, so TOP/RIGHT boundary blocks are constantly dragged off their
// edge.  The stochastic 5%-probability FixBoundary SA move cannot keep up --
// it fights a deterministic compaction that runs every single pack, so dense
// cases finish with many boundary violations (case 95: V_boundary≈26, which
// alone drives exp(2·V_rel) ≈ 3× on the contest cost).
//
// This pass closes that gap deterministically: after compaction, slide each
// boundary block onto its required edge whenever the target cell is free.
//   * never overlaps   -- the move is skipped if the target cell is occupied
//   * never grows bbox  -- R/T targets land exactly on the EXISTING bbox edge
//   * preplaced skipped -- their (x,y) is hard-locked
// The only cost is a little whitespace where a block vacates an interior slot,
// which does not change bbox area -- a great trade for cutting the exponential
// V_rel penalty.
static void boundary_repair_pass(const FloorplanInstance& inst, BTree& tree) {
    const int n = (int)tree.nodes.size();
    if (n == 0) return;

    auto cell_free = [&](int i, Real nx, Real ny) -> bool {
        Real w = tree.w[i], h = tree.h[i];
        for (int j = 0; j < n; ++j) {
            if (j == i) continue;
            if (rect_overlap(nx, ny, w, h, tree.x[j], tree.y[j], tree.w[j], tree.h[j]))
                return false;
        }
        return true;
    };

    // Two sweeps: moving one block onto its edge can free the cell another
    // boundary block needs.  Early-exits when a full sweep moves nothing.
    for (int pass = 0; pass < 2; ++pass) {
        Real Wbb = 0, Hbb = 0;
        for (int i = 0; i < n; ++i) {
            Wbb = std::max(Wbb, tree.x[i] + tree.w[i]);
            Hbb = std::max(Hbb, tree.y[i] + tree.h[i]);
        }
        bool moved = false;
        for (int i = 0; i < n; ++i) {
            const Block& b = inst.blocks[i];
            if (b.bedge == E_NONE) continue;
            if (b.is_preplaced) continue;
            const Real w = tree.w[i], h = tree.h[i];
            Real nx = tree.x[i], ny = tree.y[i];
            switch (b.bedge) {
                case E_LEFT:                 nx = 0.0;                     break;
                case E_RIGHT:                nx = Wbb - w;                 break;
                case E_BOTTOM:               ny = 0.0;                     break;
                case E_TOP:                  ny = Hbb - h;                 break;
                case C_BL:    nx = 0.0;       ny = 0.0;                    break;
                case C_BR:    nx = Wbb - w;   ny = 0.0;                    break;
                case C_TL:    nx = 0.0;       ny = Hbb - h;                break;
                case C_TR:    nx = Wbb - w;   ny = Hbb - h;                break;
                default: break;
            }
            if (nx < -1e-9 || ny < -1e-9) continue;            // would leave canvas
            if (std::abs(nx - tree.x[i]) < 1e-9 &&
                std::abs(ny - tree.y[i]) < 1e-9) continue;     // already on edge
            if (!cell_free(i, nx, ny)) continue;               // occupied -> skip
            tree.x[i] = nx;
            tree.y[i] = ny;
            moved = true;
        }
        if (!moved) break;
    }
}

PackResult Packer::pack(const FloorplanInstance& inst, BTree& tree) const {
    PackResult result;
    const int n = static_cast<int>(tree.nodes.size());
    if (n == 0) return result;

    std::vector<Seg> contour;       // empty -> ground level 0
    contour.push_back(Seg{0.0, 0.0});

    // Pre-seed the contour with every anchored (preplaced) block's footprint
    // BEFORE placing any tree block.  Preplaced blocks live at a FIXED (x,y),
    // so every tree block must be aware of them from the very start; otherwise
    // a tree block placed early in DFS order can land inside an anchor's
    // footprint, and the fixed anchor then overlaps it -- the persistent
    // cost-10 "overlap" failure on the large/dense anchored cases (93/95/98).
    // With the footprints pre-seeded, max_height_in() lifts any tree block
    // whose x-range hits an anchor up above that anchor, so tree-vs-anchor
    // overlap becomes impossible by construction.
    //
    // Seed in ascending top-edge order so a taller anchor sharing an x-range
    // with a shorter one always raises (never lowers) the contour there.
    {
        std::vector<int> anchors;
        for (int i = 0; i < n; ++i)
            if (inst.blocks[i].is_preplaced) anchors.push_back(i);
        std::sort(anchors.begin(), anchors.end(), [&](int a, int b){
            return (inst.blocks[a].y_input + inst.blocks[a].h_input)
                 < (inst.blocks[b].y_input + inst.blocks[b].h_input);
        });
        for (int i : anchors) {
            const Block& b = inst.blocks[i];
            update_contour(contour, b.x_input, b.x_input + b.w_input,
                           b.y_input + b.h_input);
        }
    }

    // We need an iterative DFS in B*-tree order: parent before children,
    // left child fully before right child (so the contour is correct).
    // Use a stack with a state flag.
    struct Frame { int v; int state; };  // state: 0 = visit self, 1 = right child, 2 = done
    std::stack<Frame> st;

    if (tree.root == NO_NODE) return result;
    st.push({tree.root, 0});

    Real bbox_w_max = 0.0;
    Real bbox_h_max = 0.0;
    bool any_overlap = false;

    while (!st.empty()) {
        Frame& f = st.top();
        int v = f.v;

        if (f.state == 0) {
            // ---- place v ----
            const Block& b = inst.blocks[v];
            Real px = 0.0, py = 0.0;
            int parent = tree.nodes[v].parent;
            bool is_left_child = false;
            if (parent != NO_NODE) is_left_child = (tree.nodes[parent].lc == v);

            // Determine x.
            if (b.is_preplaced) {
                // anchored: ignore tree-derived x,y.  The contour was already
                // pre-seeded with this footprint before the DFS (see top of
                // pack()), so we must NOT update it again here -- re-asserting
                // the anchor's top would clobber the height of any tree block
                // already stacked above this anchor's x-range and reintroduce
                // overlap.
                tree.x[v] = b.x_input;
                tree.y[v] = b.y_input;
                tree.w[v] = b.w_input;
                tree.h[v] = b.h_input;
            } else {
                if (parent == NO_NODE) {
                    px = 0.0;
                } else if (is_left_child) {
                    px = tree.x[parent] + tree.w[parent];
                } else {
                    px = tree.x[parent];
                }
                Real xr = px + tree.w[v];
                py = max_height_in(contour, px, xr);
                tree.x[v] = px;
                tree.y[v] = py;
                update_contour(contour, px, xr, py + tree.h[v]);
            }

            bbox_w_max = std::max(bbox_w_max, tree.x[v] + tree.w[v]);
            bbox_h_max = std::max(bbox_h_max, tree.y[v] + tree.h[v]);

            // descend into left child
            f.state = 1;
            int lc = tree.nodes[v].lc;
            if (lc != NO_NODE) st.push({lc, 0});
        } else if (f.state == 1) {
            f.state = 2;
            int rc = tree.nodes[v].rc;
            if (rc != NO_NODE) st.push({rc, 0});
        } else {
            st.pop();
        }
    }

    // ---- Post-pack compaction ------------------------------------------
    // Slide every non-anchored, non-right/top-pinned block as far left and
    // down as it will go.  This eliminates the fragmented-whitespace holes
    // that the contour packer naturally produces but the topology moves
    // cannot close.
    compact_left_down(inst, tree);

    // ---- Aspect-ratio balance pass (case 056 fix) ----------------------
    // After compaction, if the bbox is still badly imbalanced (tall pole or
    // wide pancake), try to slide spike blocks into emptier slots.  Cheap
    // (O(n²) per pass × up-to-60 passes) and only triggers when AR > 1.10,
    // so well-balanced packings pay nothing.  See bbox_balance_pass comment
    // block above for the full rationale.
    bbox_balance_pass(inst, tree);

    // ---- Re-compact after balance ---------------------------------------
    // bbox_balance_pass relocates spike blocks into "shelves" found in the
    // current packing.  This can leave the spike's *old* column with nothing
    // on top above some mid-level resident block -- that resident is now
    // unconstrained from above but might still be sitting at the contour-
    // packer's original y, with empty space below it.  This is exactly the
    // case-056 visible bug ("block 15 has dead space below it").  Running
    // compact_left_down again pulls those residents down to the new floor.
    compact_left_down(inst, tree);

    // ---- L-shaped hole fill (v3 fix) ------------------------------------
    // compact_left_down moves blocks only along ONE axis at a time, so a
    // block trapped in an "L-shaped" hole (blocked left AND blocked down
    // but with a free corner diagonally) never moves.  holes_fill_pass
    // does the diagonal relocation that closes those holes -- this is
    // the polish step that visibly drops bbox area on case 056.
    // After this, one more compact_left_down catches any new floaters.
    holes_fill_pass(inst, tree);
    compact_left_down(inst, tree);

    // ---- Soft-constraint repair (deterministic fixers) ------------------
    // Compaction above scatters cluster members and pulls TOP/RIGHT boundary
    // blocks off their edge every pass.  Repair both deterministically, LAST,
    // so the compaction sweeps don't immediately undo them.  Grouping first,
    // boundary second (boundary gets the final say on any shared block; in
    // practice grouping skips boundary-constrained members).  See each pass's
    // header for the full rationale.
    grouping_repair_pass(inst, tree);
    boundary_repair_pass(inst, tree);

    // Re-compute the bbox after compaction (it can only shrink).
    bbox_w_max = 0.0;
    bbox_h_max = 0.0;
    for (int i = 0; i < n; ++i) {
        bbox_w_max = std::max(bbox_w_max, tree.x[i] + tree.w[i]);
        bbox_h_max = std::max(bbox_h_max, tree.y[i] + tree.h[i]);
    }

    // Post-pass overlap check just for anchored vs. anything: tree-vs-tree
    // packing is overlap-free by construction; only anchored blocks can
    // overlap, e.g. when a tree-placed block grew into an anchored block's
    // footprint.  This is O(n^2) but n ≤ 120 so it's free.
    for (int i = 0; i < n; ++i) {
        if (!inst.blocks[i].is_preplaced) continue;
        for (int j = 0; j < n; ++j) if (j != i) {
            if (rect_overlap(tree.x[i], tree.y[i], tree.w[i], tree.h[i],
                             tree.x[j], tree.y[j], tree.w[j], tree.h[j])) {
                any_overlap = true; break;
            }
        }
        if (any_overlap) break;
    }

    result.bbox_w = bbox_w_max;
    result.bbox_h = bbox_h_max;
    result.bbox_area = bbox_w_max * bbox_h_max;
    result.overlap_free = !any_overlap;
    return result;
}

} // namespace fp
