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

PackResult Packer::pack(const FloorplanInstance& inst, BTree& tree) const {
    PackResult result;
    const int n = static_cast<int>(tree.nodes.size());
    if (n == 0) return result;

    std::vector<Seg> contour;       // empty -> ground level 0
    contour.push_back(Seg{0.0, 0.0});

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
                // anchored: ignore tree-derived x,y
                tree.x[v] = b.x_input;
                tree.y[v] = b.y_input;
                tree.w[v] = b.w_input;
                tree.h[v] = b.h_input;
                // Do not update the contour with anchored blocks here -- we
                // still need to flag overlap if a previously-placed tree block
                // intersects this anchor.  We handle that in a post-pass below
                // by just registering the rectangle into the contour anyway:
                // the anchored block *is* part of the floorplan and tree
                // blocks placed *afterwards* must respect it.
                update_contour(contour, b.x_input, b.x_input + b.w_input,
                               b.y_input + b.h_input);
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
