// moves.cpp -- Implementation of the SA move set.
#include "moves.hpp"
#include <algorithm>
#include <cmath>
#include <cassert>
#include <map>
#include <functional>

namespace fp {

namespace {
template <class RNG>
int rand_int(RNG& rng, int lo, int hi) {
    if (hi <= lo) return lo;
    return std::uniform_int_distribution<int>(lo, hi)(rng);
}

template <class RNG>
Real rand_real(RNG& rng, Real lo, Real hi) {
    return std::uniform_real_distribution<Real>(lo, hi)(rng);
}

// Clamp h/w to [ar_min, ar_max] while preserving area = a (within tol).
inline std::pair<Real, Real> sample_dims(Real area, Real ar_min, Real ar_max,
                                         std::mt19937_64& rng,
                                         Real tol = 0.005) {
    // Pick aspect ratio r = h/w in [ar_min, ar_max], possibly perturbed.
    Real r = std::exp(rand_real(rng, std::log(ar_min), std::log(ar_max)));
    // Choose nominal area in [(1-tol)a, (1+tol)a] so we land safely inside the
    // 1 % hard tolerance even after rounding.
    Real a = area * (1.0 + rand_real(rng, -tol, tol));
    Real w = std::sqrt(a / r);
    Real h = a / w;
    return {w, h};
}

inline bool block_dims_locked(const Block& b) {
    return b.is_fixed || b.is_preplaced;
}

} // anonymous

bool MoveEngine::apply_rotate(const FloorplanInstance& inst, BTree& t, Move& m) {
    const int n = inst.n_blocks;
    // Rotation must respect MIB: rotating one block in a MIB group breaks it,
    // so we either skip (return false) or rotate every block in the group.
    // For simplicity we choose any rotatable block and, if it's in a MIB group,
    // we rotate every block in that group.
    int tries = 32;
    while (tries-- > 0) {
        int v = rand_int(rng_, 0, n - 1);
        const Block& b = inst.blocks[v];
        if (block_dims_locked(b)) continue;
        m.v = v;
        m.saved_w = t.w[v]; m.saved_h = t.h[v];
        if (b.mib_group >= 0) {
            const auto& group = inst.mib_groups[b.mib_group];
            m.mib_blocks = group;
            m.saved_w_vec.assign(group.size(), 0);
            m.saved_h_vec.assign(group.size(), 0);
            for (size_t i = 0; i < group.size(); ++i) {
                m.saved_w_vec[i] = t.w[group[i]];
                m.saved_h_vec[i] = t.h[group[i]];
                std::swap(t.w[group[i]], t.h[group[i]]);
            }
        } else {
            std::swap(t.w[v], t.h[v]);
        }
        return true;
    }
    return false;
}

bool MoveEngine::apply_move(const FloorplanInstance& inst, BTree& t, Move& m) {
    const int n = inst.n_blocks;
    if (n < 2) return false;
    int tries = 64;
    while (tries-- > 0) {
        int v = rand_int(rng_, 0, n - 1);
        if (v == t.root) continue;
        int u = rand_int(rng_, 0, n - 1);
        if (u == v) continue;
        // Preplaced blocks are anchored; moving them in the tree only affects
        // the placement of their *children*, which is fine.  We allow it.
        m.v = v; m.u = u;
        // Save full state of v and u so we can revert exactly
        m.saved_parent = t.nodes[v].parent;
        m.saved_lc = t.nodes[v].lc;
        m.saved_rc = t.nodes[v].rc;
        m.as_left = std::bernoulli_distribution(0.5)(rng_);
        // We use op_move for the topology change.  Note op_move can
        // implicitly graft existing children of u onto v's slot, which makes
        // a perfect revert tricky.  Therefore we implement revert by saving
        // the *whole* tree topology before the move.  That cost is fine for
        // n ≤ 200.
        // Save the whole tree (lazy and safe).
        // We piggyback on Move::saved_w_vec to store an int-encoded snapshot.
        m.saved_w_vec.clear();
        m.saved_h_vec.clear();
        m.saved_w_vec.reserve(n * 3);
        for (int i = 0; i < n; ++i) {
            m.saved_w_vec.push_back((Real)t.nodes[i].parent);
            m.saved_w_vec.push_back((Real)t.nodes[i].lc);
            m.saved_w_vec.push_back((Real)t.nodes[i].rc);
        }
        m.saved_h_vec.push_back((Real)t.root);
        if (!t.op_move(v, u, m.as_left)) {
            // restore from snapshot just in case
            for (int i = 0; i < n; ++i) {
                t.nodes[i].parent = (int)m.saved_w_vec[3 * i];
                t.nodes[i].lc     = (int)m.saved_w_vec[3 * i + 1];
                t.nodes[i].rc     = (int)m.saved_w_vec[3 * i + 2];
            }
            t.root = (int)m.saved_h_vec[0];
            continue;
        }
        return true;
    }
    return false;
}

bool MoveEngine::apply_swap(const FloorplanInstance& inst, BTree& t, Move& m) {
    const int n = inst.n_blocks;
    if (n < 2) return false;
    int tries = 32;
    while (tries-- > 0) {
        int a = rand_int(rng_, 0, n - 1);
        int b = rand_int(rng_, 0, n - 1);
        if (a == b) continue;
        m.a = a; m.b = b;
        // snapshot whole topology for safe revert
        m.saved_w_vec.clear();
        m.saved_w_vec.reserve(n * 3);
        for (int i = 0; i < n; ++i) {
            m.saved_w_vec.push_back((Real)t.nodes[i].parent);
            m.saved_w_vec.push_back((Real)t.nodes[i].lc);
            m.saved_w_vec.push_back((Real)t.nodes[i].rc);
        }
        m.saved_h_vec.assign(1, (Real)t.root);
        t.op_swap(a, b);
        return true;
    }
    return false;
}

bool MoveEngine::apply_ar(const FloorplanInstance& inst, BTree& t, Move& m) {
    const int n = inst.n_blocks;
    int tries = 32;
    while (tries-- > 0) {
        int v = rand_int(rng_, 0, n - 1);
        const Block& b = inst.blocks[v];
        if (block_dims_locked(b)) continue;
        if (b.mib_group >= 0) {
            // Use the MibSync move instead -- changing one MIB block's AR
            // would create a violation. Recursively try.
            continue;
        }
        if (b.area_target <= 0) continue;
        m.v = v;
        m.saved_w = t.w[v];
        m.saved_h = t.h[v];
        auto [nw, nh] = sample_dims(b.area_target, b.ar_min, b.ar_max, rng_);
        t.w[v] = nw;
        t.h[v] = nh;
        return true;
    }
    return false;
}

bool MoveEngine::apply_mib(const FloorplanInstance& inst, BTree& t, Move& m) {
    if (inst.mib_groups.empty()) return false;
    int tries = 16;
    while (tries-- > 0) {
        int g = rand_int(rng_, 0, (int)inst.mib_groups.size() - 1);
        const auto& group = inst.mib_groups[g];
        if (group.empty()) continue;
        // Use the *first non-locked* block's area as the canonical area.
        Real area = -1.0;
        Real armin = 0.25, armax = 4.0;
        for (int b : group) {
            if (block_dims_locked(inst.blocks[b])) continue;
            area = inst.blocks[b].area_target;
            armin = inst.blocks[b].ar_min;
            armax = inst.blocks[b].ar_max;
            break;
        }
        if (area <= 0) continue;
        auto [nw, nh] = sample_dims(area, armin, armax, rng_);
        m.mib_blocks = group;
        m.saved_w_vec.assign(group.size(), 0);
        m.saved_h_vec.assign(group.size(), 0);
        for (size_t i = 0; i < group.size(); ++i) {
            int b = group[i];
            m.saved_w_vec[i] = t.w[b];
            m.saved_h_vec[i] = t.h[b];
            t.w[b] = nw;
            t.h[b] = nh;
        }
        return true;
    }
    return false;
}

bool MoveEngine::apply_fixb(const FloorplanInstance& inst, BTree& t, Move& m,
                            const Costs* /*prev*/) {
    // Find blocks whose boundary constraint is currently violated.  For each
    // we have several "fix" tactics (PARSAC §3.2.1):
    //   1) If there are blocks at the required boundary that have no
    //      boundary constraint, swap one of them with our violating block.
    //   2) Otherwise, move the violating block to be the right-child of one
    //      of those constrained blocks (forces it to the same edge).
    //
    // We need a current packing to determine bbox extent and which blocks are
    // already at the boundary.  We assume the packing is current.
    const int n = inst.n_blocks;
    Real Wbb = 0, Hbb = 0;
    for (int i = 0; i < n; ++i) {
        Wbb = std::max(Wbb, t.x[i] + t.w[i]);
        Hbb = std::max(Hbb, t.y[i] + t.h[i]);
    }
    auto edge_match = [&](int b, BoundaryEdge e) -> bool {
        Real x = t.x[b], y = t.y[b], w = t.w[b], h = t.h[b];
        bool L = std::abs(x) < 1e-7;
        bool B = std::abs(y) < 1e-7;
        bool R = std::abs((x + w) - Wbb) < 1e-7;
        bool T = std::abs((y + h) - Hbb) < 1e-7;
        switch (e) {
            case E_LEFT: return L; case E_RIGHT: return R;
            case E_BOTTOM: return B; case E_TOP: return T;
            case C_BL: return L && B; case C_BR: return R && B;
            case C_TL: return L && T; case C_TR: return R && T;
            default: return true;
        }
    };

    // collect violating blocks
    std::vector<int> violating;
    for (int i = 0; i < n; ++i) {
        if (inst.blocks[i].bedge != E_NONE && !edge_match(i, inst.blocks[i].bedge))
            violating.push_back(i);
    }
    if (violating.empty()) return false;
    int v = violating[rand_int(rng_, 0, (int)violating.size() - 1)];
    BoundaryEdge e = inst.blocks[v].bedge;

    // PARSAC tactic 1: find a non-constrained block currently at edge e and swap
    std::vector<int> candidates;
    for (int j = 0; j < n; ++j) {
        if (j == v) continue;
        if (inst.blocks[j].bedge != E_NONE) continue;
        if (edge_match(j, e)) candidates.push_back(j);
    }
    if (!candidates.empty()) {
        int u = candidates[rand_int(rng_, 0, (int)candidates.size() - 1)];
        m.a = v; m.b = u;
        m.kind = MoveKind::Swap;
        m.saved_w_vec.clear();
        m.saved_w_vec.reserve(n * 3);
        for (int i = 0; i < n; ++i) {
            m.saved_w_vec.push_back((Real)t.nodes[i].parent);
            m.saved_w_vec.push_back((Real)t.nodes[i].lc);
            m.saved_w_vec.push_back((Real)t.nodes[i].rc);
        }
        m.saved_h_vec.assign(1, (Real)t.root);
        t.op_swap(v, u);
        m.always_accept = true;
        return true;
    }

    // Tactic 2: move v to be the right-child of any constrained block already
    // at edge e (this anchors it on the edge in the next packing).
    std::vector<int> anchors;
    for (int j = 0; j < n; ++j) {
        if (j == v) continue;
        if (edge_match(j, e)) anchors.push_back(j);
    }
    if (anchors.empty()) return false;
    int u = anchors[rand_int(rng_, 0, (int)anchors.size() - 1)];
    bool as_left;
    // For LEFT/BOTTOM edges, the block needs the leftmost / bottom-most
    // position; for RIGHT/TOP, picking right_child of an anchor pushes v away
    // from the chosen edge.  This is heuristic -- it's just a hint.
    switch (e) {
        case E_LEFT: case C_BL: case C_TL:    as_left = false; break;
        case E_BOTTOM: case E_RIGHT: case C_BR: as_left = true;  break;
        case E_TOP: case C_TR:                as_left = false; break;
        default:                              as_left = true;  break;
    }
    m.v = v; m.u = u; m.as_left = as_left;
    m.kind = MoveKind::Move;
    m.saved_w_vec.clear();
    m.saved_w_vec.reserve(n * 3);
    for (int i = 0; i < n; ++i) {
        m.saved_w_vec.push_back((Real)t.nodes[i].parent);
        m.saved_w_vec.push_back((Real)t.nodes[i].lc);
        m.saved_w_vec.push_back((Real)t.nodes[i].rc);
    }
    m.saved_h_vec.assign(1, (Real)t.root);
    if (!t.op_move(v, u, as_left)) return false;
    m.always_accept = true;
    return true;
}

Move MoveEngine::propose(const FloorplanInstance& inst, BTree& tree, const Costs* prev) {
    Move m{};
    // Probabilities (sum to 1). The original P_FIX = 0.0005 produced ~1
    // boundary fix per 2000 iters — far too rare given that random initial
    // trees start with most boundary blocks violated. Bumped to P_FIXB +
    // P_FIXG = 0.10 so SA can chase boundary/grouping satisfaction.
    constexpr double P_FIXB = 0.05;
    constexpr double P_FIXG = 0.05;
    constexpr double P_AR   = 0.18;
    constexpr double P_MIB  = 0.05;
    constexpr double P_ROT  = 0.15;
    constexpr double P_SWP  = 0.15;
    // remainder ~0.37 = move
    double r = std::uniform_real_distribution<double>(0, 1)(rng_);
    double acc = 0.0;
    if      (r < (acc += P_FIXB)) m.kind = MoveKind::FixBoundary;
    else if (r < (acc += P_FIXG)) m.kind = MoveKind::FixGrouping;
    else if (r < (acc += P_AR))   m.kind = MoveKind::AspectRatio;
    else if (r < (acc += P_MIB))  m.kind = MoveKind::MibSync;
    else if (r < (acc += P_ROT))  m.kind = MoveKind::Rotate;
    else if (r < (acc += P_SWP))  m.kind = MoveKind::Swap;
    else                          m.kind = MoveKind::Move;

    bool ok = false;
    switch (m.kind) {
        case MoveKind::Rotate:       ok = apply_rotate(inst, tree, m); break;
        case MoveKind::Move:         ok = apply_move  (inst, tree, m); break;
        case MoveKind::Swap:         ok = apply_swap  (inst, tree, m); break;
        case MoveKind::AspectRatio:  ok = apply_ar    (inst, tree, m); break;
        case MoveKind::MibSync:      ok = apply_mib   (inst, tree, m); break;
        case MoveKind::FixBoundary:  ok = apply_fixb  (inst, tree, m, prev); break;
        case MoveKind::FixGrouping:  ok = apply_fixg  (inst, tree, m); break;
    }
    if (!ok) {
        // fall back to a swap or move
        if (apply_swap(inst, tree, m)) m.kind = MoveKind::Swap;
        else if (apply_move(inst, tree, m)) m.kind = MoveKind::Move;
    }
    return m;
}

void MoveEngine::revert(const FloorplanInstance& inst, BTree& tree, const Move& m) {
    const int n = inst.n_blocks;
    auto restore_topology = [&]() {
        for (int i = 0; i < n; ++i) {
            tree.nodes[i].parent = (int)m.saved_w_vec[3 * i];
            tree.nodes[i].lc     = (int)m.saved_w_vec[3 * i + 1];
            tree.nodes[i].rc     = (int)m.saved_w_vec[3 * i + 2];
        }
        tree.root = (int)m.saved_h_vec[0];
    };
    switch (m.kind) {
        case MoveKind::Rotate: {
            if (!m.mib_blocks.empty()) {
                for (size_t i = 0; i < m.mib_blocks.size(); ++i) {
                    tree.w[m.mib_blocks[i]] = m.saved_w_vec[i];
                    tree.h[m.mib_blocks[i]] = m.saved_h_vec[i];
                }
            } else {
                tree.w[m.v] = m.saved_w;
                tree.h[m.v] = m.saved_h;
            }
            break;
        }
        case MoveKind::AspectRatio: {
            tree.w[m.v] = m.saved_w;
            tree.h[m.v] = m.saved_h;
            break;
        }
        case MoveKind::MibSync: {
            for (size_t i = 0; i < m.mib_blocks.size(); ++i) {
                tree.w[m.mib_blocks[i]] = m.saved_w_vec[i];
                tree.h[m.mib_blocks[i]] = m.saved_h_vec[i];
            }
            break;
        }
        case MoveKind::Swap:
        case MoveKind::Move:
        case MoveKind::FixBoundary:
        case MoveKind::FixGrouping:
            if (!m.saved_w_vec.empty() && !m.saved_h_vec.empty())
                restore_topology();
            break;
    }
}

// ---------------------------------------------------------------------------
// FixGrouping: PARSAC-style constraint-fixing for grouping (cluster) groups.
// Strategy:
//   For a grouping group whose members are currently in >1 component (i.e.,
//   not all touching), pick a stray member v and move it to be the right-
//   child of another member u in the largest component. This forces v
//   directly above u in the next packing, which usually causes them to abut.
//   "Always-accept" so SA can use this to escape grouping-violated local
//   minima even when overall cost increases.
// ---------------------------------------------------------------------------
bool MoveEngine::apply_fixg(const FloorplanInstance& inst, BTree& t, Move& m) {
    if (inst.grouping_groups.empty()) return false;
    const int n = inst.n_blocks;

    // Recompute current bbox so touch tests are well-defined.
    Real Wbb = 0, Hbb = 0;
    for (int i = 0; i < n; ++i) {
        Wbb = std::max(Wbb, t.x[i] + t.w[i]);
        Hbb = std::max(Hbb, t.y[i] + t.h[i]);
    }

    auto rect_touches = [&](int a, int b) -> bool {
        Real ax = t.x[a], ay = t.y[a], aw = t.w[a], ah = t.h[a];
        Real bx = t.x[b], by = t.y[b], bw = t.w[b], bh = t.h[b];
        const Real EPS = 1e-7;
        if (std::abs((ax + aw) - bx) < EPS || std::abs((bx + bw) - ax) < EPS) {
            Real ylo = std::max(ay, by), yhi = std::min(ay + ah, by + bh);
            if (yhi - ylo > EPS) return true;
        }
        if (std::abs((ay + ah) - by) < EPS || std::abs((by + bh) - ay) < EPS) {
            Real xlo = std::max(ax, bx), xhi = std::min(ax + aw, bx + bw);
            if (xhi - xlo > EPS) return true;
        }
        return false;
    };

    // Pick a random group with size >= 2 that is currently fragmented.
    std::vector<int> candidate_groups;
    for (size_t g = 0; g < inst.grouping_groups.size(); ++g) {
        const auto& grp = inst.grouping_groups[g];
        if ((int)grp.size() < 2) continue;
        // Quick component count via union-find on touch graph.
        std::vector<int> par((int)grp.size());
        for (size_t i = 0; i < grp.size(); ++i) par[i] = (int)i;
        std::function<int(int)> find = [&](int x) -> int {
            while (par[x] != x) { par[x] = par[par[x]]; x = par[x]; }
            return x;
        };
        for (size_t i = 0; i < grp.size(); ++i)
            for (size_t j = i + 1; j < grp.size(); ++j)
                if (rect_touches(grp[i], grp[j])) {
                    int a = find((int)i), b = find((int)j);
                    if (a != b) par[a] = b;
                }
        int comps = 0;
        for (size_t i = 0; i < grp.size(); ++i) if (find((int)i) == (int)i) ++comps;
        if (comps > 1) candidate_groups.push_back((int)g);
    }
    if (candidate_groups.empty()) return false;

    int g = candidate_groups[rand_int(rng_, 0, (int)candidate_groups.size() - 1)];
    const auto& grp = inst.grouping_groups[g];

    // Group members into connected components again (we recompute so we know
    // the exact partition).
    std::vector<int> par((int)grp.size());
    for (size_t i = 0; i < grp.size(); ++i) par[i] = (int)i;
    std::function<int(int)> find = [&](int x) -> int {
        while (par[x] != x) { par[x] = par[par[x]]; x = par[x]; }
        return x;
    };
    for (size_t i = 0; i < grp.size(); ++i)
        for (size_t j = i + 1; j < grp.size(); ++j)
            if (rect_touches(grp[i], grp[j])) {
                int a = find((int)i), b = find((int)j);
                if (a != b) par[a] = b;
            }
    // Find the largest component and a stray member.
    std::map<int, std::vector<int>> comps;
    for (size_t i = 0; i < grp.size(); ++i) comps[find((int)i)].push_back(grp[i]);
    int big_root = -1; int big_size = -1;
    for (auto& kv : comps) if ((int)kv.second.size() > big_size) {
        big_size = (int)kv.second.size(); big_root = kv.first;
    }
    std::vector<int> strays;
    for (auto& kv : comps) if (kv.first != big_root)
        for (int b : kv.second) strays.push_back(b);
    if (strays.empty()) return false;

    int v = strays[rand_int(rng_, 0, (int)strays.size() - 1)];
    if (v == t.root) return false;             // moving root invalidates op_move
    if (inst.blocks[v].is_preplaced) return false;
    const auto& big = comps[big_root];
    int u = big[rand_int(rng_, 0, (int)big.size() - 1)];
    if (u == v) return false;

    // Save full topology snapshot (cheap at n <= 200).
    m.v = v; m.u = u; m.as_left = false;       // right-child = stack on top of u
    m.kind = MoveKind::FixGrouping;
    m.saved_w_vec.clear();
    m.saved_w_vec.reserve(n * 3);
    for (int i = 0; i < n; ++i) {
        m.saved_w_vec.push_back((Real)t.nodes[i].parent);
        m.saved_w_vec.push_back((Real)t.nodes[i].lc);
        m.saved_w_vec.push_back((Real)t.nodes[i].rc);
    }
    m.saved_h_vec.assign(1, (Real)t.root);
    if (!t.op_move(v, u, m.as_left)) return false;
    m.always_accept = true;
    return true;
}

} // namespace fp
