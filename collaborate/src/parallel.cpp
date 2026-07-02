// parallel.cpp -- Parallel SA runner.
#include "parallel.hpp"
#include "btree.hpp"
#include "packer.hpp"
#include "cost.hpp"

#include <thread>
#include <vector>
#include <mutex>
#include <atomic>
#include <iostream>
#include <random>
#include <algorithm>
#include <cmath>
#include <map>

namespace fp {

namespace {

// Build a constraint-aware initial tree.
//
// Random initial trees produce two failure modes:
//   1. Most boundary-constrained blocks start violated (random position).
//   2. Members of the same grouping/MIB group land far apart in the tree,
//      so even after heavy SA they tend to stay fragmented.
//
// We instead build a balanced tree biased to put related blocks nearby:
//   - Bottom-row "anchors": preplaced/fixed blocks first (root region),
//     then BL/BR/TL/TR corner-constrained blocks, then edge-constrained.
//   - For each grouping/MIB group, place members as adjacent siblings.
//   - Remaining soft blocks fill in by area (largest first, helps the
//     contour packer keep the floorplan square).
//
// The tree shape is "balanced" (random insertion at *random existing
// node*, choosing an empty slot or descending to lc) — this matches
// build_random's general behaviour but with a deliberate insertion order.
BTree make_initial(const FloorplanInstance& inst, uint64_t seed) {
    BTree t;
    t.init(inst.n_blocks);
    const int n = inst.n_blocks;
    if (n == 0) { t.root = NO_NODE; return t; }

    // ---- Initial dimensions -------------------------------------------------
    for (int i = 0; i < n; ++i) {
        const Block& b = inst.blocks[i];
        if (b.is_fixed || b.is_preplaced) {
            t.w[i] = b.w_input;
            t.h[i] = b.h_input;
        } else if (b.area_target > 0) {
            // Near-square initialisation; aspect-ratio move will explore later.
            Real s = std::sqrt(b.area_target);
            t.w[i] = s;
            t.h[i] = s;
        } else {
            t.w[i] = 1.0;
            t.h[i] = 1.0;
        }
    }
    // MIB sync: pick the first non-locked member's dims as the shared shape.
    for (const auto& g : inst.mib_groups) {
        Real W = -1, H = -1;
        for (int b : g)
            if (!inst.blocks[b].is_fixed && !inst.blocks[b].is_preplaced) {
                W = t.w[b]; H = t.h[b]; break;
            }
        if (W <= 0) continue;
        for (int b : g)
            if (!inst.blocks[b].is_fixed && !inst.blocks[b].is_preplaced) {
                t.w[b] = W; t.h[b] = H;
            }
    }

    // ---- Build ordered insertion list --------------------------------------
    // Priority for tree position (lower = inserted earlier = closer to root):
    //   0: corner-constrained (BL=4, BR=5, TL=6, TR=7)
    //   1: edge-constrained
    //   2: largest grouping/MIB members first
    //   3: largest remaining soft blocks first
    //   4: preplaced (anchored) — placed LAST so they're leaves
    //
    // Why preplaced is no longer priority 0:
    //   When a preplaced block is the root, the contour packer ignores its
    //   tree-derived position and snaps it to its input (x,y).  If that
    //   anchor sits at, say, (31, 216), then the root's descendants build
    //   their contour starting from height 216 above x∈[31, 31+w].  In
    //   practice the whole floorplan ends up offset upward — exactly the
    //   "blocks float far from the origin" symptom we see in vis_output.
    //   Making preplaced a leaf keeps it anchored at its true position
    //   without dragging the rest of the tree away from (0, 0).
    auto priority = [&](int i) -> int {
        const Block& b = inst.blocks[i];
        if (b.is_preplaced) return 4;
        if (b.bedge >= C_BL) return 0;
        if (b.bedge != E_NONE) return 1;
        if (b.group_id >= 0 || b.mib_group >= 0) return 2;
        return 3;
    };

    // Connectivity score: total weight of nets each block participates in.
    // Used as a tie-breaker so that highly-connected blocks land closer to
    // the root (and thus near the origin in the contour packer).  This is
    // motivated by case 99: the origin region is where most terminals live,
    // so blocks with many net connections should be packed there to keep
    // external HPWL low.
    std::vector<Real> conn_score(n, 0.0);
    for (const auto& net : inst.b2b_nets) {
        if (net.a >= 0 && net.a < n) conn_score[net.a] += net.w;
        if (net.b >= 0 && net.b < n) conn_score[net.b] += net.w;
    }
    for (const auto& net : inst.p2b_nets) {
        if (net.b >= 0 && net.b < n) conn_score[net.b] += net.w;
    }

    std::vector<int> order(n);
    for (int i = 0; i < n; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        int pa = priority(a), pb = priority(b);
        if (pa != pb) return pa < pb;
        // Primary tie-break: more-connected first (drives them toward origin)
        if (conn_score[a] != conn_score[b]) return conn_score[a] > conn_score[b];
        // Secondary tie-break: larger area first (helps the contour-packer)
        Real Aa = t.w[a] * t.h[a];
        Real Ab = t.w[b] * t.h[b];
        if (Aa != Ab) return Aa > Ab;
        return a < b;
    });

    // ---- Build tree by insertion -------------------------------------------
    // Root is the highest-priority block; subsequent inserts pick a uniformly
    // random *existing* node and walk down to the first free slot, with a
    // bias toward grouping/MIB siblings going next to each other.
    std::mt19937_64 rng(seed);
    int root = order[0];
    t.root = root;
    t.nodes[root] = BNode{};

    // Track which inserted blocks belong to which grouping/MIB id
    std::map<int, std::vector<int>> placed_in_group;   // grouping
    std::map<int, std::vector<int>> placed_in_mib;     // mib

    auto record = [&](int v) {
        const Block& b = inst.blocks[v];
        if (b.group_id >= 0)  placed_in_group[b.group_id].push_back(v);
        if (b.mib_group >= 0) placed_in_mib[b.mib_group].push_back(v);
    };
    record(root);

    // Track global lc/rc counts so far.  In a B*-tree:
    //   left child  → block placed to the RIGHT of its parent (extends width)
    //   right child → block placed ABOVE its parent           (extends height)
    // A pure right-spine produces a vertical stack (the bug we saw on case 80
    // and 99); a pure left-spine produces a horizontal strip.  Balancing the
    // two counts produces a roughly square initial floorplan.
    int n_lc = 0, n_rc = 0;
    auto insert_under = [&](int v, int u) {
        // Walk down to the first node with an empty slot.  Two changes vs. the
        // old version:
        //   (a) If u has an empty slot, ALWAYS use it instead of descending —
        //       the old code's 50/50 lc/rc choice could bypass an empty slot
        //       and create deep chains, which produces tall narrow floorplans.
        //   (b) When BOTH slots are empty, bias toward the under-represented
        //       direction (lc if n_lc < n_rc, rc if n_rc < n_lc).  This keeps
        //       the global lc/rc count roughly balanced → square bbox.
        while (true) {
            bool lc_empty = (t.nodes[u].lc == NO_NODE);
            bool rc_empty = (t.nodes[u].rc == NO_NODE);
            bool go_left;
            if (lc_empty && rc_empty) {
                if      (n_lc < n_rc) go_left = true;
                else if (n_rc < n_lc) go_left = false;
                else                  go_left = std::bernoulli_distribution(0.5)(rng);
            } else if (lc_empty) {
                go_left = true;
            } else if (rc_empty) {
                go_left = false;
            } else {
                // Neither slot empty: descend to the smaller subtree's child
                // (still random pick — the empty-slot guard fires next loop).
                go_left = std::bernoulli_distribution(0.5)(rng);
                u = go_left ? t.nodes[u].lc : t.nodes[u].rc;
                continue;
            }
            int& slot = go_left ? t.nodes[u].lc : t.nodes[u].rc;
            slot = v;
            t.nodes[v].parent = u;
            if (go_left) ++n_lc; else ++n_rc;
            return;
        }
    };

    for (int k = 1; k < n; ++k) {
        int v = order[k];
        const Block& bv = inst.blocks[v];

        // Anchor candidate u: prefer an already-placed member of v's grouping
        // or MIB group so they end up tree-adjacent (and thus likely abutted
        // or sharing shape after packing).
        int u = -1;
        if (bv.group_id >= 0) {
            auto it = placed_in_group.find(bv.group_id);
            if (it != placed_in_group.end() && !it->second.empty())
                u = it->second[std::uniform_int_distribution<int>(0, (int)it->second.size() - 1)(rng)];
        }
        if (u == -1 && bv.mib_group >= 0) {
            auto it = placed_in_mib.find(bv.mib_group);
            if (it != placed_in_mib.end() && !it->second.empty())
                u = it->second[std::uniform_int_distribution<int>(0, (int)it->second.size() - 1)(rng)];
        }
        if (u == -1) {
            // Random already-placed node
            u = order[std::uniform_int_distribution<int>(0, k - 1)(rng)];
        }
        insert_under(v, u);
        record(v);
    }
    return t;
}

// Build an initial tree from the ML predictor's WARM_POSITIONS.
//
// The predictor gives an approximate center (cx,cy) and shape (w,h) for every
// block.  Those positions overlap and are NOT a legal floorplan, but their
// RELATIVE arrangement encodes where the model thinks each block belongs.  We
// turn that arrangement into a B*-tree and let the contour packer compact it
// into a legal placement:
//
//   - soft-block dims : adopt the ML aspect ratio, renormalised to the exact
//                       area target, so the 1% area HARD constraint still holds.
//   - insertion order : bottom-left first (sort by x+y of the lower-left
//                       corner) so early inserts land near the origin, matching
//                       the packer's growth direction; preplaced blocks go last
//                       (kept as leaves -- same rationale as make_initial).
//   - parent / slot   : attach each block under the nearest already-placed
//                       block; LEFT child if the ML says it is to the RIGHT,
//                       RIGHT child if the ML says it is ABOVE.
//
// Any malformed result falls back to make_initial(), so a warm start can never
// make the solver worse than the baseline initial tree.
BTree make_initial_warm(const FloorplanInstance& inst, uint64_t seed) {
    const int n = inst.n_blocks;
    BTree t;
    t.init(n);
    if (n == 0) { t.root = NO_NODE; return t; }
    if (!inst.has_warm) return make_initial(inst, seed);

    auto clampr = [](Real v, Real lo, Real hi) {
        return v < lo ? lo : (v > hi ? hi : v);
    };

    // ---- dimensions --------------------------------------------------------
    for (int i = 0; i < n; ++i) {
        const Block& b = inst.blocks[i];
        if (b.is_fixed || b.is_preplaced) {
            t.w[i] = b.w_input;
            t.h[i] = b.h_input;
        } else if (b.area_target > 0) {
            Real wm = inst.warm_w[i], hm = inst.warm_h[i];
            if (wm > 1e-9 && hm > 1e-9) {
                Real aspect = clampr(hm / wm, b.ar_min, b.ar_max);   // h / w
                Real h = std::sqrt(b.area_target * aspect);
                Real w = b.area_target / h;                          // w*h == area exactly
                t.w[i] = w; t.h[i] = h;
            } else {
                Real s = std::sqrt(b.area_target);
                t.w[i] = s; t.h[i] = s;
            }
        } else {
            t.w[i] = 1.0; t.h[i] = 1.0;
        }
    }
    // MIB sync: share one shape across each MIB group (same policy as make_initial).
    for (const auto& g : inst.mib_groups) {
        Real W = -1, H = -1;
        for (int b : g)
            if (!inst.blocks[b].is_fixed && !inst.blocks[b].is_preplaced) { W = t.w[b]; H = t.h[b]; break; }
        if (W <= 0) continue;
        for (int b : g)
            if (!inst.blocks[b].is_fixed && !inst.blocks[b].is_preplaced) { t.w[b] = W; t.h[b] = H; }
    }

    // ---- lower-left corners implied by the ML centers ----------------------
    std::vector<Real> X(n), Y(n);
    for (int i = 0; i < n; ++i) {
        X[i] = inst.warm_cx[i] - t.w[i] * 0.5;
        Y[i] = inst.warm_cy[i] - t.h[i] * 0.5;
    }

    // ---- insertion order: bottom-left first, preplaced last ----------------
    std::vector<int> order(n);
    for (int i = 0; i < n; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        bool pa = inst.blocks[a].is_preplaced, pb = inst.blocks[b].is_preplaced;
        if (pa != pb) return !pa;                 // non-preplaced before preplaced
        Real ka = X[a] + Y[a], kb = X[b] + Y[b];
        if (ka != kb) return ka < kb;
        return a < b;
    });

    // ---- build tree: nearest placed anchor + directional slot --------------
    int root = order[0];
    t.root = root;
    t.nodes[root] = BNode{};
    std::vector<int> placed;
    placed.reserve(n);
    placed.push_back(root);

    for (int k = 1; k < n; ++k) {
        int v = order[k];
        // Nearest already-placed block to v's lower-left corner.
        int u = root; Real bestd = REAL_INF;
        for (int p : placed) {
            Real dx = X[v] - X[p], dy = Y[v] - Y[p];
            Real d = dx * dx + dy * dy;
            if (d < bestd) { bestd = d; u = p; }
        }
        // Descend from u to the first free slot, preferring the ML direction.
        while (true) {
            bool want_left = (X[v] - X[u]) >= (Y[v] - Y[u]);  // right -> lc, above -> rc
            bool lfree = (t.nodes[u].lc == NO_NODE);
            bool rfree = (t.nodes[u].rc == NO_NODE);
            if (want_left && lfree) { t.nodes[u].lc = v; t.nodes[v].parent = u; break; }
            if (!want_left && rfree){ t.nodes[u].rc = v; t.nodes[v].parent = u; break; }
            if (lfree)              { t.nodes[u].lc = v; t.nodes[v].parent = u; break; }
            if (rfree)              { t.nodes[u].rc = v; t.nodes[v].parent = u; break; }
            u = want_left ? t.nodes[u].lc : t.nodes[u].rc;    // both full: descend
        }
        placed.push_back(v);
    }

    if (!t.validate()) return make_initial(inst, seed);   // safety net
    return t;
}

} // namespace

ParallelResult run_parallel(const FloorplanInstance& inst,
                            const ParallelConfig& cfg, uint64_t base_seed) {
    int N = std::max(1, cfg.n_threads);
    std::vector<SAResult> results(N);
    std::vector<std::thread> threads;
    threads.reserve(N);
    std::vector<std::unique_ptr<SimulatedAnnealing>> sas(N);

    // Cross-thread early-stop atomic.  Any SA chain that reaches
    // target_contest_cost sets this; every chain checks it at the top of
    // each iter and exits if set.  Lives on the stack of run_parallel so
    // its lifetime trivially outlasts all worker threads (which we join
    // before returning).
    std::atomic<bool> shared_stop{false};

    for (int i = 0; i < N; ++i) {
        SAConfig sa_cfg = cfg.sa_cfg;
        sa_cfg.stopping.time_budget_sec = cfg.time_budget_sec;
        sas[i] = std::make_unique<SimulatedAnnealing>(
            inst, sa_cfg, base_seed + 1009u * (uint64_t)i, &shared_stop);
    }

    // When the ML predictor supplied WARM_POSITIONS, dedicate up to half the
    // threads to the ML warm-start tree and keep the rest on the diverse
    // make_initial() restarts.  Best-of selection below means the warm threads
    // can only help: if the ML start is worse, a diverse thread still wins.
    int n_warm = inst.has_warm ? std::max(1, N / 2) : 0;
    if (cfg.sa_cfg.verbose && n_warm > 0) {
        std::cerr << "[parallel] ML warm-start active on " << n_warm << "/" << N
                  << " threads\n";
    }

    for (int i = 0; i < N; ++i) {
        threads.emplace_back([i, n_warm, &inst, &results, &sas, base_seed]() {
            BTree init = (i < n_warm)
                ? make_initial_warm(inst, base_seed + 17u * (uint64_t)i)
                : make_initial(inst, base_seed + 17u * (uint64_t)i);
            results[i] = sas[i]->run(std::move(init));
        });
    }
    for (auto& th : threads) th.join();

    // Pick the best by contest cost (feasible first, then min cost).
    Evaluator ev;
    ParallelResult R;
    Real best = REAL_INF;
    int  best_idx = -1;
    int  feas_cnt = 0;
    for (int i = 0; i < N; ++i) {
        if (results[i].best_costs.feasible) ++feas_cnt;
        Real cost = ev.contest_cost(results[i].best_costs, 1.0);
        
        // Strongly prefer feasible
        if (results[i].best_costs.feasible && best_idx >= 0 && !results[best_idx].best_costs.feasible) {
            best = cost; best_idx = i;
        } else if (cost < best) {
            // tie-break: feasible wins
            if (best_idx < 0 ||
                (results[i].best_costs.feasible && !results[best_idx].best_costs.feasible) ||
                results[i].best_costs.feasible == results[best_idx].best_costs.feasible) {
                best = cost; best_idx = i;
            }
        }
    }
    if (best_idx < 0) best_idx = 0;
    R.best = std::move(results[best_idx]);
    R.best_thread = best_idx;
    R.n_feasible = feas_cnt;
    return R;
}

} // namespace fp
