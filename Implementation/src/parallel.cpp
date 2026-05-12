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

namespace fp {

namespace {

// Build a soft-block-aware initial tree:  start from a random binary tree,
// then for soft blocks pick a near-square (w,h) satisfying the area target,
// for fixed/preplaced use the input dims, for MIB members snap to a single
// shape.  This gives SA a feasible-area starting point.
BTree make_initial(const FloorplanInstance& inst, uint64_t seed) {
    BTree t;
    t.init(inst.n_blocks);
    t.build_random(seed);

    // Initial dimensions
    for (int i = 0; i < inst.n_blocks; ++i) {
        const Block& b = inst.blocks[i];
        if (b.is_fixed || b.is_preplaced) {
            t.w[i] = b.w_input;
            t.h[i] = b.h_input;
        } else if (b.area_target > 0) {
            // Near-square initialisation
            Real s = std::sqrt(b.area_target);
            t.w[i] = s;
            t.h[i] = s;
        } else {
            t.w[i] = 1.0;
            t.h[i] = 1.0;
        }
    }
    // Synchronise MIB groups: pick the first non-locked block's dims as the
    // shared shape.  If everyone in the group is locked, this is a no-op
    // (and probably an inconsistent input anyway).
    for (const auto& g : inst.mib_groups) {
        Real W = -1, H = -1;
        for (int b : g) {
            if (!inst.blocks[b].is_fixed && !inst.blocks[b].is_preplaced) {
                W = t.w[b]; H = t.h[b]; break;
            }
        }
        if (W <= 0) continue;
        for (int b : g) {
            if (!inst.blocks[b].is_fixed && !inst.blocks[b].is_preplaced) {
                t.w[b] = W; t.h[b] = H;
            }
        }
    }
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

    for (int i = 0; i < N; ++i) {
        SAConfig sa_cfg = cfg.sa_cfg;
        sa_cfg.time_budget_sec = cfg.time_budget_sec;
        sas[i] = std::make_unique<SimulatedAnnealing>(inst, sa_cfg, base_seed + 1009u * (uint64_t)i);
    }

    for (int i = 0; i < N; ++i) {
        threads.emplace_back([i, &inst, &results, &sas, base_seed]() {
            BTree init = make_initial(inst, base_seed + 17u * (uint64_t)i);
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
