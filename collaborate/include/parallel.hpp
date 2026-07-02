// parallel.hpp -- Multi-thread, multi-seed SA orchestrator.
//
// Runs N independent SA chains, each starting from a different randomly
// generated B*-tree seed.  Stops every chain at the global wall-clock budget
// and returns the chain whose best solution has the lowest contest cost.
//
// This is the simplest form of "parallel SA"; PARSAC also adds a periodic
// migration / cross-population step, which we leave as a TODO.
//
#pragma once
#include "types.hpp"
#include "sa.hpp"

namespace fp {

struct ParallelConfig {
    int    n_threads = 8;
    double time_budget_sec = 30.0;
    SAConfig sa_cfg;       // sa_cfg.stopping.time_budget_sec is overridden to time_budget_sec
};

struct ParallelResult {
    SAResult best;
    int      best_thread = -1;
    int      n_feasible = 0;
};

ParallelResult run_parallel(const FloorplanInstance& inst,
                            const ParallelConfig& cfg, uint64_t base_seed);

} // namespace fp
