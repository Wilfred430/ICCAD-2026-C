// sa.hpp -- Simulated annealing driver.
//
// Implements a Fast-SA-style schedule (Chen & Chang 2006) with three stages
// plus a constraints-aware acceptance rule (PARSAC §3.2.1).
//
//   Stage 1 (k = 1)         T = T1 = high enough that ~99% uphill moves accepted
//   Stage 2 (2 <= k <= K)   T = T1 * delta_avg / (k * c)
//   Stage 3 (k > K)         T = T1 * delta_avg / k
//
// where delta_avg is the running average of |Δcost| across all moves
// considered.  PARSAC's constraints-fixing moves are *always* accepted
// regardless of the cost change.
//
#pragma once
#include "types.hpp"
#include "btree.hpp"
#include "packer.hpp"
#include "cost.hpp"
#include "moves.hpp"

#include <atomic>
#include <chrono>

namespace fp {

struct SAConfig {
    int    n_iters_per_block = 200;     // K1 -- iterations per block per temperature step
    int    K = 7;                       // Fast-SA: end of stage 2
    double p_accept_init = 0.99;        // initial uphill acceptance
    double c_fastsa = 100.0;            // FastSA c constant
    double time_budget_sec = 30.0;      // hard wall-clock cap
    SAWeights weights;                  // SA cost weights (mutable across stages later)
    bool   verbose = false;
    int    log_every = 200;
};

struct SAResult {
    BTree   best_tree;
    Costs   best_costs;
    double  best_sa_cost = REAL_INF;
    int     iters = 0;
    double  elapsed_sec = 0.0;
};

class SimulatedAnnealing {
public:
    SimulatedAnnealing(const FloorplanInstance& inst, const SAConfig& cfg, uint64_t seed);

    // Run SA from the given initial tree (passed by value -- the algorithm
    // makes its own copies as it goes).  Stops when budget elapses.
    SAResult run(BTree initial);

    // Allow external code to ask SA to stop early (used by parallel runner
    // when a "good enough" cost is reached or wall-clock exceeded globally).
    std::atomic<bool> stop_flag{false};

private:
    const FloorplanInstance& inst_;
    SAConfig cfg_;
    uint64_t seed_;
    Packer    packer_;
    Evaluator evaluator_;
    MoveEngine engine_;
};

} // namespace fp
