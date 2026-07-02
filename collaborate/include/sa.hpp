// sa.hpp -- Simulated annealing driver.
//
// Three-stage geometric cooling (replaces the original FastSA divide-by-k*c
// schedule).  All three stages use a simple per-step geometric multiplier,
// so there are no discontinuities at stage boundaries:
//
//   Stage 1 (k = 1)            T = T1                (no cooling — exploration)
//   Stage 2 (2 <= k <= K2)     T *= alpha_stage2     (rapid cooling)
//   Stage 3 (k >  K2)          T *= alpha_stage3     (slow refinement / annealing)
//
// One "step" = n_iters_per_block * n_blocks iters of the inner SA loop.
// T1 is calibrated from a random-move probe so that exp(-Δavg/T1) =
// p_accept_init.  T is floored at T1 * T_floor_ratio to prevent stage 3
// from running into denormal/zero territory on long budgets.
//
// Three termination conditions (any one triggers exit):
//   1. Wall-clock budget elapsed         (stopping.time_budget_sec)
//   2. Stagnation + frozen T             (no best improvement for
//                                          stagnation_stages stages AND
//                                          T < T1 * T_frozen_ratio)
//   3. Cross-thread early-stop signal    (shared_stop atomic set by any
//                                          peer thread when it reaches
//                                          contest_cost <= target_contest_cost)
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

// ---- Cooling schedule (geometric, 3-stage, with reheating) ----------------
struct SACooling {
    // Step index where stage 2 ends and stage 3 begins.  Equivalent to the
    // FastSA paper's K constant.  Stage 2 is 1 .. K2-1 steps long; stage 3
    // runs until time/stagnation termination.
    int    stage2_end_k = 7;

    // Per-step geometric multipliers.  T_new = T_old * alpha.
    //   stage1 = 1.0: T held at T1 throughout stage 1 (just one step by default)
    //   stage2 = 0.85: after 6 stage-2 steps, T ≈ T1 * 0.85^6 ≈ T1 * 0.38
    //   stage3 = 0.99: gentle annealing, T halves every ~70 steps
    double alpha_stage1 = 1.0;
    double alpha_stage2 = 0.92;
    double alpha_stage3 = 0.99;

    // One-shot reheating at the stage 2 → stage 3 boundary.  The original
    // FastSA paper (Chen & Chang 2006) gets reheating implicitly from its
    // stage-3 formula `T = T1·Δavg/k` jumping ~87× higher than the end of
    // stage 2; geometric cooling needs this explicitly.  At k = stage2_end_k+1
    // we multiply T by stage3_reheat once, so SA escapes the local basin found
    // by aggressive stage-2 cooling and re-explores at a higher temperature
    // before stage 3's slow annealing.
    // NOTE: with the T = T1 * stage3_reheat formula (see sa.cpp), the value
    // is interpreted as a FRACTION of T1, not a multiplier on current T.
    //   0.5 = jump T to half of initial T (mild reheat from late stage 2)
    //   0.7 = jump T to 70% of initial T (the case-56 4.9158 setting)
    //   1.0 = jump back to full T1 (FastSA-style aggressive reheat)
    double stage3_reheat = 0.7;

    // Lower bound for T relative to T1.  T is never set below T1*T_floor_ratio.
    // Same value as stopping.T_frozen_ratio by default (see SAStopping).
    double T_floor_ratio = 1e-4;
};

// ---- Termination conditions ----------------------------------------------
struct SAStopping {
    // (1) Wall-clock budget.
    double time_budget_sec = 30.0;

    // (2) Stagnation + frozen T.
    //   stagnation_stages: # of "stages" (each n_iters_per_block * n_blocks
    //     iters long) without best-cost improvement before we consider
    //     ourselves stuck.  Set to 0 to disable.
    //   T_frozen_ratio: T below T1 * this counts as frozen.  Stagnation
    //     ALONE doesn't terminate — we also require T frozen, otherwise
    //     SA may still be in a hot exploration phase where best naturally
    //     doesn't improve for a while.
    int    stagnation_stages = 30;
    double T_frozen_ratio    = 1e-4;

    // (3) Cross-thread early-stop.
    //   target_contest_cost: if a thread's best feasible contest_cost falls
    //     at or below this, it signals all peer threads to stop too.
    //     Set to 0 or negative to disable.
    double target_contest_cost = 1.001;
};

// ---- T1 calibration ------------------------------------------------------
struct SACalibration {
    int    n_probes      = 80;     // # random moves used to estimate Δavg
    double p_accept_init = 0.90;   // target uphill-acceptance probability for T1
};

// ---- Re-anchor (best-restart) -------------------------------------------
struct SAReanchor {
    // After this many iters without best improvement, snap `current` back to
    // `best` and continue cooling.  Prevents `current` from random-walking
    // into a bad-cost region from which Metropolis can't return (especially
    // late in stage 3 when T is small).  Set to 0 to disable.
    int every_iters_per_block = 50;
};

// ---- Top-level SA config ------------------------------------------------
struct SAConfig {
    // Per-T-step iter count multiplier: actual iters_per_step = this * n_blocks.
    int n_iters_per_block = 50;

    SACooling     cooling;
    SAStopping    stopping;
    SACalibration calib;
    SAReanchor    reanchor;
    MoveProb      move_prob;
    SAWeights     weights;

    bool verbose   = false;
    int  log_every = 200;
};

struct SAResult {
    BTree   best_tree;
    Costs   best_costs;
    double  best_sa_cost = REAL_INF;
    int     iters = 0;
    double  elapsed_sec = 0.0;

    // Diagnostic: which stop condition fired (0 = time, 1 = stagnation, 2 = peer)
    int     stop_reason = 0;
};

class SimulatedAnnealing {
public:
    // shared_stop is an OPTIONAL atomic shared across all parallel SA chains;
    // when set true by any chain (because it hit target_contest_cost), all
    // chains exit at the next iter.  Pass nullptr for stand-alone runs.
    SimulatedAnnealing(const FloorplanInstance& inst,
                       const SAConfig& cfg,
                       uint64_t seed,
                       std::atomic<bool>* shared_stop = nullptr);

    // Run SA from the given initial tree (passed by value -- the algorithm
    // makes its own copies as it goes).  Stops on any of the three
    // termination conditions.
    SAResult run(BTree initial);

    // External stop hook (legacy; cross-thread now uses shared_stop instead).
    std::atomic<bool> stop_flag{false};

private:
    const FloorplanInstance& inst_;
    SAConfig cfg_;
    uint64_t seed_;
    std::atomic<bool>* shared_stop_;

    Packer    packer_;
    Evaluator evaluator_;
    MoveEngine engine_;
};

} // namespace fp
