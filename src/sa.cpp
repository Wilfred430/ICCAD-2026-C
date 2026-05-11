// sa.cpp -- Fast-SA driver.
#include "sa.hpp"
#include <cmath>
#include <random>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <thread>
#include <sstream>

namespace fp {

SimulatedAnnealing::SimulatedAnnealing(const FloorplanInstance& inst,
                                       const SAConfig& cfg, uint64_t seed)
    : inst_(inst), cfg_(cfg), seed_(seed), engine_(seed) {}

SAResult SimulatedAnnealing::run(BTree current) {
    using clock = std::chrono::steady_clock;
    auto t0 = clock::now();

    // Initial pack & evaluate
    PackResult pr = packer_.pack(inst_, current);
    Costs c = evaluator_.evaluate(inst_, current, pr);
    Real  cost = evaluator_.sa_cost(c, cfg_.weights, inst_);

    SAResult R;
    R.best_tree.copy_from(current);
    R.best_costs = c;
    R.best_sa_cost = cost;

    // ---- Calibrate T1 by sampling |Δcost| of random moves AT the initial
    //      state (always reverting), then setting T1 such that
    //      exp(-Δavg / T1) = p_accept_init.
    //
    // The previous version did a 60-step random walk that ACCEPTED every move,
    // so by step 30 the probe had drifted far from `current` and Δ samples
    // came from a high-cost region.  That over-estimated Δavg and inflated T1
    // by ~10x, which then parked stage-1 at T so high every uphill move was
    // accepted — destroying the good initial state in the first ~5K iters.
    Real delta_sum = 0.0;
    int  delta_n   = 0;
    BTree probe; probe.copy_from(current);
    for (int i = 0; i < 80; ++i) {
        Move m = engine_.propose(inst_, probe);
        PackResult pp = packer_.pack(inst_, probe);
        Costs cc = evaluator_.evaluate(inst_, probe, pp);
        Real nc = evaluator_.sa_cost(cc, cfg_.weights, inst_);
        Real d = nc - cost;
        if (d > 0) { delta_sum += d; ++delta_n; }
        engine_.revert(inst_, probe, m);   // always revert so probe stays at initial state
    }
    Real delta_avg = (delta_n > 0) ? delta_sum / delta_n : 1.0;
    Real T1 = -delta_avg / std::log(std::max(1e-6, cfg_.p_accept_init));
    if (!(T1 > 0) || std::isnan(T1) || std::isinf(T1)) T1 = 1.0;

    if (cfg_.verbose) {
        std::cerr << "[SA] delta_avg=" << delta_avg
                  << " T1=" << T1 << "\n";
    }

    // ---- Main loop ----
    std::mt19937_64 rng(seed_ ^ 0xDEADBEEFULL);
    std::uniform_real_distribution<double> U(0.0, 1.0);

    int  k = 1;
    int  iter = 0;
    int  iters_this_step = 0;
    int  iters_per_step = std::max(1, cfg_.n_iters_per_block * inst_.n_blocks);
    Real T = T1;

    // Re-anchor: if no improvement for this many iters, restore current<-best
    // and continue cooling.  Prevents `current` from random-walking into a
    // bad-cost region and never returning, which is what we observed in the
    // convergence log (best stuck at iter 1, current hovering at 4x best).
    const int reanchor_threshold =
        std::max(1, cfg_.reanchor_every_iters * std::max(1, inst_.n_blocks));
    int iters_since_improvement = 0;

    std::stringstream ss;
    ss << "log_thread/convergence_log_thread_" << std::this_thread::get_id() << ".csv";
    std::ofstream csv_file(ss.str());
    // 寫入 CSV 的標題列 (Header)
    csv_file << "Iteration,Temperature,CurrentCost,BestCost\n";
    int counter = 0;

    Real running_delta_sum = 0.0;
    int  running_delta_n = 0;

    while (true) {
        // Time / stop check
        auto t1 = clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();
        if (elapsed > cfg_.time_budget_sec) break;
        if (stop_flag.load()) break;

        // Temperature schedule (modified Fast-SA).
        //
        // Two fixes vs. the textbook FastSA:
        //   1. Stage-3 formula removed — keeps T monotone past k=K.
        //   2. T is capped at T1 — prevents the k=1→k=2 boundary spike we
        //      observed on real test data: when stage 1 accumulates large Δ
        //      samples (since at T=T1 most uphill moves get accepted), avg
        //      can grow so big that T1*avg/(2c) > T1, vaulting T *upward*
        //      at the very moment cooling is supposed to start.
        if (k == 1) {
            T = T1;
        } else {
            Real avg = (running_delta_n > 0) ? running_delta_sum / running_delta_n : delta_avg;
            Real T_raw = T1 * avg / ((Real)k * cfg_.c_fastsa);
            T = std::min(T1, std::max(1e-9, T_raw));
        }

        Move m = engine_.propose(inst_, current);
        PackResult pp = packer_.pack(inst_, current);
        Costs cc = evaluator_.evaluate(inst_, current, pp);
        Real nc = evaluator_.sa_cost(cc, cfg_.weights, inst_);
        Real d = nc - cost;

        bool accept;
        // Constraint-fixing moves (FixBoundary, FixGrouping) used to bypass
        // SA acceptance unconditionally.  They can however turn a feasible
        // tree into an infeasible one (creating preplaced overlap, etc.) —
        // the cost then jumps by w_overlap = 5000 in one step, which we
        // observed as the bimodal 4000⇄8000 oscillation in the log.
        // Now: trust always_accept only when the move does NOT introduce a
        // new hard violation; otherwise fall back to standard SA acceptance.
        bool intro_new_hard =
            (cc.overlap_violation && !c.overlap_violation) ||
            (cc.area_violation    && !c.area_violation);
        if (m.always_accept && !intro_new_hard) accept = true;
        else if (d <= 0) accept = true;
        else {
            Real prob = std::exp(-d / T);
            accept = U(rng) < prob;
        }

        if (accept) {
            cost = nc;
            c = cc;
            // Update best by *contest-relevant* metric: prefer feasible.
            bool better = false;
            if (R.best_costs.feasible && !cc.feasible) better = false;
            else if (!R.best_costs.feasible && cc.feasible) better = true;
            else better = (cost < R.best_sa_cost);
            if (better) {
                R.best_sa_cost = cost;
                R.best_costs = c;
                R.best_tree.copy_from(current);
                iters_since_improvement = 0;
            } else {
                ++iters_since_improvement;
            }
        } else {
            engine_.revert(inst_, current, m);
            ++iters_since_improvement;
        }

        // Re-anchor: if `current` has been wandering far from `best` for too
        // long, snap it back to `best` and keep cooling.  This is the standard
        // "best-restart" trick (cf. Chen & Chang 2006 §IV-C).
        if (iters_since_improvement >= reanchor_threshold) {
            current.copy_from(R.best_tree);
            PackResult pr2 = packer_.pack(inst_, current);
            c = evaluator_.evaluate(inst_, current, pr2);
            cost = evaluator_.sa_cost(c, cfg_.weights, inst_);
            iters_since_improvement = 0;
        }

        // Log every 4th iter.  IMPORTANT: log `cost` (the SA's CURRENT
        // accepted state), NOT `nc` (the candidate move's cost).
        //
        // `nc` includes proposals that were REJECTED — and B*-tree moves
        // can rearrange 10–50 blocks at once, so a rejected `nc` can be
        // 5–10× higher than the accepted `cost`.  Logging `nc` makes the
        // graph look like wild oscillation when in fact `cost` is moving
        // smoothly: the lower envelope of the noisy curve is the real SA
        // trajectory.  Logging `cost` instead matches the CSV header
        // ("CurrentCost") and gives a graph that looks like the smooth
        // descent you expect from a partition-SA-style problem.
        if(counter == 3){
            csv_file << iter << "," << T << "," << cost << "," << R.best_sa_cost << "\n";
            counter = 0;
        }else{
            counter++;
        }


        if (d > 0) { running_delta_sum += d; ++running_delta_n; }

        ++iter; ++iters_this_step;
        if (iters_this_step >= iters_per_step) {
            // Reset Δ stats at end of stage 1.  Otherwise the large deltas
            // accumulated at T=T1 (stage 1 accepts ~92% of uphill moves)
            // dominate the running average and inflate T at k=2.  We want
            // k=2 onwards to base its T on its OWN delta distribution.
            if (k == 1) {
                running_delta_sum = 0.0;
                running_delta_n   = 0;
            }
            ++k;
            iters_this_step = 0;
        }

        if (cfg_.verbose && (iter % cfg_.log_every == 0)) {
            std::cerr << "[SA] iter=" << iter
                      << " T=" << T << " cost=" << cost
                      << " best=" << R.best_sa_cost
                      << " feas=" << (c.feasible ? 1 : 0)
                      << " hpwl_gap=" << c.hpwl_gap
                      << " area_gap=" << c.area_gap
                      << " V_rel=" << c.v_relative
                      << "\n";
        }
    }
    auto tend = clock::now();
    R.elapsed_sec = std::chrono::duration<double>(tend - t0).count();
    R.iters = iter;
    return R;
}

} // namespace fp
