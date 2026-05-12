// sa.cpp -- Fast-SA driver.
#include "sa.hpp"
#include <cmath>
#include <random>
#include <iostream>
#include <iomanip>

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

    // ---- Calibrate T1 by running a small batch of random uphill moves and
    //      computing average |Δcost|, then setting T1 such that
    //      exp(-Δavg / T1) = p_accept_init.
    Real delta_sum = 0.0;
    int  delta_n   = 0;
    BTree probe; probe.copy_from(current);
    Real probe_cost = cost;
    for (int i = 0; i < 60; ++i) {
        Move m = engine_.propose(inst_, probe, nullptr, 1.0, 1.0);
        PackResult pp = packer_.pack(inst_, probe);
        Costs cc = evaluator_.evaluate(inst_, probe, pp);
        Real nc = evaluator_.sa_cost(cc, cfg_.weights, inst_);
        Real d = nc - probe_cost;
        if (d > 0) { delta_sum += d; ++delta_n; }
        probe_cost = nc;
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

    Real running_delta_sum = 0.0;
    int  running_delta_n = 0;

    while (true) {
        // Time / stop check
        auto t1 = clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();
        if (elapsed > cfg_.time_budget_sec) break;
        if (stop_flag.load()) break;

        // Temperature schedule (Fast-SA)
        if (k == 1) T = T1;
        else if (k <= cfg_.K) {
            Real avg = (running_delta_n > 0) ? running_delta_sum / running_delta_n : delta_avg;
            T = std::max(1e-9, T1 * avg / ((Real)k * cfg_.c_fastsa));
        } else {
            Real avg = (running_delta_n > 0) ? running_delta_sum / running_delta_n : delta_avg;
            T = std::max(1e-9, T1 * avg / (Real)k);
        }

        Move m = engine_.propose(inst_, current, &c, T, T1);
        PackResult pp = packer_.pack(inst_, current);
        Costs cc = evaluator_.evaluate(inst_, current, pp);
        Real nc = evaluator_.sa_cost(cc, cfg_.weights, inst_);
        Real d = nc - cost;

        bool accept;
        if (m.always_accept) accept = true;
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
            }
        } else {
            engine_.revert(inst_, current, m);
        }

        if (d > 0) { running_delta_sum += d; ++running_delta_n; }
        
        // --- SDB-AP: Adaptive Penalty Shaping ---
        // If we are consistently failing to stay within the outline, ramp up the penalty.
        if (iter % 2000 == 0 && iter > 0) {
            if (c.bbox_w > inst_.outline_w || c.bbox_h > inst_.outline_h) {
                cfg_.weights.w_outline *= 1.4; // Gradually ramp up
                if (cfg_.verbose) {
                    std::cerr << "[SA] Ramping up w_outline to " << cfg_.weights.w_outline << "\n";
                }
            }
        }

        ++iter; ++iters_this_step;
        if (iters_this_step >= iters_per_step) {
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
