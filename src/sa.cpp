// sa.cpp -- Simulated Annealing driver.
//
// Three-stage geometric cooling (see sa.hpp for the full schedule spec).
// Three independent stop conditions:
//   (1) Wall-clock elapsed > cfg_.stopping.time_budget_sec
//   (2) Stagnation (no best improvement for stagnation_stages * step_size
//                   iters) AND T frozen (T < T1 * T_frozen_ratio)
//   (3) Cross-thread shared_stop_ atomic set (by a peer chain that
//                   reached target_contest_cost)
//
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
                                       const SAConfig& cfg, uint64_t seed,
                                       std::atomic<bool>* shared_stop)
    : inst_(inst), cfg_(cfg), seed_(seed),
      shared_stop_(shared_stop), engine_(seed, cfg.move_prob) {}

SAResult SimulatedAnnealing::run(BTree current) {
    using clock = std::chrono::steady_clock;
    auto t0 = clock::now();

    // --- Initial pack & evaluate ------------------------------------------
    PackResult pr = packer_.pack(inst_, current);
    Costs c = evaluator_.evaluate(inst_, current, pr);
    Real  cost = evaluator_.sa_cost(c, cfg_.weights, inst_);

    SAResult R;
    R.best_tree.copy_from(current);
    R.best_costs = c;
    R.best_sa_cost = cost;

    // --- Calibrate T1 from random-move probe at the initial state ---------
    //
    // We propose N random moves, evaluate, and ALWAYS revert (so the probe
    // stays at the initial state).  This avoids the bias from a drifting
    // probe that the previous implementation suffered from.
    //
    // T1 is then set so that exp(-Δavg / T1) = p_accept_init, i.e.,
    //   T1 = -Δavg / ln(p_accept_init).
    Real delta_sum = 0.0;
    int  delta_n   = 0;
    {
        BTree probe; probe.copy_from(current);
        for (int i = 0; i < cfg_.calib.n_probes; ++i) {
            Move m = engine_.propose(inst_, probe);
            PackResult pp = packer_.pack(inst_, probe);
            Costs cc = evaluator_.evaluate(inst_, probe, pp);
            Real nc = evaluator_.sa_cost(cc, cfg_.weights, inst_);
            Real d = nc - cost;
            if (d > 0) { delta_sum += d; ++delta_n; }
            engine_.revert(inst_, probe, m);
        }
    }
    Real delta_avg = (delta_n > 0) ? delta_sum / delta_n : 1.0;
    Real T1 = -delta_avg / std::log(std::max(1e-6, cfg_.calib.p_accept_init));
    if (!(T1 > 0) || std::isnan(T1) || std::isinf(T1)) T1 = 1.0;
    const Real T_floor    = T1 * cfg_.cooling.T_floor_ratio;
    const Real T_frozen   = T1 * cfg_.stopping.T_frozen_ratio;

    if (cfg_.verbose) {
        std::cerr << "[SA] delta_avg=" << delta_avg
                  << " T1=" << T1
                  << " T_floor=" << T_floor
                  << " T_frozen=" << T_frozen << "\n";
    }

    // --- Main-loop bookkeeping --------------------------------------------
    std::mt19937_64 rng(seed_ ^ 0xDEADBEEFULL);
    std::uniform_real_distribution<double> U(0.0, 1.0);

    int  k = 1;
    int  iter = 0;
    int  iters_this_step = 0;
    const int iters_per_step =
        std::max(1, cfg_.n_iters_per_block * inst_.n_blocks);
    Real T = T1;

    // Re-anchor: restore current<-best when current drifts away for too long.
    const int reanchor_threshold = (cfg_.reanchor.every_iters_per_block > 0)
        ? std::max(1, cfg_.reanchor.every_iters_per_block * std::max(1, inst_.n_blocks))
        : 0;
    int iters_since_improvement = 0;

    // Stagnation termination: "stage" granularity matches the cooling step
    // (= iters_per_step iters per stage).
    const int stagnation_iter_threshold = (cfg_.stopping.stagnation_stages > 0)
        ? cfg_.stopping.stagnation_stages * iters_per_step
        : 0;

    // --- CSV logging (one file per thread, ./log_thread/) -----------------
    std::stringstream ss;
    ss << "log_thread/convergence_log_thread_" << std::this_thread::get_id() << ".csv";
    std::ofstream csv_file(ss.str());
    csv_file << "Iteration,Temperature,CurrentCost,BestCost\n";
    int log_counter = 0;

    // --- Helper: cross-thread stop ----------------------------------------
    auto peer_stop = [&]() -> bool {
        return shared_stop_ && shared_stop_->load(std::memory_order_relaxed);
    };
    auto maybe_signal_peers = [&](const Costs& cc) {
        if (!shared_stop_) return;
        if (cfg_.stopping.target_contest_cost <= 0.0) return;
        if (!cc.feasible) return;
        Real cc_cost = evaluator_.contest_cost(cc, 1.0);
        if (cc_cost <= cfg_.stopping.target_contest_cost) {
            shared_stop_->store(true, std::memory_order_relaxed);
        }
    };

    // --- Main loop --------------------------------------------------------
    while (true) {
        // ---- (1) Wall-clock budget --------------------------------------
        auto t1 = clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();
        if (elapsed > cfg_.stopping.time_budget_sec) {
            R.stop_reason = 0;
            break;
        }

        // ---- (3) Cross-thread early-stop --------------------------------
        if (peer_stop() || stop_flag.load(std::memory_order_relaxed)) {
            R.stop_reason = 2;
            break;
        }

        // ---- (2) Stagnation + frozen T ----------------------------------
        if (stagnation_iter_threshold > 0 &&
            iters_since_improvement >= stagnation_iter_threshold &&
            T < T_frozen) {
            R.stop_reason = 1;
            break;
        }

        // ---- Propose / pack / evaluate / accept -------------------------
        Move m = engine_.propose(inst_, current);
        PackResult pp = packer_.pack(inst_, current);
        Costs cc = evaluator_.evaluate(inst_, current, pp);
        Real nc = evaluator_.sa_cost(cc, cfg_.weights, inst_);
        Real d = nc - cost;

        // always_accept (FixB / FixG) is honoured ONLY when the move
        // (a) doesn't introduce a new hard violation, AND
        // (b) doesn't blow up cost catastrophically in the END-GAME phase.
        //
        // The previous `d > 5*T` check was wrong: it kicked in at every T
        // (which during late stage 3 with T~0.01 means d > 0.05 — basically
        // ANY FixB/FixG repair was rejected).  That's why case 55 ends with
        // V_rel = 0.48 — the very moves designed to clear soft violations
        // were being silently blocked.
        //
        // The new criterion only blocks late-phase pollution: T already
        // frozen AND best is already feasible AND the proposed move would
        // push cost more than 10x best.  In every other regime, FixB/FixG
        // is unconditionally accepted.
        bool intro_new_hard =
            (cc.overlap_violation && !c.overlap_violation) ||
            (cc.area_violation    && !c.area_violation);
        bool endgame_pollution =
            (T < T_frozen) && R.best_costs.feasible &&
            (d > 10.0 * std::max(R.best_sa_cost, 1.0));
        bool accept;
        if (m.always_accept && !intro_new_hard && !endgame_pollution) {
            accept = true;
        } else if (d <= 0) {
            accept = true;
        } else {
            Real prob = std::exp(-d / T);
            accept = U(rng) < prob;
        }

        if (accept) {
            cost = nc;
            c = cc;
            // Update best, preferring feasibility.
            bool better = false;
            if (R.best_costs.feasible && !cc.feasible)      better = false;
            else if (!R.best_costs.feasible && cc.feasible) better = true;
            else                                            better = (cost < R.best_sa_cost);
            if (better) {
                R.best_sa_cost = cost;
                R.best_costs = c;
                R.best_tree.copy_from(current);
                iters_since_improvement = 0;
                maybe_signal_peers(cc);
            } else {
                ++iters_since_improvement;
            }
        } else {
            engine_.revert(inst_, current, m);
            ++iters_since_improvement;
        }

        // ---- Re-anchor --------------------------------------------------
        if (reanchor_threshold > 0 &&
            iters_since_improvement >= reanchor_threshold) {
            current.copy_from(R.best_tree);
            PackResult pr2 = packer_.pack(inst_, current);
            c = evaluator_.evaluate(inst_, current, pr2);
            cost = evaluator_.sa_cost(c, cfg_.weights, inst_);
            iters_since_improvement = 0;
        }

        // ---- CSV log every 4th iter (log accepted `cost`, NOT `nc`) ----
        if (log_counter == 3) {
            csv_file << iter << "," << T << "," << cost << "," << R.best_sa_cost << "\n";
            log_counter = 0;
        } else {
            ++log_counter;
        }

        // ---- Advance step counter + apply geometric cooling at boundary -
        ++iter; ++iters_this_step;
        if (iters_this_step >= iters_per_step) {
            ++k;
            iters_this_step = 0;
            // Apply this step's cooling multiplier.
            //   Stage 1 (k==1 still doesn't happen here, since we just ++k'd).
            //   Stage 2: 2 <= k <= stage2_end_k   -> alpha_stage2
            //   Stage 3:  k >  stage2_end_k        -> alpha_stage3
            double alpha;
            if (k <= 1) {
                alpha = cfg_.cooling.alpha_stage1;
            } else if (k <= cfg_.cooling.stage2_end_k) {
                alpha = cfg_.cooling.alpha_stage2;
            } else {
                alpha = cfg_.cooling.alpha_stage3;
            }
            T = std::max(T_floor, T * alpha);

            // One-shot reheating at the stage 2 -> stage 3 boundary.
            // Reproduces FastSA's "stage-3 starts hotter than end of stage 2"
            // behaviour so SA can escape the basin found by stage-2 aggressive
            // cooling and re-explore at higher T before settling.
            if (k == cfg_.cooling.stage2_end_k + 1 &&
                cfg_.cooling.stage3_reheat > 1.0) {
                T = std::min(T1, T * cfg_.cooling.stage3_reheat);
                if (cfg_.verbose) {
                    std::cerr << "[SA] stage-3 reheat: T -> " << T << "\n";
                }
            }
        }

        // Adaptive reheating inside stage 3: if best hasn't improved for a
        // long stretch and we're in stage 3, kick T up so SA can climb out.
        if (k > cfg_.cooling.stage2_end_k &&
            cfg_.cooling.reheat_stagnation_iters > 0 &&
            iters_since_improvement >=
                cfg_.cooling.reheat_stagnation_iters * iters_per_step) {
            Real T_target = std::min(T1,
                T1 * cfg_.cooling.reheat_to_fraction_of_T1);
            if (T_target > T) {
                if (cfg_.verbose) {
                    std::cerr << "[SA] adaptive reheat: T " << T
                              << " -> " << T_target
                              << " (stagn=" << iters_since_improvement << ")\n";
                }
                T = T_target;
            }
            // Reset stagnation counter so we don't reheat every iter.
            iters_since_improvement = 0;
        }

        if (cfg_.verbose && (iter % cfg_.log_every == 0)) {
            std::cerr << "[SA] iter=" << iter
                      << " k=" << k
                      << " T=" << T << " cost=" << cost
                      << " best=" << R.best_sa_cost
                      << " feas=" << (c.feasible ? 1 : 0)
                      << " hpwl_gap=" << c.hpwl_gap
                      << " area_gap=" << c.area_gap
                      << " V_rel=" << c.v_relative
                      << " stagn=" << iters_since_improvement
                      << "\n";
        }
    }

    auto tend = clock::now();
    R.elapsed_sec = std::chrono::duration<double>(tend - t0).count();
    R.iters = iter;

    if (cfg_.verbose) {
        const char* reason = (R.stop_reason == 0) ? "time" :
                             (R.stop_reason == 1) ? "stagnation" : "peer";
        std::cerr << "[SA] stopped (" << reason
                  << ") iters=" << R.iters
                  << " elapsed=" << R.elapsed_sec << "s"
                  << " best=" << R.best_sa_cost << "\n";
    }
    return R;
}

} // namespace fp
