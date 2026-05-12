// main.cpp -- Command-line entry point.
//
// Usage:
//   floorplanner <input.txt> <output.txt> [--time SEC] [--threads N] [--seed S] [--verbose]
//
#include "types.hpp"
#include "btree.hpp"
#include "packer.hpp"
#include "cost.hpp"
#include "parser.hpp"
#include "parallel.hpp"
#include "sa.hpp"

#include <iostream>
#include <chrono>
#include <thread>
#include <string>

using namespace fp;

static void print_usage() {
    std::cerr << "Usage: floorplanner <input.txt> <output.txt>\n"
              << "                    [--time SEC] [--threads N] [--seed S]\n"
              << "                    [--iters-per-block K] [--verbose]\n";
}

int main(int argc, char** argv) {
    if (argc < 3) { print_usage(); return 1; }
    std::string in_path = argv[1];
    std::string out_path = argv[2];

    double time_budget = 30.0;
    int    n_threads   = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    uint64_t seed      = 42;
    int    ipb         = 200;
    bool   verbose     = false;

    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--time" && i + 1 < argc) time_budget = std::stod(argv[++i]);
        else if (a == "--threads" && i + 1 < argc) n_threads = std::stoi(argv[++i]);
        else if (a == "--seed" && i + 1 < argc) seed = std::stoull(argv[++i]);
        else if (a == "--iters-per-block" && i + 1 < argc) ipb = std::stoi(argv[++i]);
        else if (a == "--verbose") verbose = true;
        else { std::cerr << "unknown arg: " << a << "\n"; print_usage(); return 1; }
    }

    FloorplanInstance inst;
    std::string err;
    if (!load_instance(in_path, inst, &err)) {
        std::cerr << "[load] " << err << "\n";
        return 2;
    }
    std::cerr << "[main] loaded: blocks=" << inst.n_blocks
              << " terminals=" << inst.n_terminals
              << " b2b=" << inst.b2b_nets.size()
              << " p2b=" << inst.p2b_nets.size()
              << " groups=" << inst.grouping_groups.size()
              << " mib_groups=" << inst.mib_groups.size()
              << " baseline_hpwl=" << inst.baseline_hpwl
              << " baseline_area=" << inst.baseline_area
              << "\n";

    ParallelConfig pcfg;
    pcfg.n_threads = n_threads;
    pcfg.time_budget_sec = time_budget;
    pcfg.sa_cfg.n_iters_per_block = ipb;
    pcfg.sa_cfg.verbose = verbose;

    auto t0 = std::chrono::steady_clock::now();
    auto R = run_parallel(inst, pcfg, seed);
    auto t1 = std::chrono::steady_clock::now();

    Evaluator ev;
    Real cc = ev.contest_cost(R.best.best_costs, 1.0);
    double secs = std::chrono::duration<double>(t1 - t0).count();
    std::cerr << "[main] best thread=" << R.best_thread
              << " feasible=" << (R.best.best_costs.feasible ? 1 : 0)
              << " contest_cost=" << cc
              << " hpwl_gap=" << R.best.best_costs.hpwl_gap
              << " area_gap=" << R.best.best_costs.area_gap
              << " V_rel=" << R.best.best_costs.v_relative
              << " (n_feasible_threads=" << R.n_feasible << "/" << pcfg.n_threads << ")"
              << "  elapsed=" << secs << "s\n";

    if (!save_solution(out_path, inst, R.best.best_tree)) {
        std::cerr << "[main] cannot write " << out_path << "\n";
        return 3;
    }
    std::cerr << "[main] wrote " << out_path << "\n";
    return R.best.best_costs.feasible ? 0 : 4;
}
