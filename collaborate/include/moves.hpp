// moves.hpp -- The SA move set.
//
// Following the PARSAC paper:
//   M1  rotate v       : swap w_v <-> h_v   (illegal for fixed/preplaced)
//   M2  move    v      : detach v and re-attach as a child of a random u
//   M3  swap   a,b     : swap two nodes' tree positions
//   M4  ar     v       : pick a new (w,h) for soft block v with same area target
//                        (within the 1 % tolerance and aspect-ratio bounds)
//   M5  mib    g       : pick a new shared (w,h) for every block in MIB group g
//   M6  fix-boundary v : move the violating block toward its required edge
//                        (PARSAC's constraints-fixing move; always accepted)
//
// All moves return enough info to support reverting if SA rejects them.
//
#pragma once
#include "types.hpp"
#include "btree.hpp"
#include "packer.hpp"
#include "cost.hpp"

#include <random>
#include <variant>

namespace fp {

enum class MoveKind { Rotate, Move, Swap, AspectRatio, MibSync, FixBoundary, FixGrouping };

// Tunable move-mix probabilities and AR-move parameters.
// Centralised here so they can be tweaked from sa.hpp's SAConfig without
// rebuilding (in principle just re-linking; in practice still a rebuild
// since we're statically linking, but it keeps every knob in one place).
//
// Constraints:
//   * Six explicit probabilities below must satisfy p_fixb + p_fixg + p_ar
//     + p_mib + p_rot + p_swp <= 1.0; remainder goes to MoveKind::Move (the
//     subtree-graft, biggest-Δ move).
//   * Set p_ar / p_rot higher for low-Δ exploration; set p_fixb / p_fixg
//     higher when V_boundary / V_grouping stay high after SA.
struct MoveProb {
    // Move-mix rebalanced after the deterministic boundary/grouping repair
    // passes (packer.cpp) took over most constraint satisfaction.  The
    // stochastic FixBoundary/FixGrouping moves now do less unique work, so we
    // shift that probability budget to quality-improving moves: AspectRatio
    // (area/HPWL shaping) and the big-Δ subtree Move (the remainder).
    double p_fixb = 0.02;   // FixBoundary  (was 0.05; repair pass handles most)
    double p_fixg = 0.02;   // FixGrouping  (was 0.05; repair pass handles most)
    double p_ar   = 0.22;   // AspectRatio  (was 0.18; more (w,h) shaping)
    double p_mib  = 0.05;   // MibSync      (sync (w,h) across whole MIB group)
    double p_rot  = 0.15;   // Rotate       (swap w↔h)
    double p_swp  = 0.15;   // Swap         (exchange two tree positions)
    // remainder ≈ 0.39 goes to MoveKind::Move (subtree graft, biggest Δ)

    // AspectRatio move samples area in [a·(1−tol_ar), a·(1+tol_ar)].
    // Hard area tolerance is 1%; keep tol_ar < 0.01 for margin.
    double tol_ar = 0.005;

    // SA-side aspect-ratio clamp.  Block's own ar_min/ar_max comes from the
    // contest input (typically 0.25–4.0, i.e. up to 4:1 extreme).  Letting SA
    // explore the full range produces very elongated blocks that take a lot
    // of width and disrupt cluster abutment — exactly the case-56 visual
    // symptom.  We tighten the SA-side range to a subset:
    //   eff_ar_min = max(b.ar_min, 1/sa_ar_clamp)
    //   eff_ar_max = min(b.ar_max,   sa_ar_clamp)
    //   2.0  = blocks stay within [1:2 .. 2:1]  (recommended)
    //   3.0  = mild clamp [1:3 .. 3:1]
    //   1.0  = forced square (probably too aggressive, area_target won't fit)
    //   0    = disable clamp, use block's full ar_min/ar_max range (legacy)
    double sa_ar_clamp = 2.0;
};

struct Move {
    MoveKind kind;

    // Common payload
    int v = -1;              // primary block id

    // For Move
    int u = -1;
    bool as_left = true;
    // saved for revert
    int saved_parent = -1;
    int saved_lc = -1;
    int saved_rc = -1;       // for swap we save both nodes' BNode

    // For Swap
    int a = -1, b = -1;

    // For AspectRatio / MibSync
    Real saved_w = 0.0, saved_h = 0.0;
    // For MibSync: saved widths/heights for every block in the group
    std::vector<int>  mib_blocks;
    std::vector<Real> saved_w_vec, saved_h_vec;

    // For FixBoundary
    bool always_accept = false;
};

class MoveEngine {
public:
    explicit MoveEngine(uint64_t seed, MoveProb prob = {})
        : rng_(seed), prob_(prob) {}

    // Sample and apply a random move.  Returns the move so SA can revert.
    Move propose(const FloorplanInstance& inst, BTree& tree,
                 const Costs* prev_costs = nullptr);

    // Revert a move (undo).
    void revert(const FloorplanInstance& inst, BTree& tree, const Move& m);

    // Allow SA to swap probabilities mid-run (useful for adaptive move mix).
    void set_prob(const MoveProb& p) { prob_ = p; }
    const MoveProb& prob() const { return prob_; }

private:
    std::mt19937_64 rng_;
    MoveProb prob_;

    bool apply_rotate(const FloorplanInstance& inst, BTree& t, Move& m);
    bool apply_move  (const FloorplanInstance& inst, BTree& t, Move& m);
    bool apply_swap  (const FloorplanInstance& inst, BTree& t, Move& m);
    bool apply_ar    (const FloorplanInstance& inst, BTree& t, Move& m);
    bool apply_mib   (const FloorplanInstance& inst, BTree& t, Move& m);
    bool apply_fixb  (const FloorplanInstance& inst, BTree& t, Move& m,
                      const Costs* prev_costs);
    bool apply_fixg  (const FloorplanInstance& inst, BTree& t, Move& m);
};

} // namespace fp
