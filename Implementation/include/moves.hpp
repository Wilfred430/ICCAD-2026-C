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

enum class MoveKind { Rotate, Move, Swap, AspectRatio, MibSync, FixBoundary };

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
    explicit MoveEngine(uint64_t seed) : rng_(seed) {}

    // Sample and apply a random move.  Returns the move so SA can revert.
    Move propose(const FloorplanInstance& inst, BTree& tree,
                 const Costs* prev_costs = nullptr, Real temp = 0.0, Real t1 = 1.0);

    // Revert a move (undo).
    void revert(const FloorplanInstance& inst, BTree& tree, const Move& m);

private:
    std::mt19937_64 rng_;

    bool apply_rotate(const FloorplanInstance& inst, BTree& t, Move& m);
    bool apply_move  (const FloorplanInstance& inst, BTree& t, Move& m, const Costs* prev);
    bool apply_swap  (const FloorplanInstance& inst, BTree& t, Move& m, const Costs* prev);
    bool apply_ar    (const FloorplanInstance& inst, BTree& t, Move& m);
    bool apply_mib   (const FloorplanInstance& inst, BTree& t, Move& m);
    bool apply_fixb  (const FloorplanInstance& inst, BTree& t, Move& m,
                      const Costs* prev_costs);
};

} // namespace fp
