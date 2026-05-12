// btree.hpp -- B*-tree representation for compact rectangular floorplans.
//
// Reference: Chen & Chang, "Modern floorplanning based on B*-tree and fast
// simulated annealing" (TCAD 2006) and PARSAC (Mostafa et al. 2024).
//
// Conventions:
//   - left child  : placed immediately to the right of the parent
//                   (x_lc = x_p + w_p), at the highest y where it fits.
//   - right child : placed above the parent, sharing the parent's x
//                   (x_rc = x_p), at the highest y where it fits.
//
// Anchored (preplaced) blocks ignore the position dictated by the tree and
// are anchored at their input (x,y); their children are placed relative to
// the anchored position.  See PARSAC §3.2.2.
//
// We store the tree as an indexed array (no raw pointers) so the whole
// search state can be copied / serialised in one memcpy.
//
#pragma once

#include "types.hpp"
#include <vector>
#include <cstdint>

namespace fp {

struct BNode {
    int parent = NO_NODE;
    int lc     = NO_NODE;     // left  child
    int rc     = NO_NODE;     // right child
};

// B*-tree paired with the live block geometry.  Multiple BTree instances
// can run in parallel (one per SA thread).
struct BTree {
    // Tree topology.  nodes[i] holds the parent / children of block i.
    std::vector<BNode> nodes;
    int root = NO_NODE;

    // Per-block live geometry (mirrors blocks' x,y,w,h but is the working
    // copy used by SA -- the instance-level Block holds *targets* and *constraints*).
    std::vector<Real> x, y, w, h;

    void init(int n_blocks);

    // Build a default tree: a left-spine of all blocks (root = 0).
    // This is a legal initial state but usually a bad one; it gives the SA
    // something to chew on.
    void build_default();

    // Random initial tree (random binary tree on n nodes), useful as a seed
    // for parallel multi-restart.
    void build_random(uint64_t seed);

    // Detach node v from its parent (the parent's child slot pointing at v
    // is set to NO_NODE).  Does not touch v's children.
    void detach(int v);

    // Insert v as the left or right child of u.  Whatever was occupying the
    // u->lc / u->rc slot before is grafted onto v (left side preserved).
    void insert_left(int u, int v);
    void insert_right(int u, int v);

    // Random move primitives (used by the SA move set in moves.cpp).
    // They modify the tree in place; the caller is responsible for re-packing
    // and computing the cost delta.
    void op_swap(int a, int b);            // M3
    void op_rotate(int v);                 // M1: swap w,h of v (geometry only)
    bool op_move(int v, int u, bool as_left); // M2: move v to be u's child

    // Validate the tree topology (returns true if all reachable from root
    // and no cycles).  Useful when debugging move bugs.
    bool validate() const;

    // Deep copy.
    void copy_from(const BTree& other);
};

} // namespace fp
