// packer.hpp -- Contour-based packing.  Walks the B*-tree in DFS order,
// assigning (x,y) to every block while maintaining a horizontal contour
// (skyline) of currently-placed blocks.  Anchored (preplaced) blocks ignore
// the tree-derived position and snap to their input (x,y); their children
// then build a normal B*-tree subtree relative to that anchor.  This is the
// scheme described in PARSAC §3.2.2.
//
// The packer also returns the achieved bounding box and reports whether the
// packing is overlap-free with respect to anchored blocks (the only way an
// overlap can sneak in given a valid tree).
//
#pragma once
#include "types.hpp"
#include "btree.hpp"

namespace fp {

struct PackResult {
    Real bbox_w = 0.0;
    Real bbox_h = 0.0;
    Real bbox_area = 0.0;
    bool overlap_free = true;     // false iff an anchored block overlaps a tree-placed block
};

class Packer {
public:
    // Pack the tree.  Reads w,h from tree; writes x,y into tree.
    // The instance is needed to know which blocks are preplaced (their input
    // x,y are the anchor) and which terminals exist (currently unused by the
    // packer but kept here so we can extend with terminal-aware compaction).
    PackResult pack(const FloorplanInstance& inst, BTree& tree) const;
};

} // namespace fp
