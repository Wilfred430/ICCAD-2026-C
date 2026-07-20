"""Pure-Python B*-tree packer -- turns a (parent, direction) topology into
actual (x, y, w, h) geometry, for fast prototyping/scoring of the tree
generator's samples without needing the compiled C++ binary.

This ports the CORE placement rule from `collaborate/src/packer.cpp`
(confirmed by cross-checking against `fp_sol` on real training cases --
see the project memory / WINNING_STRATEGY.md for the validation numbers):

    left  child (direction 0): x = parent.x + parent.w   (touch parent's right)
    right child (direction 1): x = parent.x              (touch parent's top)
    both:                       y = current contour height in [x, x+w)

Also ports the full repair pipeline from packer.cpp: `compact_left_down`
(slide every non-anchored block as far left/down as it will go),
`bbox_balance_pass` (relocate the worst "spike" block to pull a tall/thin
bbox toward square), `holes_fill_pass` (diagonal relocation into L-shaped
whitespace compact_left_down can't reach), `grouping_repair_pass` (reattach
isolated group members) and `boundary_repair_pass` (snap boundary blocks
onto their required edge). Validated 2026-07-08/09 on the full 100-case
validation set: adding the full 4-pass pipeline dropped Total Score
(e^(n/12)-weighted) from 13.77 (compact_left_down only) to 5.13 -- a 62.7%
reduction, from only porting repair passes packer.cpp already had (no
model retraining) -- confirming the contour representation was never
structurally broken, it was just missing this repair pipeline (see
ICCAD_code/6_ML_Generative_BTree.md §6.6 for the full before/after numbers,
including an earlier wrong conclusion that got corrected by this test).
This module is for quickly ranking/visualizing samples from the ML
topology model; the real submission path still goes through the actual
C++ `floorplanner` binary as the source of truth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class PackResult:
    x: Dict[int, float]
    y: Dict[int, float]
    w: Dict[int, float]
    h: Dict[int, float]
    bbox_w: float
    bbox_h: float
    overlap_free: bool


class _Contour:
    """Horizontal skyline as a sorted list of (x_start, x_end, height)."""

    def __init__(self):
        self.segs: List[Tuple[float, float, float]] = [(0.0, float("inf"), 0.0)]

    def height_in(self, x0: float, x1: float) -> float:
        h = 0.0
        for a, b, ch in self.segs:
            if b <= x0 or a >= x1:
                continue
            h = max(h, ch)
        return h

    def set_height(self, x0: float, x1: float, new_h: float) -> None:
        nc = []
        for a, b, ch in self.segs:
            if b <= x0 or a >= x1:
                nc.append((a, b, ch))
                continue
            if a < x0:
                nc.append((a, x0, ch))
            if b > x1:
                nc.append((x1, b, ch))
        nc.append((x0, x1, new_h))
        nc.sort()
        # collapse adjacent equal-height runs
        merged = []
        for seg in nc:
            if merged and abs(merged[-1][2] - seg[2]) < 1e-9 and abs(merged[-1][1] - seg[0]) < 1e-9:
                merged[-1] = (merged[-1][0], seg[1], merged[-1][2])
            else:
                merged.append(seg)
        self.segs = merged


def build_lc_rc(root: int, parent_id, direction, n: int, gen_order=None):
    """parent_id/direction: [n] (original-block-id-indexed, parent_id[root]=-1)
    -> (lc, rc) dicts, block_id -> child block_id or None.

    A B*-tree node has AT MOST one left child and one right child.  An
    undertrained (or still-sampling-badly) generator can predict two
    different blocks for the same (parent, direction) slot -- nothing in
    the pointer-network's raw output structurally forbids it, only the
    training signal discourages it.  Left as-is, the second block would
    silently overwrite the first in the dict, leaving it un-attached (never
    visited by the packer's DFS) and crashing downstream code that assumes
    every block got an (x, y).  We repair this deterministically instead of
    crashing: any block that loses the slot race falls back to the nearest
    placed node (preferring the most recently placed) with a free lc/rc
    slot.  This guarantees a fully-connected, valid tree every time, so the
    pipeline degrades gracefully (worse layout) rather than failing outright
    when the model is weak or still training.
    """
    lc: Dict[int, Optional[int]] = {i: None for i in range(n)}
    rc: Dict[int, Optional[int]] = {i: None for i in range(n)}
    placed = [root]
    placed_set = {root}
    order = list(gen_order) if gen_order is not None else [i for i in range(n) if i != root]

    for c in order:
        if c == root:
            continue
        p, d = int(parent_id[c]), int(direction[c])
        attached = False
        if p in placed_set:
            if d == 0 and lc[p] is None:
                lc[p] = c; attached = True
            elif d == 1 and rc[p] is None:
                rc[p] = c; attached = True
            elif lc[p] is None:
                lc[p] = c; attached = True
            elif rc[p] is None:
                rc[p] = c; attached = True
        if not attached:
            # fallback: most-recently-placed node with a free slot, else scan all
            for cand in reversed(placed):
                if lc[cand] is None:
                    lc[cand] = c; attached = True; break
                if rc[cand] is None:
                    rc[cand] = c; attached = True; break
        if not attached:
            raise RuntimeError(f"pack_tree: no free slot for block {c} (tree structurally full)")
        placed.append(c)
        placed_set.add(c)

    return lc, rc


def _shelf_pack(members: List[int], dims: Dict[int, Tuple[float, float]], first: Optional[int] = None):
    """Deterministic next-fit-decreasing-height shelf packing. In the default
    (`first=None`) call, items are placed in globally height-descending
    order, which guarantees every item touches its shelf-mates AND the
    shelf below: the first item of each new shelf is always that shelf's
    tallest, and (being first) always starts at x=0 -- exactly where the
    first item of the NEXT shelf also starts, so consecutive shelves always
    share a positive-length touching edge at x=0. The whole cluster comes
    out as ONE connected component by construction in this case.

    If `first` is given, that member is placed before all others (still
    x=0/y=0) regardless of its own height -- needed when a specific member
    (e.g. a preplaced one) MUST land at local (0, 0) so its DFS-placed
    position can double as the whole cluster's bbox origin. This can BREAK
    the connectivity guarantee above (a short forced-first item may not
    reach the height a later, taller shelf-mate defines) -- callers using
    `first` must verify connectivity themselves (see `_offsets_connected`)
    and fall back to not collapsing the cluster if it fails.

    Returns (offsets, bbox_w, bbox_h, first_id) where offsets[i] is relative
    to the bbox's own (0, 0) corner and `first_id` is whichever member ended
    up there (== `first` when given).
    """
    total_area = sum(dims[i][0] * dims[i][1] for i in members)
    target_w = math.sqrt(total_area) if total_area > 0 else 1.0
    rest = sorted((i for i in members if i != first), key=lambda i: (-dims[i][1], i))
    items = ([first] if first is not None else []) + rest
    offsets: Dict[int, Tuple[float, float]] = {}
    cur_x = cur_y = shelf_h = max_w = 0.0
    for i in items:
        w, h = dims[i]
        if cur_x > 0 and cur_x + w > target_w + 1e-6:
            cur_y += shelf_h
            cur_x, shelf_h = 0.0, 0.0
        offsets[i] = (cur_x, cur_y)
        cur_x += w
        shelf_h = max(shelf_h, h)
        max_w = max(max_w, cur_x)
    total_h = cur_y + shelf_h
    return offsets, max_w, total_h, items[0]


def _offsets_connected(members, offsets, dims) -> bool:
    """Verify a `_shelf_pack` result (using LOCAL `offsets` as positions) is
    a single connected component, via the same touch+union-find approach as
    `_grouping_repair_pass.components()`. Needed only when `_shelf_pack` was
    called with a forced `first` item, since that can break the
    height-descending connectivity guarantee."""
    tx = {i: offsets[i][0] for i in members}
    ty = {i: offsets[i][1] for i in members}
    parent = {i: i for i in members}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for idx_a in range(len(members)):
        for idx_b in range(idx_a + 1, len(members)):
            a, b = members[idx_a], members[idx_b]
            if _touches(a, b, tx, ty, dims):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb
    return len({find(i) for i in members}) == 1


def _collapse_clusters(root, lc, rc, dims, is_preplaced, preplaced_xy, cluster_id, boundary_code):
    """By-construction grouping fix (2026-07-09): instead of scattering group
    members across the tree and gathering them post-hoc afterwards (limited
    by whatever whitespace happens to survive compaction -- see
    `_grouping_repair_pass`, which tops out around V_group~4-5/case on dense
    packings because there's rarely a free adjacent cell to gather into),
    collapse each *eligible* cluster into a single 'super-block' tree node
    BEFORE the main DFS ever runs. The super-block's internal layout is a
    `_shelf_pack`, which is connected by construction, so V_group=0 for any
    cluster this function collapses -- no repair pass needed afterwards.

    A cluster is eligible iff every member has boundary_code 0, or exactly
    one member has a LEFT (1) / BOTTOM (8) / bottom-left-corner (9) code
    (see the in-loop comment for why only these three are safe by
    construction; RIGHT (2) / TOP (4) and multi-member boundary conflicts
    still fall back to the old post-hoc path) -- and at most one member is
    preplaced (more than one is a hard position conflict between two
    absolute anchors; a preplaced member with a DIFFERENT boundary-coded
    member in the same cluster is also a fallback case, see below).
    Ineligible clusters are left untouched -- their members stay
    independent tree nodes, exactly the pre-existing behaviour,
    so this is strictly additive: it can only remove grouping violations it
    is able to fully resolve, never introduce new ones.

    Returns (new_root, new_lc, new_rc, coll_dims, new_is_preplaced,
    new_preplaced_xy, member_offset, absorbed) where `coll_dims` substitutes
    the collapsed bbox size at each anchor id, `member_offset` maps every
    absorbed (non-anchor) member -> (anchor_id, dx, dy) relative to the
    anchor's own final position, and `absorbed` is the set of member ids
    that no longer exist as independent tree nodes.
    """
    boundary_code = boundary_code or {}
    groups: Dict[int, List[int]] = {}
    for i in dims:
        cid = cluster_id.get(i, 0)
        if cid and cid > 0:
            groups.setdefault(cid, []).append(i)

    parent_of: Dict[int, Tuple[int, bool]] = {}
    for p, c in lc.items():
        if c is not None:
            parent_of[c] = (p, True)
    for p, c in rc.items():
        if c is not None:
            parent_of[c] = (p, False)

    depth = {root: 0}
    stack = [root]
    while stack:
        v = stack.pop()
        for c in (lc.get(v), rc.get(v)):
            if c is not None and c not in depth:
                depth[c] = depth[v] + 1
                stack.append(c)

    new_dims = dict(dims)
    new_is_preplaced = dict(is_preplaced)
    new_preplaced_xy = dict(preplaced_xy)
    member_offset: Dict[int, Tuple[int, float, float]] = {}
    absorbed: set = set()

    # A preplaced anchor's footprint balloons from its own small size to the
    # whole cluster's bbox -- the ORIGINAL preplaced positions are only
    # guaranteed mutually non-overlapping at their small, individual sizes,
    # not after one of them grows. Track every fixed (preplaced) footprint
    # so a ballooned bbox that would collide with an unrelated preplaced
    # block (or another already-accepted ballooned cluster) gets rejected
    # -- that specific cluster falls back to the old post-hoc path instead
    # of producing an unresolvable hard-constraint overlap.
    cluster_member_set = {i for members in groups.values() for i in members}
    fixed_boxes = [
        (preplaced_xy[i][0], preplaced_xy[i][1], dims[i][0], dims[i][1])
        for i in dims
        if is_preplaced.get(i, False) and i not in cluster_member_set
    ]

    def overlaps_box(a, b) -> bool:
        ax0, ay0, aw0, ah0 = a
        bx0, by0, bw0, bh0 = b
        return (ax0 < bx0 + bw0 - 1e-7 and bx0 < ax0 + aw0 - 1e-7 and
                ay0 < by0 + bh0 - 1e-7 and by0 < ay0 + ah0 - 1e-7)

    for members in groups.values():
        if len(members) < 2:
            continue

        # By-construction boundary support is limited to a SINGLE member
        # requiring LEFT (1) / BOTTOM (8) / bottom-left corner (9): these
        # are the only codes that constrain a block's OWN (x, y) position
        # and never its FAR edge (`_boundary_repair_pass`/`_boundary_wall_
        # slide`'s satisfied()/slide checks for these bits only ever look
        # at x[i]/y[i], never dims[i]) -- so forcing that member to anchor
        # the shelf-pack (its local offset guaranteed (0, 0), same trick as
        # the preplaced case below) makes its own true corner coincide with
        # the bbox's own corner, and the existing post-hoc passes work
        # correctly on the ballooned anchor with NO further changes needed
        # (boundary_code stays keyed by the anchor's own original id).
        # RIGHT (2) / TOP (4) need the block's FAR edge to align with the
        # bbox's far edge, which a ballooned bbox does NOT preserve for a
        # non-anchor member -- those, and any cluster with more than one
        # boundary-coded member, still fall back to the old post-hoc path.
        bc_members = [i for i in members if boundary_code.get(i, 0)]
        if any(boundary_code.get(i, 0) not in (0, 1, 8, 9) for i in members):
            continue
        if len(bc_members) > 1:
            continue
        pre_members = [i for i in members if is_preplaced.get(i, False)]
        if len(pre_members) > 1:
            continue
        if bc_members and pre_members and bc_members[0] != pre_members[0]:
            continue  # two different forced-first candidates -- ambiguous, fall back

        # The anchor's DFS-placed (x, y) doubles as the WHOLE cluster's bbox
        # origin (see `member_offset` below), which only works if the anchor
        # sits at local (0, 0) in the shelf-pack -- a rectangle can only be
        # reserved starting at a node's own placed corner and growing in
        # +x/+y, never behind it. A preplaced position (or an eligible
        # boundary requirement) is a hard constraint, so that member MUST be
        # the anchor -- force it first and verify the forced ordering didn't
        # break connectivity; anything else, take whichever member the
        # natural (always-connected) ordering put first.
        forced = pre_members[0] if pre_members else (bc_members[0] if bc_members else None)
        if forced is not None:
            anchor = forced
            offsets, bw, bh, _ = _shelf_pack(members, dims, first=anchor)
            if not _offsets_connected(members, offsets, dims):
                continue
            if anchor in pre_members:
                px, py = preplaced_xy[anchor]
                candidate_box = (px, py, bw, bh)  # ax, ay == (0, 0) here
                if any(overlaps_box(candidate_box, fb) for fb in fixed_boxes):
                    continue
                fixed_boxes.append(candidate_box)
        else:
            offsets, bw, bh, anchor = _shelf_pack(members, dims)

        ax, ay = offsets[anchor]  # always (0.0, 0.0) by construction now
        for i in members:
            if i != anchor:
                dx, dy = offsets[i][0] - ax, offsets[i][1] - ay
                member_offset[i] = (anchor, dx, dy)
                absorbed.add(i)
                new_is_preplaced.pop(i, None)
                new_preplaced_xy.pop(i, None)

        new_dims[anchor] = (bw, bh)
        if pre_members:
            px, py = preplaced_xy[anchor]
            new_preplaced_xy[anchor] = (px - ax, py - ay)
            new_is_preplaced[anchor] = True

    if not absorbed:
        return root, lc, rc, new_dims, new_is_preplaced, new_preplaced_xy, {}, set()

    # Sever every tree edge touching an absorbed member (as parent or as
    # child), collecting the surviving children of absorbed parents (plain
    # blocks, or another cluster's anchor) that now need a new home.
    new_lc = dict(lc)
    new_rc = dict(rc)
    dangling: List[int] = []
    for m in absorbed:
        for child in (new_lc.get(m), new_rc.get(m)):
            if child is not None and child not in absorbed:
                dangling.append(child)
        new_lc[m] = None
        new_rc[m] = None
        p = parent_of.get(m)
        if p is not None:
            pid, is_left = p
            if is_left and new_lc.get(pid) == m:
                new_lc[pid] = None
            elif not is_left and new_rc.get(pid) == m:
                new_rc[pid] = None

    # Reattach every dangling node to the first tree node (BFS from root)
    # with a free lc/rc slot -- shallower dangling nodes get first pick.
    dangling.sort(key=lambda i: depth.get(i, 1 << 30))
    for d in dangling:
        bfs = [root]
        seen = {root}
        placed = False
        while bfs and not placed:
            v = bfs.pop(0)
            if new_lc.get(v) is None:
                new_lc[v] = d
                placed = True
                break
            if new_rc.get(v) is None:
                new_rc[v] = d
                placed = True
                break
            for c in (new_lc.get(v), new_rc.get(v)):
                if c is not None and c not in seen:
                    seen.add(c)
                    bfs.append(c)
        if not placed:
            raise RuntimeError("pack_tree: no free slot to reattach collapsed-cluster remainder")

    return root, new_lc, new_rc, new_dims, new_is_preplaced, new_preplaced_xy, member_offset, absorbed


def pack_btree(
    root: int,
    lc: Dict[int, Optional[int]],
    rc: Dict[int, Optional[int]],
    dims: Dict[int, Tuple[float, float]],
    is_preplaced: Optional[Dict[int, bool]] = None,
    preplaced_xy: Optional[Dict[int, Tuple[float, float]]] = None,
    compact: bool = True,
    baseline_area: Optional[float] = None,
    cluster_id: Optional[Dict[int, int]] = None,
    boundary_code: Optional[Dict[int, int]] = None,
    boundary_push_past: bool = True,
    group_by_construction: bool = True,
) -> PackResult:
    n = len(dims)
    is_preplaced = dict(is_preplaced or {})
    preplaced_xy = dict(preplaced_xy or {})

    member_offset: Dict[int, Tuple[int, float, float]] = {}
    absorbed: set = set()
    coll_dims = dict(dims)
    if group_by_construction and cluster_id is not None:
        (root, lc, rc, coll_dims, is_preplaced, preplaced_xy,
         member_offset, absorbed) = _collapse_clusters(
            root, lc, rc, dims, is_preplaced, preplaced_xy, cluster_id, boundary_code)

    surviving = [i for i in range(n) if i not in absorbed]

    contour = _Contour()
    x: Dict[int, float] = {}
    y: Dict[int, float] = {}

    # Pre-seed preplaced (anchored) footprints, ascending top-edge order,
    # exactly as packer.cpp does -- so tree-placed blocks that share an
    # x-range with an anchor get lifted above it instead of overlapping.
    anchors = [i for i in surviving if is_preplaced.get(i, False)]
    anchors.sort(key=lambda i: preplaced_xy[i][1] + coll_dims[i][1])
    for i in anchors:
        ax, ay = preplaced_xy[i]
        aw, ah = coll_dims[i]
        x[i], y[i] = ax, ay
        contour.set_height(ax, ax + aw, ay + ah)

    # Iterative DFS: parent before children, left child fully before right
    # child (matches packer.cpp's stack-based traversal).
    stack = [(root, 0)]
    while stack:
        v, state = stack.pop()
        if state == 0:
            if v not in x:  # not a pre-seeded anchor
                w_v, h_v = coll_dims[v]
                if v == root:
                    px = 0.0
                else:
                    parent = None
                    is_left = False
                    for p_id, child in lc.items():
                        if child == v:
                            parent, is_left = p_id, True
                            break
                    if parent is None:
                        for p_id, child in rc.items():
                            if child == v:
                                parent, is_left = p_id, False
                                break
                    px = (x[parent] + coll_dims[parent][0]) if is_left else x[parent]
                py = contour.height_in(px, px + w_v)
                x[v], y[v] = px, py
                contour.set_height(px, px + w_v, py + h_v)
            stack.append((v, 1))
            if lc.get(v) is not None:
                stack.append((lc[v], 0))
        elif state == 1:
            stack.append((v, 2))
            if rc.get(v) is not None:
                stack.append((rc[v], 0))
        # state == 2: done, nothing to do

    if compact:
        _compact_left_down(surviving, x, y, coll_dims, is_preplaced)
        _bbox_balance_pass(surviving, x, y, coll_dims, is_preplaced, baseline_area)
        _compact_left_down(surviving, x, y, coll_dims, is_preplaced)
        _holes_fill_pass(surviving, x, y, coll_dims, is_preplaced)
        _compact_left_down(surviving, x, y, coll_dims, is_preplaced)
        # Boundary pins blocks to walls (aggressive), THEN grouping gathers
        # the free (non-boundary) members into one connected component around
        # wherever the group's pinned members ended up. Boundary runs once
        # (it's internally iterated); grouping then runs to convergence.
        # Grouping skips boundary blocks so it can't re-open a boundary
        # violation -- the two cooperate (boundary pins, grouping gathers)
        # instead of thrashing. (Only clusters `_collapse_clusters` left
        # untouched -- boundary-coded ones -- still need this at all.)
        if boundary_code is not None:
            _boundary_repair_pass(surviving, x, y, coll_dims, is_preplaced, boundary_code,
                                  push_past=boundary_push_past)
        if cluster_id is not None:
            for _ in range(4):
                if not _grouping_repair_pass(surviving, x, y, coll_dims, is_preplaced,
                                             cluster_id, boundary_code):
                    break
        # Final compaction to RECLAIM the interior area that the aggressive
        # boundary push opened up (lifting an interior block to a TOP/RIGHT
        # edge leaves an empty column/row below/beside it). Pin the boundary
        # AND grouping-cluster blocks (treat like preplaced) so this compaction
        # only pulls the truly-free blocks toward the origin -- it must not
        # drag a boundary block off its wall or scatter a just-gathered group.
        if boundary_code is not None or cluster_id is not None:
            pinned = dict(is_preplaced)
            for i in surviving:
                if (boundary_code and boundary_code.get(i, 0)) or \
                   (cluster_id and cluster_id.get(i, 0)):
                    pinned[i] = True
            _compact_left_down(surviving, x, y, coll_dims, pinned)
            # Constraint-preserving reclaim (2026-07-09): the pin above froze
            # boundary+cluster blocks, but they CAN still move in ways that
            # keep their constraint AND reclaim area:
            #   * a whole cluster slides toward the origin as ONE rigid body
            #     (relative positions unchanged -> V_group unchanged);
            #   * a boundary block slides ALONG its wall (a LEFT block drops in
            #     y keeping x=0; a BOTTOM block slides left keeping y=0) toward
            #     free space, staying on its edge.
            if cluster_id is not None:
                _rigid_group_compact(surviving, x, y, coll_dims, is_preplaced, cluster_id)
            if boundary_code is not None:
                _boundary_wall_slide(surviving, x, y, coll_dims, is_preplaced, boundary_code)
            # A final free-block compaction to fill anything the above opened.
            _compact_left_down(surviving, x, y, coll_dims, pinned)

    # Expand collapsed-cluster members back to individual absolute positions
    # (their own true dims, not the anchor's substituted bbox size).
    for i, (anchor, dx, dy) in member_offset.items():
        x[i] = x[anchor] + dx
        y[i] = y[anchor] + dy

    bbox_w = max(x[i] + dims[i][0] for i in range(n))
    bbox_h = max(y[i] + dims[i][1] for i in range(n))
    overlap_free = _check_overlap_free(range(n), x, y, dims)

    return PackResult(x=x, y=y,
                       w={i: dims[i][0] for i in range(n)},
                       h={i: dims[i][1] for i in range(n)},
                       bbox_w=bbox_w, bbox_h=bbox_h, overlap_free=overlap_free)


def _rigid_group_compact(ids, x, y, dims, is_preplaced, cluster_id, passes: int = 6):
    """Slide each grouping-cluster toward the origin as ONE rigid body (all
    members shift by the same delta), so relative positions -- and therefore
    V_group -- are unchanged, but trapped whitespace beside/below the group is
    reclaimed. Alternates a leftward and a downward rigid slide until fixpoint.
    Never overlaps any non-group block (or a preplaced block)."""
    ids = list(ids)
    groups = {}
    for i in ids:
        cid = cluster_id.get(i, 0)
        if cid and cid > 0:
            groups.setdefault(cid, []).append(i)

    def max_shift(members, axis):
        """Largest non-negative delta the group can move in -axis without
        overlapping any block outside the group."""
        mset = set(members)
        best = min((x[i] if axis == 0 else y[i]) for i in members)  # can't go past 0
        for i in members:
            for j in ids:
                if j in mset:
                    continue
                # does j block i's path in the -axis direction?
                if axis == 0:  # moving left: need y-overlap, j to the left of i
                    ylo, yhi = max(y[i], y[j]), min(y[i] + dims[i][1], y[j] + dims[j][1])
                    if yhi - ylo <= 1e-9:
                        continue
                    if x[j] + dims[j][0] <= x[i] + 1e-9:
                        best = min(best, x[i] - (x[j] + dims[j][0]))
                else:  # moving down: need x-overlap, j below i
                    xlo, xhi = max(x[i], x[j]), min(x[i] + dims[i][0], x[j] + dims[j][0])
                    if xhi - xlo <= 1e-9:
                        continue
                    if y[j] + dims[j][1] <= y[i] + 1e-9:
                        best = min(best, y[i] - (y[j] + dims[j][1]))
        return max(0.0, best)

    for g in groups.values():
        if any(is_preplaced.get(i, False) for i in g):
            continue  # a preplaced member pins the group's position
        for _ in range(passes):
            moved = False
            dl = max_shift(g, 0)
            if dl > 1e-9:
                for i in g:
                    x[i] -= dl
                moved = True
            dd = max_shift(g, 1)
            if dd > 1e-9:
                for i in g:
                    y[i] -= dd
                moved = True
            if not moved:
                break


def _boundary_wall_slide(ids, x, y, dims, is_preplaced, boundary_code, passes: int = 4):
    """Slide each boundary block ALONG its wall toward free space, preserving
    edge contact: LEFT (x=0) / RIGHT blocks drop in y; BOTTOM (y=0) / TOP blocks
    slide left in x. Reclaims area along the walls that the pinned final compact
    couldn't. Never overlaps and never leaves the wall."""
    ids = list(ids)
    for _ in range(passes):
        moved = False
        for i in ids:
            bc = boundary_code.get(i, 0)
            if not bc or is_preplaced.get(i, False):
                continue
            w, h = dims[i]
            # Which axis may this block slide along without leaving its wall?
            # LEFT(1)/RIGHT(2) fix x -> slide in y (toward 0).
            # BOTTOM(8)/TOP(4) fix y -> slide in x (toward 0).
            slide_y = bool(bc & 1 or bc & 2) and not (bc & 4 or bc & 8)
            slide_x = bool(bc & 4 or bc & 8) and not (bc & 1 or bc & 2)
            if slide_y and y[i] > 1e-9:
                best = 0.0
                for j in ids:
                    if j == i:
                        continue
                    xlo, xhi = max(x[i], x[j]), min(x[i] + w, x[j] + dims[j][0])
                    if xhi - xlo <= 1e-9:
                        continue
                    top = y[j] + dims[j][1]
                    if top <= y[i] + 1e-9 and top > best:
                        best = top
                if best < y[i] - 1e-9:
                    y[i] = best
                    moved = True
            elif slide_x and x[i] > 1e-9:
                best = 0.0
                for j in ids:
                    if j == i:
                        continue
                    ylo, yhi = max(y[i], y[j]), min(y[i] + h, y[j] + dims[j][1])
                    if yhi - ylo <= 1e-9:
                        continue
                    right = x[j] + dims[j][0]
                    if right <= x[i] + 1e-9 and right > best:
                        best = right
                if best < x[i] - 1e-9:
                    x[i] = best
                    moved = True
        if not moved:
            break


def _compact_left_down(ids, x, y, dims, is_preplaced, passes: int = 12):
    """Port of packer.cpp's compact_left_down: slide every non-preplaced
    block as far left/down as it goes without overlap, alternating axes
    until a fixpoint (or `passes` iterations)."""
    ids = list(ids)

    def overlaps_y(i, j):
        ay, ah = y[i], dims[i][1]
        by, bh = y[j], dims[j][1]
        return not (ay + 1e-9 >= by + bh or by + 1e-9 >= ay + ah)

    def overlaps_x(i, j):
        ax, aw = x[i], dims[i][0]
        bx, bw = x[j], dims[j][0]
        return not (ax + 1e-9 >= bx + bw or bx + 1e-9 >= ax + aw)

    for _ in range(passes):
        changed = False
        for i in sorted(ids, key=lambda k: y[k]):
            if is_preplaced.get(i, False):
                continue
            best_y = 0.0
            for j in ids:
                if j == i or not overlaps_x(i, j):
                    continue
                cand = y[j] + dims[j][1]
                if cand <= y[i] + 1e-9 and cand > best_y:
                    best_y = cand
            if best_y < y[i] - 1e-9:
                y[i] = best_y
                changed = True
        for i in sorted(ids, key=lambda k: x[k]):
            if is_preplaced.get(i, False):
                continue
            best_x = 0.0
            for j in ids:
                if j == i or not overlaps_y(i, j):
                    continue
                cand = x[j] + dims[j][0]
                if cand <= x[i] + 1e-9 and cand > best_x:
                    best_x = cand
            if best_x < x[i] - 1e-9:
                x[i] = best_x
                changed = True
        if not changed:
            break


def _relocate_one_spike(ids, x, y, dims, is_preplaced, tall, long_dim, short_dim):
    """Port of packer.cpp's relocate_one_spike: find the movable block whose
    extreme edge defines the long dimension, and try to slide it to a
    'shelf' (another block's trailing edge) that shortens the long dim."""
    spike, spike_ext = -1, -1.0
    for i in ids:
        if is_preplaced.get(i, False):
            continue
        ext = (y[i] + dims[i][1]) if tall else (x[i] + dims[i][0])
        if ext > spike_ext:
            spike_ext, spike = ext, i

    if spike < 0 or spike_ext < long_dim - 1e-6:
        return False

    i = spike
    cw, ch = dims[i]
    cur_long = (y[i] + ch) if tall else (x[i] + cw)

    short_coords = {0.0}
    for j in ids:
        if j == i:
            continue
        short_coords.add((x[j] + dims[j][0]) if tall else (y[j] + dims[j][1]))

    self_short_extent = cw if tall else ch
    self_long_extent = ch if tall else cw

    best_long = cur_long
    best_short = x[i] if tall else y[i]

    for sc in sorted(short_coords):
        if sc < -1e-9 or sc + self_short_extent > short_dim + 1e-6:
            continue
        lc = 0.0
        for j in ids:
            if j == i:
                continue
            j_short_lo = x[j] if tall else y[j]
            j_short_hi = j_short_lo + (dims[j][0] if tall else dims[j][1])
            if j_short_hi <= sc + 1e-9 or j_short_lo >= sc + self_short_extent - 1e-9:
                continue
            j_long_hi = (y[j] if tall else x[j]) + (dims[j][1] if tall else dims[j][0])
            lc = max(lc, j_long_hi)
        candidate_long_end = lc + self_long_extent
        if candidate_long_end < best_long - 1e-6:
            best_long, best_short = candidate_long_end, sc

    if best_long >= cur_long - 1e-6:
        return False
    if tall:
        x[i], y[i] = best_short, best_long - ch
    else:
        y[i], x[i] = best_short, best_long - cw
    return True


def _bbox_balance_pass(ids, x, y, dims, is_preplaced, baseline_area: Optional[float] = None):
    """Port of packer.cpp's bbox_balance_pass: repeatedly relocate the worst
    'spike' block to pull a tall/thin (or wide/flat) bbox toward square."""
    ids = list(ids)
    n = len(ids)
    if n < 2:
        return

    def compute_bbox():
        bw = max(x[i] + dims[i][0] for i in ids)
        bh = max(y[i] + dims[i][1] for i in ids)
        return bw, bh

    init_w, init_h = compute_bbox()
    target_side = math.sqrt(baseline_area) if baseline_area and baseline_area > 0 \
        else math.sqrt(max(init_w * init_h, 1.0))

    max_passes = min(n, 60)
    no_improve = 0
    for _ in range(max_passes):
        bbox_w, bbox_h = compute_bbox()
        if bbox_w <= 0 or bbox_h <= 0:
            return
        ar = bbox_h / bbox_w
        if 1.0 / 1.10 < ar < 1.10:
            return
        tall = ar > 1.0
        long_dim = bbox_h if tall else bbox_w
        short_dim = bbox_w if tall else bbox_h
        short_dim_cap = max(short_dim, target_side)

        if not _relocate_one_spike(ids, x, y, dims, is_preplaced, tall, long_dim, short_dim_cap):
            no_improve += 1
            if no_improve >= 2:
                return
        else:
            no_improve = 0


def _relocate_to_min_corner(ids, x, y, dims, i):
    """Port of packer.cpp's relocate_to_min_corner: try every candidate
    (x, y) formed by other blocks' trailing edges, pick the one that tucks
    block i closest to the origin without overlapping anything."""
    cw, ch = dims[i]
    cur_max = max(x[i] + cw, y[i] + ch)
    cur_sum = (x[i] + cw) + (y[i] + ch)
    if cur_max < 1e-6:
        return False

    xs = {0.0}
    ys = {0.0}
    for j in ids:
        if j == i:
            continue
        xs.add(x[j] + dims[j][0])
        ys.add(y[j] + dims[j][1])
    xs = sorted(xs)
    ys = sorted(ys)

    def overlaps_anything(cx, cy):
        for j in ids:
            if j == i:
                continue
            if cx + cw <= x[j] + 1e-9 or x[j] + dims[j][0] <= cx + 1e-9:
                continue
            if cy + ch <= y[j] + 1e-9 or y[j] + dims[j][1] <= cy + 1e-9:
                continue
            return True
        return False

    best_x, best_y = x[i], y[i]
    best_max, best_sum = cur_max, cur_sum

    for cx in xs:
        if cx < -1e-9 or cx + cw > best_max + 1e-6:
            break
        for cy in ys:
            if cy < -1e-9 or cy + ch > best_max + 1e-6:
                break
            new_max = max(cx + cw, cy + ch)
            new_sum = (cx + cw) + (cy + ch)
            if new_max > best_max + 1e-9:
                continue
            if new_max > best_max - 1e-9 and new_sum >= best_sum - 1e-9:
                continue
            if overlaps_anything(cx, cy):
                continue
            best_max, best_sum, best_x, best_y = new_max, new_sum, cx, cy

    if abs(best_x - x[i]) < 1e-9 and abs(best_y - y[i]) < 1e-9:
        return False
    x[i], y[i] = best_x, best_y
    return True


def _holes_fill_pass(ids, x, y, dims, is_preplaced):
    """Port of packer.cpp's holes_fill_pass: diagonal relocation that closes
    L-shaped whitespace compact_left_down can't reach (axis-only moves).

    NOTE: the C++ version also skips boundary-pinned and grouping-cluster
    blocks (their positions are constraint-critical). This Python port
    doesn't have boundary/grouping info at this layer, so it only skips
    preplaced blocks -- occasionally disturbing a boundary/grouping block
    is possible, but contest_cost.py will still score any resulting V_rel
    violation correctly, so this is a safe simplification for prototyping.
    """
    ids = list(ids)
    n = len(ids)
    if n < 2:
        return

    order = [i for i in ids if not is_preplaced.get(i, False)]
    if not order:
        return
    order.sort(key=lambda i: max(x[i] + dims[i][0], y[i] + dims[i][1]), reverse=True)

    for _ in range(2):
        any_moved = False
        for i in order:
            if _relocate_to_min_corner(ids, x, y, dims, i):
                any_moved = True
        if not any_moved:
            break


def _touches(a, b, x, y, dims, eps=1e-7) -> bool:
    ax, ay, aw, ah = x[a], y[a], dims[a][0], dims[a][1]
    bx, by, bw, bh = x[b], y[b], dims[b][0], dims[b][1]
    if abs((ax + aw) - bx) < eps or abs((bx + bw) - ax) < eps:
        ylo, yhi = max(ay, by), min(ay + ah, by + bh)
        if yhi - ylo > eps:
            return True
    if abs((ay + ah) - by) < eps or abs((by + bh) - ay) < eps:
        xlo, xhi = max(ax, bx), min(ax + aw, bx + bw)
        if xhi - xlo > eps:
            return True
    return False


def _grouping_repair_pass(ids, x, y, dims, is_preplaced, cluster_id, boundary_code=None):
    """Reattach isolated group members (touching no sibling) by sliding them
    flush against a sibling whenever a free adjacent slot exists.

    Skips both preplaced AND boundary-constrained blocks (2026-07-09): the
    aggressive boundary repair pins boundary blocks to walls, and if grouping
    were allowed to drag them back toward their group it would re-open
    boundary violations. So grouping only relocates the FREE members, pulling
    them to wherever the group's pinned members ended up -- the two repairs
    then cooperate instead of fighting (boundary pins, grouping gathers)."""
    ids = list(ids)
    boundary_code = boundary_code or {}
    groups: Dict[int, List[int]] = {}
    for i in ids:
        cid = cluster_id.get(i, 0)
        if cid and cid > 0:
            groups.setdefault(cid, []).append(i)

    def cell_free(i, nx, ny):
        w, h = dims[i]
        for j in ids:
            if j == i:
                continue
            if (nx < x[j] + dims[j][0] and x[j] < nx + w and
                    ny < y[j] + dims[j][1] and y[j] < ny + h):
                return False
        return True

    def movable(i):
        return not is_preplaced.get(i, False) and not boundary_code.get(i, 0)

    def components(g):
        """Connected components of group g under edge-adjacency (union-find)."""
        parent = {i: i for i in g}

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for a_idx in range(len(g)):
            for b_idx in range(a_idx + 1, len(g)):
                a, b = g[a_idx], g[b_idx]
                if _touches(a, b, x, y, dims):
                    ra, rb = find(a), find(b)
                    if ra != rb:
                        parent[ra] = rb
        comps: Dict[int, List[int]] = {}
        for i in g:
            comps.setdefault(find(i), []).append(i)
        return list(comps.values())

    total_moved = False
    for g in groups.values():
        if len(g) <= 1:
            continue
        for _ in range(6):
            comps = components(g)
            if len(comps) <= 1:
                break  # already one connected component -> V_group=0 for this group
            # Core = largest component; try to attach every other component's
            # movable members flush against a core member.
            comps.sort(key=len, reverse=True)
            core = comps[0]
            core_cx = sum(x[i] + dims[i][0] / 2 for i in core) / len(core)
            core_cy = sum(y[i] + dims[i][1] / 2 for i in core) / len(core)
            moved = False
            for comp in comps[1:]:
                for s in comp:
                    if not movable(s):
                        continue
                    sw, sh = dims[s]
                    best = None
                    best_d = None
                    for t in core:
                        tx, ty = x[t], y[t]
                        tw, th = dims[t]
                        for nx, ny in ((tx + tw, ty), (tx - sw, ty),
                                       (tx, ty + th), (tx, ty - sh),
                                       (tx + tw, ty + th - sh), (tx - sw, ty + th - sh)):
                            if nx < -1e-9 or ny < -1e-9:
                                continue
                            if not cell_free(s, nx, ny):
                                continue
                            d = (nx + sw / 2 - core_cx) ** 2 + (ny + sh / 2 - core_cy) ** 2
                            if best_d is None or d < best_d:
                                best_d, best = d, (nx, ny)
                    if best is not None:
                        x[s], y[s] = best
                        moved = total_moved = True
            if not moved:
                break
    return total_moved


def _boundary_repair_pass(ids, x, y, dims, is_preplaced, boundary_code, rounds: int = 6,
                          push_past: bool = True):
    """Stronger-than-packer.cpp boundary repair. The C++ version only snaps a
    block onto its edge when the exact target cell is free, so a block whose
    edge is blocked stays violated -- this is the dominant V_rel source
    (2026-07-09 diagnosis: boundary was 141 of 191 total soft violations).

    This version instead SCANS ALONG the required wall for the first
    overlap-free slot:
      * LEFT (bit 1, x=0) / BOTTOM (bit 8, y=0): the wall coordinate is fixed
        at 0, so a free slot ALWAYS exists (stack above/beside everything) ->
        these are GUARANTEED satisfiable and this pass will always place them.
      * RIGHT (bit 2) / TOP (bit 4): the wall coordinate is the current bbox
        extent, which shifts as blocks move; we align to it and, if no free
        slot exists at that coordinate, push the block just past the current
        extent so it BECOMES the new edge (guaranteeing contact this round).
    Corners (two bits) require touching both walls simultaneously -- handled
    by only accepting a slot that satisfies every set bit.

    The whole thing iterates `rounds` times because moving one block changes
    the bbox and frees/occupies cells for others. Only non-preplaced blocks
    move. Never introduces overlap (every placement is overlap-checked).
    """
    ids = list(ids)
    if not ids:
        return

    def cell_free(i, nx, ny):
        w, h = dims[i]
        for j in ids:
            if j == i:
                continue
            if (nx + 1e-9 < x[j] + dims[j][0] and x[j] + 1e-9 < nx + w and
                    ny + 1e-9 < y[j] + dims[j][1] and y[j] + 1e-9 < ny + h):
                return False
        return True

    def satisfied(i, bc, wbb, hbb):
        w, h = dims[i]
        if bc & 1 and abs(x[i]) > 1e-6:
            return False
        if bc & 8 and abs(y[i]) > 1e-6:
            return False
        if bc & 2 and abs((x[i] + w) - wbb) > 1e-6:
            return False
        if bc & 4 and abs((y[i] + h) - hbb) > 1e-6:
            return False
        return True

    for _ in range(rounds):
        wbb = max(x[i] + dims[i][0] for i in ids)
        hbb = max(y[i] + dims[i][1] for i in ids)
        moved = False
        # Corners (2 bits set) first -- they are the most constrained.
        order = sorted(
            (i for i in ids
             if boundary_code.get(i, 0) and not is_preplaced.get(i, False)),
            key=lambda i: -bin(boundary_code.get(i, 0)).count("1"),
        )
        for i in order:
            bc = boundary_code[i]
            if satisfied(i, bc, wbb, hbb):
                continue
            w, h = dims[i]

            # Fixed coordinates implied by the set bits (None = free axis).
            fx = 0.0 if (bc & 1) else (wbb - w if (bc & 2) else None)
            fy = 0.0 if (bc & 8) else (hbb - h if (bc & 4) else None)

            # Candidate positions for the free axis = other blocks' edges.
            xs = sorted({0.0} | {x[j] for j in ids} | {x[j] + dims[j][0] for j in ids}
                        | {wbb - w})
            ys = sorted({0.0} | {y[j] for j in ids} | {y[j] + dims[j][1] for j in ids}
                        | {hbb - h})

            placed = False
            cand_x = [fx] if fx is not None else xs
            cand_y = [fy] if fy is not None else ys
            for nx in cand_x:
                if nx < -1e-9:
                    continue
                for ny in cand_y:
                    if ny < -1e-9:
                        continue
                    if cell_free(i, nx, ny):
                        if abs(nx - x[i]) > 1e-9 or abs(ny - y[i]) > 1e-9:
                            x[i], y[i] = nx, ny
                            moved = placed = True
                        else:
                            placed = True  # already there and now counts as ok
                        break
                if placed:
                    break

            # RIGHT/TOP with no free slot at the current edge: push the block
            # just past everything so it defines the new edge (guaranteed
            # contact) -- but this grows the bbox (area cost). `push_past`
            # gates it so we can A/B "guarantee boundary" vs "keep area tight".
            if push_past and not placed and (bc & 2 or bc & 4):
                nx = (wbb) if (bc & 2) else (0.0 if (bc & 1) else x[i])
                ny = (hbb) if (bc & 4) else (0.0 if (bc & 8) else y[i])
                # For RIGHT push: place flush past the rightmost, keep a free y.
                if bc & 2 and not (bc & 4):
                    nx = wbb  # sticks out -> becomes new right edge
                    ny = _lowest_free_y(ids, x, y, dims, i, nx)
                elif bc & 4 and not (bc & 2):
                    ny = hbb
                    nx = _lowest_free_x(ids, x, y, dims, i, ny)
                if nx >= -1e-9 and ny >= -1e-9 and cell_free(i, nx, ny):
                    x[i], y[i] = nx, ny
                    moved = True
        if not moved:
            break


def _lowest_free_y(ids, x, y, dims, i, nx):
    """Lowest y >= 0 where block i placed at (nx, y) overlaps nothing."""
    w, h = dims[i]
    cands = sorted({0.0} | {y[j] + dims[j][1] for j in ids if j != i})
    for ny in cands:
        if ny < -1e-9:
            continue
        ok = True
        for j in ids:
            if j == i:
                continue
            if (nx + 1e-9 < x[j] + dims[j][0] and x[j] + 1e-9 < nx + w and
                    ny + 1e-9 < y[j] + dims[j][1] and y[j] + 1e-9 < ny + h):
                ok = False
                break
        if ok:
            return ny
    return 0.0


def _lowest_free_x(ids, x, y, dims, i, ny):
    """Lowest x >= 0 where block i placed at (x, ny) overlaps nothing."""
    w, h = dims[i]
    cands = sorted({0.0} | {x[j] + dims[j][0] for j in ids if j != i})
    for nx in cands:
        if nx < -1e-9:
            continue
        ok = True
        for j in ids:
            if j == i:
                continue
            if (nx + 1e-9 < x[j] + dims[j][0] and x[j] + 1e-9 < nx + w and
                    ny + 1e-9 < y[j] + dims[j][1] and y[j] + 1e-9 < ny + h):
                ok = False
                break
        if ok:
            return nx
    return 0.0


def _check_overlap_free(ids, x, y, dims) -> bool:
    ids = list(ids)
    for a_idx in range(len(ids)):
        i = ids[a_idx]
        for j in ids[a_idx + 1:]:
            if (x[i] < x[j] + dims[j][0] and x[j] < x[i] + dims[i][0] and
                    y[i] < y[j] + dims[j][1] and y[j] < y[i] + dims[i][1]):
                return False
    return True


def compute_hpwl(x, y, dims, b2b, p2b, pins_pos) -> Tuple[float, float]:
    """Weighted centroid-Manhattan HPWL, matching cost.cpp's definition
    (see CLAUDE.md gotcha #1): cx_i = x_i + w_i/2.  Returns (hpwl_int, hpwl_ext)."""
    cx = {i: x[i] + 0.5 * dims[i][0] for i in x}
    cy = {i: y[i] + 0.5 * dims[i][1] for i in y}

    hpwl_int = 0.0
    for row in b2b.tolist():
        a, b, w = int(row[0]), int(row[1]), row[2]
        hpwl_int += w * (abs(cx[a] - cx[b]) + abs(cy[a] - cy[b]))

    hpwl_ext = 0.0
    for row in p2b.tolist():
        pin, b, w = int(row[0]), int(row[1]), row[2]
        px, py = float(pins_pos[pin][0]), float(pins_pos[pin][1])
        hpwl_ext += w * (abs(px - cx[b]) + abs(py - cy[b]))

    return hpwl_int, hpwl_ext
