"""Pure-Python B*-tree packer -- turns a (parent, direction) topology into
actual (x, y, w, h) geometry, for fast prototyping/scoring of the tree
generator's samples without needing the compiled C++ binary.

This ports the CORE placement rule from `collaborate/src/packer.cpp`
(confirmed by cross-checking against `fp_sol` on real training cases --
see the project memory / WINNING_STRATEGY.md for the validation numbers):

    left  child (direction 0): x = parent.x + parent.w   (touch parent's right)
    right child (direction 1): x = parent.x              (touch parent's top)
    both:                       y = current contour height in [x, x+w)

Also ports `compact_left_down` (slide every non-anchored block as far left
and down as it will go without overlapping) since the raw contour pack
alone leaves visible whitespace.  The C++ packer additionally runs
`bbox_balance_pass`, `holes_fill_pass`, `grouping_repair_pass` and
`boundary_repair_pass` (see packer.cpp) -- those are NOT ported here.  This
module is for quickly ranking/visualizing samples from the ML topology
model; the real submission path always goes through the actual C++
`floorplanner` binary, which has the full repair pipeline.
"""

from __future__ import annotations

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


def pack_btree(
    root: int,
    lc: Dict[int, Optional[int]],
    rc: Dict[int, Optional[int]],
    dims: Dict[int, Tuple[float, float]],
    is_preplaced: Optional[Dict[int, bool]] = None,
    preplaced_xy: Optional[Dict[int, Tuple[float, float]]] = None,
    compact: bool = True,
) -> PackResult:
    n = len(dims)
    is_preplaced = is_preplaced or {}
    preplaced_xy = preplaced_xy or {}

    contour = _Contour()
    x: Dict[int, float] = {}
    y: Dict[int, float] = {}

    # Pre-seed preplaced (anchored) footprints, ascending top-edge order,
    # exactly as packer.cpp does -- so tree-placed blocks that share an
    # x-range with an anchor get lifted above it instead of overlapping.
    anchors = [i for i in range(n) if is_preplaced.get(i, False)]
    anchors.sort(key=lambda i: preplaced_xy[i][1] + dims[i][1])
    for i in anchors:
        ax, ay = preplaced_xy[i]
        aw, ah = dims[i]
        x[i], y[i] = ax, ay
        contour.set_height(ax, ax + aw, ay + ah)

    # Iterative DFS: parent before children, left child fully before right
    # child (matches packer.cpp's stack-based traversal).
    stack = [(root, 0)]
    while stack:
        v, state = stack.pop()
        if state == 0:
            if v not in x:  # not a pre-seeded anchor
                w_v, h_v = dims[v]
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
                    px = (x[parent] + dims[parent][0]) if is_left else x[parent]
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
        _compact_left_down(range(n), x, y, dims, is_preplaced)

    bbox_w = max(x[i] + dims[i][0] for i in range(n))
    bbox_h = max(y[i] + dims[i][1] for i in range(n))
    overlap_free = _check_overlap_free(range(n), x, y, dims)

    return PackResult(x=x, y=y,
                       w={i: dims[i][0] for i in range(n)},
                       h={i: dims[i][1] for i in range(n)},
                       bbox_w=bbox_w, bbox_h=bbox_h, overlap_free=overlap_free)


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
