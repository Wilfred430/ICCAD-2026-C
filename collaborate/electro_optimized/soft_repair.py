"""Deterministic soft-constraint repair passes, run AFTER legalization.

Mirrors the C++ SA solver's boundary / grouping repair logic, but in numpy and
operating on an already-overlap-free placement.  Every move is guarded: a block
is only relocated if its destination cell is free, so zero overlap is preserved.

  * boundary_snap   : slide each boundary block so its required edge coincides
                      with the layout bbox extreme (V_boundary).
  * grouping_repair : abut isolated cluster members to a connected sibling so the
                      group forms one connected component (V_grouping).

Preplaced blocks are never moved.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-6


def _free(idx, nx, ny, x, y, w, h, ignore=None):
    """True if placing block idx at (nx,ny) overlaps no other block."""
    N = len(x)
    l2, r2, b2, t2 = nx, nx + w[idx], ny, ny + h[idx]
    for j in range(N):
        if j == idx or (ignore is not None and j in ignore):
            continue
        if (l2 < x[j] + w[j] - _EPS and x[j] < r2 - _EPS and
                b2 < y[j] + h[j] - _EPS and y[j] < t2 - _EPS):
            return False
    return True


def _get_overlapping(idx, nx, ny, x, y, w, h):
    """Return indices of all blocks overlapping with block idx at (nx, ny)."""
    N = len(x)
    l2, r2, b2, t2 = nx, nx + w[idx], ny, ny + h[idx]
    overlapped = []
    for j in range(N):
        if j == idx:
            continue
        if (l2 < x[j] + w[j] - _EPS and x[j] < r2 - _EPS and
                b2 < y[j] + h[j] - _EPS and y[j] < t2 - _EPS):
            overlapped.append(j)
    return overlapped


def _slot_along_y(i, X, x, y, w, h, ymn, ymx, floor=None):
    """Fix block i's x to X; find the NEAREST free y in [ymn, ymx-h_i].

    Min-displacement: among the discrete candidate slots (stay, the two wall
    ends, and the tight-pack positions just above/below each column neighbour)
    pick the free one closest to the block's current y -- this keeps the
    boundary block on its wall while moving it as little as possible, so
    satisfying the boundary constraint costs the least possible wirelength."""
    R = X + w[i]
    cands = [y[i], ymn, ymx - h[i]]
    for j in range(len(x)):
        if j == i:
            continue
        if x[j] < R - _EPS and X < x[j] + w[j] - _EPS:    # shares the column
            cands.append(y[j] + h[j])                      # above j
            cands.append(y[j] - h[i])                      # below j
    lo = ymn if floor is None else max(ymn, floor)
    valid = sorted({c for c in cands if lo - _EPS <= c <= ymx - h[i] + _EPS},
                   key=lambda c: abs(c - y[i]))
    for yy in valid:
        if _free(i, X, yy, x, y, w, h):
            return yy
    return None


def _slot_along_x(i, Y, x, y, w, h, xmn, xmx, floor=None):
    """Fix block i's y to Y; find the NEAREST free x in [xmn, xmx-w_i]
    (min-displacement; mirror of _slot_along_y)."""
    T = Y + h[i]
    cands = [x[i], xmn, xmx - w[i]]
    for j in range(len(x)):
        if j == i:
            continue
        if y[j] < T - _EPS and Y < y[j] + h[j] - _EPS:    # shares the row
            cands.append(x[j] + w[j])
            cands.append(x[j] - w[i])
    lo = xmn if floor is None else max(xmn, floor)
    valid = sorted({c for c in cands if lo - _EPS <= c <= xmx - w[i] + _EPS},
                   key=lambda c: abs(c - x[i]))
    for xx in valid:
        if _free(i, xx, Y, x, y, w, h):
            return xx
    return None


def boundary_snap(x, y, w, h, bcode, is_pre, passes=3, floor=None, push_past=False, clust_id=None, mib_id=None, wide_swap=False):
    """Slide each boundary block onto its required bbox edge, searching along the
    wall for a free slot (not just the exact current spot).  With `floor` set,
    movable blocks are kept at corner >= floor (first-quadrant containment).

    `push_past` (2026-07-14, default False = original behaviour unchanged):
    when the wall-scan finds no free slot, push the block just past the
    current opposite extent instead of leaving the violation unresolved --
    always free by definition, but grows the bbox (area cost). Validated as a
    PORTFOLIO candidate (try both, keep whichever scores better on real cost)
    via `electro_parallel.boundary_pushpast_variant` -- full-100 with the
    portfolio: Total Score 2.8233 -> 2.5611. Never call this as an
    unconditional replacement (forcing push_past=True always regressed to
    3.0668 in isolated testing)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    w = np.asarray(w, float); h = np.asarray(h, float)
    bcode = np.asarray(bcode).astype(int); is_pre = np.asarray(is_pre, bool)
    N = len(x)
    for _ in range(passes):
        xmn = x.min(); xmx = (x + w).max(); ymn = y.min(); ymx = (y + h).max()
        moved = False
        for i in range(N):
            c = int(bcode[i])
            if c == 0 or is_pre[i]:
                continue
            want_x = (c & 1) or (c & 2)
            want_y = (c & 4) or (c & 8)
            X = xmn if (c & 1) else (xmx - w[i]) if (c & 2) else x[i]
            Y = ymn if (c & 8) else (ymx - h[i]) if (c & 4) else y[i]

            if want_x and want_y:                      # corner: one exact spot
                if _free(i, X, Y, x, y, w, h) and (abs(X - x[i]) > _EPS or abs(Y - y[i]) > _EPS):
                    x[i], y[i] = X, Y; moved = True
            elif want_x:                               # left/right wall: slide y
                yy = _slot_along_y(i, X, x, y, w, h, ymn, ymx, floor)
                if yy is None and push_past:
                    yy = ymx if _free(i, X, ymx, x, y, w, h) else None
                if yy is not None and (abs(X - x[i]) > _EPS or abs(yy - y[i]) > _EPS):
                    x[i], y[i] = X, yy; moved = True
            elif want_y:                               # top/bottom wall: slide x
                xx = _slot_along_x(i, Y, x, y, w, h, xmn, xmx, floor)
                if xx is None and push_past:
                    xx = xmx if _free(i, xmx, Y, x, y, w, h) else None
                if xx is not None and (abs(xx - x[i]) > _EPS or abs(Y - y[i]) > _EPS):
                    x[i], y[i] = xx, Y; moved = True
        if not moved:
            break
            
    # Strict Zero-Overlap Swap Pass — try multiple wall positions
    for _swap_round in range(3 if wide_swap else 1):
        xmn = x.min(); xmx = (x + w).max(); ymn = y.min(); ymx = (y + h).max()
        any_swapped = False
        for i in range(N):
            c = int(bcode[i])
            if c == 0 or is_pre[i]:
                continue
                
            ok = True
            if c & 1 and abs(x[i] - xmn) >= 1e-6: ok = False
            if c & 2 and abs(x[i] + w[i] - xmx) >= 1e-6: ok = False
            if c & 4 and abs(y[i] + h[i] - ymx) >= 1e-6: ok = False
            if c & 8 and abs(y[i] - ymn) >= 1e-6: ok = False
            if ok:
                continue
                
            want_x = (c & 1) or (c & 2)
            want_y = (c & 4) or (c & 8)
            
            # Generate candidate wall positions (not just one)
            wall_positions = []
            if want_x and want_y:
                # Corner: only one position
                X = xmn if (c & 1) else (xmx - w[i])
                Y = ymn if (c & 8) else (ymx - h[i])
                wall_positions.append((X, Y))
            elif want_x:
                X = xmn if (c & 1) else (xmx - w[i])
                # Slide along the wall: try current y, wall ends, and tight-pack slots
                cands_y = {y[i], ymn, ymx - h[i]}
                R = X + w[i]
                for j2 in range(N):
                    if j2 == i: continue
                    if x[j2] < R - _EPS and X < x[j2] + w[j2] - _EPS:
                        cands_y.add(y[j2] + h[j2])
                        cands_y.add(y[j2] - h[i])
                for yy in sorted(cands_y, key=lambda c2: abs(c2 - y[i])):
                    if ymn - _EPS <= yy <= ymx - h[i] + _EPS:
                        wall_positions.append((X, yy))
            elif want_y:
                Y = ymn if (c & 8) else (ymx - h[i])
                cands_x = {x[i], xmn, xmx - w[i]}
                T = Y + h[i]
                for j2 in range(N):
                    if j2 == i: continue
                    if y[j2] < T - _EPS and Y < y[j2] + h[j2] - _EPS:
                        cands_x.add(x[j2] + w[j2])
                        cands_x.add(x[j2] - w[i])
                for xx in sorted(cands_x, key=lambda c2: abs(c2 - x[i])):
                    if xmn - _EPS <= xx <= xmx - w[i] + _EPS:
                        wall_positions.append((xx, Y))
            
            swapped = False
            for X, Y in wall_positions:
                if swapped:
                    break
                for j in range(N):
                    if j == i or is_pre[j]: continue
                    if mib_id is not None and int(mib_id[j]) != 0: continue
                    
                    cj = int(bcode[j])
                    if cj != 0:
                        if not wide_swap: continue
                        if not _bnd_ok(j, x[i], y[i], w, h, bcode, xmn, xmx, ymn, ymx):
                            continue
                            
                    if not _free(i, X, Y, x, y, w, h, ignore=[i, j]): continue
                    if not _free(j, x[i], y[i], x, y, w, h, ignore=[i, j]): continue
                        
                    old_xi, old_yi = x[i], y[i]
                    x[i], y[i] = X, Y
                    x[j], y[j] = old_xi, old_yi
                    swapped = True
                    any_swapped = True
                    break
                
            if swapped:
                is_pre[i] = True

        if not any_swapped:
            break

    return x, y


def _touch(i, j, x, y, w, h):
    """True if blocks i,j share an edge segment of positive length (abut)."""
    ox = min(x[i] + w[i], x[j] + w[j]) - max(x[i], x[j])   # x-overlap length
    oy = min(y[i] + h[i], y[j] + h[j]) - max(y[i], y[j])   # y-overlap length
    share_v = (abs(x[i] + w[i] - x[j]) < 1e-4 or abs(x[j] + w[j] - x[i]) < 1e-4) and oy > 1e-4
    share_h = (abs(y[i] + h[i] - y[j]) < 1e-4 or abs(y[j] + h[j] - y[i]) < 1e-4) and ox > 1e-4
    return share_v or share_h


def _components(members, x, y, w, h):
    """Connected components of `members` under the abut relation (union-find)."""
    parent = {m: m for m in members}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for ai in range(len(members)):
        for bi in range(ai + 1, len(members)):
            i, j = members[ai], members[bi]
            if _touch(i, j, x, y, w, h):
                parent[find(i)] = find(j)
    comps = {}
    for m in members:
        comps.setdefault(find(m), []).append(m)
    return list(comps.values())


def soft_violation_counts(x, y, w, h, bcode, clust_id, mib_id):
    """Compute (V_boundary, V_grouping, V_mib, n_soft) exactly as the evaluator."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    w = np.asarray(w, float); h = np.asarray(h, float)
    bcode = np.asarray(bcode).astype(int)
    clust_id = np.asarray(clust_id).astype(int)
    mib_id = np.asarray(mib_id).astype(int)
    N = len(x)

    # boundary
    xmn, xmx = x.min(), (x + w).max()
    ymn, ymx = y.min(), (y + h).max()
    vb = 0
    n_boundary = int((bcode != 0).sum())
    for i in range(N):
        c = int(bcode[i])
        if c == 0:
            continue
        ok = True
        if c & 1 and abs(x[i] - xmn) >= 1e-6: ok = False
        if c & 2 and abs(x[i] + w[i] - xmx) >= 1e-6: ok = False
        if c & 4 and abs(y[i] + h[i] - ymx) >= 1e-6: ok = False
        if c & 8 and abs(y[i] - ymn) >= 1e-6: ok = False
        if not ok:
            vb += 1

    # grouping
    vg = 0; n_grp = 0
    for g in range(1, (int(clust_id.max()) if clust_id.size else 0) + 1):
        mem = np.where(clust_id == g)[0].tolist()
        n_grp += max(0, len(mem) - 1)
        if len(mem) > 1:
            vg += len(_components(mem, x, y, w, h)) - 1

    # mib
    vm = 0; n_mib = 0
    for g in range(1, (int(mib_id.max()) if mib_id.size else 0) + 1):
        mem = np.where(mib_id == g)[0].tolist()
        n_mib += max(0, len(mem) - 1)
        shapes = {(round(float(w[i]), 4), round(float(h[i]), 4)) for i in mem}
        vm += len(shapes) - 1

    n_soft = max(1, n_boundary + n_grp + n_mib)
    return vb, vg, vm, n_soft


def _bnd_ok(i, nx, ny, w, h, bcode, xmn, xmx, ymn, ymx):
    """Check if placing block i at (nx, ny) satisfies its boundary constraint."""
    c = int(bcode[i])
    if c == 0:
        return True
    if c & 1 and abs(nx - xmn) >= 1e-6:
        return False
    if c & 2 and abs(nx + w[i] - xmx) >= 1e-6:
        return False
    if c & 4 and abs(ny + h[i] - ymx) >= 1e-6:
        return False
    if c & 8 and abs(ny - ymn) >= 1e-6:
        return False
    return True


def grouping_repair(x, y, w, h, clust_id, is_pre, passes=4, floor=None, push_past=False, bcode=None, mib_id=None):
    x = np.asarray(x, float); y = np.asarray(y, float)
    w = np.asarray(w, float); h = np.asarray(h, float)
    clust_id = np.asarray(clust_id).astype(int); is_pre = np.asarray(is_pre, bool)
    G = int(clust_id.max()) if clust_id.size else 0
    fl = -np.inf if floor is None else floor
    has_bcode = bcode is not None
    if has_bcode:
        bcode = np.asarray(bcode).astype(int)
    if mib_id is not None:
        mib_id = np.asarray(mib_id).astype(int)

    xmx_curr = (x + w).max()
    ymx_curr = (y + h).max()

    for _ in range(passes):
        any_move = False
        # Recompute bbox at each pass for boundary checks
        if has_bcode:
            bnd_xmn = x.min(); bnd_xmx = (x + w).max()
            bnd_ymn = y.min(); bnd_ymx = (y + h).max()
        for g in range(1, G + 1):
            members = np.where(clust_id == g)[0].tolist()
            if len(members) <= 1:
                continue
            comps = _components(members, x, y, w, h)
            if len(comps) <= 1:
                continue
            comps.sort(key=len, reverse=True)
            main = set(comps[0])           # keep the largest component anchored
            for comp in comps[1:]:
                for i in comp:
                    if is_pre[i]:
                        continue
                    # Min-displacement: among every free abut slot against a
                    # main-component sibling, pick the one nearest i's current
                    # position (Manhattan = the HPWL metric) so reconnecting the
                    # cluster perturbs wirelength as little as possible.
                    candidates = []
                    for s in main:
                        # left of s
                        nx = x[s] - w[i]
                        clip_y_min = y[s] - h[i] + 1e-3
                        clip_y_max = y[s] + h[s] - 1e-3
                        ny_clip = np.clip(y[i], clip_y_min, clip_y_max) if clip_y_min <= clip_y_max else y[s]
                        for ny in (y[s], y[s] + h[s] - h[i], y[s] + 0.5 * (h[s] - h[i]), ny_clip):
                            candidates.append((nx, ny))
                        
                        # right of s
                        nx = x[s] + w[s]
                        for ny in (y[s], y[s] + h[s] - h[i], y[s] + 0.5 * (h[s] - h[i]), ny_clip):
                            candidates.append((nx, ny))
                        
                        # below s
                        ny = y[s] - h[i]
                        clip_x_min = x[s] - w[i] + 1e-3
                        clip_x_max = x[s] + w[s] - 1e-3
                        nx_clip = np.clip(x[i], clip_x_min, clip_x_max) if clip_x_min <= clip_x_max else x[s]
                        for nx in (x[s], x[s] + w[s] - w[i], x[s] + 0.5 * (w[s] - w[i]), nx_clip):
                            candidates.append((nx, ny))
                        
                        # above s
                        ny = y[s] + h[s]
                        for nx in (x[s], x[s] + w[s] - w[i], x[s] + 0.5 * (w[s] - w[i]), nx_clip):
                            candidates.append((nx, ny))

                    best, best_d = None, None
                    for nx, ny in candidates:
                        if nx < fl or ny < fl:          # keep first-quadrant
                            continue
                        if has_bcode and not _bnd_ok(i, nx, ny, w, h, bcode, bnd_xmn, bnd_xmx, bnd_ymn, bnd_ymx):
                            continue
                        if _free(i, nx, ny, x, y, w, h):
                            d = abs(nx - x[i]) + abs(ny - y[i])
                            if best_d is None or d < best_d:
                                best, best_d = (nx, ny), d
                    
                    # Try zero-overlap swap if no free slot is found
                    best_swap, best_swap_d, best_swap_j = None, None, None
                    if best is None:
                        s_xmn = x.min(); s_xmx = (x + w).max()
                        s_ymn = y.min(); s_ymx = (y + h).max()
                        for nx, ny in candidates:
                            if nx < fl or ny < fl:
                                continue
                            if has_bcode and not _bnd_ok(i, nx, ny, w, h, bcode, bnd_xmn, bnd_xmx, bnd_ymn, bnd_ymx):
                                continue
                            
                            overlapped = _get_overlapping(i, nx, ny, x, y, w, h)
                            if len(overlapped) == 1:
                                j = overlapped[0]
                                if is_pre[j]:
                                    continue
                                if clust_id[j] != 0:
                                    continue
                                
                                # j's boundary constraint must be satisfied at i's old position
                                if not _bnd_ok(j, x[i], y[i], w, h, bcode if has_bcode else np.zeros_like(x), s_xmn, s_xmx, s_ymn, s_ymx):
                                    continue
                                    
                                if not _free(i, nx, ny, x, y, w, h, ignore=[i, j]):
                                    continue
                                if not _free(j, x[i], y[i], x, y, w, h, ignore=[i, j]):
                                    continue
                                    
                                d = abs(nx - x[i]) + abs(ny - y[i])
                                if best_swap_d is None or d < best_swap_d:
                                    best_swap, best_swap_d, best_swap_j = (nx, ny), d, j

                    if best is not None:
                        x[i], y[i] = best
                        main.add(i)
                        any_move = True
                    elif best_swap is not None:
                        j = best_swap_j
                        old_xi, old_yi = x[i], y[i]
                        x[i], y[i] = best_swap
                        x[j], y[j] = old_xi, old_yi
                        main.add(i)
                        any_move = True
                    elif push_past:
                        # Rebuild candidates with axis information: 'H' (horizontal abutment) or 'V' (vertical)
                        candidates_with_axis = []
                        for s in main:
                            # left of s
                            nx = x[s] - w[i]
                            clip_y_min = y[s] - h[i] + 1e-3
                            clip_y_max = y[s] + h[s] - 1e-3
                            ny_clip = np.clip(y[i], clip_y_min, clip_y_max) if clip_y_min <= clip_y_max else y[s]
                            for ny in (y[s], y[s] + h[s] - h[i], y[s] + 0.5 * (h[s] - h[i]), ny_clip):
                                candidates_with_axis.append((nx, ny, 'H'))
                            
                            # right of s
                            nx = x[s] + w[s]
                            for ny in (y[s], y[s] + h[s] - h[i], y[s] + 0.5 * (h[s] - h[i]), ny_clip):
                                candidates_with_axis.append((nx, ny, 'H'))
                            
                            # below s
                            ny = y[s] - h[i]
                            clip_x_min = x[s] - w[i] + 1e-3
                            clip_x_max = x[s] + w[s] - 1e-3
                            nx_clip = np.clip(x[i], clip_x_min, clip_x_max) if clip_x_min <= clip_x_max else x[s]
                            for nx in (x[s], x[s] + w[s] - w[i], x[s] + 0.5 * (w[s] - w[i]), nx_clip):
                                candidates_with_axis.append((nx, ny, 'V'))
                            
                            # above s
                            ny = y[s] + h[s]
                            for nx in (x[s], x[s] + w[s] - w[i], x[s] + 0.5 * (w[s] - w[i]), nx_clip):
                                candidates_with_axis.append((nx, ny, 'V'))
                        
                        best_pp, best_pp_d, best_pp_j, best_pp_axis = None, None, None, None
                        for nx, ny, axis in candidates_with_axis:
                            if nx < fl or ny < fl:
                                continue
                            overlapped = _get_overlapping(i, nx, ny, x, y, w, h)
                            if len(overlapped) == 1:
                                j = overlapped[0]
                                if not is_pre[j]:
                                    if mib_id is not None and int(mib_id[j]) != 0:
                                        continue
                                    d = abs(nx - x[i]) + abs(ny - y[i])
                                    if best_pp_d is None or d < best_pp_d:
                                        best_pp, best_pp_d, best_pp_j, best_pp_axis = (nx, ny), d, j, axis
                        
                        if best_pp is not None:
                            x[i], y[i] = best_pp
                            j = best_pp_j
                            if best_pp_axis == 'H':
                                x[j] = xmx_curr
                                xmx_curr += w[j]
                            else:
                                y[j] = ymx_curr
                                ymx_curr += h[j]
                            main.add(i)
                            any_move = True
        if not any_move:
            break
    return x, y
