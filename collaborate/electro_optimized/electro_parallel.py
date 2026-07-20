"""Parallel multi-start worker, kept in its own importable module.

The contest harness loads electro_optimizer.py from a file path under the name
"optimizer_module", which is NOT importable by name -- so a worker function
defined there can't be pickled to a multiprocessing worker.  Defining the worker
here (electro_parallel is a normal module on sys.path, because electro_optimizer
inserts its own directory) makes it picklable, so the fork pool works.

Per-case inputs are stashed in the module global `WORK` *before* the pool is
created, so the fork inherits them and we never pickle the connectivity tensors.
"""
from __future__ import annotations

import numpy as np

from analytical_place import place
from legalize import legalize, remove_overlap, legalize_qinfer
from soft_repair import boundary_snap, grouping_repair

WORK = None   # per-case inputs, set by the parent before forking the pool


def compute_hpwl_numpy(x, y, w, h, P):
    cx = x + 0.5 * w
    cy = y + 0.5 * h
    wl = 0.0
    
    def to_np(arr):
        if hasattr(arr, "cpu"):
            return arr.cpu().numpy()
        return arr

    b2b = P.get("b2b")
    if b2b is not None and len(b2b) > 0:
        b2b_np = to_np(b2b)
        ia = b2b_np[:, 0].astype(int)
        ib = b2b_np[:, 1].astype(int)
        wb = b2b_np[:, 2]
        wl += (wb * (np.abs(cx[ia] - cx[ib]) + np.abs(cy[ia] - cy[ib]))).sum()
        
    p2b = P.get("p2b")
    if p2b is not None and len(p2b) > 0 and P.get("pins") is not None and len(P["pins"]) > 0:
        p2b_np = to_np(p2b)
        pins_np = to_np(P["pins"])
        pi = np.clip(p2b_np[:, 0].astype(int), 0, len(pins_np) - 1)
        bi = np.clip(p2b_np[:, 1].astype(int), 0, len(cx) - 1)
        wp = p2b_np[:, 2]
        px = pins_np[pi, 0]
        py = pins_np[pi, 1]
        wl += (wp * (np.abs(cx[bi] - px) + np.abs(cy[bi] - py))).sum()
        
    return wl


def post_place_repair(positions, P):
    """Run post-placement legalization, grouping/boundary repair, and final overlap removal."""
    x = np.array([p[0] for p in positions], dtype=float)
    y = np.array([p[1] for p in positions], dtype=float)
    w = np.array([p[2] for p in positions], dtype=float)
    h = np.array([p[3] for p in positions], dtype=float)
    is_pre = P["is_pre"].copy()
    clust_id, mib_id, bcode = P["clust_id"], P["mib_id"], P["bcode"]
    floor = 0.0 if P.get("nonneg", False) else None

    import os
    enable_qinfer = os.environ.get("ELECTRO_QINFER_LEGAL", "0") == "1"
    
    if enable_qinfer:
        # Run legalization using standard separation graph
        x1, y1 = legalize(x.copy(), y.copy(), w, h, is_pre, floor=floor)
        for _ in range(P["rounds"]):
            x1, y1 = grouping_repair(x1, y1, w, h, clust_id, is_pre, floor=floor, bcode=bcode, mib_id=mib_id)
            x1, y1 = boundary_snap(x1, y1, w, h, bcode, is_pre, floor=floor, clust_id=clust_id, mib_id=mib_id)
        x1, y1 = grouping_repair(x1, y1, w, h, clust_id, is_pre, floor=floor, bcode=bcode, mib_id=mib_id)
        x1, y1 = remove_overlap(x1, y1, w, h, is_pre, nonneg=P.get("nonneg", False))
        
        # Run legalization using QinFer continuous optimization
        x2, y2 = legalize_qinfer(x.copy(), y.copy(), w, h, is_pre, floor=floor)
        for _ in range(P["rounds"]):
            x2, y2 = grouping_repair(x2, y2, w, h, clust_id, is_pre, floor=floor, bcode=bcode, mib_id=mib_id)
            x2, y2 = boundary_snap(x2, y2, w, h, bcode, is_pre, floor=floor, clust_id=clust_id, mib_id=mib_id)
        x2, y2 = grouping_repair(x2, y2, w, h, clust_id, is_pre, floor=floor, bcode=bcode, mib_id=mib_id)
        x2, y2 = remove_overlap(x2, y2, w, h, is_pre, nonneg=P.get("nonneg", False))
        
        wl1 = compute_hpwl_numpy(x1, y1, w, h, P)
        wl2 = compute_hpwl_numpy(x2, y2, w, h, P)
        print(f"[debug-qinfer] wl1 (standard): {wl1:.4f} | wl2 (qinfer): {wl2:.4f}")
        
        if wl2 < wl1:
            x, y = x2, y2
        else:
            x, y = x1, y1
    else:
        x, y = legalize(x, y, w, h, is_pre, floor=floor)
        for _ in range(P["rounds"]):
            x, y = grouping_repair(x, y, w, h, clust_id, is_pre, floor=floor, bcode=bcode, mib_id=mib_id)
            x, y = boundary_snap(x, y, w, h, bcode, is_pre, floor=floor, clust_id=clust_id, mib_id=mib_id)
        x, y = grouping_repair(x, y, w, h, clust_id, is_pre, floor=floor, bcode=bcode, mib_id=mib_id)
        x, y = remove_overlap(x, y, w, h, is_pre, nonneg=P.get("nonneg", False))
        
    return x, y, w, h


def run_start(seed, P):
    """One independent start: place + legalize + iterated min-displacement repair."""
    positions, _ = place(
        P["n"], P["area"], P["b2b"], P["p2b"], P["pins"], P["cons"], P["tp"],
        iters=P["iters"], lr=P["lr"], device=P["device"], seed=seed,
        init_centers=P["init"],
    )
    return post_place_repair(positions, P)


def run_start_diag(seed, P):
    """One independent start returning the layout and the convergence needs_extension flag."""
    positions, diag = place(
        P["n"], P["area"], P["b2b"], P["p2b"], P["pins"], P["cons"], P["tp"],
        iters=P["iters"], lr=P["lr"], device=P["device"], seed=seed,
        init_centers=P["init"],
    )
    layout = post_place_repair(positions, P)
    return layout, diag.get("needs_extension", False)


def run_start_jacobi_diag(seed, P):
    """One start using Jacobi initialization, returning layout and needs_extension (2026-07-18)."""
    import os
    old = os.environ.get("ELECTRO_INIT", "random")
    os.environ["ELECTRO_INIT"] = "jacobi"
    try:
        positions, diag = place(
            P["n"], P["area"], P["b2b"], P["p2b"], P["pins"], P["cons"], P["tp"],
            iters=P["iters"], lr=P["lr"], device=P["device"], seed=seed,
            init_centers=None,  # force random->jacobi, not ML
        )
        layout = post_place_repair(positions, P)
        return layout, diag.get("needs_extension", False)
    finally:
        os.environ["ELECTRO_INIT"] = old



def run_start_with_iters(seed, P, iters):
    """One start with custom placement iterations."""
    P_custom = P.copy()
    P_custom["iters"] = iters
    return run_start(seed, P_custom)


def boundary_pushpast_variant(start, P):
    """Boundary push-past portfolio candidate (2026-07-14): re-run the
    grouping/boundary repair loop on an already-repaired layout with
    `boundary_snap(push_past=True)`, returned as an ADDITIONAL candidate.
    solve() ranks it against the original (push_past=False) layout by the
    full cost proxy, so it is only ever chosen when net better -- push-past
    guarantees boundary contact for blocks the plain wall-scan couldn't
    place, at the cost of growing the bbox. Validated full-100 (on top of
    this file's MIB fix): Total Score 2.7138 baseline -> see merge test.
    Strictly additive: never call unconditionally (forcing push_past=True
    always regressed to 3.0668 in isolated pre-MIB-fix testing)."""
    x, y, w, h = start
    is_pre = P["is_pre"].copy()
    clust_id, mib_id, bcode = P["clust_id"], P["mib_id"], P["bcode"]
    nonneg = P.get("nonneg", False)
    floor = 0.0 if nonneg else None
    x, y = x.copy(), y.copy()
    for _ in range(P["rounds"]):
        x, y = grouping_repair(x, y, w, h, clust_id, is_pre, floor=floor, bcode=bcode, mib_id=mib_id)
        x, y = boundary_snap(x, y, w, h, bcode, is_pre, floor=floor, push_past=True, clust_id=clust_id, mib_id=mib_id)
    # Final grouping repair pass to clean up any swaps that perturbed clusters
    # Pass bcode so boundary-constrained blocks stay on their required walls
    x, y = grouping_repair(x, y, w, h, clust_id, is_pre, floor=floor, bcode=bcode, mib_id=mib_id)
    x, y = remove_overlap(x, y, w, h, is_pre, nonneg=nonneg)
    return x, y, w, h


def grouping_pushpast_variant(start, P):
    """Grouping push-past portfolio candidate: re-run the grouping/boundary
    repair loop on an already-repaired layout with `grouping_repair(push_past=True)`,
    returned as an ADDITIONAL candidate."""
    x, y, w, h = start
    is_pre = P["is_pre"].copy()
    clust_id, mib_id, bcode = P["clust_id"], P["mib_id"], P["bcode"]
    nonneg = P.get("nonneg", False)
    floor = 0.0 if nonneg else None
    x, y = x.copy(), y.copy()
    for _ in range(P["rounds"]):
        x, y = grouping_repair(x, y, w, h, clust_id, is_pre, floor=floor, push_past=True, bcode=bcode, mib_id=mib_id)
        x, y = boundary_snap(x, y, w, h, bcode, is_pre, floor=floor, clust_id=clust_id, mib_id=mib_id)
    # Final grouping repair pass to clean up any swaps that perturbed clusters
    # Pass bcode so boundary-constrained blocks stay on their required walls
    x, y = grouping_repair(x, y, w, h, clust_id, is_pre, floor=floor, bcode=bcode, mib_id=mib_id)
    x, y = remove_overlap(x, y, w, h, is_pre, nonneg=nonneg)
    return x, y, w, h
def boundary_wideswap_variant(start, P):
    """Boundary wide-swap portfolio candidate (2026-07-16): re-run the
    grouping/boundary repair loop with `boundary_snap(wide_swap=True)`,
    returned as an ADDITIONAL candidate. Widens the Strict Zero-Overlap Swap
    Pass's candidate pool (swap with any non-MIB block whose own boundary
    requirement still holds after the swap, not just fully-free blocks) and
    iterates it to convergence. Strictly additive: solve() ranks it against
    the plain layout by the full cost proxy."""
    x, y, w, h = start
    is_pre = P["is_pre"].copy()
    clust_id, mib_id, bcode = P["clust_id"], P["mib_id"], P["bcode"]
    nonneg = P.get("nonneg", False)
    floor = 0.0 if nonneg else None
    x, y = x.copy(), y.copy()
    for _ in range(P["rounds"]):
        x, y = grouping_repair(x, y, w, h, clust_id, is_pre, floor=floor, bcode=bcode, mib_id=mib_id)
        x, y = boundary_snap(x, y, w, h, bcode, is_pre, floor=floor, clust_id=clust_id, mib_id=mib_id, wide_swap=True)
    # Final grouping repair pass to clean up any swaps that perturbed clusters
    # Pass bcode so boundary-constrained blocks stay on their required walls
    x, y = grouping_repair(x, y, w, h, clust_id, is_pre, floor=floor, bcode=bcode, mib_id=mib_id)
    x, y = remove_overlap(x, y, w, h, is_pre, nonneg=nonneg)
    return x, y, w, h



def run_start_jacobi(seed, P):
    """One start using Jacobi graph-layout initialization (2026-07-18).

    Runs place() with ELECTRO_INIT=jacobi (neighbor-averaging warm start on the
    b2b connectivity graph, preplaced blocks as anchors).  Same iters as the
    normal start, only the init strategy differs.  Used as a portfolio candidate
    alongside the random-init start -- solve() keeps whichever is better.
    """
    import os
    old = os.environ.get("ELECTRO_INIT", "random")
    os.environ["ELECTRO_INIT"] = "jacobi"
    try:
        positions, _ = place(
            P["n"], P["area"], P["b2b"], P["p2b"], P["pins"], P["cons"], P["tp"],
            iters=P["iters"], lr=P["lr"], device=P["device"], seed=seed,
            init_centers=None,  # force random->jacobi, not ML
        )
        return post_place_repair(positions, P)
    finally:
        os.environ["ELECTRO_INIT"] = old


def run_start_jacobi_with_iters(seed, P, iters):
    """Jacobi-init start with custom placement iterations."""
    import os
    old = os.environ.get("ELECTRO_INIT", "random")
    os.environ["ELECTRO_INIT"] = "jacobi"
    try:
        P_custom = P.copy()
        P_custom["iters"] = iters
        positions, _ = place(
            P_custom["n"], P_custom["area"], P_custom["b2b"], P_custom["p2b"],
            P_custom["pins"], P_custom["cons"], P_custom["tp"],
            iters=P_custom["iters"], lr=P_custom["lr"], device=P_custom["device"],
            seed=seed, init_centers=None,
        )
        return post_place_repair(positions, P_custom)
    finally:
        os.environ["ELECTRO_INIT"] = old


def pool_init(threads=1):
    """Give each worker its share of cores (cores/nproc threads).  Threads are set
    AFTER the fork, so the parent never holds a live OpenMP pool across fork."""
    try:
        import torch
        torch.set_num_threads(max(1, int(threads)))
    except Exception:
        pass


def seed_worker(seed):
    return run_start(seed, WORK)


def seed_worker_diag(seed):
    return run_start_diag(seed, WORK)

