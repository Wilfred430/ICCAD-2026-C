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
from legalize import legalize, remove_overlap
from soft_repair import boundary_snap, grouping_repair

WORK = None   # per-case inputs, set by the parent before forking the pool


def run_start(seed, P):
    """One independent start: place + legalize + iterated min-displacement repair."""
    positions, _ = place(
        P["n"], P["area"], P["b2b"], P["p2b"], P["pins"], P["cons"], P["tp"],
        iters=P["iters"], lr=P["lr"], device=P["device"], seed=seed,
        init_centers=P["init"],
    )
    x = np.array([p[0] for p in positions], dtype=float)
    y = np.array([p[1] for p in positions], dtype=float)
    w = np.array([p[2] for p in positions], dtype=float)
    h = np.array([p[3] for p in positions], dtype=float)
    is_pre, clust_id, bcode = P["is_pre"], P["clust_id"], P["bcode"]
    # nonneg keeps the WHOLE chain (legalize + both repairs + final cleanup)
    # floored at 0, so blocks never drift far below the wall and get shoved back
    # -- the incremental floor that makes first-quadrant containment cheap rather
    # than exploding (a post-hoc floor-only-at-the-end shove cascades the legalizer).
    floor = 0.0 if P.get("nonneg", False) else None
    x, y = legalize(x, y, w, h, is_pre, floor=floor)
    for _ in range(P["rounds"]):
        x, y = grouping_repair(x, y, w, h, clust_id, is_pre, floor=floor)
        x, y = boundary_snap(x, y, w, h, bcode, is_pre, floor=floor)
    # final hard-feasibility net; nonneg also enforces the x=0/y=0 canvas walls
    x, y = remove_overlap(x, y, w, h, is_pre, nonneg=P.get("nonneg", False))
    return x, y, w, h


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
