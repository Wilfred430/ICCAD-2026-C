#!/usr/bin/env python3
"""Contest entry point: analytical placement + legalization + soft-constraint repair.

Per case:
    analytical global placement  (WL + spreading + grouping + boundary terms)
      -> legalize to exactly zero overlap
      -> grouping_repair  (abut isolated cluster members)
      -> boundary_snap    (slide boundary blocks onto bbox edges)
      -> (x, y, w, h)

Hard constraints by construction: no overlap, soft-block area exact, MIB same
shape, fixed dims locked, preplaced pinned.  Soft constraints reduced by the
analytical penalties + the two repair passes.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
from typing import List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --- SUBMISSION DEFAULT: first-quadrant containment ------------------------
# The contest harness imports this module and calls solve() with no env vars.
# We want the SUBMITTED behaviour to keep every block in the first quadrant
# (x, y >= 0, the contest's origin convention) rather than the negative-coord-
# drifted layout that scores marginally better locally.  These two knobs turn
# on the in-optimization lower-wall clamp (CLAMP) and the floor-aware
# legalize+repair chain (NONNEG), which together guarantee non-negative output.
# Both are still overridable: e.g. ELECTRO_CLAMP=0 ELECTRO_NONNEG=0 reproduces
# the lower-cost (but negative-coord) configuration.
os.environ.setdefault("ELECTRO_CLAMP", "1")
os.environ.setdefault("ELECTRO_NONNEG", "1")

from iccad2026_evaluate import FloorplanOptimizer
from legalize import verify_overlap
from soft_repair import soft_violation_counts
import electro_parallel


def _edges_np(b2b, p2b, pins, n):
    """Pull the valid (non-padding) edges/pins out as numpy for HPWL scoring."""
    def valid(t):
        if t is None or t.numel() == 0:
            return None
        a = t.cpu().numpy()
        a = a[a[:, 0] != -1]
        return a if len(a) else None
    return valid(b2b), valid(p2b), valid(pins)


def _hpwl(cx, cy, eb, ep, pv):
    """Contest HPWL (center-to-center Manhattan, b2b + p2b) for seed ranking."""
    n = len(cx)
    wl = 0.0
    if eb is not None:
        i = np.clip(eb[:, 0].astype(int), 0, n - 1)
        j = np.clip(eb[:, 1].astype(int), 0, n - 1)
        wl += float((eb[:, 2] * (np.abs(cx[i] - cx[j]) + np.abs(cy[i] - cy[j]))).sum())
    if ep is not None and pv is not None and len(pv):
        pi = np.clip(ep[:, 0].astype(int), 0, len(pv) - 1)
        bi = np.clip(ep[:, 1].astype(int), 0, n - 1)
        wl += float((ep[:, 2] * (np.abs(pv[pi, 0] - cx[bi]) + np.abs(pv[pi, 1] - cy[bi]))).sum())
    return wl


class MyOptimizer(FloorplanOptimizer):
    def __init__(self, verbose: bool = False):
        super().__init__(verbose=verbose)
        self.iters = int(os.environ.get("ELECTRO_ITERS", "600"))
        # CPU by default.  This is a SMALL problem (n<=120) run for 600 sequential
        # iterations of tiny ops, so a GPU is ~6x SLOWER here (kernel-launch
        # overhead dominates) -- and on a laptop it would run on the display GPU
        # and freeze the screen.  GPU only pays off with seed-BATCHING (TODO);
        # opt in then with ELECTRO_DEVICE=cuda.
        self.device = os.environ.get("ELECTRO_DEVICE", "cpu")
        self.lr = float(os.environ.get("ELECTRO_LR", "0.02"))
        # Rounds of (grouping_repair -> boundary_snap).  Each repair is now
        # min-displacement, but the two fight over blocks that are both a cluster
        # member AND a boundary block; one round leaves boundary blocks freshly
        # snapped off their cluster.  Iterating lets them settle: full-100 score
        # 3.733 (1 round) -> 3.568 (2) -> 3.545 (3) -> 3.545 (4, saturated).
        self.repair_rounds = int(os.environ.get("ELECTRO_REPAIR_ROUNDS", "3"))
        # Multi-start: keep the best of N seeds.  More seeds -> lower quality score
        # (subset 1->2.54, 3->2.16, 8->2.07 with ML) but ~Nx runtime.  The contest
        # runtime penalty (R^0.3, UNCAPPED on the slow side) usually makes seeds=1
        # win the runtime-adjusted total unless the field's median runtime is very
        # high.  Default 1 (fast); raise it when runtime is cheap / median is high.
        self.seeds = int(os.environ.get("ELECTRO_SEEDS", "1"))
        # Multi-start seeds in parallel fork processes.  OFF by default: on CPU
        # the place loop is dispatch/OpenMP-bound, and forked workers oversubscribe
        # the OpenMP runtime (N workers x M threads); the real seed-batching speedup
        # belongs on the GPU.  ELECTRO_PARALLEL=1 to opt in.
        self.parallel = os.environ.get("ELECTRO_PARALLEL", "0") == "1"
        # ML warm-start: use the trained FloorplanTransformer's predicted block
        # centers as the analytical placer's init (instead of random).  Lazily
        # loaded; falls back to random init if weights/model are unavailable.
        self.ml_init = os.environ.get("ELECTRO_ML_INIT", "1") == "1"
        self._predictor = None

    def solve(
        self,
        block_count: int,
        area_targets: torch.Tensor,
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
        constraints: torch.Tensor,
        target_positions: Optional[torch.Tensor] = None,
    ) -> List[Tuple[float, float, float, float]]:
        if block_count == 0:
            return []
        t0 = time.time()

        cons = constraints[:block_count].cpu().numpy()
        is_pre = (cons[:, 1] != 0).astype(bool)
        mib_id = cons[:, 2].astype(int) if cons.shape[1] > 2 else np.zeros(block_count, int)
        clust_id = cons[:, 3].astype(int) if cons.shape[1] > 3 else np.zeros(block_count, int)
        bcode = cons[:, 4].astype(int) if cons.shape[1] > 4 else np.zeros(block_count, int)
        eb, ep, pv = _edges_np(b2b_connectivity, p2b_connectivity, pins_pos, block_count)

        # ML warm-start only helps WITH multi-start (jitter around the prediction);
        # a single pure-ML start is worse than a single random start, so for
        # seeds==1 we use random init.
        use_ml = self.ml_init and self.seeds > 1
        init_centers = self._ml_centers(
            block_count, area_targets, constraints, target_positions,
            b2b_connectivity, p2b_connectivity, pins_pos) if use_ml else None

        nseeds = max(1, self.seeds)
        P = {
            "n": block_count, "area": area_targets, "b2b": b2b_connectivity,
            "p2b": p2b_connectivity, "pins": pins_pos, "cons": constraints,
            "tp": target_positions, "iters": self.iters, "lr": self.lr,
            "device": self.device, "init": init_centers, "is_pre": is_pre,
            "clust_id": clust_id, "bcode": bcode, "rounds": self.repair_rounds,
            "nonneg": os.environ.get("ELECTRO_NONNEG", "0") == "1",
        }

        # Multi-start: each seed lands in a different basin; run them in parallel
        # processes (independent -> embarrassingly parallel) and keep the
        # lowest-cost-proxy result.  Each worker is single-threaded by default:
        # the parent runs the ML model (initialising the OpenMP pool), so forked
        # workers that spin up >1 thread can deadlock (libgomp fork hazard).  On a
        # 48-core box this is fine -- run MANY single-thread seeds in parallel at
        # the same wall-clock.  ELECTRO_WORKER_THREADS>1 opts into multi-thread
        # workers (only safe if the parent never touched OpenMP, e.g. ML_INIT=0).
        # The solver can't see the GT baseline, so we rank by
        # exp(2*V_rel)*(hpwl/mean + area/mean), mirroring contest cost.  CUDA can't
        # be forked, so on GPU we run seeds sequentially (fast anyway).
        starts = None
        if self.parallel and nseeds > 1 and self.device == "cpu":
            try:
                electro_parallel.WORK = P
                nproc = min(nseeds, os.cpu_count() or 1)
                threads = int(os.environ.get("ELECTRO_WORKER_THREADS", "1"))
                ctx = mp.get_context("fork")
                with ctx.Pool(nproc, initializer=electro_parallel.pool_init,
                              initargs=(threads,)) as pool:
                    starts = pool.map(electro_parallel.seed_worker, range(nseeds))
            except Exception as e:
                sys.stderr.write(f"[electro] parallel failed ({e}); sequential\n")
                starts = None
            finally:
                electro_parallel.WORK = None
        if starts is None:
            starts = [electro_parallel.run_start(s, P) for s in range(nseeds)]

        cands = []
        for (x, y, w, h) in starts:
            vb, vg, vm, nsoft = soft_violation_counts(x, y, w, h, bcode, clust_id, mib_id)
            vrel = (vb + vg + vm) / nsoft
            hpwl = _hpwl(x + 0.5 * w, y + 0.5 * h, eb, ep, pv)
            area = (float((x + w).max() - x.min()) * float((y + h).max() - y.min()))
            cands.append((x, y, w, h, vrel, hpwl, area))

        mh = np.mean([c[5] for c in cands]) or 1.0
        ma = np.mean([c[6] for c in cands]) or 1.0
        x, y, w, h, vrel, hpwl, area = min(
            cands, key=lambda c: np.exp(2.0 * c[4]) * (c[5] / mh + c[6] / ma))

        ov = verify_overlap(x, y, w, h)
        soft = ((cons[:, 0] == 0) & (cons[:, 1] == 0))
        at = area_targets[:block_count].cpu().numpy()
        drift = np.abs(w * h - at) / np.maximum(at, 1e-9)
        max_drift = float(drift[soft].max()) if soft.any() else 0.0
        dt = time.time() - t0
        sys.stderr.write(
            f"[electro] n={block_count} t={dt:.3f}s seeds={self.seeds} "
            f"resid_overlap={ov:.3g} max_area_drift={max_drift:.4f} "
            f"V_rel={vrel:.3f}\n"
        )
        return [(float(x[i]), float(y[i]), float(w[i]), float(h[i]))
                for i in range(block_count)]

    def _ml_centers(self, block_count, area_targets, constraints, target_positions,
                    b2b, p2b, pins):
        """Predicted block centers [N,2] (raw coords) from the trained model, or
        None if the model/weights are unavailable or the case is too big."""
        if self._predictor is None:
            try:
                # Find the `ml/` package: explicit env override, then bundled next
                # to this file (submission layout), then the PARENT directory (the
                # dev-tree layout, where ml/ sits beside electro/).  All paths are
                # relative to __file__ -- no machine-specific absolute paths.
                here = os.path.dirname(os.path.abspath(__file__))
                ml_dir = None
                for d in (os.environ.get("ELECTRO_ML_DIR"), here,
                          os.path.dirname(here)):
                    if d and os.path.isdir(os.path.join(d, "ml")):
                        ml_dir = d
                        break
                if ml_dir is None:
                    raise FileNotFoundError("ml/ package not found")
                if ml_dir not in sys.path:
                    sys.path.insert(0, ml_dir)
                from ml.predict import Predictor
                wts = os.environ.get(
                    "ELECTRO_ML_WEIGHTS",
                    os.path.join(ml_dir, "ml", "weights", "floorplan_v2.pt"))
                self._predictor = Predictor(wts, device=self.device)
            except Exception as e:
                sys.stderr.write(f"[electro] ML init unavailable: {e}\n")
                self._predictor = False
        if not self._predictor:
            return None
        try:
            pred = self._predictor.predict(
                block_count, area_targets, constraints, target_positions,
                b2b, p2b, pins)
            if pred is None:
                return None
            return torch.tensor([[p[0], p[1]] for p in pred.positions],
                                dtype=torch.float32)
        except Exception as e:
            sys.stderr.write(f"[electro] ML predict failed: {e}\n")
            return None

