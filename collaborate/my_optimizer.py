#!/usr/bin/env python3
"""
my_optimizer.py — ICCAD 2026 FloorSet Challenge submission.

This file is the contest-compliant wrapper around our C++ floorplanner.
The contest framework (iccad2026_evaluate.py) imports this module, finds
the MyOptimizer class (subclass of FloorplanOptimizer), instantiates it,
and calls optimizer.solve() once per test case.

Per-case pipeline inside solve():
    tensors  →  *.txt  →  ./floorplanner subprocess  →  *.sol  →  positions

----------------------------------------------------------------------
INSTALLATION
----------------------------------------------------------------------

1. Build the C++ solver (in our floorplanner repo):
       cd <floorplanner repo>
       make
   This produces an executable named `floorplanner`.

2. Copy this file into the contest folder:
       cp my_optimizer.py /path/to/FloorSet/iccad2026contest/

3. Tell us where the binary is — choose one:
       a) Copy the binary next to my_optimizer.py inside iccad2026contest/.
       b) Set the FLOORPLANNER_BIN environment variable.

4. Validate, then evaluate (from FloorSet/iccad2026contest/):
       python iccad2026_evaluate.py --validate my_optimizer.py
       python iccad2026_evaluate.py --evaluate my_optimizer.py --test-id 0
       python iccad2026_evaluate.py --evaluate my_optimizer.py

Environment variables (all optional):
    FLOORPLANNER_BIN     path to the C++ binary  (default: ./floorplanner)
    FLOORPLANNER_THREADS thread count for the solver        (default: 8)
    FLOORPLANNER_TIME    per-case time budget; expression in 'n'
                         where n = block count               (default: '5+0.5*n')
    FLOORPLANNER_SEED    base RNG seed; case i uses seed+i   (default: 1)
    FLOORPLANNER_KEEP    if set to '1', keep intermediate files for inspection
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import torch

# Contest framework — this import works because, when the framework
# loads my_optimizer.py via importlib, it has already added the
# iccad2026contest/ directory and its parent to sys.path (per
# iccad2026_evaluate.py).
from iccad2026_evaluate import FloorplanOptimizer


# =============================================================================
# Boundary code conversion
# =============================================================================
#
# Official encoding (from iccad2026_evaluate.py boundary check):
#     code is a BITMASK — 1=left, 2=right, 4=top, 8=bottom
#     corners are sums:   5=TL (1+4), 9=BL (1+8), 6=TR (2+4), 10=BR (2+8)
#     code == 0           means no boundary constraint
#
# Our C++ enum (BoundaryEdge in include/types.hpp):
#     -1=none, 0=L, 1=R, 2=B, 3=T, 4=BL, 5=BR, 6=TL, 7=TR
# =============================================================================

_BOUNDARY_BITMASK_TO_ENUM = {
    0:  -1,  # no constraint
    1:   0,  # left
    2:   1,  # right
    4:   3,  # top
    8:   2,  # bottom
    5:   6,  # left + top   = TL
    9:   4,  # left + bottom = BL
    6:   7,  # right + top   = TR
    10:  5,  # right + bottom = BR
}


def _convert_boundary(code) -> int:
    """Bitmask boundary code -> our enum value."""
    return _BOUNDARY_BITMASK_TO_ENUM.get(int(code), -1)


# =============================================================================
# Tensor → text format conversion
# =============================================================================

def _filter_padding(t: torch.Tensor, pad_col: int = 0, pad_value: float = -1.0) -> torch.Tensor:
    """Drop rows where the given column equals pad_value (typically -1)."""
    if t is None or t.numel() == 0:
        return t
    keep = t[:, pad_col] != pad_value
    return t[keep]


def _write_txt(
    out_path: Path,
    block_count: int,
    area_targets: torch.Tensor,
    b2b_connectivity: torch.Tensor,
    p2b_connectivity: torch.Tensor,
    pins_pos: torch.Tensor,
    constraints: torch.Tensor,
    target_positions: Optional[torch.Tensor],
) -> None:
    """Convert one test case (tensors) to our C++ solver's text format."""

    # Number of valid pins (pins_pos is -1 padded)
    if pins_pos is None or pins_pos.numel() == 0:
        n_pins = 0
        pins = []
    else:
        n_pins = int((pins_pos[:, 0] != -1).sum().item())
        pins = pins_pos[:n_pins].tolist()

    # b2b / p2b: drop -1 padded rows
    valid_b2b = _filter_padding(b2b_connectivity)
    valid_p2b = _filter_padding(p2b_connectivity)

    # Group blocks by mib_id (col 2) and cluster_id (col 3)
    # Both use 1-indexed group IDs; 0 = not in any group.
    mib_groups: dict[int, list[int]] = {}
    cluster_groups: dict[int, list[int]] = {}
    for i in range(block_count):
        mid = int(constraints[i, 2].item()) if constraints.shape[1] > 2 else 0
        cid = int(constraints[i, 3].item()) if constraints.shape[1] > 3 else 0
        if mid > 0:
            mib_groups.setdefault(mid, []).append(i)
        if cid > 0:
            cluster_groups.setdefault(cid, []).append(i)

    # Stable order so the same case always produces the same .txt
    sorted_mibs = [v for _, v in sorted(mib_groups.items())]
    sorted_clusters = [v for _, v in sorted(cluster_groups.items())]

    # Per-block lookup of "which mib group am I in" by ordinal position
    # (matches the order we'll write them in)
    block_to_mib_ordinal = {}
    for ord_, group in enumerate(sorted_mibs):
        for b in group:
            block_to_mib_ordinal[b] = ord_
    block_to_cluster_ordinal = {}
    for ord_, group in enumerate(sorted_clusters):
        for b in group:
            block_to_cluster_ordinal[b] = ord_

    # ---- Baseline estimation ------------------------------------------------
    #
    # The C++ solver normalises area / HPWL by these baselines so that SA's
    # sa_cost stays in a reasonable scale (~1-10 per term).  Without them
    # cost.cpp falls back to abase = hbase = 1.0 and raw area_bbox (~50000)
    # dominates -- T1 calibrates to ~50000 instead of ~5, and Metropolis's
    # exp(-Δ/T) collapses to ~1 (SA accepts everything = random walk).
    #
    # Heuristics:
    #   baseline_area = sum(block_area) * 1.10  (10% whitespace headroom)
    #   baseline_hpwl = total_net_weight * (sqrt(area) * 0.5)
    #                   (assume each net spans ~half the bbox side)
    total_area = float(area_targets[:block_count].sum().item()) if block_count > 0 else 0.0
    baseline_area = total_area * 1.10 if total_area > 0 else 1.0

    side = math.sqrt(baseline_area) if baseline_area > 0 else 1.0
    avg_edge = side * 0.5

    total_net_weight = 0.0
    if valid_b2b is not None and valid_b2b.numel() > 0:
        total_net_weight += float(valid_b2b[:, 2].sum().item())
    if valid_p2b is not None and valid_p2b.numel() > 0:
        total_net_weight += float(valid_p2b[:, 2].sum().item())
    baseline_hpwl = (total_net_weight * avg_edge) if total_net_weight > 0 else side

    with open(out_path, "w") as f:
        f.write("# emitted by my_optimizer.py for the ICCAD 2026 contest\n")
        f.write(f"N_BLOCKS    {block_count}\n")
        f.write(f"N_TERMINALS {n_pins}\n")
        # Baselines: the framework computes its OWN baselines from ground
        # truth for the final contest_cost score, so the values here ONLY
        # affect the solver's internal SA cost function (which needs them
        # to keep magnitudes in a sane scale -- see comment above).
        f.write(f"BASELINE_HPWL {baseline_hpwl:.6f}\n")
        f.write(f"BASELINE_AREA {baseline_area:.6f}\n")
        f.write("OUTLINE 0.0 0.0\n")

        if n_pins > 0:
            f.write("TERMINALS\n")
            for i, (px, py) in enumerate(pins):
                f.write(f"{i} {float(px):.10f} {float(py):.10f}\n")

        f.write("BLOCKS\n")
        for i in range(block_count):
            ai = float(area_targets[i].item())
            isf = int(constraints[i, 0].item() != 0)
            isp = int(constraints[i, 1].item() != 0)

            # target_positions columns are (x, y, w, h). Defaults to -1.
            # For preplaced blocks all four are set; for fixed-shape blocks
            # only w and h are set; for soft blocks all four are -1.
            if target_positions is not None and target_positions.numel() > 0:
                tx = float(target_positions[i, 0].item())
                ty = float(target_positions[i, 1].item())
                tw = float(target_positions[i, 2].item())
                th = float(target_positions[i, 3].item())
            else:
                tx = ty = tw = th = -1.0

            # Our text format expects (w_in, h_in, x_in, y_in) for the
            # locked geometry of fixed/preplaced blocks, and zeros for
            # soft blocks.
            if isp:
                wi, hi, xi, yi = tw, th, tx, ty
            elif isf:
                wi, hi = tw, th
                xi, yi = 0.0, 0.0
            else:
                wi = hi = xi = yi = 0.0

            mid_ord = block_to_mib_ordinal.get(i, -1)
            cid_ord = block_to_cluster_ordinal.get(i, -1)

            be = _convert_boundary(
                constraints[i, 4].item() if constraints.shape[1] > 4 else 0)

            # ar_min, ar_max are an SA neighbourhood hint, not a hard
            # constraint. v9 doesn't constrain aspect ratio; we use a
            # permissive band that's wide enough to find any reasonable
            # shape but narrow enough that SA doesn't waste time on
            # 100:1 slivers.
            f.write(
                f"{i} {ai:.10f} {isf} {isp} "
                f"{wi:.10f} {hi:.10f} {xi:.10f} {yi:.10f} "
                f"{mid_ord} {cid_ord} {be} 0.10 10.00\n"
            )

        # Edges
        n_b2b = int(valid_b2b.shape[0]) if valid_b2b.numel() > 0 else 0
        f.write(f"B2B {n_b2b}\n")
        for k in range(n_b2b):
            a = int(valid_b2b[k, 0].item())
            b = int(valid_b2b[k, 1].item())
            w = float(valid_b2b[k, 2].item())
            f.write(f"{a} {b} {w:.10f}\n")

        n_p2b = int(valid_p2b.shape[0]) if valid_p2b.numel() > 0 else 0
        f.write(f"P2B {n_p2b}\n")
        for k in range(n_p2b):
            t = int(valid_p2b[k, 0].item())
            b = int(valid_p2b[k, 1].item())
            w = float(valid_p2b[k, 2].item())
            f.write(f"{t} {b} {w:.10f}\n")

        # Soft groups (note: v9 'cluster' is what we call "grouping")
        f.write(f"GROUPS {len(sorted_clusters)}\n")
        for g in sorted_clusters:
            f.write(f"{len(g)} " + " ".join(str(b) for b in g) + "\n")
        f.write(f"MIB {len(sorted_mibs)}\n")
        for g in sorted_mibs:
            f.write(f"{len(g)} " + " ".join(str(b) for b in g) + "\n")

        f.write("END\n")


# =============================================================================
# .sol parsing
# =============================================================================

def _parse_sol(sol_path: Path, block_count: int,
               area_targets: torch.Tensor) -> List[Tuple[float, float, float, float]]:
    """
    Parse our solver's .sol file into the contest's expected
    List[(x, y, w, h)] of length block_count.

    If a row is missing in the .sol (shouldn't happen with a healthy
    solver, but be defensive), fill with a square at the origin sized
    to the block's area target so the framework can still compute a
    score (it'll likely be infeasible due to overlap, scored as M=10).
    """
    rows: dict[int, Tuple[float, float, float, float]] = {}
    if sol_path.exists():
        with open(sol_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("N_BLOCKS"):
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    bid = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                except ValueError:
                    continue
                rows[bid] = (x, y, w, h)

    out: List[Tuple[float, float, float, float]] = []
    for i in range(block_count):
        if i in rows:
            out.append(rows[i])
        else:
            ai = float(area_targets[i].item()) if i < area_targets.numel() else 1.0
            s = math.sqrt(max(ai, 1e-9))
            out.append((0.0, 0.0, s, s))
    return out


# =============================================================================
# MyOptimizer class — what the contest framework loads
# =============================================================================

class MyOptimizer(FloorplanOptimizer):
    """
    Wraps our C++ floorplanner as a contest-compliant FloorplanOptimizer.

    See module docstring for installation and environment variables.
    """

    def __init__(self, verbose: bool = False):
        super().__init__(verbose=verbose)

        # Resolve the binary. Prefer FLOORPLANNER_BIN env var; else look
        # next to this file; else fall back to the working dir.
        env_bin = os.environ.get("FLOORPLANNER_BIN")
        if env_bin:
            self.binary = Path(env_bin).resolve()
        else:
            here = Path(__file__).resolve().parent
            cand = here / "floorplanner"
            self.binary = cand if cand.exists() else Path("./floorplanner").resolve()

        self.threads = int(os.environ.get("FLOORPLANNER_THREADS", "8"))
        self.time_expr = os.environ.get("FLOORPLANNER_TIME", "8+1.0*n")
        self.seed = int(os.environ.get("FLOORPLANNER_SEED", "1"))
        self.keep = os.environ.get("FLOORPLANNER_KEEP", "0") == "1"

        # Feasibility-triggered budget escalation.  When the solver returns
        # rc=4 (ran fine but the best solution is still infeasible -> cost M=10),
        # re-run that one case with a larger budget instead of shipping a
        # cost-10 result.  Keeps the aggressive short budget on the easy cases
        # while rescuing the few large/dense ones.  Disable with
        # FLOORPLANNER_ESCALATE=0.
        self.escalate = os.environ.get("FLOORPLANNER_ESCALATE", "1") != "0"

        # Workdir survives across solve() calls so we can inspect after a run
        self.workdir = Path(tempfile.mkdtemp(prefix="my_optimizer_"))
        self._call_idx = 0

        if self.verbose:
            print(f"[my_optimizer] binary  = {self.binary}", file=sys.stderr)
            print(f"[my_optimizer] threads = {self.threads}", file=sys.stderr)
            print(f"[my_optimizer] time    = {self.time_expr}", file=sys.stderr)
            print(f"[my_optimizer] workdir = {self.workdir}", file=sys.stderr)

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
        # Slice off any padding the dataset may have applied
        area = area_targets[:block_count]
        cons = constraints[:block_count]
        tp = target_positions[:block_count] if target_positions is not None else None

        # Per-case time budget — expression in n. Safety bounds applied.
        try:
            time_s = float(eval(self.time_expr, {"__builtins__": {}}, {"n": block_count}))
        except Exception:
            time_s = 8.0 + 1.0 * block_count
        time_s = max(1.0, min(time_s, 300.0))

        idx = self._call_idx
        self._call_idx += 1
        in_txt = self.workdir / f"case_{idx:03d}.txt"
        out_sol = self.workdir / f"case_{idx:03d}.sol"

        _write_txt(in_txt, block_count, area, b2b_connectivity,
                   p2b_connectivity, pins_pos, cons, tp)

        if self.verbose:
            print(f"[my_optimizer] case {idx}: n={block_count} budget={time_s:.1f}s",
                  file=sys.stderr)

        positions = self._run_solver(in_txt, out_sol, block_count, time_s, idx, area)

        if not self.keep:
            in_txt.unlink(missing_ok=True)
            out_sol.unlink(missing_ok=True)

        return positions

    def _run_solver(self, in_txt: Path, out_sol: Path, block_count: int,
                    base_time_s: float, idx: int,
                    area: torch.Tensor) -> List[Tuple[float, float, float, float]]:
        """Run the C++ solver, escalating the time budget on infeasibility.

        The binary returns rc=0 (feasible), rc=4 (ran fine but best solution
        infeasible -> contest cost M=10), or another code on a real error.
        Because an infeasible case costs the maximum penalty, on the few hard
        cases it is always worth re-running with more time; the cheap cases
        keep the fast budget and never escalate.  The same input file (and any
        WARM_POSITIONS hint already appended to it) is reused on every attempt.
        """
        # Build the escalation ladder.  Rung 1 is the configured budget; rung 2
        # is a "safe" budget known to legalise the large/dense cases.
        budgets = [base_time_s]
        if self.escalate:
            floor = float(os.environ.get("FLOORPLANNER_ESCALATE_FLOOR", "60"))
            cap   = float(os.environ.get("FLOORPLANNER_ESCALATE_CAP", "90"))
            safe  = min(cap, max(floor, 0.6 * block_count))
            if safe > base_time_s * 1.3:
                budgets.append(safe)

        best_effort: Optional[List[Tuple[float, float, float, float]]] = None
        for attempt, budget in enumerate(budgets):
            # Vary the seed across escalation rungs so each retry is an
            # INDEPENDENT draw, not just more time on the same (possibly
            # unlucky) 8-thread population.  Feasibility on large/dense
            # anchored cases is strongly seed-dependent, so independent rungs
            # multiply the odds of landing a feasible arrangement.
            cmd = [
                str(self.binary), str(in_txt), str(out_sol),
                "--time",    f"{budget:g}",
                "--threads", str(self.threads),
                "--seed",    str(self.seed + idx + attempt * 7919),
            ]
            try:
                res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            except FileNotFoundError:
                sys.stderr.write(
                    f"[my_optimizer] solver binary not found: {self.binary}\n"
                    f"  Set FLOORPLANNER_BIN, or place the binary next to my_optimizer.py.\n"
                )
                return self._fallback(block_count, area)

            if res.returncode == 0:
                return _parse_sol(out_sol, block_count, area)          # feasible

            if res.returncode == 4:
                # Infeasible, but a real best-effort layout was written.  Keep
                # it and escalate to a larger budget if one remains.
                best_effort = _parse_sol(out_sol, block_count, area)
                if self.verbose:
                    nxt = (f"; escalating to {budgets[attempt + 1]:g}s"
                           if attempt + 1 < len(budgets) else "")
                    sys.stderr.write(
                        f"[my_optimizer] case {idx}: infeasible at {budget:g}s{nxt}\n"
                    )
                continue

            sys.stderr.write(
                f"[my_optimizer] case {idx}: solver exited with rc={res.returncode}\n"
                + (res.stderr[-2000:] if res.stderr else "") + "\n"
            )
            return self._fallback(block_count, area)

        # Exhausted the ladder while still infeasible: return the real layout
        # (cost M either way, but a true placement is never worse than the
        # overlapping-squares emergency fallback).
        return best_effort if best_effort is not None else self._fallback(block_count, area)

    @staticmethod
    def _fallback(block_count: int, area: torch.Tensor):
        """Emergency placement when the solver fails — guaranteed
        infeasible (overlap), so the framework will score this case as
        M=10. Better than crashing the whole run."""
        return [
            (0.0, 0.0,
             math.sqrt(max(float(area[i].item()), 1e-9)),
             math.sqrt(max(float(area[i].item()), 1e-9)))
            for i in range(block_count)
        ]

    def __del__(self):
        if not getattr(self, "keep", False):
            try:
                shutil.rmtree(self.workdir, ignore_errors=True)
            except Exception:
                pass
