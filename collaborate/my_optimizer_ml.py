#!/usr/bin/env python3
"""my_optimizer_ml.py -- ML-augmented FloorplanOptimizer.

This is a drop-in replacement for my_optimizer.py.  It extends the
baseline `MyOptimizer` (which wraps the C++ SA solver) with a
front-loaded ML predictor that gives the C++ solver a warm-start hint.

Pipeline
--------

    case tensors                            ┌─ Predictor (ml/predict.py) ──┐
        │                                   │   Graph-Transformer          │
        ▼                                   │   inference (~50 ms)         │
   ┌─────────────────────┐                  └──────────────┬───────────────┘
   │  baseline _write_txt│                                 │
   │  (BLOCKS, NETS, …)  │ <───── append WARM_POSITIONS ───┘
   └─────────┬───────────┘
             ▼
        case_NNN.txt
             │
             ▼
   ┌───────────────────────┐
   │  C++ floorplanner SA  │ ← reads WARM_POSITIONS if compiled with the
   │  --warm-from-file     │   companion patch (see ML_FLOORPLAN.md §4);
   └─────────┬─────────────┘   otherwise silently ignores it
             ▼
   case_NNN.sol -> positions returned to the contest framework

Configuration via environment variables (in addition to MyOptimizer's):

    FLOORPLANNER_ML_WEIGHTS   path to ml/weights/floorplan_v1.pt
                              (default: ml/weights/floorplan_v1.pt next
                               to this file).  If file missing, the ML
                               step is silently skipped and the original
                               C++-only pipeline runs.

    FLOORPLANNER_ML_DEVICE    cpu / cuda (default cpu).  Inference on
                              CPU is plenty fast (~50 ms per case for
                              n ≤ 128 blocks).

    FLOORPLANNER_ML_VERBOSE   1 to print predictor diagnostics.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch

# Import everything from the baseline -- we only ADD a warm-start step;
# nothing else changes.
from my_optimizer import (
    MyOptimizer as _BaselineOptimizer,
    _write_txt,
    _parse_sol,
)

try:
    from ml.predict import Predictor, Prediction
except ImportError:
    Predictor = None
    Prediction = None


# ---------------------------------------------------------------------------
# Where to dump the ML prediction so the C++ side (or a future warm-start
# hook) can pick it up.  We append a new section to the .txt input that
# the current C++ parser silently skips (parser.cpp's dispatch loop has
# no `else: error` branch on unknown tokens).
# ---------------------------------------------------------------------------

def _append_warm_positions(
    txt_path: Path,
    prediction: "Prediction",
    block_count: int,
) -> None:
    """Append a WARM_POSITIONS section to an existing case_NNN.txt.

    Format:
        WARM_POSITIONS  <N>
        <block_id>  <cx>  <cy>  <w>  <h>
        ...
        WARM_PRIORITY  <N>
        <block_id_in_insertion_order>  ...
    """
    if prediction is None:
        return
    with open(txt_path, "a") as f:
        f.write(f"WARM_POSITIONS {block_count}\n")
        for i, (cx, cy, w, h) in enumerate(prediction.positions):
            f.write(f"{i} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        f.write(f"WARM_PRIORITY {len(prediction.priority)}\n")
        f.write(" ".join(str(b) for b in prediction.priority) + "\n")


# ---------------------------------------------------------------------------
# Subclass
# ---------------------------------------------------------------------------

class MyOptimizer(_BaselineOptimizer):
    """ML-augmented optimizer.

    Falls back to baseline if:
        * `ml/` package not importable
        * predictor weights file missing
        * any exception thrown during predict
    """

    def __init__(self, verbose: bool = False):
        super().__init__(verbose=verbose)

        # Locate predictor weights.
        default_weights = Path(__file__).resolve().parent / "ml" / "weights" / "floorplan_v1.pt"
        self.ml_weights = Path(os.environ.get("FLOORPLANNER_ML_WEIGHTS", str(default_weights)))
        self.ml_device  = os.environ.get("FLOORPLANNER_ML_DEVICE", "cpu")
        self.ml_verbose = os.environ.get("FLOORPLANNER_ML_VERBOSE", "0") == "1"

        self.predictor: Optional["Predictor"] = None
        if Predictor is not None:
            try:
                self.predictor = Predictor(str(self.ml_weights), device=self.ml_device)
                if not self.predictor.available():
                    self.predictor = None  # treat as disabled
            except Exception as e:
                sys.stderr.write(f"[my_optimizer_ml] predictor init failed: {e}\n")
                self.predictor = None

        if self.verbose:
            status = "enabled" if self.predictor else "disabled"
            sys.stderr.write(f"[my_optimizer_ml] ML predictor: {status}\n")

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

        # Slice off any padding (consistent with baseline).
        area = area_targets[:block_count]
        cons = constraints[:block_count]
        tp   = target_positions[:block_count] if target_positions is not None else None

        # ---- ML predict (best-effort) ----
        prediction: Optional["Prediction"] = None
        if self.predictor is not None:
            try:
                prediction = self.predictor.predict(
                    block_count=block_count,
                    area_targets=area,
                    constraints=cons,
                    target_positions=tp,
                    b2b_connectivity=b2b_connectivity,
                    p2b_connectivity=p2b_connectivity,
                    pins_pos=pins_pos,
                )
                if self.ml_verbose and prediction is not None:
                    pos0 = prediction.positions[0]
                    sys.stderr.write(
                        f"[ML] case n={block_count}: block0 pred=({pos0[0]:.1f},{pos0[1]:.1f}) "
                        f"prio[0..5]={prediction.priority[:5]}\n"
                    )
            except Exception as e:
                sys.stderr.write(f"[ML] predict failed: {e}; falling back\n")
                prediction = None

        # ---- Run the baseline solver pipeline -----------------------------
        # We replicate the relevant chunk of MyOptimizer.solve() here so we
        # can interpose between _write_txt and the subprocess call.

        try:
            time_s = float(eval(self.time_expr, {"__builtins__": {}}, {"n": block_count}))
        except Exception:
            time_s = 8.0 + 1.0 * block_count
        time_s = max(1.0, min(time_s, 300.0))

        idx = self._call_idx
        self._call_idx += 1
        in_txt  = self.workdir / f"case_{idx:03d}.txt"
        out_sol = self.workdir / f"case_{idx:03d}.sol"

        # Baseline .txt (BLOCKS, NETS, GROUPS, etc.)
        _write_txt(in_txt, block_count, area, b2b_connectivity,
                   p2b_connectivity, pins_pos, cons, tp)

        # ML annotation -- silently ignored by current parser; the C++ patch
        # in ML_FLOORPLAN.md §4 makes make_initial use it.
        if prediction is not None:
            _append_warm_positions(in_txt, prediction, block_count)

        if self.verbose:
            sys.stderr.write(
                f"[my_optimizer_ml] case {idx}: n={block_count} budget={time_s:.1f}s"
                f" ml={'on' if prediction else 'off'}\n"
            )

        # Shared solver runner with feasibility-triggered budget escalation
        # (defined on the baseline MyOptimizer).  The WARM_POSITIONS hint we
        # appended to in_txt above is reused on every escalation attempt.
        positions = self._run_solver(in_txt, out_sol, block_count, time_s, idx, area)

        if not self.keep:
            in_txt.unlink(missing_ok=True)
            out_sol.unlink(missing_ok=True)

        return positions
