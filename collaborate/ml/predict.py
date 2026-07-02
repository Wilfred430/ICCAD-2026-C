"""Inference helper used by my_optimizer_ml.py at solve-time.

Provides a `Predictor` class that:

  1. Loads the trained Transformer once at process start.
  2. For each contest case, accepts the tensors that
     FloorSet's FloorplanOptimizer.solve() receives.
  3. Returns predicted (cx, cy, w, h) per block as a Python list, plus
     a `priority` ordering (sorted by predicted (cy, cx)) that the C++
     solver can use to seed its B*-tree insertion order.

If the model file isn't available (no `weights/floorplan_v1.pt` yet),
Predictor returns None from .predict() and the caller falls back to
the existing random-init pipeline.  This keeps the integration
"strictly additive" -- ML never *breaks* the baseline, only augments it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch

from .data import (
    BLOCK_FEAT_DIM,
    F_AREA, F_AREA_LOG, F_IS_FIXED, F_IS_PREPLACED,
    F_HAS_MIB, F_HAS_CLUSTER,
    F_BND_LEFT, F_BND_RIGHT, F_BND_TOP, F_BND_BOTTOM,
    F_DEG_B2B, F_DEG_P2B,
    F_W_HINT, F_H_HINT, F_X_HINT, F_Y_HINT,
)
from .model import FloorplanTransformer


@dataclass
class Prediction:
    """One per-case prediction.

    positions: list of (cx, cy, w, h) tuples, length = block_count
    priority:  list of block indices in suggested insertion order
               (lowest first -> closest to B*-tree root).
    """
    positions: List[Tuple[float, float, float, float]]
    priority:  List[int]


class Predictor:
    def __init__(self, weights_path: str, device: str = "cpu"):
        self.weights_path = Path(weights_path)
        self.device = device
        self.model: Optional[FloorplanTransformer] = None
        self.config: dict = {}
        if self.weights_path.exists():
            self._load()
        else:
            print(f"[predictor] weights not found at {self.weights_path} "
                  f"-- ML disabled, falling back to baseline")

    def _load(self):
        ckpt = torch.load(self.weights_path, map_location=self.device,
                          weights_only=False)
        self.config = ckpt.get("config", {})
        hidden = self.config.get("hidden_dim", 128)
        layers = self.config.get("n_layers", 4)
        heads  = self.config.get("n_heads", 4)
        self.model = FloorplanTransformer(
            hidden_dim=hidden, n_layers=layers, n_heads=heads,
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        print(f"[predictor] loaded {self.weights_path} "
              f"(hidden={hidden}, layers={layers}, heads={heads})")

    def available(self) -> bool:
        return self.model is not None

    # ---- feature construction at inference -----------------------------------

    @staticmethod
    def _build_features(
        block_count:      int,
        area_targets:     torch.Tensor,    # [N]
        constraints:      torch.Tensor,    # [N, K] cols: is_fixed, is_preplaced, mib, cluster, bcode
        target_positions: Optional[torch.Tensor],  # [N, 4] (x, y, w, h) or -1
        b2b:              torch.Tensor,    # [B, 3]
        p2b:              torch.Tensor,    # [P, 3]
    ) -> torch.Tensor:
        """Build [N, 16] feature tensor matching ml/data.py's schema."""
        feat = torch.zeros((block_count, BLOCK_FEAT_DIM), dtype=torch.float32)

        area = area_targets[:block_count].clamp(min=0.0)
        feat[:, F_AREA]         = area
        feat[:, F_AREA_LOG]     = torch.log(area + 1.0)
        feat[:, F_IS_FIXED]     = constraints[:block_count, 0].float()
        feat[:, F_IS_PREPLACED] = constraints[:block_count, 1].float()
        if constraints.shape[1] > 2:
            feat[:, F_HAS_MIB]     = (constraints[:block_count, 2] > 0).float()
        if constraints.shape[1] > 3:
            feat[:, F_HAS_CLUSTER] = (constraints[:block_count, 3] > 0).float()
        if constraints.shape[1] > 4:
            bcode = constraints[:block_count, 4].long()
            feat[:, F_BND_LEFT]   = ((bcode & 1) > 0).float()
            feat[:, F_BND_RIGHT]  = ((bcode & 2) > 0).float()
            feat[:, F_BND_TOP]    = ((bcode & 4) > 0).float()
            feat[:, F_BND_BOTTOM] = ((bcode & 8) > 0).float()

        # Connectivity degrees.
        if b2b is not None and b2b.numel() > 0:
            valid_b2b = b2b[b2b[:, 0] != -1]
            if valid_b2b.numel() > 0:
                ids = valid_b2b[:, :2].long().flatten()
                ids = ids.clamp(0, block_count - 1)
                deg = torch.zeros(block_count, dtype=torch.float32)
                deg.scatter_add_(0, ids, torch.ones_like(ids, dtype=torch.float32))
                feat[:, F_DEG_B2B] = torch.log(deg + 1.0)

        if p2b is not None and p2b.numel() > 0:
            valid_p2b = p2b[p2b[:, 0] != -1]
            if valid_p2b.numel() > 0:
                ids = valid_p2b[:, 1].long().clamp(0, block_count - 1)
                deg = torch.zeros(block_count, dtype=torch.float32)
                deg.scatter_add_(0, ids, torch.ones_like(ids, dtype=torch.float32))
                feat[:, F_DEG_P2B] = torch.log(deg + 1.0)

        # Geometry hints from target_positions (-1 means unset).
        if target_positions is not None:
            tp = target_positions[:block_count]
            for i in range(block_count):
                x, y, w, h = float(tp[i, 0]), float(tp[i, 1]), float(tp[i, 2]), float(tp[i, 3])
                if w > 0 and h > 0:
                    feat[i, F_W_HINT] = w
                    feat[i, F_H_HINT] = h
                if x >= 0 and y >= 0:
                    feat[i, F_X_HINT] = x + 0.5 * max(w, 0.0)
                    feat[i, F_Y_HINT] = y + 0.5 * max(h, 0.0)

        return feat

    # ---- public inference API ------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        block_count:      int,
        area_targets:     torch.Tensor,
        constraints:      torch.Tensor,
        target_positions: Optional[torch.Tensor],
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos:         torch.Tensor,
    ) -> Optional[Prediction]:
        """Return predicted (cx, cy, w, h) per block, or None if ML disabled."""
        if not self.available():
            return None

        max_n = self.config.get("max_blocks", 128)
        max_t = self.config.get("max_terms", 512)

        if block_count > max_n:
            print(f"[predictor] case has {block_count} blocks > max {max_n}; "
                  f"falling back to baseline")
            return None

        feat = self._build_features(
            block_count, area_targets, constraints, target_positions,
            b2b_connectivity, p2b_connectivity,
        )
        blocks_feat = torch.zeros((1, max_n, BLOCK_FEAT_DIM), dtype=torch.float32)
        blocks_feat[0, :block_count] = feat
        blocks_mask = torch.zeros((1, max_n), dtype=torch.bool)
        blocks_mask[0, :block_count] = True

        # Terminals
        if pins_pos is not None and pins_pos.numel() > 0:
            valid_terms = pins_pos[pins_pos[:, 0] != -1]
        else:
            valid_terms = torch.zeros((0, 2), dtype=torch.float32)
        t_use = min(valid_terms.shape[0], max_t)
        terms = torch.zeros((1, max_t, 2), dtype=torch.float32)
        if t_use > 0:
            terms[0, :t_use] = valid_terms[:t_use]
        terms_mask = torch.zeros((1, max_t), dtype=torch.bool)
        terms_mask[0, :t_use] = True

        pred_pos, pred_dim = self.model(
            blocks_feat.to(self.device),
            blocks_mask.to(self.device),
            terms.to(self.device),
            terms_mask.to(self.device),
        )
        pred_pos = pred_pos[0, :block_count].cpu()
        pred_dim = pred_dim[0, :block_count].cpu()

        # Enforce hard constraints in post-processing:
        #   * fixed/preplaced: copy w, h (and x, y for preplaced) from input
        positions: List[Tuple[float, float, float, float]] = []
        for i in range(block_count):
            cx, cy = float(pred_pos[i, 0]), float(pred_pos[i, 1])
            w,  h  = float(pred_dim[i, 0]), float(pred_dim[i, 1])
            is_fixed     = bool(constraints[i, 0].item())
            is_preplaced = bool(constraints[i, 1].item())
            if (is_fixed or is_preplaced) and target_positions is not None:
                tw = float(target_positions[i, 2])
                th = float(target_positions[i, 3])
                if tw > 0: w = tw
                if th > 0: h = th
            if is_preplaced and target_positions is not None:
                tx = float(target_positions[i, 0])
                ty = float(target_positions[i, 1])
                if tx >= 0 and ty >= 0:
                    cx = tx + 0.5 * w
                    cy = ty + 0.5 * h
            # Snap area to target for soft blocks (1% tolerance).
            if not is_fixed and not is_preplaced:
                a_tgt = float(area_targets[i])
                if a_tgt > 0 and w > 0 and h > 0:
                    a_pred = w * h
                    scale = (a_tgt / a_pred) ** 0.5
                    w *= scale
                    h *= scale
            positions.append((cx, cy, max(w, 1e-3), max(h, 1e-3)))

        # Priority: sort by (cy ascending, cx ascending) so blocks closer
        # to (0, 0) come earlier -- matches B*-tree's natural origin growth.
        order = sorted(range(block_count),
                       key=lambda i: (positions[i][1], positions[i][0]))
        return Prediction(positions=positions, priority=order)
