"""Supervised training for the floorplan position predictor.

Usage:
    python -m ml.train \\
        --data /path/to/LiteTensorDataTest \\
        --out  ml/weights/floorplan_v1.pt \\
        --epochs 5 --batch 32 --lr 1e-3

What the loss optimises
-----------------------

    L_total = L_pos + λ_dim * L_dim + λ_area * L_area_consistency

  L_pos   = smooth L1 on (cx, cy)  -- positions, normalised by case bbox
  L_dim   = smooth L1 on (w, h)    -- dimensions, only on SOFT blocks
                                     (fixed/preplaced are forced via input
                                      features so we don't penalise them)
  L_area  = L1 on |w*h - area_target| / area_target  -- soft block area
                                                      tolerance is 1% so
                                                      this keeps predictions
                                                      consistent

Normalising by per-case bbox size means a position error of "20 grid
units" on a 200-wide layout gets the same weight as "2 grid units" on a
20-wide layout.  Without this normalisation big layouts would dominate
the loss.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split

from .data  import FloorSetLiteDataset, F_IS_FIXED, F_IS_PREPLACED, F_AREA
from .model import FloorplanTransformer


def per_case_bbox(target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """For each batch item, compute (max_x_extent, max_y_extent) used to
    normalise positions / dimensions.  target=[B,N,4], mask=[B,N]."""
    cx, cy, w, h = target[..., 0], target[..., 1], target[..., 2], target[..., 3]
    # block extent = cx + w/2 (right edge), cy + h/2 (top edge)
    right = cx + 0.5 * w
    top   = cy + 0.5 * h
    right = right.masked_fill(~mask, 0.0)
    top   = top.masked_fill(~mask, 0.0)
    bbox_w, _ = right.max(dim=1)
    bbox_h, _ = top.max(dim=1)
    # Avoid divide-by-zero on degenerate cases.
    return torch.stack([bbox_w.clamp(min=1.0), bbox_h.clamp(min=1.0)], dim=1)  # [B, 2]


def compute_loss(
    pred_pos: torch.Tensor,
    pred_dim: torch.Tensor,
    target:   torch.Tensor,
    blocks_feat: torch.Tensor,
    blocks_mask: torch.Tensor,
    lambda_dim: float = 0.5,
    lambda_area: float = 0.2,
    case_weight: Optional[torch.Tensor] = None,
) -> dict:
    """Per-case losses, combined into a (optionally size-weighted) batch mean.

    case_weight: optional [B] tensor.  The batch loss is the case_weight-weighted
    mean of per-case losses, so passing n_blocks**P emphasises large cases.
    case_weight = ones (or None) weights every case equally; case_weight =
    n_blocks reproduces the original per-block averaging.
    """
    bbox = per_case_bbox(target, blocks_mask)  # [B, 2]
    # Normalise positions and dimensions by case bbox for scale-invariance.
    norm_xy = bbox.unsqueeze(1)             # [B, 1, 2]

    pred_pos_n = pred_pos / norm_xy
    pred_dim_n = pred_dim / norm_xy
    tgt_pos_n  = target[..., :2] / norm_xy
    tgt_dim_n  = target[..., 2:] / norm_xy

    # ---- per-case position loss (mean over a case's own blocks) ----
    n_per_case = blocks_mask.float().sum(1).clamp(min=1.0)          # [B]
    pos_err = F.smooth_l1_loss(pred_pos_n, tgt_pos_n, reduction="none").sum(-1)
    pos_err = pos_err.masked_fill(~blocks_mask, 0.0)
    L_pos_case = pos_err.sum(1) / n_per_case                        # [B]

    # ---- per-case dimension loss (SOFT blocks only) ----
    is_locked = (blocks_feat[..., F_IS_FIXED] > 0.5) | (blocks_feat[..., F_IS_PREPLACED] > 0.5)
    soft_mask = blocks_mask & ~is_locked
    n_soft = soft_mask.float().sum(1).clamp(min=1.0)                # [B]
    dim_err = F.smooth_l1_loss(pred_dim_n, tgt_dim_n, reduction="none").sum(-1)
    dim_err = dim_err.masked_fill(~soft_mask, 0.0)
    L_dim_case = dim_err.sum(1) / n_soft                            # [B]

    # ---- per-case area-consistency (SOFT blocks only) ----
    pred_area = pred_dim[..., 0] * pred_dim[..., 1]
    tgt_area  = blocks_feat[..., F_AREA].clamp(min=1.0)
    area_rel_err = ((pred_area - tgt_area).abs() / tgt_area).masked_fill(~soft_mask, 0.0)
    L_area_case = area_rel_err.sum(1) / n_soft                      # [B]

    per_case = L_pos_case + lambda_dim * L_dim_case + lambda_area * L_area_case  # [B]

    w = torch.ones_like(per_case) if case_weight is None else case_weight
    wsum = w.sum().clamp(min=1e-8)
    return {
        "loss":  (w * per_case).sum()    / wsum,
        "L_pos": (w * L_pos_case).sum()  / wsum,
        "L_dim": (w * L_dim_case).sum()  / wsum,
        "L_area": (w * L_area_case).sum() / wsum,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="path to LiteTensorDataTest")
    parser.add_argument("--out",  default="ml/weights/floorplan_v1.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch",  type=int, default=16)
    parser.add_argument("--lr",     type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads",  type=int, default=4)
    parser.add_argument("--max-blocks", type=int, default=128)
    parser.add_argument("--max-terms",  type=int, default=512)
    parser.add_argument("--val-frac",   type=float, default=0.05)
    parser.add_argument("--workers",    type=int, default=2)
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    # ---- big-case emphasis -------------------------------------------------
    # The v10 contest score is dominated by the largest cases, but they are a
    # minority of the training data.  size-power P makes each case count like
    # n_blocks**P:
    #   P = 1.0  -> reproduces the original per-block averaging (v2 baseline)
    #   P = 2.0  -> quadratic emphasis on big cases (recommended start)
    #   P = 3.0  -> strong emphasis (small cases nearly ignored)
    # The emphasis is applied EITHER via the loss (default) OR, with
    # --weighted-sampling, by over-sampling big cases (equivalent target, lower
    # gradient variance, but needs a one-time block-count scan that is cached).
    parser.add_argument("--size-power", type=float, default=1.0,
                        help="emphasise big cases as n_blocks**P (1.0=v2 baseline)")
    parser.add_argument("--weighted-sampling", action="store_true",
                        help="apply the n**size_power emphasis by over-sampling "
                             "big cases instead of by loss weighting")
    parser.add_argument("--init-from", default=None,
                        help="checkpoint (.pt) to FINE-TUNE from instead of "
                             "training from scratch; loads its weights and "
                             "architecture.  Use a lower --lr (e.g. 3e-4) and "
                             "fewer --epochs when fine-tuning.")
    args = parser.parse_args()

    print(f"[train] device={args.device}")
    dataset = FloorSetLiteDataset(
        args.data,
        max_blocks=args.max_blocks,
        max_terms=args.max_terms,
    )
    print(f"[train] dataset cases = {len(dataset)}")

    n_val = max(1, int(args.val_frac * len(dataset)))
    n_tr  = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_tr, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    # Apply the big-case emphasis EITHER by over-sampling (sampler) OR by loss
    # weighting -- never both, so the effective emphasis is exactly n**size_power.
    if args.weighted_sampling and args.size_power != 0.0:
        counts = dataset.block_counts()                         # [len(dataset)], cached scan
        w_all  = counts.float() ** args.size_power
        train_w = w_all[torch.as_tensor(train_set.indices)]
        sampler = WeightedRandomSampler(train_w, num_samples=len(train_set.indices),
                                        replacement=True)
        train_loader = DataLoader(train_set, batch_size=args.batch,
                                  sampler=sampler, num_workers=args.workers)
        train_loss_power = 0.0     # emphasis comes from sampling; keep loss uniform
        print(f"[train] big-case emphasis: WEIGHTED SAMPLING with n**{args.size_power}")
    else:
        train_loader = DataLoader(train_set, batch_size=args.batch,
                                  shuffle=True, num_workers=args.workers)
        train_loss_power = args.size_power
        print(f"[train] big-case emphasis: LOSS WEIGHTING with n**{args.size_power}")
    val_loader   = DataLoader(val_set,   batch_size=args.batch,
                              shuffle=False, num_workers=args.workers)

    # Build the model.  When fine-tuning (--init-from) we adopt the checkpoint's
    # architecture so the weights load cleanly, regardless of the size flags.
    init_state = None
    if args.init_from:
        ckpt = torch.load(args.init_from, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config", {})
        args.hidden = cfg.get("hidden_dim", args.hidden)
        args.layers = cfg.get("n_layers",   args.layers)
        args.heads  = cfg.get("n_heads",    args.heads)
        init_state  = ckpt["model_state"]
        print(f"[train] fine-tuning from {args.init_from} "
              f"(epoch={ckpt.get('epoch','?')}, val_loss={ckpt.get('val_loss','?')}, "
              f"arch hidden={args.hidden} layers={args.layers} heads={args.heads})")

    model = FloorplanTransformer(
        hidden_dim=args.hidden,
        n_layers=args.layers,
        n_heads=args.heads,
    ).to(args.device)
    if init_state is not None:
        model.load_state_dict(init_state)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] model params = {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(train_loader))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(args.epochs):
        # ---- train ----
        model.train()
        t0 = time.time()
        running = {"loss": 0.0, "L_pos": 0.0, "L_dim": 0.0, "L_area": 0.0, "n": 0}
        for batch in train_loader:
            blocks_feat = batch["blocks_feat"].to(args.device)
            blocks_mask = batch["blocks_mask"].to(args.device)
            terms       = batch["terms"].to(args.device)
            terms_mask  = batch["terms_mask"].to(args.device)
            target      = batch["target"].to(args.device)

            n_blocks    = batch["n_blocks"].to(args.device).float()
            case_weight = n_blocks ** train_loss_power     # power 0 -> all ones (uniform)

            pred_pos, pred_dim = model(blocks_feat, blocks_mask, terms, terms_mask)
            losses = compute_loss(pred_pos, pred_dim, target, blocks_feat, blocks_mask,
                                  case_weight=case_weight)

            opt.zero_grad()
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            bs = blocks_feat.shape[0]
            for k in ("loss", "L_pos", "L_dim", "L_area"):
                running[k] += losses[k].item() * bs
            running["n"] += bs

        tr = {k: running[k] / running["n"] for k in ("loss", "L_pos", "L_dim", "L_area")}

        # ---- validate ----
        model.eval()
        v_running = {"loss": 0.0, "L_pos": 0.0, "n": 0}
        with torch.no_grad():
            for batch in val_loader:
                blocks_feat = batch["blocks_feat"].to(args.device)
                blocks_mask = batch["blocks_mask"].to(args.device)
                terms       = batch["terms"].to(args.device)
                terms_mask  = batch["terms_mask"].to(args.device)
                target      = batch["target"].to(args.device)
                n_blocks    = batch["n_blocks"].to(args.device).float()
                case_weight = n_blocks ** args.size_power
                pred_pos, pred_dim = model(blocks_feat, blocks_mask, terms, terms_mask)
                losses = compute_loss(pred_pos, pred_dim, target, blocks_feat, blocks_mask,
                                      case_weight=case_weight)
                bs = blocks_feat.shape[0]
                v_running["loss"]  += losses["loss"].item() * bs
                v_running["L_pos"] += losses["L_pos"].item() * bs
                v_running["n"]     += bs
        val = {k: v_running[k] / v_running["n"] for k in ("loss", "L_pos")}

        dt = time.time() - t0
        print(f"[epoch {epoch+1}/{args.epochs}] {dt:5.1f}s  "
              f"train_loss={tr['loss']:.4f} (pos={tr['L_pos']:.4f} dim={tr['L_dim']:.4f} area={tr['L_area']:.4f}) "
              f"| val_loss={val['loss']:.4f} pos={val['L_pos']:.4f}")

        if val["loss"] < best_val:
            best_val = val["loss"]
            ckpt = {
                "model_state":   model.state_dict(),
                "config": {
                    "hidden_dim": args.hidden,
                    "n_layers":   args.layers,
                    "n_heads":    args.heads,
                    "max_blocks": args.max_blocks,
                    "max_terms":  args.max_terms,
                },
                "epoch":       epoch + 1,
                "val_loss":    val["loss"],
            }
            torch.save(ckpt, out_path)
            print(f"[train] saved {out_path} (val_loss={best_val:.4f})")

    print(f"[train] done; best val_loss={best_val:.4f}")


if __name__ == "__main__":
    main()
