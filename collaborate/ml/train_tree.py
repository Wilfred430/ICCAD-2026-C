"""Supervised (teacher-forced) training for the B*-tree generator.

Usage:
    python -m ml.train_tree \\
        --data /path/to/ICCAD-C-FloorSet-official \\
        --out  ml/weights/tree_v1.pt \\
        --epochs 3 --batch 16 --lr 1e-3

What the loss optimises
------------------------

At every generation step t (t = 1 .. n_blocks-1; step 0 is the root and has
no parent), the model predicts:

    L_ptr = cross-entropy over "which earlier step is this step's parent"
    L_dir = binary cross-entropy over the attach direction (0/1)

    L_total = L_ptr + lambda_dir * L_dir

This directly supervises the *topology* (the discrete B*-tree structure),
not (cx, cy, w, h) -- avoiding the mode-collapse failure of coordinate
regression (see WINNING_STRATEGY.md Section 1: averaging two valid mirrored
topologies produces an overlapping, invalid one; averaging two valid parent-
pointer distributions is fine because cross-entropy training doesn't blend
labels, it makes the model *choose*).

Only cases with `has_tree=True` (TRAIN-format cases that fit within
--max-blocks) contribute; TEST-format cases have no tree_sol and are
skipped automatically via the batch mask.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split

from .data import FloorSetLiteDataset
from .model_tree import TreeGenerator


def compute_tree_loss(block_logits, ptr_logits, dir_logits, gen_order, parent_step, direction,
                       step_mask, lambda_dir: float = 0.5, lambda_block: float = 1.0, case_weight=None):
    """
    block_logits: [B, N, N]  (block_logits[b,t,:] scores over block ids)
    ptr_logits:   [B, N, N]  (ptr_logits[b,t,:] scores over earlier steps)
    dir_logits:   [B, N]
    gen_order:    [B, N]  ground-truth block id placed at step t
    parent_step:  [B, N]  ground truth parent STEP index; -1 for root/padding
    direction:    [B, N]
    step_mask:    [B, N]  True for real steps (root included)
    """
    B, N, _ = ptr_logits.shape
    # Every real step (including root) has a block-selection label.
    block_loss = F.cross_entropy(
        block_logits.reshape(B * N, N), gen_order.clamp(min=0).reshape(B * N), reduction="none",
    ).reshape(B, N)
    block_loss = block_loss.masked_fill(~step_mask, 0.0)

    # Root (step 0) has no parent/direction to predict -- exclude via
    # supervised_mask (real step AND not the root).
    is_root = torch.zeros_like(step_mask)
    is_root[:, 0] = True
    supervised = step_mask & ~is_root                      # [B, N]

    parent_step_safe = parent_step.clamp(min=0)             # avoid -1 indexing into CE
    ptr_loss = F.cross_entropy(
        ptr_logits.reshape(B * N, N),
        parent_step_safe.reshape(B * N),
        reduction="none",
    ).reshape(B, N)
    ptr_loss = ptr_loss.masked_fill(~supervised, 0.0)

    dir_loss = F.binary_cross_entropy_with_logits(dir_logits, direction, reduction="none")
    dir_loss = dir_loss.masked_fill(~supervised, 0.0)

    n_per_case = step_mask.float().sum(1).clamp(min=1.0)         # [B]
    n_sup_case = supervised.float().sum(1).clamp(min=1.0)        # [B]
    L_block_case = block_loss.sum(1) / n_per_case
    L_ptr_case   = ptr_loss.sum(1) / n_sup_case
    L_dir_case   = dir_loss.sum(1) / n_sup_case
    per_case = lambda_block * L_block_case + L_ptr_case + lambda_dir * L_dir_case

    w = torch.ones_like(per_case) if case_weight is None else case_weight
    wsum = w.sum().clamp(min=1e-8)
    return {
        "loss":    (w * per_case).sum()     / wsum,
        "L_block": (w * L_block_case).sum() / wsum,
        "L_ptr":   (w * L_ptr_case).sum()   / wsum,
        "L_dir":   (w * L_dir_case).sum()   / wsum,
    }


def block_accuracy(block_logits, gen_order, step_mask):
    pred = block_logits.argmax(dim=-1)
    correct = (pred == gen_order) & step_mask
    return correct.float().sum() / step_mask.float().sum().clamp(min=1.0)


def ptr_accuracy(ptr_logits, parent_step, step_mask):
    """Fraction of supervised steps where argmax(ptr_logits) == true parent."""
    is_root = torch.zeros_like(step_mask)
    is_root[:, 0] = True
    supervised = step_mask & ~is_root
    pred = ptr_logits.argmax(dim=-1)
    correct = (pred == parent_step) & supervised
    return correct.float().sum() / supervised.float().sum().clamp(min=1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="path to FloorSet root (contains floorset_lite/)")
    parser.add_argument("--out",  default="ml/weights/tree_v1.pt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch",  type=int, default=16)
    parser.add_argument("--lr",     type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--ctx-layers", type=int, default=4)
    parser.add_argument("--dec-layers", type=int, default=4)
    parser.add_argument("--heads",  type=int, default=4)
    parser.add_argument("--lambda-dir", type=float, default=0.5)
    parser.add_argument("--max-blocks", type=int, default=128)
    parser.add_argument("--max-terms",  type=int, default=512)
    parser.add_argument("--val-frac",   type=float, default=0.02)
    parser.add_argument("--workers",    type=int, default=2)
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit-cases", type=int, default=None,
                        help="cap dataset size (debug / smoke test)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="stop after this many optimizer steps (debug / smoke test)")
    parser.add_argument("--size-power", type=float, default=1.0,
                        help="emphasise big cases as n_blocks**P in the loss "
                             "(same convention as train.py; big cases dominate "
                             "the e^n-weighted contest score)")
    parser.add_argument("--weighted-sampling", action="store_true")
    parser.add_argument("--init-from", default=None)
    args = parser.parse_args()

    print(f"[train_tree] device={args.device}")
    dataset = FloorSetLiteDataset(args.data, max_blocks=args.max_blocks, max_terms=args.max_terms)
    if args.limit_cases:
        dataset = torch.utils.data.Subset(dataset, range(min(args.limit_cases, len(dataset))))
    print(f"[train_tree] dataset cases = {len(dataset)}")

    n_val = max(1, int(args.val_frac * len(dataset)))
    n_tr  = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_tr, n_val], generator=torch.Generator().manual_seed(42),
    )

    base_dataset = dataset.dataset if isinstance(dataset, torch.utils.data.Subset) else dataset
    if args.weighted_sampling and args.size_power != 0.0 and isinstance(base_dataset, FloorSetLiteDataset):
        counts = base_dataset.block_counts()
        # torch.utils.data.random_split / Subset indices are relative to `dataset`
        # (possibly itself a Subset from --limit-cases); resolve down to base indices.
        def to_base_idx(i):
            j = train_set.indices[i]
            return dataset.indices[j] if isinstance(dataset, torch.utils.data.Subset) else j
        train_w = torch.tensor([float(counts[to_base_idx(i)]) for i in range(len(train_set))]) ** args.size_power
        sampler = WeightedRandomSampler(train_w, num_samples=len(train_set), replacement=True)
        train_loader = DataLoader(train_set, batch_size=args.batch, sampler=sampler, num_workers=args.workers)
        train_loss_power = 0.0
        print(f"[train_tree] big-case emphasis: WEIGHTED SAMPLING with n**{args.size_power}")
    else:
        train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=args.workers)
        train_loss_power = args.size_power
        print(f"[train_tree] big-case emphasis: LOSS WEIGHTING with n**{args.size_power}")
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False, num_workers=args.workers)

    init_state = None
    if args.init_from:
        ckpt = torch.load(args.init_from, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config", {})
        args.hidden     = cfg.get("hidden_dim",  args.hidden)
        args.ctx_layers = cfg.get("n_ctx_layers", args.ctx_layers)
        args.dec_layers = cfg.get("n_dec_layers", args.dec_layers)
        args.heads      = cfg.get("n_heads",      args.heads)
        init_state      = ckpt["model_state"]
        print(f"[train_tree] fine-tuning from {args.init_from}")

    model = TreeGenerator(
        hidden_dim=args.hidden, n_ctx_layers=args.ctx_layers, n_dec_layers=args.dec_layers,
        n_heads=args.heads, max_blocks=args.max_blocks,
    ).to(args.device)
    if init_state is not None:
        model.load_state_dict(init_state)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train_tree] model params = {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    total_steps = args.epochs * len(train_loader)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(total_steps, 1))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    global_step = 0

    def run_batch(batch):
        blocks_feat = batch["blocks_feat"].to(args.device)
        blocks_mask = batch["blocks_mask"].to(args.device)
        terms       = batch["terms"].to(args.device)
        terms_mask  = batch["terms_mask"].to(args.device)
        has_tree    = batch["has_tree"].to(args.device)
        gen_order   = batch["gen_order"].to(args.device)
        parent_step = batch["parent_step"].to(args.device)
        direction   = batch["direction"].to(args.device)
        n_blocks    = batch["n_blocks"].to(args.device)

        N = blocks_feat.shape[1]
        step_idx  = torch.arange(N, device=args.device).unsqueeze(0)
        step_mask = (step_idx < n_blocks.unsqueeze(1)) & has_tree.unsqueeze(1)  # [B, N]

        if not step_mask.any():
            return None  # whole batch lacks usable trees (e.g. TEST-format only)

        block_logits, ptr_logits, dir_logits = model(
            blocks_feat, blocks_mask, terms, terms_mask, gen_order.clamp(min=0), step_mask)
        case_weight = (n_blocks.float() ** train_loss_power) * has_tree.float()
        losses = compute_tree_loss(block_logits, ptr_logits, dir_logits, gen_order, parent_step,
                                    direction, step_mask, lambda_dir=args.lambda_dir,
                                    case_weight=case_weight)
        b_acc = block_accuracy(block_logits, gen_order, step_mask)
        p_acc = ptr_accuracy(ptr_logits, parent_step, step_mask)
        return losses, b_acc, p_acc

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        running = {"loss": 0.0, "L_block": 0.0, "L_ptr": 0.0, "L_dir": 0.0, "b_acc": 0.0, "p_acc": 0.0, "n": 0}
        for batch in train_loader:
            out = run_batch(batch)
            if out is None:
                continue
            losses, b_acc, p_acc = out
            opt.zero_grad()
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            global_step += 1

            bs = batch["blocks_feat"].shape[0]
            running["loss"]    += losses["loss"].item()    * bs
            running["L_block"] += losses["L_block"].item() * bs
            running["L_ptr"]   += losses["L_ptr"].item()   * bs
            running["L_dir"]   += losses["L_dir"].item()   * bs
            running["b_acc"]   += b_acc.item() * bs
            running["p_acc"]   += p_acc.item() * bs
            running["n"]       += bs

            if args.max_steps and global_step >= args.max_steps:
                break

        n = max(running["n"], 1)
        tr = {k: running[k] / n for k in ("loss", "L_block", "L_ptr", "L_dir", "b_acc", "p_acc")}

        model.eval()
        v_running = {"loss": 0.0, "b_acc": 0.0, "p_acc": 0.0, "n": 0}
        with torch.no_grad():
            for batch in val_loader:
                out = run_batch(batch)
                if out is None:
                    continue
                losses, b_acc, p_acc = out
                bs = batch["blocks_feat"].shape[0]
                v_running["loss"]  += losses["loss"].item() * bs
                v_running["b_acc"] += b_acc.item() * bs
                v_running["p_acc"] += p_acc.item() * bs
                v_running["n"]     += bs
        vn = max(v_running["n"], 1)
        val = {"loss": v_running["loss"] / vn, "b_acc": v_running["b_acc"] / vn, "p_acc": v_running["p_acc"] / vn}

        dt = time.time() - t0
        print(f"[epoch {epoch+1}/{args.epochs}] {dt:5.1f}s  "
              f"train_loss={tr['loss']:.4f} (block={tr['L_block']:.4f} ptr={tr['L_ptr']:.4f} dir={tr['L_dir']:.4f} "
              f"b_acc={tr['b_acc']:.3f} p_acc={tr['p_acc']:.3f}) "
              f"| val_loss={val['loss']:.4f} val_b_acc={val['b_acc']:.3f} val_p_acc={val['p_acc']:.3f}")

        if val["loss"] < best_val:
            best_val = val["loss"]
            ckpt = {
                "model_state": model.state_dict(),
                "config": {
                    "hidden_dim":   args.hidden,
                    "n_ctx_layers": args.ctx_layers,
                    "n_dec_layers": args.dec_layers,
                    "n_heads":      args.heads,
                    "max_blocks":   args.max_blocks,
                    "max_terms":    args.max_terms,
                },
                "epoch": epoch + 1,
                "val_loss": val["loss"],
                "val_b_acc": val["b_acc"],
                "val_p_acc": val["p_acc"],
            }
            torch.save(ckpt, out_path)
            print(f"[train_tree] saved {out_path} (val_loss={best_val:.4f}, "
                  f"val_b_acc={val['b_acc']:.3f}, val_p_acc={val['p_acc']:.3f})")

        if args.max_steps and global_step >= args.max_steps:
            break

    print(f"[train_tree] done; best val_loss={best_val:.4f}")


if __name__ == "__main__":
    main()
