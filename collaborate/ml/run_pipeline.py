"""One-command end-to-end demo of the generative B*-tree pipeline.

    python -m ml.run_pipeline

does all of the following:

  1. Trains the TreeGenerator (ml/model_tree.py) on a slice of the real 1M
     training set if no checkpoint exists yet (or --retrain is passed).
  2. Loads a genuinely unseen validation case (LiteTensorDataTest -- TEST
     format, no tree_sol, so this is a fair blind-inference test).
  3. Autoregressively samples K candidate B*-tree topologies from the model
     (ml/model_tree.py's generate()) -- no ground truth used.
  4. Packs each sampled topology into real (x, y, w, h) geometry via the
     validated Python port of the C++ packer (ml/pack_tree.py).
  5. Scores every sample with the REAL contest cost formula (ml/contest_cost.py,
     ported from src/cost.cpp: feasibility, HPWL_gap/Area_gap vs. the dataset's
     own baseline `metrics`, V_rel from grouping/MIB/boundary, contest Cost)
     and prints a ranked table.
  6. Writes the best sample's geometry as a `.sol`-style text file for
     manual cross-checking against the official evaluator.

Usage:
    python -m ml.run_pipeline --case 0 --samples 8
    python -m ml.run_pipeline --retrain --train-cases 5000 --epochs 3
    python -m ml.run_pipeline --case 5 --samples 16 --temperature 0.8
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from .data import FloorSetLiteDataset, case_to_features
from .model_tree import TreeGenerator
from .pack_tree import build_lc_rc, pack_btree
from .contest_cost import evaluate as evaluate_cost
from .train_tree import compute_tree_loss, block_accuracy, ptr_accuracy


def quick_train(args, device) -> Path:
    print(f"[pipeline] no checkpoint at {args.weights} -- training a fresh one "
          f"({args.train_cases} cases, {args.epochs} epochs)")
    dataset = FloorSetLiteDataset(args.data, max_blocks=args.max_blocks, max_terms=args.max_terms)
    n_use = min(args.train_cases, len(dataset))
    subset = torch.utils.data.Subset(dataset, range(n_use))
    n_val = max(1, int(0.05 * n_use))
    n_tr = n_use - n_val
    train_set, val_set = random_split(subset, [n_tr, n_val], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=args.batch, shuffle=False, num_workers=0)

    model = TreeGenerator(hidden_dim=args.hidden, n_ctx_layers=args.ctx_layers,
                           n_dec_layers=args.dec_layers, n_heads=args.heads,
                           max_blocks=args.max_blocks).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[pipeline] model params = {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(args.epochs * len(train_loader), 1))

    def run_batch(batch):
        blocks_feat = batch["blocks_feat"].to(device)
        blocks_mask = batch["blocks_mask"].to(device)
        terms       = batch["terms"].to(device)
        terms_mask  = batch["terms_mask"].to(device)
        has_tree    = batch["has_tree"].to(device)
        gen_order   = batch["gen_order"].to(device)
        parent_step = batch["parent_step"].to(device)
        direction   = batch["direction"].to(device)
        n_blocks    = batch["n_blocks"].to(device)
        N = blocks_feat.shape[1]
        step_idx = torch.arange(N, device=device).unsqueeze(0)
        step_mask = (step_idx < n_blocks.unsqueeze(1)) & has_tree.unsqueeze(1)
        if not step_mask.any():
            return None
        block_logits, ptr_logits, dir_logits = model(
            blocks_feat, blocks_mask, terms, terms_mask, gen_order.clamp(min=0), step_mask)
        losses = compute_tree_loss(block_logits, ptr_logits, dir_logits, gen_order,
                                    parent_step, direction, step_mask,
                                    case_weight=n_blocks.float() ** args.size_power)
        return losses, block_accuracy(block_logits, gen_order, step_mask), \
            ptr_accuracy(ptr_logits, parent_step, step_mask)

    best_val = float("inf")
    out_path = Path(args.weights)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        for batch in train_loader:
            out = run_batch(batch)
            if out is None:
                continue
            losses, _, _ = out
            opt.zero_grad()
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

        model.eval()
        vl, vb, vp, vn = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                out = run_batch(batch)
                if out is None:
                    continue
                losses, b_acc, p_acc = out
                bs = batch["blocks_feat"].shape[0]
                vl += losses["loss"].item() * bs
                vb += b_acc.item() * bs
                vp += p_acc.item() * bs
                vn += bs
        vn = max(vn, 1)
        print(f"[pipeline] epoch {epoch+1}/{args.epochs}  {time.time()-t0:5.1f}s  "
              f"val_loss={vl/vn:.4f}  val_block_acc={vb/vn:.3f}  val_ptr_acc={vp/vn:.3f}")
        if vl / vn < best_val:
            best_val = vl / vn
            torch.save({
                "model_state": model.state_dict(),
                "config": {
                    "hidden_dim": args.hidden, "n_ctx_layers": args.ctx_layers,
                    "n_dec_layers": args.dec_layers, "n_heads": args.heads,
                    "max_blocks": args.max_blocks, "max_terms": args.max_terms,
                },
                "epoch": epoch + 1, "val_loss": best_val,
            }, out_path)
    print(f"[pipeline] saved trained model -> {out_path}")
    return out_path


def load_case_raw(val_root: str, case_idx: int):
    """Load one validation case WITHOUT going through FloorSetLiteDataset's
    padding, so we keep the raw block count and can build geometry hints
    exactly like the real contest framework's target_positions would."""
    import glob
    configs = sorted(Path(val_root).glob("config_*"),
                      key=lambda p: int(p.name.split("_")[1]))
    cfg = configs[case_idx]
    data_files = sorted(glob.glob(str(cfg / "litedata_*.pth")))
    data = torch.load(data_files[0], weights_only=False)[0]
    label_path = data_files[0].replace("litedata_", "litelabel_")
    label = torch.load(label_path, weights_only=False)[0]
    blocks, b2b, p2b, pins_pos = data
    metrics, geometry = label
    return blocks, b2b, p2b, pins_pos, metrics, geometry, cfg.name


def build_dims_and_hints(blocks, geometry):
    """For fixed/preplaced blocks (hard input constraints), read their true
    (w,h[,x,y]) from geometry -- this mirrors what target_positions supplies
    at real contest runtime, NOT the solution we're trying to produce (see
    module docstring).  Soft blocks get a placeholder square w=h=sqrt(area);
    the tree model only decides topology, not per-block aspect ratio."""
    n = blocks.shape[0]
    dims, is_preplaced, preplaced_xy = {}, {}, {}
    for i in range(n):
        is_fixed_i     = bool(blocks[i, 1] > 0.5)
        is_preplaced_i = bool(blocks[i, 2] > 0.5)
        is_preplaced[i] = is_preplaced_i
        if is_fixed_i or is_preplaced_i:
            xs, ys = geometry[i, :, 0], geometry[i, :, 1]
            w, h = float(xs.max() - xs.min()), float(ys.max() - ys.min())
            dims[i] = (max(w, 1e-3), max(h, 1e-3))
            if is_preplaced_i:
                preplaced_xy[i] = (float(xs.min()), float(ys.min()))
        else:
            area = max(float(blocks[i, 0]), 1.0)
            side = math.sqrt(area)
            dims[i] = (side, side)
    return dims, is_preplaced, preplaced_xy


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="d:/ICCAD-2026-C/ICCAD-C-FloorSet-official",
                        help="FloorSet root (contains floorset_lite/, used for training)")
    parser.add_argument("--val", default="d:/ICCAD-2026-C/ICCAD-C-FloorSet-official/LiteTensorDataTest",
                        help="validation set root (100 config_* cases)")
    parser.add_argument("--weights", default="ml/weights/tree_v1.pt")
    parser.add_argument("--retrain", action="store_true", help="train even if --weights already exists")
    parser.add_argument("--case", type=int, default=0, help="validation case index (0-99)")
    parser.add_argument("--samples", type=int, default=8, help="number of topologies to sample")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-sol", default="ml/pipeline_best.sol")
    # training knobs (only used if a checkpoint has to be trained)
    parser.add_argument("--train-cases", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--ctx-layers", type=int, default=3)
    parser.add_argument("--dec-layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--max-blocks", type=int, default=128)
    parser.add_argument("--max-terms", type=int, default=512)
    parser.add_argument("--size-power", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    gen = torch.Generator(device=args.device).manual_seed(args.seed)

    weights_path = Path(args.weights)
    if args.retrain or not weights_path.exists():
        weights_path = quick_train(args, args.device)
    else:
        print(f"[pipeline] loading existing checkpoint {weights_path}")

    ckpt = torch.load(weights_path, map_location=args.device, weights_only=False)
    cfg = ckpt["config"]
    model = TreeGenerator(hidden_dim=cfg["hidden_dim"], n_ctx_layers=cfg["n_ctx_layers"],
                           n_dec_layers=cfg["n_dec_layers"], n_heads=cfg["n_heads"],
                           max_blocks=cfg["max_blocks"]).to(args.device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"\n[pipeline] loading validation case #{args.case} from {args.val}")
    blocks, b2b, p2b, pins_pos, metrics, geometry, cfg_name = load_case_raw(args.val, args.case)
    n = blocks.shape[0]
    print(f"[pipeline] case={cfg_name}  n_blocks={n}  "
          f"baseline_area={float(metrics[0]):.1f}  "
          f"baseline_hpwl_int={float(metrics[6]):.2f}  baseline_hpwl_ext={float(metrics[7]):.3f}")

    feat = case_to_features(blocks, b2b, p2b, geometry)  # fixed/preplaced hints only, no cheating
    max_n, max_t = cfg["max_blocks"], cfg["max_terms"]
    blocks_feat = torch.zeros((1, max_n, feat.shape[1]))
    blocks_feat[0, :n] = feat
    blocks_mask = torch.zeros((1, max_n), dtype=torch.bool)
    blocks_mask[0, :n] = True
    t_use = min(pins_pos.shape[0], max_t)
    terms = torch.zeros((1, max_t, 2))
    terms[0, :t_use] = pins_pos[:t_use]
    terms_mask = torch.zeros((1, max_t), dtype=torch.bool)
    terms_mask[0, :t_use] = True

    dims, is_preplaced, preplaced_xy = build_dims_and_hints(blocks, geometry)

    baseline_area = float(metrics[0])
    baseline_hpwl = float(metrics[6]) + float(metrics[7])

    print(f"\n[pipeline] sampling {args.samples} topologies (temperature={args.temperature})...")
    results = []
    t0 = time.time()
    for s in range(args.samples):
        out = model.generate(blocks_feat.to(args.device), blocks_mask.to(args.device),
                              terms.to(args.device), terms_mask.to(args.device),
                              n_blocks=n, temperature=args.temperature, sample=True, generator=gen)
        root = int(out["gen_order"][0])
        lc, rc = build_lc_rc(root, out["parent_id"], out["direction"], n, gen_order=out["gen_order"].tolist())
        pack = pack_btree(root, lc, rc, dims, is_preplaced, preplaced_xy)
        cc = evaluate_cost(pack.x, pack.y, dims, blocks, b2b, p2b, pins_pos,
                            baseline_area, baseline_hpwl)
        results.append({"sample": s, "pack": pack, "cost": cc})
    dt = time.time() - t0

    # Real contest ranking: infeasible (Cost=10) always loses; among feasible
    # samples, lower Cost wins (matches contest_cost's own semantics).
    results.sort(key=lambda r: r["cost"].cost)
    print(f"\n[pipeline] sampled + packed {args.samples} topologies in {dt:.1f}s\n")
    print(f"{'sample':>6} {'feasible':>9} {'bbox_w':>8} {'bbox_h':>8} {'area_gap':>9} "
          f"{'hpwl_gap':>9} {'V_rel':>6} {'Cost':>8}")
    for r in results:
        p, cc = r["pack"], r["cost"]
        print(f"{r['sample']:>6} {str(cc.feasible):>9} {p.bbox_w:>8.1f} {p.bbox_h:>8.1f} "
              f"{cc.area_gap:>+9.2%} {cc.hpwl_gap:>+9.2%} {cc.v_relative:>6.2f} {cc.cost:>8.4f}")

    best = results[0]
    bc = best["cost"]
    print(f"\n[pipeline] best sample = #{best['sample']}  feasible={bc.feasible}  "
          f"area_gap={bc.area_gap:+.2%}  hpwl_gap={bc.hpwl_gap:+.2%}  "
          f"V_rel={bc.v_relative:.3f}  Cost={bc.cost:.4f}")
    print("[pipeline] Cost formula (src/cost.cpp): (1 + 0.5*(HPWL_gap+Area_gap)) * "
          "exp(2*V_rel) * max(0.7, RT^0.3); RT (runtime factor) is fixed at 1.0 here "
          "since we have no offline reference runtime -- real submissions can score "
          "up to ~30% lower via RT<1. Cost<1.0 means beating the baseline solver.")

    out_path = Path(args.out_sol)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for i in range(n):
            p = best["pack"]
            f.write(f"{i} {p.x[i]:.3f} {p.y[i]:.3f} {p.w[i]:.3f} {p.h[i]:.3f}\n")
    print(f"[pipeline] wrote best solution -> {out_path}  (id x y w h, matches CLAUDE.md .sol convention)")
    print("\n[pipeline] NOTE: soft-block dimensions here are a placeholder square "
          "(w=h=sqrt(area)) -- this model predicts TOPOLOGY only, not aspect ratio. "
          "Cost above uses the real formula and real baseline metrics, but skips "
          "bbox_balance_pass/holes_fill_pass/grouping_repair/boundary_repair (see "
          "pack_tree.py) and assumes RT=1.0, so treat it as a lower bound on what "
          "the actual C++ submission path would score, not the final number.")


if __name__ == "__main__":
    main()
