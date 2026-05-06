#!/usr/bin/env python3
"""
floorset_to_txt.py

Convert a FloorSet-Lite (or iccad2026 contest) sample loaded via the official
PyTorch DataLoaders (primeLoader.py / liteLoader.py) into the plain-text
format read by our C++ solver.

Two ways to run:

  # 1) From inside the IntelLabs/FloorSet repository (ideal):
  #    Run their dataloader, then call our function on the batch tensors.

  # 2) Stand-alone, given a single pickle that already holds the tensors as
  #    a dict with the official keys.  This matches the per-sample structure
  #    described in the FloorSet README:
  #
  #      area_target           : Tensor[n_blocks]
  #      b2b_connectivity      : Tensor[n_b2b_edges, 3]   (i, j, weight)
  #      p2b_connectivity      : Tensor[n_p2b_edges, 3]   (terminal, block, weight)
  #      pins_pos              : Tensor[n_terminals, 2]
  #      placement_constraints : Tensor[n_blocks, 5]      [fixed, preplaced, MIB, cluster, boundary]
  #      fp_sol                : Tensor[n_blocks, 4]      (w, h, x, y) for each block (ground truth)
  #      metrics_sol           : Tensor[8]                (area, ..., b2b_wl, p2b_wl)
  #
  # The exact field names in the iccad2026 release of FloorSet may vary slightly
  # (e.g. nested per-constraint tensors).  Adjust the readers below if you see a
  # KeyError -- they are guarded by .get() with fallbacks where possible.

Usage:
    python floorset_to_txt.py SAMPLE.pkl OUT.txt
    python floorset_to_txt.py --validation INDEX OUT.txt    (uses iccad2026_evaluate.py)

Author: skeleton, adjust as needed for the actual repo layout.
"""

import argparse
import os
import pickle
import sys

import numpy as np


def to_np(x):
    """Robust convert torch.Tensor / numpy / list to numpy."""
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


def write_txt(out_path, sample, baseline_hpwl=0.0, baseline_area=0.0):
    """Write one sample to text format compatible with our C++ solver."""
    area_target = to_np(sample["area_target"]).reshape(-1).astype(float)
    n_blocks = int(area_target.shape[0])
    pins = to_np(sample["pins_pos"]).reshape(-1, 2).astype(float) \
        if "pins_pos" in sample else np.zeros((0, 2))
    n_terminals = int(pins.shape[0])

    b2b = to_np(sample["b2b_connectivity"]).reshape(-1, 3).astype(float) \
        if "b2b_connectivity" in sample else np.zeros((0, 3))
    p2b = to_np(sample["p2b_connectivity"]).reshape(-1, 3).astype(float) \
        if "p2b_connectivity" in sample else np.zeros((0, 3))

    pc = to_np(sample["placement_constraints"]).reshape(n_blocks, -1).astype(int) \
        if "placement_constraints" in sample else np.zeros((n_blocks, 5), dtype=int)
    # pc columns: [fixed, preplaced, MIB, cluster, boundary] per the FloorSet docs
    is_fixed = pc[:, 0]
    is_pre   = pc[:, 1]
    mib_flag = pc[:, 2]
    grp_flag = pc[:, 3]
    bnd_code = pc[:, 4]    # 0..7 if set, -1 (or 0) if not

    # ground-truth dimensions / locations -- used as input for fixed and preplaced
    fp_sol = to_np(sample["fp_sol"]).reshape(n_blocks, 4).astype(float) \
        if "fp_sol" in sample else np.zeros((n_blocks, 4))
    # fp_sol layout: w, h, x, y

    # try a few common keys for the boundary-code if FloorSet wraps it differently
    bedges = bnd_code.copy()
    # convention in our C++: -1 == none; FloorSet uses 0-based codes when active
    # heuristic: if column 4 of placement_constraints is a flag, the actual code
    # may live elsewhere -- let it pass through and adjust on real data.
    bedges = np.where(bedges == 0, -1, bedges)

    # Group ids: FloorSet ships per-block group IDs as auxiliary tensors.  If
    # not present, fall back to constructing trivial groups from flags.
    if "grouping_groups" in sample:
        grouping_groups = [list(map(int, g)) for g in sample["grouping_groups"]]
    else:
        # Group together blocks with identical positive group_id if available
        gid = to_np(sample.get("group_id", np.full(n_blocks, -1))).astype(int)
        groups = {}
        for i, g in enumerate(gid):
            if g < 0: continue
            groups.setdefault(int(g), []).append(i)
        grouping_groups = list(groups.values())

    if "mib_groups" in sample:
        mib_groups = [list(map(int, g)) for g in sample["mib_groups"]]
    else:
        mid = to_np(sample.get("mib_id", np.full(n_blocks, -1))).astype(int)
        groups = {}
        for i, g in enumerate(mid):
            if g < 0: continue
            groups.setdefault(int(g), []).append(i)
        mib_groups = list(groups.values())

    with open(out_path, "w") as f:
        f.write("# FloorSet-Lite text dump produced by floorset_to_txt.py\n")
        f.write(f"N_BLOCKS    {n_blocks}\n")
        f.write(f"N_TERMINALS {n_terminals}\n")
        f.write(f"BASELINE_HPWL {baseline_hpwl:.10f}\n")
        f.write(f"BASELINE_AREA {baseline_area:.10f}\n")
        # OUTLINE is informational only
        if "outline" in sample:
            ow, oh = sample["outline"][:2]
            f.write(f"OUTLINE {float(ow):.10f} {float(oh):.10f}\n")

        if n_terminals > 0:
            f.write("TERMINALS\n")
            for i, (x, y) in enumerate(pins):
                f.write(f"{i} {x:.10f} {y:.10f}\n")

        f.write("BLOCKS\n")
        for i in range(n_blocks):
            wi, hi, xi, yi = fp_sol[i]
            isf = int(is_fixed[i])
            isp = int(is_pre[i])
            # MIB / group ids: -1 if not part of any
            mib_id = -1
            for q, g in enumerate(mib_groups):
                if i in g: mib_id = q; break
            grp_id = -1
            for p, g in enumerate(grouping_groups):
                if i in g: grp_id = p; break
            be = int(bedges[i]) if bedges[i] >= 0 else -1
            # default soft-block aspect-ratio band; tighten if you have real bounds
            armin, armax = 0.25, 4.0
            f.write(f"{i} {area_target[i]:.10f} {isf} {isp} "
                    f"{wi:.10f} {hi:.10f} {xi:.10f} {yi:.10f} "
                    f"{mib_id} {grp_id} {be} {armin:.4f} {armax:.4f}\n")

        f.write(f"B2B {b2b.shape[0]}\n")
        for i in range(b2b.shape[0]):
            a, b, w = b2b[i]
            f.write(f"{int(a)} {int(b)} {w:.10f}\n")

        f.write(f"P2B {p2b.shape[0]}\n")
        for i in range(p2b.shape[0]):
            t, b, w = p2b[i]
            f.write(f"{int(t)} {int(b)} {w:.10f}\n")

        f.write(f"GROUPS {len(grouping_groups)}\n")
        for g in grouping_groups:
            f.write(f"{len(g)} " + " ".join(str(int(x)) for x in g) + "\n")
        f.write(f"MIB {len(mib_groups)}\n")
        for g in mib_groups:
            f.write(f"{len(g)} " + " ".join(str(int(x)) for x in g) + "\n")

        f.write("END\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp", help="path to a single-sample .pkl file")
    ap.add_argument("out", help="path to the output .txt file")
    ap.add_argument("--baseline-hpwl", type=float, default=0.0)
    ap.add_argument("--baseline-area", type=float, default=0.0)
    args = ap.parse_args()

    with open(args.inp, "rb") as f:
        sample = pickle.load(f)
    if isinstance(sample, (list, tuple)):
        sample = sample[0]

    # Auto-fill baselines if metrics_sol present
    bhpwl = args.baseline_hpwl
    barea = args.baseline_area
    if "metrics_sol" in sample and (bhpwl == 0 or barea == 0):
        m = to_np(sample["metrics_sol"]).reshape(-1).astype(float)
        # m: [area, num_pins, num_total_nets, num_b2b_nets, num_p2b_nets,
        #     num_hardconstraints, b2b_weighted_wl, p2b_weighted_wl]
        if barea == 0 and m.size >= 1: barea = float(m[0])
        if bhpwl == 0 and m.size >= 8: bhpwl = float(m[6] + m[7])

    write_txt(args.out, sample, baseline_hpwl=bhpwl, baseline_area=barea)
    print(f"wrote {args.out} (n_blocks={int(to_np(sample['area_target']).reshape(-1).shape[0])})")


if __name__ == "__main__":
    main()
