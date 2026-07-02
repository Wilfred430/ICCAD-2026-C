"""FloorSet dataset loader for the ML predictor.

The FloorSet Lite dataset is laid out as:

    LiteTensorDataTest/
        config_<n>/
            litedata_<k>.pth     # list[list[Tensor]] -- inputs
            litelabel_<k>.pth    # list[list[Tensor]] -- ground-truth

Each *case* has:
    inputs:
        [0] blocks      [N, 6]   = (area, is_fixed, is_preplaced,
                                    mib_id, cluster_id, boundary_code)
        [1] b2b         [B, 3]   = (block_a, block_b, weight)
        [2] p2b         [P, 3]   = (pin_id, block_b, weight)
        [3] pins_pos    [T, 2]   = (px, py)
    labels:
        [0] metrics     [8]      = global stats (ignored at training)
        [1] geometry    [N, 5, 2] = polygon vertices (5 = closed rectangle)

The TRAIN format (1M-case `floorset_lite/worker_*/layouts*.th`) additionally
ships `tree_sol` [N-1, 3] = (parent_id, child_id, direction) -- the B*-tree
edge list that produced `fp_sol`.  See `tree_sol_to_sequence()` below for how
this is turned into generation-order training targets for `model_tree.py`.

This file converts each case into a flat per-block feature tensor
suitable for a Transformer model, plus a target (cx, cy, w, h) per block
(for the coordinate-regression model in `model.py`) and, when available,
B*-tree generation-order targets (for the generative model in
`model_tree.py`).
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset


# ---- feature schema ---------------------------------------------------------
#
# Per-block feature vector (16 dims).  Order matters because predict.py
# constructs the same vector at inference time.
BLOCK_FEAT_DIM = 16

# Indices into the 16-dim feature vector (for readability):
F_AREA          = 0   # area_target (raw)
F_AREA_LOG      = 1   # log(area+1) -- numerical stability
F_IS_FIXED      = 2
F_IS_PREPLACED  = 3
F_HAS_MIB       = 4
F_HAS_CLUSTER   = 5
F_BND_LEFT      = 6
F_BND_RIGHT     = 7
F_BND_TOP       = 8
F_BND_BOTTOM    = 9
F_DEG_B2B       = 10  # log(1 + b2b edge count)
F_DEG_P2B       = 11  # log(1 + p2b edge count)
F_W_HINT        = 12  # 0 for soft blocks; w from label for fixed/preplaced
F_H_HINT        = 13
F_X_HINT        = 14  # 0 for soft blocks; x from label for preplaced
F_Y_HINT        = 15


# ---- helpers ----------------------------------------------------------------

def _rect_from_polygon(vertices: torch.Tensor) -> Tuple[float, float, float, float]:
    """Vertices [5, 2] -> (cx, cy, w, h) bounding rectangle.

    The FloorSet polygons are closed rectangles (4 unique + 1 repeat), so
    this is just min/max over x and y.
    """
    x = vertices[:, 0]
    y = vertices[:, 1]
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    return (
        0.5 * (x_min + x_max),  # cx
        0.5 * (y_min + y_max),  # cy
        x_max - x_min,           # w
        y_max - y_min,           # h
    )


def case_to_features(
    blocks: torch.Tensor,
    b2b: torch.Tensor,
    p2b: torch.Tensor,
    geometry: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build the per-block feature tensor [N, 16].

    `geometry` (optional, [N, 5, 2]) is used for fixed/preplaced hints
    during training; at inference my_optimizer_ml.py supplies a stub
    geometry derived from target_positions instead.
    """
    n = blocks.shape[0]
    feat = torch.zeros((n, BLOCK_FEAT_DIM), dtype=torch.float32)

    area = blocks[:, 0]
    feat[:, F_AREA]         = area
    feat[:, F_AREA_LOG]     = torch.log(area + 1.0)
    feat[:, F_IS_FIXED]     = blocks[:, 1]
    feat[:, F_IS_PREPLACED] = blocks[:, 2]
    feat[:, F_HAS_MIB]      = (blocks[:, 3] > 0).float()
    feat[:, F_HAS_CLUSTER]  = (blocks[:, 4] > 0).float()

    # Boundary code bitmask: 1=L, 2=R, 4=T, 8=B (and corner sums).
    bcode = blocks[:, 5].long()
    feat[:, F_BND_LEFT]   = ((bcode & 1) > 0).float()
    feat[:, F_BND_RIGHT]  = ((bcode & 2) > 0).float()
    feat[:, F_BND_TOP]    = ((bcode & 4) > 0).float()
    feat[:, F_BND_BOTTOM] = ((bcode & 8) > 0).float()

    # Connectivity degree: how many b2b/p2b nets each block participates in.
    if b2b is not None and b2b.numel() > 0:
        ids_a = b2b[:, 0].long().clamp(0, n - 1)
        ids_b = b2b[:, 1].long().clamp(0, n - 1)
        deg = torch.zeros(n, dtype=torch.float32)
        deg.scatter_add_(0, ids_a, torch.ones_like(ids_a, dtype=torch.float32))
        deg.scatter_add_(0, ids_b, torch.ones_like(ids_b, dtype=torch.float32))
        feat[:, F_DEG_B2B] = torch.log(deg + 1.0)

    if p2b is not None and p2b.numel() > 0:
        ids_b = p2b[:, 1].long().clamp(0, n - 1)
        deg = torch.zeros(n, dtype=torch.float32)
        deg.scatter_add_(0, ids_b, torch.ones_like(ids_b, dtype=torch.float32))
        feat[:, F_DEG_P2B] = torch.log(deg + 1.0)

    # Geometry hints for fixed / preplaced (zero for soft).
    if geometry is not None:
        for i in range(n):
            cx, cy, w, h = _rect_from_polygon(geometry[i])
            is_fixed     = bool(feat[i, F_IS_FIXED] > 0.5)
            is_preplaced = bool(feat[i, F_IS_PREPLACED] > 0.5)
            if is_fixed or is_preplaced:
                feat[i, F_W_HINT] = w
                feat[i, F_H_HINT] = h
            if is_preplaced:
                feat[i, F_X_HINT] = cx
                feat[i, F_Y_HINT] = cy

    return feat


def case_to_targets(geometry: torch.Tensor) -> torch.Tensor:
    """Geometry [N, 5, 2] -> targets [N, 4] = (cx, cy, w, h)."""
    n = geometry.shape[0]
    tgt = torch.zeros((n, 4), dtype=torch.float32)
    for i in range(n):
        cx, cy, w, h = _rect_from_polygon(geometry[i])
        tgt[i] = torch.tensor([cx, cy, w, h])
    return tgt


# ---- B*-tree generation-order targets ---------------------------------------
#
# `tree_sol` ships as an edge list [N-1, 3] = (parent_id, child_id, direction),
# direction in {0, 1}: 0 = "left child" (placed to the right of parent,
# touching parent's right edge), 1 = "right child" (placed directly above
# parent, same x).  Both use the running horizontal contour to pick y -- see
# collaborate/src/packer.cpp::Packer::pack() for the reference C++ packer
# (this is our own SA solver's packer; the semantics were confirmed to match
# by decoding a sample of `tree_sol` against `fp_sol` and cross-checking the
# left/right-child x-formula, which reproduced ~50-80% of blocks exactly --
# the residual gap is Intel's own post-pack compaction, which we don't need
# to reproduce here since we only want the *topology* labels, not geometry).
#
# The edge rows are already in DFS preorder (parent appears before either
# child; a node's "direction 0" child and its whole subtree are fully listed
# before its "direction 1" child).  That means simply prepending the root
# gives a valid generation order where every parent has already appeared
# by the time its child is emitted -- exactly what an autoregressive model
# needs for teacher forcing.
def tree_sol_to_sequence(tree_sol: torch.Tensor, n: int):
    """tree_sol [N-1, 3] -> (gen_order [N], parent_step [N], direction [N]).

    gen_order[t]    = block id placed at generation step t (gen_order[0] = root)
    parent_step[t]  = index (into gen_order) of step t's parent; -1 for root
    direction[t]    = 0/1 direction bit used to attach step t to its parent;
                       0 for root (unused, always masked out in the loss)
    """
    rows = tree_sol.tolist()
    child_ids = {int(r[1]) for r in rows}
    root = next(i for i in range(n) if i not in child_ids)

    gen_order    = torch.full((n,), -1, dtype=torch.int64)
    parent_step  = torch.full((n,), -1, dtype=torch.int64)
    direction    = torch.zeros(n, dtype=torch.float32)

    gen_order[0] = root
    step_of_block = {root: 0}
    for t, (p, c, d) in enumerate(rows, start=1):
        p, c, d = int(p), int(c), int(d)
        gen_order[t]   = c
        parent_step[t] = step_of_block[p]
        direction[t]   = float(d)
        step_of_block[c] = t

    return gen_order, parent_step, direction


# ---- Dataset class ---------------------------------------------------------

class FloorSetLiteDataset(Dataset):
    """Iterate cases across either dataset format that FloorSet ships.

    The class auto-detects which layout it's pointed at:

        ┌── TEST format ─────────────────────────────────┐
        │ <root>/config_<n>/litedata_<k>.pth   (inputs)  │  ~100 cases total
        │ <root>/config_<n>/litelabel_<k>.pth  (labels)  │
        └────────────────────────────────────────────────┘

        ┌── TRAIN format (LiteTensorData_v2) ────────────┐
        │ <root>/floorset_lite/worker_<n>/layouts<k>.pt  │  ~1.1M cases total
        │   each file = tuple of 7 tensors packing 112    │
        │   cases (matches Intel's lite_dataset.py)       │
        └────────────────────────────────────────────────┘

    Args:
        root: path to either dataset root.
        max_blocks: pad / truncate each case to this many blocks (default 128).
        max_terms:  pad / truncate terminals to this many (default 512).

    Returns a dict per case:
        blocks_feat : [max_blocks, 16]   features
        blocks_mask : [max_blocks]       True for valid rows (non-padding)
        terms       : [max_terms, 2]     terminal (x, y), zero-padded
        terms_mask  : [max_terms]
        target      : [max_blocks, 4]    (cx, cy, w, h); zero for padding
    """

    # Format codes
    FORMAT_TEST  = "test"   # config_<n>/litedata + litelabel
    FORMAT_TRAIN = "train"  # floorset_lite/worker_<n>/layouts<k>.pt (7-tuple)

    def __init__(
        self,
        root: str,
        max_blocks: int = 128,
        max_terms: int = 512,
    ):
        self.root = Path(root)
        self.max_blocks = max_blocks
        self.max_terms = max_terms

        # ---- auto-detect format -----------------------------------------
        # FloorSet v2 ships files as `layouts_<start_idx>.th` (note: .th
        # extension, not .pt -- legacy torch convention).  Older FloorSet
        # builds used .pt.  Glob both to be safe.
        train_files = sorted(
            list((self.root / "floorset_lite").glob("worker_*/layouts*.th"))
            + list((self.root / "floorset_lite").glob("worker_*/layouts*.pt"))
        )
        if train_files:
            self.format = self.FORMAT_TRAIN
            self.files: List[Tuple[Path, Path]] = [(p, p) for p in train_files]  # label = data file
            # Each layouts<k>.pt holds 112 cases (FloorSet's lite convention).
            # We probe the first file once to get the actual count in case it
            # ever changes; cached so we don't re-read.
            probe = torch.load(train_files[0], weights_only=False)
            self.cases_per_file = len(probe[0])  # outer dim of any tensor in the 7-tuple
            print(f"[dataset] format=TRAIN, {len(train_files)} files × {self.cases_per_file} cases = "
                  f"{len(train_files) * self.cases_per_file} total")
        else:
            self.format = self.FORMAT_TEST
            self.files = []
            for config_dir in sorted(self.root.glob("config_*")):
                for data_path in sorted(config_dir.glob("litedata_*.pth")):
                    label_path = data_path.parent / data_path.name.replace("litedata_", "litelabel_")
                    if label_path.exists():
                        self.files.append((data_path, label_path))
            self.cases_per_file = None
            print(f"[dataset] format=TEST, {len(self.files)} (data, label) pairs")

        # ---- build flat index of (file_index, case_index_within_file) ----
        self.index: List[Tuple[int, int]] = []
        if self.format == self.FORMAT_TRAIN:
            for fi in range(len(self.files)):
                for ci in range(self.cases_per_file):
                    self.index.append((fi, ci))
        else:
            for fi, (dp, _) in enumerate(self.files):
                n_cases = len(torch.load(dp, weights_only=False))
                for ci in range(n_cases):
                    self.index.append((fi, ci))

        # cache last-loaded file (huge speedup, files are large)
        self._cache_fi = -1
        self._cache_data = None
        self._cache_label = None

    def __len__(self) -> int:
        return len(self.index)

    def block_counts(self, cache: bool = True, verbose: bool = True) -> torch.Tensor:
        """Per-case block count as an int tensor of length len(self).

        Used for size-weighted sampling so the model spends more capacity on
        the large cases (which dominate the v10 contest score).  For the
        ~1M-case TRAIN format this scans every file once (a few minutes) and
        caches the result to <root>/.block_counts_<N>.pt; later runs load it
        instantly.
        """
        cache_path = self.root / f".block_counts_{len(self)}.pt"
        if cache and cache_path.exists():
            counts = torch.load(cache_path)
            if int(counts.numel()) == len(self):
                if verbose:
                    print(f"[dataset] loaded cached block counts from {cache_path}")
                return counts

        if verbose:
            print(f"[dataset] scanning {len(self.files)} files for block counts "
                  f"(one-time; cached afterwards)...")
        counts = torch.zeros(len(self), dtype=torch.int32)
        for idx, (fi, ci) in enumerate(self.index):
            self._load_file(fi)              # cached: each file is loaded once
            if self.format == self.FORMAT_TRAIN:
                counts[idx] = int(self._cache_data[0][ci].shape[0])
            else:
                counts[idx] = int(self._cache_data[ci][0].shape[0])
            if verbose and idx % 50000 == 0 and idx > 0:
                print(f"  ...{idx}/{len(self)} cases", flush=True)
        if cache:
            torch.save(counts, cache_path)
            if verbose:
                print(f"[dataset] cached block counts -> {cache_path}")
        return counts

    def _load_file(self, fi: int):
        if fi == self._cache_fi:
            return
        dp, lp = self.files[fi]
        self._cache_data  = torch.load(dp, weights_only=False)
        if self.format == self.FORMAT_TEST:
            self._cache_label = torch.load(lp, weights_only=False)
        else:
            # TRAIN format: data and labels are in the SAME 7-tuple file.
            self._cache_label = None
        self._cache_fi = fi

    def _unpack_case(self, ci: int):
        """Returns (blocks, b2b, p2b, pins_pos, geometry, tree_sol)."""
        if self.format == self.FORMAT_TRAIN:
            t = self._cache_data
            # 7-tuple per FloorSet's lite_dataset.py
            blocks      = t[0][ci]          # [N, 6]: area + 5 constraint cols
            b2b         = t[1][ci]          # [B, 3]
            p2b         = t[2][ci]          # [P, 3]
            pins_pos    = t[3][ci]          # [T, 2]
            tree_sol    = t[4][ci]          # [N-1, 3]: (parent, child, direction)
            fp_sol      = t[5][ci]          # [N, 4]: (w, h, x, y)
            # metrics  = t[6][ci]  -- unused

            # Convert (w, h, x, y) -> 5-vertex polygon to match test loader
            # (so case_to_features / case_to_targets stay uniform).
            n = blocks.shape[0]
            geometry = torch.zeros((n, 5, 2), dtype=torch.float32)
            geometry[:, 0, 0] = fp_sol[:, 2]                    # x_min
            geometry[:, 0, 1] = fp_sol[:, 3]                    # y_min
            geometry[:, 1, 0] = fp_sol[:, 2]                    # x_min
            geometry[:, 1, 1] = fp_sol[:, 3] + fp_sol[:, 1]     # y_max
            geometry[:, 2, 0] = fp_sol[:, 2] + fp_sol[:, 0]     # x_max
            geometry[:, 2, 1] = fp_sol[:, 3] + fp_sol[:, 1]     # y_max
            geometry[:, 3, 0] = fp_sol[:, 2] + fp_sol[:, 0]     # x_max
            geometry[:, 3, 1] = fp_sol[:, 3]                    # y_min
            geometry[:, 4]    = geometry[:, 0]                  # closing vertex
            return blocks, b2b, p2b, pins_pos, geometry, tree_sol

        # TEST format -- no tree_sol available (labels are metrics + geometry only)
        case_in    = self._cache_data[ci]
        case_label = self._cache_label[ci]
        blocks   = case_in[0]
        b2b      = case_in[1]
        p2b      = case_in[2]
        pins_pos = case_in[3]
        geometry = case_label[1]
        return blocks, b2b, p2b, pins_pos, geometry, None

    def __getitem__(self, idx: int):
        fi, ci = self.index[idx]
        self._load_file(fi)

        blocks, b2b, p2b, pins_pos, geometry, tree_sol = self._unpack_case(ci)

        n = blocks.shape[0]
        n_use = min(n, self.max_blocks)

        feat = case_to_features(blocks[:n_use], b2b, p2b, geometry[:n_use])
        tgt  = case_to_targets(geometry[:n_use])

        # Pad blocks
        blocks_feat = torch.zeros((self.max_blocks, BLOCK_FEAT_DIM), dtype=torch.float32)
        blocks_feat[:n_use] = feat
        blocks_mask = torch.zeros(self.max_blocks, dtype=torch.bool)
        blocks_mask[:n_use] = True

        target = torch.zeros((self.max_blocks, 4), dtype=torch.float32)
        target[:n_use] = tgt

        # Pad terminals
        t = pins_pos.shape[0] if pins_pos is not None else 0
        t_use = min(t, self.max_terms)
        terms = torch.zeros((self.max_terms, 2), dtype=torch.float32)
        if t_use > 0:
            terms[:t_use] = pins_pos[:t_use]
        terms_mask = torch.zeros(self.max_terms, dtype=torch.bool)
        terms_mask[:t_use] = True

        # B*-tree generation-order targets (TRAIN format only, and only when
        # the case fits within max_blocks -- truncating a tree loses parent
        # references for blocks past the cutoff, so we just mark it invalid
        # rather than emit a broken partial tree).
        has_tree = tree_sol is not None and n <= self.max_blocks
        gen_order   = torch.full((self.max_blocks,), -1, dtype=torch.int64)
        parent_step = torch.full((self.max_blocks,), -1, dtype=torch.int64)
        direction   = torch.zeros(self.max_blocks, dtype=torch.float32)
        if has_tree:
            go, ps, dr = tree_sol_to_sequence(tree_sol, n)
            gen_order[:n]   = go
            parent_step[:n] = ps
            direction[:n]   = dr

        return {
            "blocks_feat": blocks_feat,
            "blocks_mask": blocks_mask,
            "terms":       terms,
            "terms_mask":  terms_mask,
            "target":      target,
            "n_blocks":    n_use,
            "has_tree":    has_tree,
            "gen_order":   gen_order,
            "parent_step": parent_step,
            "direction":   direction,
        }
