#!/usr/bin/env python3
"""
Visualize floorplan optimization results using the official visualize.py.

Usage:
    # Visualize ground truth only for test case 0:
    python my_visualize.py --test-id 0

    # Compare ground truth vs optimizer solution side by side:
    python my_visualize.py --solutions my_optimizer_solutions.json --test-id 0

    # Visualize all cases in a solutions file and save to disk:
    python my_visualize.py --solutions my_optimizer_solutions.json --all --save-dir ./vis_output

    # Save without displaying (headless):
    python my_visualize.py --solutions my_optimizer_solutions.json --all --save-dir ./vis_output --no-show

How to generate solutions JSON:
    python iccad2026_evaluate.py --evaluate my_optimizer.py --save-solutions
    # This creates my_optimizer_solutions.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from litetestLoader import FloorplanDatasetLiteTest
from visualize import get_hard_color


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_solutions(solutions_path: str) -> Dict[int, dict]:
    """Load solutions JSON saved by iccad2026_evaluate.py --save-solutions."""
    with open(solutions_path) as f:
        data = json.load(f)
    by_id = {}
    for sol in data.get('solutions', []):
        tid = sol['test_id']
        by_id[tid] = {
            'positions': [tuple(p) for p in sol['positions']],
            'block_count': sol['block_count'],
        }
    return by_id


def gt_positions_from_polygons(polygons, block_count: int) -> List[Tuple]:
    """Extract (x, y, w, h) from padded polygon tensor (ground truth)."""
    out = []
    for i in range(block_count):
        block = polygons[i]
        valid = block[block[:, 0] != -1]
        if len(valid) > 0:
            x_min, y_min = valid.min(dim=0).values
            x_max, y_max = valid.max(dim=0).values
            out.append((float(x_min), float(y_min),
                        float(x_max - x_min), float(y_max - y_min)))
        else:
            out.append((0.0, 0.0, 1.0, 1.0))
    return out


def xywh_to_fp_sol(positions: List[Tuple]) -> List[List]:
    """Convert [(x, y, w, h), ...] → [[w, h, x, y], ...] for visualize_lite."""
    return [[w, h, x, y] for (x, y, w, h) in positions]


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _draw_floorplan(ax, fp_sol, b2b_conn, p2b_conn, pins_pos, constraints,
                    block_count: int, title: str = ""):
    """
    Draw a floorplan on *ax*.

    fp_sol elements are [w, h, x, y] — same convention as visualize_lite().
    """
    from shapely.geometry import Polygon
    from matplotlib.patches import Circle, Polygon as MplPolygon

    W, H = 0.0, 0.0
    all_polys: Dict[int, Polygon] = {}

    for ind in range(min(block_count, len(fp_sol))):
        elem = fp_sol[ind]
        w, h, x, y = elem[0], elem[1], elem[2], elem[3]
        if w <= 0 or h <= 0:
            continue

        poly = Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)])
        all_polys[ind] = poly

        hard_const = constraints[ind] if ind < len(constraints) else [0, 0, 0, 0, 0]
        face_color, label_text = get_hard_color(hard_const)

        patch = MplPolygon(
            list(poly.exterior.coords),
            closed=True, fill=True,
            edgecolor='black', facecolor=face_color,
            label=label_text, alpha=0.35,
        )
        ax.add_patch(patch)

        W = max(W, x + w)
        H = max(H, y + h)
        ax.annotate(str(ind + 1), (x, y), fontsize=5, color='black')

    # Pins
    pin_r = max(W, H) * 0.005 if max(W, H) > 0 else 0.5
    for pi in range(pins_pos.shape[0]):
        px, py = float(pins_pos[pi][0]), float(pins_pos[pi][1])
        if px == -1.0:
            continue
        ax.add_patch(Circle((px, py), radius=pin_r, color='g', zorder=3))

    # B2B wires (red)
    for edge in b2b_conn:
        src, dst = int(edge[0]), int(edge[1])
        if src == -1 or dst == -1:
            continue
        if src in all_polys and dst in all_polys:
            p1, p2 = all_polys[src], all_polys[dst]
            ax.plot([p1.bounds[0], p2.bounds[0]], [p1.bounds[1], p2.bounds[1]],
                    color='red', linewidth=0.15, alpha=0.5)

    # P2B wires (blue)
    for edge in p2b_conn:
        src_pin, dst_blk = int(edge[0]), int(edge[1])
        if src_pin == -1 or dst_blk == -1:
            continue
        if src_pin < pins_pos.shape[0] and dst_blk in all_polys:
            px, py = float(pins_pos[src_pin][0]), float(pins_pos[src_pin][1])
            pb = all_polys[dst_blk]
            ax.plot([px, pb.bounds[0]], [py, pb.bounds[1]],
                    color='blue', linewidth=0.1, alpha=0.4)

    margin = 1.25
    ax.set_xlim(0, W * margin)
    ax.set_ylim(0, H * margin)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title, fontsize=9)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(),
              loc='upper right', title='Constraints', fontsize=6)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def visualize_case(
    dataset: FloorplanDatasetLiteTest,
    test_id: int,
    optimizer_positions: Optional[List[Tuple]] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Visualize one test case.

    Shows ground truth on the left; if *optimizer_positions* is provided,
    shows the optimizer solution on the right.
    """
    sample = dataset[test_id]
    inputs, labels = sample['input'], sample['label']
    area_target, b2b_conn, p2b_conn, pins_pos, constraints = inputs
    polygons, metrics = labels

    block_count = int((area_target != -1).sum().item())

    gt_pos = gt_positions_from_polygons(polygons, block_count)
    gt_fp_sol = xywh_to_fp_sol(gt_pos)

    has_opt = optimizer_positions is not None and len(optimizer_positions) > 0
    ncols = 2 if has_opt else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 7))
    if ncols == 1:
        axes = [axes]

    _draw_floorplan(axes[0], gt_fp_sol, b2b_conn, p2b_conn, pins_pos, constraints,
                    block_count,
                    title=f"Ground Truth — Case {test_id}  ({block_count} blocks)")

    if has_opt:
        opt_fp_sol = xywh_to_fp_sol(list(optimizer_positions)[:block_count])
        _draw_floorplan(axes[1], opt_fp_sol, b2b_conn, p2b_conn, pins_pos, constraints,
                        block_count,
                        title=f"Optimizer Solution — Case {test_id}")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved → {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize FloorSet-Lite floorplan results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--solutions', '-s', default=None,
                        help='Solutions JSON produced by --save-solutions')
    parser.add_argument('--test-id', '-t', type=int, default=None,
                        help='Single test case ID (0-99)')
    parser.add_argument('--all', '-a', action='store_true',
                        help='Visualize every case found in the solutions file '
                             '(or all 100 validation cases when no --solutions)')
    parser.add_argument('--data-path', '-d', default='../',
                        help='Path to FloorSet data root (default: ../)')
    parser.add_argument('--save-dir', default=None,
                        help='Directory for output PNG files')
    parser.add_argument('--no-show', action='store_true',
                        help='Skip interactive display (useful with --save-dir)')
    args = parser.parse_args()

    if args.test_id is None and not args.all:
        parser.error("Provide --test-id N or --all")

    # Headless backend when not showing
    if args.no_show:
        matplotlib.use('Agg')

    print("Loading validation dataset …")
    dataset = FloorplanDatasetLiteTest(args.data_path)
    print(f"  {len(dataset)} validation cases available")

    solutions: Dict[int, dict] = {}
    if args.solutions:
        solutions = load_solutions(args.solutions)
        print(f"  Loaded {len(solutions)} optimizer solutions from {args.solutions}")

    # Which test IDs to process
    if args.all:
        if solutions:
            test_ids = sorted(solutions.keys())
        else:
            test_ids = list(range(len(dataset)))
    else:
        test_ids = [args.test_id]

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        print(f"  Output directory: {args.save_dir}")

    show = not args.no_show

    for tid in test_ids:
        print(f"Visualizing case {tid} …")
        opt_pos = solutions[tid]['positions'] if tid in solutions else None

        save_path = None
        if args.save_dir:
            tag = "_compare" if opt_pos else "_gt"
            save_path = os.path.join(args.save_dir, f"case_{tid:03d}{tag}.png")

        try:
            visualize_case(dataset, tid, opt_pos, save_path, show)
        except Exception as exc:
            print(f"  ERROR on case {tid}: {exc}")

    print("Done.")


if __name__ == '__main__':
    main()
