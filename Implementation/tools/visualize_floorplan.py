#!/usr/bin/env python3
"""
visualize_floorplan.py -- Floorplan Layout Visualizer

Reads a solver input (.txt) and solution (.sol) file, then generates a
layout image showing block placements, constraints, overlaps, pins, and
connectivity in the official white-background style.

Usage:
    python3 tools/visualize_floorplan.py benchmarks/toy.txt benchmarks/toy.sol
    python3 tools/visualize_floorplan.py benchmarks/toy.txt benchmarks/toy.sol -o plots/toy.png

For batch mode (all cases from eval):
    python3 tools/visualize_floorplan.py --batch eval_results/ -o plots/
    python3 tools/visualize_floorplan.py --batch eval_results/ -o plots/ --max-cases 100
"""

import argparse
import os
import sys
from pathlib import Path
from itertools import combinations

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for WSL
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Rectangle, Circle
    from matplotlib.lines import Line2D
except ImportError:
    print("Error: matplotlib is required. Install with: pip install matplotlib",
          file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Constraint colors (matches the official visualize.py palette)
# ---------------------------------------------------------------------------
CONSTRAINT_COLORS = {
    "soft":      ("silver",       "no constraint"),
    "fixed":     ("violet",       "fixed"),
    "preplaced": ("gray",         "preplaced"),
    "mib":       ("darkgreen",    "MIB"),
    "cluster":   ("red",          "cluster"),
    "boundary":  ("goldenrod",    "boundary"),
}

BOUNDARY_NAMES = {
    -1: "", 0: "",
    1: "left", 2: "right", 4: "top", 8: "bottom",
    5: "top-left", 9: "bottom-left", 6: "top-right", 10: "bottom-right",
}


def get_block_style(block_info):
    """Return (facecolor, label) based on block constraint type."""
    if block_info.get("is_preplaced"):
        return CONSTRAINT_COLORS["preplaced"]
    if block_info.get("is_fixed"):
        return CONSTRAINT_COLORS["fixed"]
    if block_info.get("grp_id", -1) >= 0:
        return CONSTRAINT_COLORS["cluster"]
    if block_info.get("mib_grp", -1) >= 0:
        return CONSTRAINT_COLORS["mib"]
    bedge = block_info.get("bedge", -1)
    if bedge > 0:
        return CONSTRAINT_COLORS["boundary"]
    return CONSTRAINT_COLORS["soft"]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_input(path):
    """Parse the .txt input file."""
    data = {
        "n_blocks": 0, "n_terminals": 0,
        "outline_w": 0.0, "outline_h": 0.0,
        "baseline_hpwl": 0.0, "baseline_area": 0.0,
        "blocks": [], "terminals": [],
        "b2b": [], "p2b": [],
        "groups": [], "mib_groups": [],
    }

    with open(path, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1; continue
        tokens = line.split()
        kw = tokens[0]

        if kw == "N_BLOCKS":
            data["n_blocks"] = int(tokens[1])
        elif kw == "N_TERMINALS":
            data["n_terminals"] = int(tokens[1])
        elif kw == "OUTLINE":
            data["outline_w"] = float(tokens[1])
            data["outline_h"] = float(tokens[2])
        elif kw == "BASELINE_HPWL":
            data["baseline_hpwl"] = float(tokens[1])
        elif kw == "BASELINE_AREA":
            data["baseline_area"] = float(tokens[1])
        elif kw == "TERMINALS":
            i += 1
            for _ in range(data["n_terminals"]):
                while i < len(lines) and (lines[i].strip().startswith("#") or not lines[i].strip()):
                    i += 1
                t = lines[i].strip().split()
                data["terminals"].append((float(t[1]), float(t[2])))
                i += 1
            continue
        elif kw == "BLOCKS":
            i += 1
            for _ in range(data["n_blocks"]):
                while i < len(lines) and (lines[i].strip().startswith("#") or not lines[i].strip()):
                    i += 1
                t = lines[i].strip().split()
                data["blocks"].append({
                    "id": int(t[0]), "area": float(t[1]),
                    "is_fixed": int(t[2]) != 0, "is_preplaced": int(t[3]) != 0,
                    "w_in": float(t[4]), "h_in": float(t[5]),
                    "x_in": float(t[6]), "y_in": float(t[7]),
                    "mib_grp": int(t[8]), "grp_id": int(t[9]),
                    "bedge": int(t[10]),
                    "ar_min": float(t[11]), "ar_max": float(t[12]),
                })
                i += 1
            continue
        elif kw == "B2B":
            n = int(tokens[1]); i += 1
            for _ in range(n):
                while i < len(lines) and (lines[i].strip().startswith("#") or not lines[i].strip()):
                    i += 1
                t = lines[i].strip().split()
                data["b2b"].append((int(t[0]), int(t[1]), float(t[2])))
                i += 1
            continue
        elif kw == "P2B":
            n = int(tokens[1]); i += 1
            for _ in range(n):
                while i < len(lines) and (lines[i].strip().startswith("#") or not lines[i].strip()):
                    i += 1
                t = lines[i].strip().split()
                data["p2b"].append((int(t[0]), int(t[1]), float(t[2])))
                i += 1
            continue
        elif kw == "GROUPS":
            n = int(tokens[1]); i += 1
            for _ in range(n):
                while i < len(lines) and (lines[i].strip().startswith("#") or not lines[i].strip()):
                    i += 1
                t = lines[i].strip().split()
                cnt = int(t[0])
                data["groups"].append([int(t[j+1]) for j in range(cnt)])
                i += 1
            continue
        elif kw == "MIB":
            n = int(tokens[1]); i += 1
            for _ in range(n):
                while i < len(lines) and (lines[i].strip().startswith("#") or not lines[i].strip()):
                    i += 1
                t = lines[i].strip().split()
                cnt = int(t[0])
                data["mib_groups"].append([int(t[j+1]) for j in range(cnt)])
                i += 1
            continue
        elif kw == "END":
            break
        i += 1
    return data


def parse_solution(path):
    """Parse the .sol output file."""
    blocks = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("N_BLOCKS"):
                continue
            t = line.split()
            if len(t) >= 5:
                blocks.append({
                    "id": int(t[0]),
                    "x": float(t[1]), "y": float(t[2]),
                    "w": float(t[3]), "h": float(t[4]),
                })
    return blocks


# ---------------------------------------------------------------------------
# Overlap detection
# ---------------------------------------------------------------------------

def find_overlaps(sol_blocks, tol=1e-4):
    """Find all pairwise overlapping block pairs."""
    overlaps = []
    for a, b in combinations(sol_blocks, 2):
        x1 = max(a["x"], b["x"])
        y1 = max(a["y"], b["y"])
        x2 = min(a["x"] + a["w"], b["x"] + b["w"])
        y2 = min(a["y"] + a["h"], b["y"] + b["h"])
        if x2 - x1 > tol and y2 - y1 > tol:
            overlaps.append((a["id"], b["id"], (x1, y1, x2 - x1, y2 - y1)))
    return overlaps


# ---------------------------------------------------------------------------
# Visualization (official white-background style)
# ---------------------------------------------------------------------------

def visualize(input_data, sol_blocks, output_path=None, case_label=""):
    """Generate a floorplan layout image in official style."""
    n = len(sol_blocks)
    
    # --- Dynamic Figsize calculation ---
    # Base size grows with sqrt(n) to keep density reasonable
    base_scale = 8 + (n ** 0.5) * 0.8
    outline_w = input_data["outline_w"]
    outline_h = input_data["outline_h"]
    max_x = max([b["x"] + b["w"] for b in sol_blocks]) if sol_blocks else 1.0
    max_y = max([b["y"] + b["h"] for b in sol_blocks]) if sol_blocks else 1.0
    
    content_w = max(outline_w, max_x)
    content_h = max(outline_h, max_y)
    aspect = content_h / content_w if content_w > 0 else 1.0
    
    # Adjust width and height based on aspect ratio
    if aspect > 1:
        fw, fh = base_scale, base_scale * aspect
    else:
        fw, fh = base_scale / aspect, base_scale
    
    # Cap size to prevent excessive memory usage
    fw, fh = min(fw, 30), min(fh, 40)
    
    fig, ax = plt.subplots(1, 1, figsize=(fw, fh))

    block_info_map = {b["id"]: b for b in input_data["blocks"]}
    sol_map = {b["id"]: b for b in sol_blocks}
    legend_labels = {}

    # --- Global Scale Factor for UI elements ---
    scale_factor = base_scale / 8.0  # Scale relative to a standard 8-inch figure

    # --- Draw blocks ---
    for sb in sol_blocks:
        bid = sb["id"]
        x, y, w, h = sb["x"], sb["y"], sb["w"], sb["h"]

        info = block_info_map.get(bid, {
            "is_fixed": False, "is_preplaced": False,
            "mib_grp": -1, "grp_id": -1, "bedge": -1
        })
        color, label = get_block_style(info)

        # --- Draw main block ---
        rect = Rectangle((x, y), w, h, linewidth=1.0,
                          edgecolor="black", facecolor=color, alpha=0.6, zorder=4)
        ax.add_patch(rect)

        # Block ID label centered - font size scales with figure scale
        fs = max(4, (10 - (n / 25)) * scale_factor)
        ax.annotate(str(bid), (x + w/2, y + h/2), fontsize=fs, color="black", 
                    weight="bold", ha="center", va="center", zorder=5)

        legend_labels[label] = color

    # --- Draw overlaps ---
    overlaps = find_overlaps(sol_blocks)
    for _, _, (ox, oy, ow, oh) in overlaps:
        rect = Rectangle((ox, oy), ow, oh, linewidth=2 * scale_factor,
                          edgecolor="red", facecolor="red", alpha=0.35,
                          hatch="///", zorder=10)
        ax.add_patch(rect)
    if overlaps:
        legend_labels["OVERLAP"] = "red"

    # --- Draw terminals (green dots) ---
    # Use plot with markersize instead of Circle to keep constant screen size
    for tx, ty in input_data["terminals"]:
        ax.plot(tx, ty, "go", markersize=4 * scale_factor, zorder=5)

    # --- Draw B2B connectivity (red lines) ---
    for src, dst, wt in input_data["b2b"]:
        if src in sol_map and dst in sol_map:
            s, d = sol_map[src], sol_map[dst]
            scx, scy = s["x"] + s["w"] / 2, s["y"] + s["h"] / 2
            dcx, dcy = d["x"] + d["w"] / 2, d["y"] + d["h"] / 2
            ax.plot([scx, dcx], [scy, dcy], color="r", 
                    linewidth=0.15 * scale_factor, alpha=0.6, zorder=2)

    # --- Draw P2B connectivity (blue lines) ---
    for pin_id, blk_id, wt in input_data["p2b"]:
        if pin_id < len(input_data["terminals"]) and blk_id in sol_map:
            px, py = input_data["terminals"][pin_id]
            b = sol_map[blk_id]
            bcx, bcy = b["x"] + b["w"] / 2, b["y"] + b["h"] / 2
            ax.plot([px, bcx], [py, bcy], color="b", 
                    linewidth=0.1 * scale_factor, alpha=0.5, zorder=1)

    # --- Draw outline ---
    if outline_w > 0 and outline_h > 0:
        outline_rect = Rectangle((0, 0), outline_w, outline_h,
                                  linewidth=2 * scale_factor, edgecolor="black",
                                  facecolor="none", linestyle="-")
        ax.add_patch(outline_rect)

    # --- Axis limits & Aspect Ratio ---
    content_w = max(outline_w, max_x)
    content_h = max(outline_h, max_y)
    margin = 0.05
    ax.set_xlim(-content_w * margin, content_w * (1 + margin))
    ax.set_ylim(-content_h * margin, content_h * (1 + margin))
    ax.set_aspect("equal", adjustable="box")
    
    # Scale tick labels
    ax.tick_params(axis='both', which='major', labelsize=10 * scale_factor)

    # --- Title ---
    title = f"Floorplan Layout {case_label} (n={n})"
    if overlaps:
        title += f" -- OVERLAP: {len(overlaps)} pairs"
    ax.set_title(title, fontsize=14 * scale_factor, fontweight="bold", pad=20 * scale_factor)

    # --- Legend (Move Outside to the Right) ---
    handles = [mpatches.Patch(facecolor=c, edgecolor="black",
                              label=lbl, alpha=0.4)
               for lbl, c in legend_labels.items()]
    if input_data["terminals"]:
        handles.append(Line2D([0], [0], marker="o", color="w",
                              markerfacecolor="g", markersize=6 * scale_factor,
                              label="terminal", linestyle="None"))
    
    # Place legend to the right of the axes
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1),
              title="Placement Constraints", fontsize=9 * scale_factor, 
              title_fontsize=10 * scale_factor, borderaxespad=0.)

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
                    exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"[OK] Saved: {output_path}", file=sys.stderr)
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Floorplan Layout Visualizer")
    parser.add_argument("input_txt", nargs="?", help="Input .txt file")
    parser.add_argument("solution_sol", nargs="?", help="Solution .sol file")
    parser.add_argument("-o", "--output", default=None, help="Output image path")
    parser.add_argument("--batch", default=None,
                        help="Batch mode: directory containing case_NNN.txt/sol pairs")
    parser.add_argument("--max-cases", type=int, default=100,
                        help="Max cases to visualize in batch mode (default: 100)")
    args = parser.parse_args()

    if args.batch:
        batch_dir = Path(args.batch)
        txt_files = sorted(batch_dir.glob("case_*.txt"))
        out_dir = Path(args.output) if args.output else Path("plots")
        out_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for txt_path in txt_files:
            sol_path = txt_path.with_suffix(".sol")
            if not sol_path.exists():
                continue
            case_name = txt_path.stem
            inp = parse_input(str(txt_path))
            sol = parse_solution(str(sol_path))
            out_file = out_dir / f"{case_name}.png"
            visualize(inp, sol, output_path=str(out_file), case_label=case_name)
            count += 1
            if count >= args.max_cases:
                break
        print(f"[DONE] Generated {count} floorplan images in {out_dir}/",
              file=sys.stderr)

    elif args.input_txt and args.solution_sol:
        inp = parse_input(args.input_txt)
        sol = parse_solution(args.solution_sol)
        case_label = Path(args.input_txt).stem
        visualize(inp, sol, output_path=args.output, case_label=case_label)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
