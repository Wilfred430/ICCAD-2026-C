#!/usr/bin/env python3
"""
verify_solution.py

Re-implements the v9 contest cost in pure Python so you can sanity-check the
C++ solver before sending anything to the official iccad2026_evaluate.py.

Usage:
    python verify_solution.py instance.txt solution.txt
"""

import argparse
import math
import sys


def read_instance(path):
    inst = {
        "n_blocks": 0, "n_terminals": 0,
        "baseline_hpwl": 0.0, "baseline_area": 0.0,
        "terminals": [], "blocks": [], "b2b": [], "p2b": [],
        "groups": [], "mib": [],
    }
    with open(path) as f:
        toks = []
        for line in f:
            line = line.split('#', 1)[0]
            toks.extend(line.split())
    i = 0
    def take(n):
        nonlocal i
        out = toks[i:i+n]; i += n; return out
    while i < len(toks):
        kw = toks[i]; i += 1
        if kw == "N_BLOCKS":
            inst["n_blocks"] = int(toks[i]); i += 1
        elif kw == "N_TERMINALS":
            inst["n_terminals"] = int(toks[i]); i += 1
        elif kw == "BASELINE_HPWL":
            inst["baseline_hpwl"] = float(toks[i]); i += 1
        elif kw == "BASELINE_AREA":
            inst["baseline_area"] = float(toks[i]); i += 1
        elif kw == "OUTLINE":
            i += 2
        elif kw == "TERMINALS":
            for _ in range(inst["n_terminals"]):
                tid, x, y = take(3)
                inst["terminals"].append({"id": int(tid), "x": float(x), "y": float(y)})
        elif kw == "BLOCKS":
            for _ in range(inst["n_blocks"]):
                fields = take(13)
                bid, area, isf, isp, wi, hi, xi, yi, mib, gid, bedge, armin, armax = fields
                inst["blocks"].append({
                    "id": int(bid), "area": float(area),
                    "is_fixed": int(isf) != 0, "is_preplaced": int(isp) != 0,
                    "wi": float(wi), "hi": float(hi),
                    "xi": float(xi), "yi": float(yi),
                    "mib": int(mib), "gid": int(gid), "bedge": int(bedge),
                })
        elif kw == "B2B":
            m = int(toks[i]); i += 1
            for _ in range(m):
                a, b, w = take(3)
                inst["b2b"].append((int(a), int(b), float(w)))
        elif kw == "P2B":
            m = int(toks[i]); i += 1
            for _ in range(m):
                t, b, w = take(3)
                inst["p2b"].append((int(t), int(b), float(w)))
        elif kw == "GROUPS":
            P = int(toks[i]); i += 1
            for _ in range(P):
                sz = int(toks[i]); i += 1
                g = [int(toks[i + j]) for j in range(sz)]
                i += sz
                inst["groups"].append(g)
        elif kw == "MIB":
            Q = int(toks[i]); i += 1
            for _ in range(Q):
                sz = int(toks[i]); i += 1
                g = [int(toks[i + j]) for j in range(sz)]
                i += sz
                inst["mib"].append(g)
        elif kw == "END":
            break
    return inst


def read_solution(path):
    with open(path) as f:
        toks = []
        for line in f:
            toks.extend(line.split('#', 1)[0].split())
    i = 0
    n = 0
    blocks = []
    while i < len(toks):
        if toks[i] == "N_BLOCKS":
            n = int(toks[i+1]); i += 2; continue
        bid = int(toks[i]); x = float(toks[i+1]); y = float(toks[i+2])
        w = float(toks[i+3]); h = float(toks[i+4]); i += 5
        blocks.append((bid, x, y, w, h))
    return blocks


def overlap(a, b, eps=1e-7):
    return (a[1] + eps < b[1] + b[3] and b[1] + eps < a[1] + a[3]
            and a[2] + eps < b[2] + b[4] and b[2] + eps < a[2] + a[4])


def touches(a, b, eps=1e-7):
    ax, ay, aw, ah = a[1], a[2], a[3], a[4]
    bx, by, bw, bh = b[1], b[2], b[3], b[4]
    if abs((ax + aw) - bx) < eps or abs((bx + bw) - ax) < eps:
        ylo, yhi = max(ay, by), min(ay + ah, by + bh)
        if yhi - ylo > eps: return True
    if abs((ay + ah) - by) < eps or abs((by + bh) - ay) < eps:
        xlo, xhi = max(ax, bx), min(ax + aw, bx + bw)
        if xhi - xlo > eps: return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instance"); ap.add_argument("solution")
    args = ap.parse_args()
    inst = read_instance(args.instance)
    sol  = read_solution(args.solution)
    sol.sort(key=lambda r: r[0])
    n = len(sol)
    bbox_w = max(b[1] + b[3] for b in sol)
    bbox_h = max(b[2] + b[4] for b in sol)
    bbox_a = bbox_w * bbox_h

    # Hard checks
    overlap_v = False
    for i in range(n):
        for j in range(i+1, n):
            if overlap(sol[i], sol[j]):
                overlap_v = True
                print(f"OVERLAP between blocks {sol[i][0]} and {sol[j][0]}")
                break
        if overlap_v: break

    area_v = False
    for k, b in enumerate(inst["blocks"]):
        if b["is_fixed"] or b["is_preplaced"]: continue
        sx, sy, sw, sh = sol[b["id"]][1:]
        a = sw * sh
        if b["area"] > 0 and abs(a - b["area"]) / b["area"] > 0.01 + 1e-9:
            area_v = True
            print(f"AREA VIOLATION block {b['id']}: actual={a}, target={b['area']}")
            break

    fixed_v = False
    for b in inst["blocks"]:
        if not b["is_fixed"]: continue
        sw, sh = sol[b["id"]][3], sol[b["id"]][4]
        if abs(sw - b["wi"]) > 1e-6 or abs(sh - b["hi"]) > 1e-6:
            fixed_v = True
            print(f"FIXED VIOLATION block {b['id']}")
            break

    pre_v = False
    for b in inst["blocks"]:
        if not b["is_preplaced"]: continue
        sx, sy, sw, sh = sol[b["id"]][1:]
        if (abs(sw - b["wi"]) > 1e-6 or abs(sh - b["hi"]) > 1e-6 or
            abs(sx - b["xi"]) > 1e-6 or abs(sy - b["yi"]) > 1e-6):
            pre_v = True
            print(f"PREPLACED VIOLATION block {b['id']}")
            break

    feasible = not (overlap_v or area_v or fixed_v or pre_v)

    # HPWL (centroid Manhattan, v9)
    hpwl_int = 0.0
    for a, b, w in inst["b2b"]:
        sa = sol[a]; sb = sol[b]
        ca = (sa[1] + sa[3] / 2, sa[2] + sa[4] / 2)
        cb = (sb[1] + sb[3] / 2, sb[2] + sb[4] / 2)
        hpwl_int += w * (abs(ca[0] - cb[0]) + abs(ca[1] - cb[1]))
    hpwl_ext = 0.0
    for t, b, w in inst["p2b"]:
        T = inst["terminals"][t]
        sb = sol[b]
        cb = (sb[1] + sb[3] / 2, sb[2] + sb[4] / 2)
        hpwl_ext += w * (abs(cb[0] - T["x"]) + abs(cb[1] - T["y"]))
    hpwl_total = hpwl_int + hpwl_ext

    # Soft constraints
    Vg = 0; Vm = 0; Vb = 0; Nsoft = 0
    # grouping
    for g in inst["groups"]:
        if len(g) <= 1: continue
        # connected components via shared edges
        par = list(range(len(g)))
        def find(x):
            while par[x] != x: par[x] = par[par[x]]; x = par[x]
            return x
        for i in range(len(g)):
            for j in range(i+1, len(g)):
                if touches(sol[g[i]], sol[g[j]]):
                    a = find(i); bb = find(j)
                    if a != bb: par[a] = bb
        comps = sum(1 for i in range(len(g)) if find(i) == i)
        Vg += max(0, comps - 1)
        Nsoft += len(g) - 1
    # MIB
    for g in inst["mib"]:
        if len(g) <= 1: continue
        shapes = set()
        for b in g:
            shapes.add((round(sol[b][3], 6), round(sol[b][4], 6)))
        Vm += max(0, len(shapes) - 1)
        Nsoft += len(g) - 1
    # boundary
    for b in inst["blocks"]:
        if b["bedge"] < 0: continue
        Nsoft += 1
        sx, sy, sw, sh = sol[b["id"]][1:]
        L = abs(sx) < 1e-7
        B = abs(sy) < 1e-7
        R = abs((sx + sw) - bbox_w) < 1e-7
        T = abs((sy + sh) - bbox_h) < 1e-7
        e = b["bedge"]
        ok = ((e == 0 and L) or (e == 1 and R) or (e == 2 and B) or (e == 3 and T)
              or (e == 4 and L and B) or (e == 5 and R and B)
              or (e == 6 and L and T) or (e == 7 and R and T))
        if not ok: Vb += 1

    V_rel = (Vg + Vm + Vb) / Nsoft if Nsoft > 0 else 0.0

    base_h = inst["baseline_hpwl"] or hpwl_total
    base_a = inst["baseline_area"] or bbox_a
    hpwl_gap = (hpwl_total - base_h) / base_h if base_h > 0 else 0.0
    area_gap = (bbox_a - base_a) / base_a if base_a > 0 else 0.0

    if not feasible:
        cost = 10.0
    else:
        alpha, beta, gamma = 0.5, 2.0, 0.3
        rf = max(0.7, 1.0)  # runtime_factor=1 for self-check
        cost = (1.0 + alpha * (hpwl_gap + area_gap)) * math.exp(beta * V_rel) * rf

    print("=== summary ===")
    print(f"feasible:   {feasible}")
    print(f"bbox:       {bbox_w:.4f} x {bbox_h:.4f}  area={bbox_a:.4f}")
    print(f"baseline_a: {base_a:.4f}  area_gap={area_gap:+.4f}")
    print(f"HPWL:       {hpwl_total:.4f} = int {hpwl_int:.4f} + ext {hpwl_ext:.4f}")
    print(f"baseline_h: {base_h:.4f}  hpwl_gap={hpwl_gap:+.4f}")
    print(f"V: grouping={Vg} mib={Vm} boundary={Vb} / Nsoft={Nsoft} -> V_rel={V_rel:.4f}")
    print(f"contest_cost (rf=1): {cost:.4f}")


if __name__ == "__main__":
    main()
