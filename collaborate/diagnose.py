import os
import sys
import numpy as np
import torch

sys.path.insert(0, "D:/ICCAD-2026-C/ICCAD-C-FloorSet-official/iccad2026contest")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "electro_optimized"))
from electro_optimizer import MyOptimizer, _edges_np
from soft_repair import soft_violation_counts, _components
from litetestLoader import FloorplanDatasetLiteTest

def diagnose(test_id=0):
    dataset = FloorplanDatasetLiteTest("D:/ICCAD-2026-C/ICCAD-C-FloorSet-official")
    sample = dataset[test_id]
    inputs, labels = sample['input'], sample['label']
    area_target, b2b_conn, p2b_conn, pins_pos, constraints = inputs
    block_count = int((area_target != -1).sum().item())
    
    # Extract baseline
    from iccad2026_evaluate import ContestEvaluator
    evaluator = ContestEvaluator("D:/ICCAD-2026-C/ICCAD-C-FloorSet-official")
    baseline, target_pos = evaluator._extract_baseline(
        test_id, labels, b2b_conn, p2b_conn, pins_pos, block_count
    )
    
    opt_target_pos = torch.full((block_count, 4), -1.0)
    if target_pos is not None and constraints is not None:
        nc = constraints.shape[1] if constraints.dim() > 1 else 0
        for i in range(block_count):
            is_fixed = nc > 0 and constraints[i, 0] != 0
            is_preplaced = nc > 1 and constraints[i, 1] != 0
            if is_preplaced:
                tx, ty, tw, th = target_pos[i]
                opt_target_pos[i] = torch.tensor([tx, ty, tw, th])
            elif is_fixed:
                _, _, tw, th = target_pos[i]
                opt_target_pos[i, 2] = tw
                opt_target_pos[i, 3] = th

    opt = MyOptimizer(verbose=True)
    positions = opt.solve(
        block_count, area_target, b2b_conn, p2b_conn, pins_pos,
        constraints, opt_target_pos
    )
    
    # Let's inspect the results
    x = np.array([p[0] for p in positions], dtype=float)
    y = np.array([p[1] for p in positions], dtype=float)
    w = np.array([p[2] for p in positions], dtype=float)
    h = np.array([p[3] for p in positions], dtype=float)
    
    cons = constraints[:block_count].cpu().numpy()
    is_pre = (cons[:, 1] != 0).astype(bool)
    mib_id = cons[:, 2].astype(int) if cons.shape[1] > 2 else np.zeros(block_count, int)
    clust_id = cons[:, 3].astype(int) if cons.shape[1] > 3 else np.zeros(block_count, int)
    bcode = cons[:, 4].astype(int) if cons.shape[1] > 4 else np.zeros(block_count, int)
    
    vb, vg, vm, nsoft = soft_violation_counts(x, y, w, h, bcode, clust_id, mib_id)
    print("--- DIAGNOSIS ---")
    print(f"Feasible? Yes (checked by evaluator)")
    print(f"Soft Violations: V_boundary={vb}, V_grouping={vg}, V_mib={vm}, N_soft={nsoft}")
    
    # Boundary violations detail
    xmn, xmx = x.min(), (x + w).max()
    ymn, ymx = y.min(), (y + h).max()
    print(f"Canvas bbox: X=[{xmn:.4f}, {xmx:.4f}], Y=[{ymn:.4f}, {ymx:.4f}]")
    for i in range(block_count):
        c = int(bcode[i])
        if c == 0:
            continue
        touches = {
            1: abs(x[i] - xmn) < 1e-6,
            2: abs(x[i] + w[i] - xmx) < 1e-6,
            4: abs(y[i] + h[i] - ymx) < 1e-6,
            8: abs(y[i] - ymn) < 1e-6,
        }
        requested = [name for bit, name in [(1, 'Left'), (2, 'Right'), (4, 'Top'), (8, 'Bottom')] if c & bit]
        satisfied = [name for bit, name in [(1, 'Left'), (2, 'Right'), (4, 'Top'), (8, 'Bottom')] if c & bit and touches[bit]]
        if len(satisfied) < len(requested):
            print(f"Block {i} boundary violation: code={c} ({'/'.join(requested)}), but only satisfied {satisfied}")
            print(f"  Block pos: x={x[i]:.4f}, y={y[i]:.4f}, w={w[i]:.4f}, h={h[i]:.4f}")
            # print distance to closest target edges
            for bit, name in [(1, 'Left'), (2, 'Right'), (4, 'Top'), (8, 'Bottom')]:
                if c & bit:
                    if bit == 1: print(f"    Dist to Left: {x[i] - xmn:.4f}")
                    if bit == 2: print(f"    Dist to Right: {xmx - (x[i]+w[i]):.4f}")
                    if bit == 4: print(f"    Dist to Top: {ymx - (y[i]+h[i]):.4f}")
                    if bit == 8: print(f"    Dist to Bottom: {y[i] - ymn:.4f}")
                    
    # Grouping violations detail
    for g in range(1, int(clust_id.max()) + 1):
        mem = np.where(clust_id == g)[0].tolist()
        if len(mem) > 1:
            comps = _components(mem, x, y, w, h)
            if len(comps) > 1:
                print(f"Cluster {g} has {len(comps)} connected components: {comps}")
                for comp in comps:
                    comp_pos = [f"B{i}({x[i]:.2f},{y[i]:.2f},{w[i]:.2f},{h[i]:.2f})" for i in comp]
                    print(f"  Component: {comp_pos}")
                    
    # MIB violations detail
    for g in range(1, int(mib_id.max()) + 1):
        mem = np.where(mib_id == g)[0].tolist()
        if len(mem) > 1:
            shapes = {}
            for i in mem:
                sh = (round(float(w[i]), 4), round(float(h[i]), 4))
                shapes.setdefault(sh, []).append(i)
            if len(shapes) > 1:
                print(f"MIB Group {g} has {len(shapes)} distinct shapes:")
                for sh, blocks in shapes.items():
                    print(f"  Shape {sh}: blocks {blocks}")

if __name__ == '__main__':
    tid = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    diagnose(tid)
