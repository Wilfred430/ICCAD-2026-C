import os
import sys
import time
import numpy as np

# Add contest paths
sys.path.insert(0, "D:/ICCAD-2026-C/ICCAD-C-FloorSet-official/iccad2026contest")
sys.path.insert(0, "D:/ICCAD-2026-C/collaborate/electro_optimized")

from iccad2026_evaluate import ContestEvaluator
from soft_repair import soft_violation_counts

def run_evaluation(env_config, test_ids=None):
    # Set env variables
    old_env = {}
    for k, v in env_config.items():
        old_env[k] = os.environ.get(k)
        os.environ[k] = str(v)
        
    try:
        evaluator = ContestEvaluator("D:/ICCAD-2026-C/ICCAD-C-FloorSet-official", verbose=False)
        res = evaluator.evaluate("D:/ICCAD-2026-C/collaborate/electro_optimized/electro_optimizer.py", test_ids=test_ids)
        
        # Calculate soft violation totals across all test cases
        tot_vb, tot_vg, tot_vm = 0, 0, 0
        for r in res.test_results:
            if r.error is None and r.positions is not None:
                # We need constraints and other info to count violations
                sample = evaluator.dataset[r.test_id]
                inputs = sample['input']
                area_target, b2b_conn, p2b_conn, pins_pos, constraints = inputs
                block_count = r.block_count
                
                x = np.array([p[0] for p in r.positions])
                y = np.array([p[1] for p in r.positions])
                w = np.array([p[2] for p in r.positions])
                h = np.array([p[3] for p in r.positions])
                
                cons = constraints[:block_count].cpu().numpy()
                is_pre = (cons[:, 1] != 0).astype(bool)
                mib_id = cons[:, 2].astype(int) if cons.shape[1] > 2 else np.zeros(block_count, int)
                clust_id = cons[:, 3].astype(int) if cons.shape[1] > 3 else np.zeros(block_count, int)
                bcode = cons[:, 4].astype(int) if cons.shape[1] > 4 else np.zeros(block_count, int)
                
                vb, vg, vm, _ = soft_violation_counts(x, y, w, h, bcode, clust_id, mib_id)
                tot_vb += vb
                tot_vg += vg
                tot_vm += vm
                
        return {
            "total_score": res.total_score,
            "avg_cost": res.summary["avg_cost"],
            "avg_runtime": res.summary["avg_runtime"],
            "num_feasible": res.summary["num_feasible"],
            "num_tests": res.summary["num_tests"],
            "tot_v_boundary": tot_vb,
            "tot_v_grouping": tot_vg,
            "tot_v_mib": tot_vm
        }
    finally:
        # Restore env
        for k in env_config.keys():
            if old_env[k] is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old_env[k]

def main():
    # Example config sweep: sweep ELECTRO_MIB_SHAPE
    configs = [
        {"ELECTRO_MIB_SHAPE": 0.0},
        {"ELECTRO_MIB_SHAPE": 0.1},
        {"ELECTRO_MIB_SHAPE": 0.25},
        {"ELECTRO_MIB_SHAPE": 0.5},
    ]
    
    # We can run on a subset of 10 cases to be fast
    test_ids = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80]
    print(f"Sweeping configs over {len(test_ids)} cases: {test_ids}")
    
    for cfg in configs:
        t0 = time.time()
        res = run_evaluation(cfg, test_ids=test_ids)
        dt = time.time() - t0
        print(f"Config: {cfg}")
        print(f"  Score: {res['total_score']:.4f} | Avg Cost: {res['avg_cost']:.4f} | Feasible: {res['num_feasible']}/{res['num_tests']}")
        print(f"  V_boundary: {res['tot_v_boundary']} | V_grouping: {res['tot_v_grouping']} | V_mib: {res['tot_v_mib']}")
        print(f"  Avg Runtime: {res['avg_runtime']:.3f}s | Real time: {dt:.1f}s")
        print("-" * 50)

if __name__ == '__main__':
    main()
