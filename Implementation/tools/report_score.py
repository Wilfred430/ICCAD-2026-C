#!/usr/bin/env python3
import json
import math
import sys
import glob
from pathlib import Path

def process_file(path, results_list):
    try:
        with open(path, "r") as f:
            data = json.load(f)
            test_results = data.get("test_results", [])
            results_list.extend([res for res in test_results if res["block_count"] > 0])
    except Exception as e:
        print(f"Warning: Could not read {path}: {e}")

def calculate_weighted_score():
    all_results = []
    
    # 1. 優先找 eval_results/ 下的分散結果 (eval-quick 產生的)
    quick_files = sorted(glob.glob("eval_results/results_*.json"))
    for f in quick_files:
        process_file(f, all_results)
    
    # 2. 如果沒有分散結果，找單一結果檔 (make eval 產生的)
    if not all_results:
        main_json = Path("eval_results/my_optimizer_results.json")
        if not main_json.exists():
            main_json = Path("../ICCAD-C-FloorSet-official/iccad2026contest/my_optimizer_results.json")
        
        if main_json.exists():
            process_file(main_json, all_results)

    if not all_results:
        print("Error: No evaluation results found in eval_results/ or official dir.")
        return

    # 去重 (避免重複計算同一個 test_id)
    unique_results = {}
    for res in all_results:
        unique_results[res["test_id"]] = res
    
    sorted_ids = sorted(unique_results.keys())
    
    total_weighted_cost = 0.0
    total_weight = 0.0
    
    print(f"{'ID':<4} {'N':<4} {'Feasible':<10} {'Cost':<10} {'HPWL_Gap':<10} {'Area_Gap':<10} {'V_rel':<10}")
    print("-" * 65)

    for tid in sorted_ids:
        res = unique_results[tid]
        n = res["block_count"]
        cost = res["cost"]
        h_gap = res["hpwl_gap"]
        a_gap = res["area_gap"]
        v_rel = res["violations_relative"]
        is_f = "Yes" if res["is_feasible"] else "No (10.0)"
        
        weight = math.exp(n / 10.0)
        
        print(f"{tid:<4} {n:<4} {is_f:<10} {cost:<10.4f} {h_gap:<10.4f} {a_gap:<10.4f} {v_rel:<10.4f}")
        
        total_weighted_cost += cost * weight
        total_weight += weight

    if total_weight > 0:
        final_score = total_weighted_cost / total_weight
        print("-" * 65)
        print(f"Summary: {len(sorted_ids)} cases analyzed.")
        print(f"Final Weighted Score (Total Score): {final_score:.6f}")
    else:
        print("Could not calculate score.")

if __name__ == "__main__":
    calculate_weighted_score()
