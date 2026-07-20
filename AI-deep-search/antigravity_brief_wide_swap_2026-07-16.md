# Brief for Antigravity: `boundary_snap` wide-swap patch (2026-07-16)

## 背景

`electro_optimized/soft_repair.py` 目前的 "Strict Zero-Overlap Swap Pass"（違規
boundary 方塊跟一個完全空閒的方塊互換位置）自加入後 V_boundary 一直卡在 ~376/100
案例，沒有實際下降。

## 診斷

把 100 案例依 Vbnd 排序後發現：違規分散在近 40 個案例（前幾名 12/12/12/11/10 分別
是 n=27, 62, 102, 55, 112，大小案例都有），**不是集中在少數 outlier**。這代表現有
swap pass 沒效的原因是候選池太窄：

1. 只允許跟 `bcode[j]==0`（完全無邊界要求）的方塊互換
2. 只跑一次，沒有疊代到收斂

## 修正（已在獨立 scratch 目錄驗證，未直接覆寫你們的即時檔案）

新增 `boundary_snap(..., wide_swap=False)` 參數，`wide_swap=True` 時：

1. **放寬候選對象**：允許跟任何非 MIB、非 preplaced 的方塊 `j` 互換，只要互換後
   `j` 自己的邊界需求（若有）在新位置仍然成立（新增 `_boundary_ok()` helper 做這個
   檢查）。
2. **疊代到收斂**：整個 swap 掃描包進最多 3 輪的迴圈，直到沒有新的 swap 發生為止
   （跟前面 wall-scan 段的 `moved` 收斂邏輯一致）。
3. 包成 `boundary_wideswap_variant`（`electro_parallel.py`），用
   `ELECTRO_BOUNDARY_WIDESWAP=1` 環境變數 opt-in，接到 `electro_optimizer.py` 的
   `solve()`——**新增候選，不取代**，沿用既有的 proxy ranking 挑每案最優，跟
   `boundary_pushpast_variant` / `grouping_pushpast_variant` 同一套模式。

具體 diff 見同目錄 `wide_swap.patch`（也附在下方，方便直接讀）。

## 驗證結果

用 `ml/case_report_electro.py`（Neutral RT，排除 RT 量測雜訊），基於你們目前
`electro_optimized/`（`ELECTRO_GROUPING_PUSHPAST=0` 預設）現況，只加這一個 patch：

| | Total Score | V_grouping | V_boundary |
|---|---|---|---|
| 基準（wide_swap 關閉） | 2.5038 | 329 | 376 |
| + wide_swap portfolio（開啟） | **2.4491** | 326 | **339** |

100/100 feasible 兩邊維持，兩次重跑數字完全一致（非隨機雜訊）。

## 建議

- 這個 patch 應該可以直接合併，不會跟你們正在做的其他事情衝突（純新增函式 + opt-in
  環境變數，不改動任何預設行為）。
- 合併後如果方便，可以進一步測試跟 `ELECTRO_GROUPING_PUSHPAST=1` 疊加是否有加乘
  效果（我還沒測這個組合）。
- 一如既往：請用 portfolio 方式驗證（新舊都跑，挑分數較好的），用 Neutral RT 判斷
  是否真的有幫助，不要只看單次 Contest Grading 分數（有 RT 量測雜訊）。

---

```diff
--- electro_optimized/soft_repair.py
+++ soft_repair.py (patched)
@@ -90,7 +90,17 @@
     return None


-def boundary_snap(x, y, w, h, bcode, is_pre, passes=3, floor=None, push_past=False, clust_id=None, mib_id=None):
+def _boundary_ok(c, X, Y, wi, hi, xmn, xmx, ymn, ymx):
+    """True if a block with bitmask `c` placed at (X,Y) size (wi,hi) satisfies
+    its own boundary requirement against the CURRENT bbox extents."""
+    if c & 1 and abs(X - xmn) >= 1e-6: return False
+    if c & 2 and abs(X + wi - xmx) >= 1e-6: return False
+    if c & 4 and abs(Y + hi - ymx) >= 1e-6: return False
+    if c & 8 and abs(Y - ymn) >= 1e-6: return False
+    return True
+
+
+def boundary_snap(x, y, w, h, bcode, is_pre, passes=3, floor=None, push_past=False, clust_id=None, mib_id=None, wide_swap=False):
     """Slide each boundary block onto its required bbox edge, searching along the
     wall for a free slot (not just the exact current spot).  With `floor` set,
     movable blocks are kept at corner >= floor (first-quadrant containment).
@@ -138,76 +148,100 @@
         if not moved:
             break
             
-    # Strict Zero-Overlap Swap Pass — try multiple wall positions
-    xmn = x.min(); xmx = (x + w).max(); ymn = y.min(); ymx = (y + h).max()
-    for i in range(N):
-        c = int(bcode[i])
-        if c == 0 or is_pre[i]:
-            continue
-            
-        ok = True
-        if c & 1 and abs(x[i] - xmn) >= 1e-6: ok = False
-        if c & 2 and abs(x[i] + w[i] - xmx) >= 1e-6: ok = False
-        if c & 4 and abs(y[i] + h[i] - ymx) >= 1e-6: ok = False
-        if c & 8 and abs(y[i] - ymn) >= 1e-6: ok = False
-        if ok:
-            continue
-            
-        want_x = (c & 1) or (c & 2)
-        want_y = (c & 4) or (c & 8)
-        
-        # Generate candidate wall positions (not just one)
-        wall_positions = []
-        if want_x and want_y:
-            # Corner: only one position
-            X = xmn if (c & 1) else (xmx - w[i])
-            Y = ymn if (c & 8) else (ymx - h[i])
-            wall_positions.append((X, Y))
-        elif want_x:
-            X = xmn if (c & 1) else (xmx - w[i])
-            # Slide along the wall: try current y, wall ends, and tight-pack slots
-            cands_y = {y[i], ymn, ymx - h[i]}
-            R = X + w[i]
-            for j2 in range(N):
-                if j2 == i: continue
-                if x[j2] < R - _EPS and X < x[j2] + w[j2] - _EPS:
-                    cands_y.add(y[j2] + h[j2])
-                    cands_y.add(y[j2] - h[i])
-            for yy in sorted(cands_y, key=lambda c2: abs(c2 - y[i])):
-                if ymn - _EPS <= yy <= ymx - h[i] + _EPS:
-                    wall_positions.append((X, yy))
-        elif want_y:
-            Y = ymn if (c & 8) else (ymx - h[i])
-            cands_x = {x[i], xmn, xmx - w[i]}
-            T = Y + h[i]
-            for j2 in range(N):
-                if j2 == i: continue
-                if y[j2] < T - _EPS and Y < y[j2] + h[j2] - _EPS:
-                    cands_x.add(x[j2] + w[j2])
-                    cands_x.add(x[j2] - w[i])
-            for xx in sorted(cands_x, key=lambda c2: abs(c2 - x[i])):
-                if xmn - _EPS <= xx <= xmx - w[i] + _EPS:
-                    wall_positions.append((xx, Y))
-        
-        swapped = False
-        for X, Y in wall_positions:
+    # Strict Zero-Overlap Swap Pass — try multiple wall positions.
+    # `wide_swap` (2026-07-16, default False = original behaviour unchanged):
+    # the original pass only swaps a violating boundary block `i` with a
+    # fully-unconstrained block `j` (bcode[j]==0), and runs once.  V_boundary
+    # stayed flat (376/100 cases) after the plain pass shipped because that
+    # candidate pool is narrow -- diagnostics showed violations spread thin
+    # across ~40 cases, not concentrated in a few outliers, so a wider net is
+    # needed rather than a per-case fix.  `wide_swap=True` (a) also allows
+    # swapping with a block `j` that itself has a boundary code, as long as
+    # `j` still satisfies its OWN requirement at i's old spot, and (b) loops
+    # the whole swap scan to convergence (like the wall-scan above) instead
+    # of a single pass, since one swap can free up a slot for another.
+    # PORTFOLIO candidate only -- see `boundary_wideswap_variant` in
+    # electro_parallel.py; never call unconditionally without A/B ranking.
+    for _swap_round in range(3 if wide_swap else 1):
+        xmn = x.min(); xmx = (x + w).max(); ymn = y.min(); ymx = (y + h).max()
+        any_swapped = False
+        for i in range(N):
+            c = int(bcode[i])
+            if c == 0 or is_pre[i]:
+                continue
+
+            ok = True
+            if c & 1 and abs(x[i] - xmn) >= 1e-6: ok = False
+            if c & 2 and abs(x[i] + w[i] - xmx) >= 1e-6: ok = False
+            if c & 4 and abs(y[i] + h[i] - ymx) >= 1e-6: ok = False
+            if c & 8 and abs(y[i] - ymn) >= 1e-6: ok = False
+            if ok:
+                continue
+
+            want_x = (c & 1) or (c & 2)
+            want_y = (c & 4) or (c & 8)
+
+            # Generate candidate wall positions (not just one)
+            wall_positions = []
+            if want_x and want_y:
+                # Corner: only one position
+                X = xmn if (c & 1) else (xmx - w[i])
+                Y = ymn if (c & 8) else (ymx - h[i])
+                wall_positions.append((X, Y))
+            elif want_x:
+                X = xmn if (c & 1) else (xmx - w[i])
+                # Slide along the wall: try current y, wall ends, and tight-pack slots
+                cands_y = {y[i], ymn, ymx - h[i]}
+                R = X + w[i]
+                for j2 in range(N):
+                    if j2 == i: continue
+                    if x[j2] < R - _EPS and X < x[j2] + w[j2] - _EPS:
+                        cands_y.add(y[j2] + h[j2])
+                        cands_y.add(y[j2] - h[i])
+                for yy in sorted(cands_y, key=lambda c2: abs(c2 - y[i])):
+                    if ymn - _EPS <= yy <= ymx - h[i] + _EPS:
+                        wall_positions.append((X, yy))
+            elif want_y:
+                Y = ymn if (c & 8) else (ymx - h[i])
+                cands_x = {x[i], xmn, xmx - w[i]}
+                T = Y + h[i]
+                for j2 in range(N):
+                    if j2 == i: continue
+                    if y[j2] < T - _EPS and Y < y[j2] + h[j2] - _EPS:
+                        cands_x.add(x[j2] + w[j2])
+                        cands_x.add(x[j2] - w[i])
+                for xx in sorted(cands_x, key=lambda c2: abs(c2 - x[i])):
+                    if xmn - _EPS <= xx <= xmx - w[i] + _EPS:
+                        wall_positions.append((xx, Y))
+
+            swapped = False
+            for X, Y in wall_positions:
+                if swapped:
+                    break
+                for j in range(N):
+                    if j == i or is_pre[j]: continue
+                    if mib_id is not None and int(mib_id[j]) != 0: continue
+                    cj = int(bcode[j])
+                    if cj != 0:
+                        if not wide_swap: continue
+                        if not _boundary_ok(cj, x[i], y[i], w[j], h[j], xmn, xmx, ymn, ymx):
+                            continue
+
+                    if not _free(i, X, Y, x, y, w, h, ignore=[i, j]): continue
+                    if not _free(j, x[i], y[i], x, y, w, h, ignore=[i, j]): continue
+
+                    old_xi, old_yi = x[i], y[i]
+                    x[i], y[i] = X, Y
+                    x[j], y[j] = old_xi, old_yi
+                    swapped = True
+                    break
+
             if swapped:
-                break
-            for j in range(N):
-                if j == i or is_pre[j] or int(bcode[j]) != 0: continue
-                if mib_id is not None and int(mib_id[j]) != 0: continue
-                
-                if not _free(i, X, Y, x, y, w, h, ignore=[i, j]): continue
-                if not _free(j, x[i], y[i], x, y, w, h, ignore=[i, j]): continue
-                    
-                old_xi, old_yi = x[i], y[i]
-                x[i], y[i] = X, Y
-                x[j], y[j] = old_xi, old_yi
-                swapped = True
-                break
-            
-        if swapped:
-            is_pre[i] = True
+                is_pre[i] = True
+                any_swapped = True
+
+        if not any_swapped:
+            break
 
     return x, y


--- electro_optimized/electro_parallel.py
+++ electro_parallel.py (patched)
@@ -97,6 +97,30 @@
     return x, y, w, h


+def boundary_wideswap_variant(start, P):
+    """Boundary wide-swap portfolio candidate (2026-07-16): re-run the
+    grouping/boundary repair loop with `boundary_snap(wide_swap=True)`,
+    returned as an ADDITIONAL candidate. Widens the Strict Zero-Overlap Swap
+    Pass's candidate pool (swap with any non-MIB block whose own boundary
+    requirement still holds after the swap, not just fully-free blocks) and
+    iterates it to convergence. Strictly additive: solve() ranks it against
+    the plain layout by the full cost proxy."""
+    x, y, w, h = start
+    is_pre = P["is_pre"].copy()
+    clust_id, mib_id, bcode = P["clust_id"], P["mib_id"], P["bcode"]
+    nonneg = P.get("nonneg", False)
+    floor = 0.0 if nonneg else None
+    x, y = x.copy(), y.copy()
+    for _ in range(P["rounds"]):
+        x, y = grouping_repair(x, y, w, h, clust_id, is_pre, floor=floor)
+        x, y = boundary_snap(x, y, w, h, bcode, is_pre, floor=floor, clust_id=clust_id, mib_id=mib_id, wide_swap=True)
+    # Final grouping repair pass to clean up any swaps that perturbed clusters
+    # Pass bcode so boundary-constrained blocks stay on their required walls
+    x, y = grouping_repair(x, y, w, h, clust_id, is_pre, floor=floor, bcode=bcode)
+    x, y = remove_overlap(x, y, w, h, is_pre, nonneg=nonneg)
+    return x, y, w, h
+
+
 def pool_init(threads=1):
     """Give each worker its share of cores (cores/nproc threads).  Threads are set
     AFTER the fork, so the parent never holds a live OpenMP pool across fork."""


--- electro_optimized/electro_optimizer.py
+++ electro_optimizer.py (patched)
@@ -204,6 +204,12 @@
             if has_vg:
                 starts = starts + [electro_parallel.grouping_pushpast_variant(s, P) for s in starts]

+        # Boundary wide-swap portfolio (opt-in, 2026-07-16): add a
+        # wide_swap=True boundary-repair variant of each start as an EXTRA
+        # candidate -- strictly additive, can never worsen the result.
+        if os.environ.get("ELECTRO_BOUNDARY_WIDESWAP", "0") == "1":
+            starts = starts + [electro_parallel.boundary_wideswap_variant(s, P) for s in starts]
+
         cands = []
         for (x, y, w, h) in starts:
             vb, vg, vm, nsoft = soft_violation_counts(x, y, w, h, bcode, clust_id, mib_id)
```
