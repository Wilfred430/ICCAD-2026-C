# 傳統演算法策略總整理（B*-tree + SA + Contour Packing）

本文件整理目前專案中「非 ML」的傳統演算法策略與實作細節，涵蓋資料建模、初始解策略、B*-tree 佈局、Contour Packing、Simulated Annealing（SA）流程、移動集合、成本函數與多執行緒多起點策略。內容以現行程式碼為準。

---

## 1. 端到端流程（Pipeline）

```mermaid
graph TD
    A[官方評測框架 iccad2026_evaluate.py] -->|import MyOptimizer| B[my_optimizer.py]
    B -->|tensors -> txt| C[case_XXX.txt]
    B -->|subprocess| D[floorplanner (C++)]
    D -->|parse + SA| E[多執行緒 SA chains]
    E -->|best feasible| F[case_XXX.sol]
    F -->|回傳| A
```

- Python 層只負責 I/O 轉換、baseline 估算與呼叫 C++ solver。
- C++ 層進行「B*-tree + Contour Packing + SA」搜尋，輸出 .sol。

---

## 2. 資料模型與約束

### 2.1 Block / Net / Terminal
- **Block**：包含面積目標、固定/預放置標記、(w,h,x,y) 鎖定值、group/MIB id、boundary 指定、ar_min/ar_max。
- **Net**：
  - B2B：block-to-block，權重 w。
  - P2B：pin/terminal-to-block，權重 w。
- **Terminal**：固定座標 (x, y)。

### 2.2 Boundary encoding
- 官方輸入為 bitmask：1=L, 2=R, 4=T, 8=B，角落為加總。
- 內部 enum 為 0..7（L/R/B/T/BL/BR/TL/TR）。

### 2.3 Hard vs Soft constraints
- **Hard（不可違反）**：
  - Block overlap
  - Soft block 面積誤差 > 1%
  - Fixed block 尺寸不一致
  - Preplaced block 尺寸或位置不一致
- **Soft（可違反但要罰分）**：
  - Grouping（同群 blocks 應貼齊）
  - MIB（同群 blocks 應同形）
  - Boundary（貼邊）

---

## 3. 初始解策略（Constraint-aware Initial Tree）

目標：避免亂數初始樹導致 boundary / grouping / MIB 從一開始就嚴重破壞。

### 3.1 初始尺寸
- Fixed/Preplaced：直接使用 input w/h。
- Soft block：初始化為近似正方形 $w=h=\sqrt{area}$。
- MIB group：以第一個可調 block 的 w/h 作為該群一致尺寸。

### 3.2 插入順序優先權
優先權越低越早插（越接近 root）：
1. Corner constrained (BL/BR/TL/TR)
2. Edge constrained
3. 有 grouping 或 MIB 的 block（group 內優先）
4. 一般 soft block（面積大者先）
5. Preplaced block（最後插，成為葉節點）

**理由**：
- Preplaced 若作為 root，會把整棵樹的 packing anchor 拉到遠離原點的位置，造成 bbox 偏移。
- 角落/邊界約束 block 先插入可讓 SA 起點更接近可行解。

### 3.3 插入策略
- 每次插入選一個已插入節點作為 anchor。
- 若有同 group/MIB 已插入，優先貼近該成員。
- 插入時平衡 left/right 子樹數量，避免 layout 變成「太高」或「太寬」。

---

## 4. B*-tree 表示法

### 4.1 Tree semantics
- **Left child**：放在父節點右側（x = parent.x + parent.w）。
- **Right child**：放在父節點上方（x = parent.x）。
- **Preplaced block**：忽略 tree 對 (x,y) 的指派，直接錨定在 input。

### 4.2 Topology operations
- `Move`：將節點 v 拔出並插入 u 的子節點（subtree graft）。
- `Swap`：交換兩個節點在 tree 中的位置。
- `Rotate`：交換該 block 的 w/h。

---

## 5. Contour Packing + Compaction

### 5.1 Contour packing
- 使用 skyline contour（有序 segment list）。
- 對每個節點 v：
  - 計算 x 位置（基於 parent 與 left/right child 定義）。
  - 查詢 contour 在 [x, x+w) 最高高度得到 y。
  - 更新 contour。

### 5.2 Preplaced handling
- Preplaced block 直接使用 (x_input, y_input, w_input, h_input)。
- 仍更新 contour，以便後續 block 避開。

### 5.3 Post-pack compaction
- 對所有非 preplaced block 做 **左下滑動**（left/down compaction）。
- 3 輪：先按 y 排序下移，再按 x 排序左移。
- 目的：消除 contour packing 的碎片空隙，縮小 bbox。

---

## 6. 成本函數設計

### 6.1 HPWL
- **Internal**：block-to-block centroid Manhattan。
- **External**：block centroid 到 terminal。

$$
HPWL_{int} = \sum w_{ij} (|cx_i - cx_j| + |cy_i - cy_j|)
$$
$$
HPWL_{ext} = \sum w_{tj} (|cx_j - x_t| + |cy_j - y_t|)
$$

### 6.2 BBox area
$$
Area = W_{bbox} \cdot H_{bbox}
$$

### 6.3 Soft constraints
- **Grouping**：同 group blocks 需連成單一連通分量（以邊貼齊判定）。
- **MIB**：同 group blocks 需擁有相同 (w,h)。
- **Boundary**：指定 block 必須貼齊 bbox 的某邊或角。

### 6.4 Hard constraints
- overlap、面積誤差、fixed/preplaced 不一致。

### 6.5 SA cost
SA 使用平滑可加的 cost：

$$
C_{SA} = w_{area} \frac{Area}{A_{base}} + w_{hpwl,int} \frac{HPWL_{int}}{H_{base}} + w_{hpwl,ext} \frac{HPWL_{ext}}{H_{base}} 
+ w_{group} V_g + w_{mib} V_m + w_{bound} V_b + Penalties
$$

Penalty：
- overlap：$w_{overlap} \cdot (0.10 + overlap\_area / A_{base})$
- area drift：$w_{softarea} \cdot (0.10 + drift)$

### 6.6 Contest cost
$$
C_{contest} = (1 + 0.5(HPWL\_gap + Area\_gap)) \cdot e^{2 V_{rel}} \cdot \max(0.7, RT^{0.3})
$$
若 infeasible，直接 $C=10$。

---

## 7. SA 引擎（Simulated Annealing）

### 7.1 T1 校正
- 以 **固定初始 tree** 做 N 次 random move probe。
- 只統計 uphill move 的平均 Δ。
- 設定 $T_1 = -\Delta_{avg} / \ln(p\_{accept})$。

### 7.2 Cooling schedule
三段式幾何降溫：
1. Stage1：T = T1
2. Stage2：$T \leftarrow T \cdot \alpha_2$
3. Stage3：$T \leftarrow T \cdot \alpha_3$，並在進入 Stage3 時進行一次 reheat

### 7.3 接受準則
- FixBoundary / FixGrouping 若 **不引入新的 hard violation** → always accept。
- 其他 move → Metropolis：
  - 若 $\Delta \le 0$ 直接接受。
  - 否則以 $\exp(-\Delta/T)$ 的機率接受。

### 7.4 Re-anchor
- 若長時間無改進，將 current 強制回到 best。
- 避免低溫時 current 漂離可行解。

### 7.5 停止條件
- 超過 time budget。
- Stagnation + T frozen。
- 其他 thread 已達目標 contest cost。

---

## 8. Move set（SA 鄰域）

| Move | 預設機率 | 說明 | always_accept |
|---|---|---|---|
| Rotate | 0.15 | 單一 block 旋轉 (w,h)；MIB 成員需整組旋轉 | 否 |
| Move | ~0.37 | 把 subtree graft 到另一節點 | 否 |
| Swap | 0.15 | 交換兩個節點的 tree 位置 | 否 |
| AspectRatio | 0.18 | soft block 重抽 (w,h)，area 允許微擾 | 否 |
| MibSync | 0.05 | MIB group 同步抽 (w,h) | 否 |
| FixBoundary | 0.05 | 修邊界：swap 或 move 到貼邊節點 | 部分 |
| FixGrouping | 0.05 | 修群：把游離 block graft 到主 component | 是 |

補充：
- FixBoundary 有兩種策略：
  - 若找到「已貼邊、且無邊界限制」的 block → swap（always_accept）。
  - 否則 move 到貼邊 block 的子節點（Metropolis 決定）。
- FixGrouping 會將 stray block 移到最大 component 內某節點的左右子樹。

---

## 9. Parallel Multi-Start Strategy

- 多執行緒，每條 thread 使用不同 seed。
- 每條 thread 用自己的初始 tree + SA run。
- 最終挑選 contest_cost 最小且 feasible 的解。
- 若任一 thread 達到 target contest cost（預設 1.001），其他 thread 提前停止。

---

## 10. 目前實際預設參數（核心）

### 10.1 SAConfig
- `n_iters_per_block = 50`
- Cooling: `stage2_end_k = 7`, `alpha_stage2 = 0.92`, `alpha_stage3 = 0.99`, `stage3_reheat = 0.7`
- Calibration: `n_probes = 80`, `p_accept_init = 0.90`
- Reanchor: `every_iters_per_block = 50`
- Stopping: `stagnation_stages = 30`, `T_frozen_ratio = 1e-4`, `target_contest_cost = 1.001`

### 10.2 MoveProb
- `p_fixb = 0.05`, `p_fixg = 0.05`, `p_ar = 0.18`, `p_mib = 0.05`, `p_rot = 0.15`, `p_swp = 0.15`
- `tol_ar = 0.005`, `sa_ar_clamp = 2.0`

### 10.3 SAWeights
- `w_area = 1.0`
- `w_hpwl_int = 1.0`, `w_hpwl_ext = 1.0`
- `w_overlap = 5000.0`, `w_softarea = 5000.0`
- `w_group = 300.0`, `w_mib = 80.0`, `w_bound = 80.0`
- `w_outline = 0.0`

---

## 11. 策略總結（Why this works）

1. **B*-tree + contour packing** 在 n ≤ 200 的空間內非常快，可在 SA loop 中大量重算。
2. **多階段幾何降溫 + reheat** 兼顧快速收斂與再探索。
3. **FixBoundary / FixGrouping move** 讓 SA 可以「跨越 constraint barrier」，有效提高可行率。
4. **初始 tree 的約束感知** 大幅降低 SA 起點的可行性距離。
5. **multi-start + parallel** 提升脫離 local minima 的機率。

---

如果你希望我再補上「參數調整經驗」或「case 觀察規則」，告訴我想要的深度與格式即可。
