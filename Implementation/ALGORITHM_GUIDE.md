# 演算法流程與參數調整對照表

> 對應 commit: 2026-05-05 (CHANGES_2026-05-05.log 完成後的版本)
>
> 兩份 source tree（要記得同步）：
> * **本機開發版**：`C:/Gozzz/3_Junior/EDA專題/code/`
> * **WSL 編譯版**：`/home/pop/2026_EDA_contest/`（編譯後產生 `floorplanner` binary）
> * **Contest 部署版**：`/home/pop/IntelLabs_Floorset/FloorSet/iccad2026contest/`（放 `my_optimizer.py` 與 `floorplanner` binary）

---

## Part 1 — 端到端執行流程

### 一張圖看完整 pipeline

```
┌──────────────────────────────────────────────────────────────────────┐
│  Contest Framework  (iccad2026_evaluate.py)                          │
│  /home/pop/IntelLabs_Floorset/FloorSet/iccad2026contest/             │
│                                                                       │
│  importlib → MyOptimizer().solve(block_count, area_targets, ...)     │
└──────────────────────────────────────────────────────────────────────┘
                               │
                               ▼  (per case)
┌──────────────────────────────────────────────────────────────────────┐
│  my_optimizer.py    (Python wrapper, contest-compliant 入口)         │
│  iccad2026contest/my_optimizer.py                                    │
│                                                                       │
│  ① _estimate_baselines()       → 算 baseline_hpwl / baseline_area    │
│  ② _write_txt()                → tensor 轉文字檔 case_NNN.txt        │
│  ③ subprocess.run(floorplanner)→ 呼叫 C++ solver                     │
│  ④ _parse_sol()                → 讀 case_NNN.sol 回傳 [(x,y,w,h)]    │
└──────────────────────────────────────────────────────────────────────┘
                               │
                               ▼  (system call)
┌──────────────────────────────────────────────────────────────────────┐
│  C++ floorplanner   (binary，原始碼 in 2026_EDA_contest/)            │
│                                                                       │
│  src/main.cpp                                                         │
│   ├─ load_instance()          [parser.cpp]   讀 .txt → FloorplanInst │
│   └─ run_parallel()           [parallel.cpp] N 個 thread 平行 SA     │
│        │                                                              │
│        │  per thread:                                                 │
│        │  ┌──────────────────────────────────────────────────────┐   │
│        │  │ make_initial()    [parallel.cpp]  constraint-aware   │   │
│        │  │   ↓ 建出有 priority 的初始 B*-tree                    │   │
│        │  │ SimulatedAnnealing.run()  [sa.cpp]                   │   │
│        │  │   ↓ Fast-SA 主迴圈：                                   │   │
│        │  │   ┌─────────────────────────────────────────────┐    │   │
│        │  │   │ MoveEngine.propose() [moves.cpp]            │    │   │
│        │  │   │   六種 move：Rotate / Move / Swap /          │    │   │
│        │  │   │             AspectRatio / MibSync /         │    │   │
│        │  │   │             FixBoundary / FixGrouping       │    │   │
│        │  │   │     ↓                                       │    │   │
│        │  │   │ Packer.pack()   [packer.cpp] 重新算 (x,y)    │    │   │
│        │  │   │     ↓                                       │    │   │
│        │  │   │ Evaluator.evaluate()/sa_cost() [cost.cpp]   │    │   │
│        │  │   │     ↓                                       │    │   │
│        │  │   │ Metropolis accept/reject + revert if reject  │    │   │
│        │  │   └─────────────────────────────────────────────┘    │   │
│        │  └──────────────────────────────────────────────────────┘   │
│        │                                                              │
│        └─ 收集所有 thread 結果，挑 contest_cost 最低且 feasible       │
│                                                                       │
│  save_solution()              [parser.cpp]  寫出 .sol                │
└──────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  iccad2026_evaluate.py 計算 contest cost：                           │
│   Cost = (1 + 0.5(HPWL_gap + Area_gap)) × exp(2·V_rel)               │
│          × max(0.7, RuntimeFactor^0.3)         (feasible)            │
│        = M = 10                                (infeasible)          │
└──────────────────────────────────────────────────────────────────────┘
```

### 各檔案的職責

| 檔案 | 角色 | 何時動 |
|---|---|---|
| `my_optimizer.py` | Contest framework 與 C++ binary 之間的 Python 黏合層 | 改 baseline 估算公式、改傳給 C++ 的欄位、改時間預算 |
| `main.cpp` | C++ 入口，解析 CLI 參數，呼叫 `run_parallel()` | 加新的 CLI flag |
| `parser.cpp` | `.txt` → `FloorplanInstance`、`BTree` → `.sol` | 改 IO 格式 |
| `parallel.cpp` | N-thread SA 排程 + `make_initial()` 初始樹 | 改初始樹策略、改 thread 間溝通 |
| `sa.cpp` | Fast-SA 主迴圈、溫度排程、Metropolis 判斷 | 改溫度公式、加退火階段 |
| `moves.cpp` | 七種 move（Rotate / Move / Swap / AR / MIB / FixBoundary / FixGrouping） + propose() | 加新 move、改 move 機率 |
| `packer.cpp` | B\*-tree contour packing：依 tree 算 (x,y)、回傳 bbox | 改 pack 規則、加 anchored block 處理 |
| `cost.cpp` | HPWL / area / 軟硬性違規計算、SA cost 與 contest cost | 改 cost 公式 |
| `cost.hpp` | `SAWeights` struct（所有 SA cost 權重） | **最常動的檔案**，調 cost weight |
| `btree.cpp` | B\*-tree 拓樸操作（detach / insert / swap / rotate） | 改 tree primitives |
| `types.hpp` | `Block` / `Net` / `FloorplanInstance` 等 schema | 加 Block 屬性 |

---

## Part 2 — 演算法本體（流程細節）

### 2.1 Fast-SA 主迴圈（`sa.cpp::run()`）

```
1. 對 initial tree 做 pack + evaluate，記為 best
2. 校正 T1：跑 60 個隨機 move，平均 |Δcost| → T1 = -Δavg / log(p_accept_init)
3. while (elapsed < time_budget):
     依 stage k 算 T：
       k = 1            : T = T1
       k <= K           : T = T1 * Δavg / (k * c_fastsa)         (Stage 2: 急速降溫)
       k > K            : T = T1 * Δavg / k                       (Stage 3: 慢退火)
     m = engine.propose(...)   ← 從 7 種 move 抽一個
     pack + evaluate + sa_cost
     accept = (m.always_accept) || (Δ <= 0) || (rand() < exp(-Δ/T))
     if (accept) update current; if (better) update best
     else        engine.revert(m)
     iters_this_step++
     if (iters_this_step >= n_iters_per_block * n_blocks): k++
4. 回傳 best_tree, best_costs
```

關鍵點：
* **`always_accept` 只給 FixBoundary / FixGrouping 用**。其他 move 必須走 Metropolis。
* **降溫公式來自 PARSAC § 3.3 / Fast-SA 論文**。`Stage 2` 急速冷卻是讓搜尋很快收斂到「合理盆地」；`Stage 3` 慢慢精修。
* **K = 7、c_fastsa = 100** 是論文建議值，沒特別原因不要動。

### 2.2 B\*-tree Packing（`packer.cpp`）

```
DFS pre-order 走 tree：
  if v 是 preplaced:                    snap 到 (x_input, y_input)
  else if v 是 left child of parent:    px = x_parent + w_parent
                                        py = max contour height in [px, px+w]
  else if v 是 right child of parent:   px = x_parent
                                        py = max contour height in [px, px+w]
  else (root):                          px = py = 0
  update contour with the new block top
```

關鍵點：
* **Left child = 往右擴張**；**Right child = 往上堆疊**。樹形決定 floorplan 大致形狀。
* **Contour 是 sorted vector 表示的天際線**。`O(log n)` 的 lookup + `O(n)` 的 update。
* **不會自己破壞 tree**。Packing 只讀拓樸 + (w,h)，寫 (x,y)。Determinism 是 SA 正確性的基礎。

### 2.3 Cost 計算（`cost.cpp`）

```
hpwl_int = Σ_b2b w_ij · |cx_i - cx_j| + |cy_i - cy_j|        ← 重心-重心曼哈頓
hpwl_ext = Σ_p2b w_tj · |cx_b - x_t|  + |cy_b - y_t|
area_bbox = bbox_w * bbox_h
hpwl_gap  = (hpwl - baseline) / baseline                     ← 用我們估的 baseline
area_gap  = (area - baseline) / baseline
V_grouping = Σ_p (連通分量數 - 1)                              ← 越分裂越大
V_mib      = Σ_q (該 group 內不同 (w,h) 數 - 1)
V_boundary = Σ_b (block_b 沒貼到指定邊界 → +1)
V_rel = (V_g + V_m + V_b) / N_soft

# 給 SA 用的 cost（連續可微，方便退火）：
sa_cost = w_area·(area/abase) + w_hpwl·(hpwl/hbase)
        + w_group·V_g + w_mib·V_m + w_bound·V_b
        + (overlap_violation ? w_overlap : 0)
        + (area_violation    ? w_softarea : 0)
        + (fixed_violation   ? w_overlap : 0)
        + (preplaced_violation ? w_overlap : 0)

# 比賽計分用的 cost（不可微，只在最後挑最佳解時用）：
contest_cost = (1 + 0.5·(hpwl_gap + area_gap)) · exp(2·V_rel) · max(0.7, RT^0.3)
             = 10  if infeasible
```

關鍵點：
* **`sa_cost` 與 `contest_cost` 是兩個不同函數**。SA 用 sa_cost 探索（連續、可加），最後排序挑解時用 contest_cost。
* **`w_overlap` / `w_softarea` 必須超大**（5000）才能保證 SA 不會接受違反硬性約束的解。
* **軟性權重 `w_group/w_mib/w_bound = 80`**：為了模擬 contest 的 `exp(2·V_rel)`。

### 2.4 七種 Move（`moves.cpp`）

| 編號 | 名稱 | 機率 | 動作 | always_accept |
|------|------|------|------|---------------|
| M1 | **Rotate** | 0.15 | 對某 block 交換 (w, h)；MIB group 必須整組一起轉 | ✗ |
| M2 | **Move** | 0.37 | 把 v 拔起，重新接到 u 的 left/right child | ✗ |
| M3 | **Swap** | 0.15 | 在樹上對換 a 與 b 的位置 | ✗ |
| M4 | **AspectRatio** | 0.18 | 對 soft block 重抽 (w, h)，保持 area 在 1% 內 | ✗ |
| M5 | **MibSync** | 0.05 | 對整個 MIB group 同步抽新的 (w, h) | ✗ |
| M6 | **FixBoundary** | 0.05 | 找違反 boundary 的 block，把它跟邊界上的 block 互換或掛在邊界 block 下 | **✓** |
| M7 | **FixGrouping** | 0.05 | 找分裂的 cluster，把游離的 block 接到主 component 上方 | **✓** |

Revert 機制：每個 move 在執行前會把整棵樹的 (parent, lc, rc) 與 root 都存進 `Move::saved_w_vec/h_vec`，reject 時就用 `restore_topology()` 還原。對 n ≤ 200 來說 3·n integer 的拷貝完全可接受，換來「絕對正確的 revert」。

### 2.5 Initial Tree（`parallel.cpp::make_initial`）

```
按 priority 排序 → 一個一個插：
  priority 0: preplaced
           1: corner-constrained (BL/BR/TL/TR)
           2: edge-constrained
           3: 有 group_id 或 mib_group 的 block（大 area 優先）
           4: 一般 soft block（大 area 優先）

插入時：
  if 有同 group/MIB 已被插入：     u = 隨機抽該 group 的成員（提高貼齊機率）
  else                              u = 隨機抽任一已插入的 block
  從 u 開始，random walk left/right child 到第一個空 slot 插入
```

關鍵點：**這份初始樹會強烈影響 SA 起點**。N 個 thread 用不同 seed → 不同的 priority 內排序 + 不同的 walk 方向 → 不同的起點 → SA 多重重啟（multi-start）。

---

## Part 3 — 參數速查表

> **格式**：`參數名` → 檔案:行號（預設值）｜遇到什麼狀況時調｜怎麼調

### 3.1 比賽框架介面

| 參數 | 位置 | 預設 | 說明 |
|------|------|------|------|
| `FLOORPLANNER_BIN` | env var | `./floorplanner` | 指定 binary 路徑 |
| `FLOORPLANNER_THREADS` | env var | `8` | 平行 SA chain 數量 |
| `FLOORPLANNER_TIME` | env var | `8+1.0*n` | 每 case 時間預算（秒），表達式可用變數 `n` = block 數 |
| `FLOORPLANNER_SEED` | env var | `1` | 基礎 RNG seed；case i 用 `seed + i` |
| `FLOORPLANNER_KEEP` | env var | `0` | `1` = 保留中介 `.txt`/`.sol` 方便除錯 |

### 3.2 Python wrapper (`my_optimizer.py`)

| 參數 | 位置 | 預設 | 何時調 | 怎麼調 |
|------|------|------|--------|--------|
| Baseline whitespace 比例 | `my_optimizer.py:137` | `1.10` | `area_gap` 經常為負 → baseline 估太鬆，調小到 `1.05`；經常為正過大 → 調大到 `1.15` | 直接改 `total_area * 1.10` 那行常數 |
| Baseline HPWL 倍率 | `my_optimizer.py:144` | `0.5` (avg_edge_len = side·0.5) | HPWL 跟 area 在 sa_cost 內失衡時 | 改 `side * 0.5` 的 `0.5` |
| Aspect ratio 範圍 | `my_optimizer.py:265` | `0.33 / 3.00` | block 看起來太瘦長，容易留空隙 → 收緊到 `0.5 / 2.0`；某些 case 需要極端形狀 → 放寬到 `0.25 / 4.0` | 直接改字串 |
| 預設時間預算公式 | `my_optimizer.py:373` | `8+1.0*n` | 想多花時間求精準 → `10+2.0*n`；想跑得快 → `5+0.5*n` | 也可直接 export `FLOORPLANNER_TIME` |

### 3.3 SA Cost 權重 (`cost.hpp` `SAWeights`，48–60 行)

| 參數 | 預設 | 說明 / 何時動 |
|------|------|---------------|
| `w_area` | `1.0` | bbox area / baseline_area。**area_gap 一直很大（floorplan 太鬆）**：調到 `2.0–3.0`。 |
| `w_hpwl` | `1.0` | 同上，HPWL。**HPWL 連線太長**：調到 `2.0–3.0`。 |
| `w_overlap` | `5000.0` | overlap 硬罰。SA 出現 overlap=true 的解被接受 → 調到 `10000`。 |
| `w_softarea` | `5000.0` | soft block area 超過 1% 容忍度。同 `w_overlap` 邏輯。 |
| `w_group` | `80.0` | grouping 違規 (per V)。**V_rel 高（>0.5）**：調到 `120–200`。 |
| `w_mib` | `80.0` | MIB 違規 (per V)。同 group。 |
| `w_bound` | `80.0` | boundary 違規 (per V)。同 group。 |
| `w_outline` | `0.0` | 預留欄位（v9 沒有 fixed outline）。如果之後想 penalize aspect ratio，可從這個欄位接手。 |

> **判斷方向**：
> * 看 `iccad2026_evaluate.py` 跑出的 `hpwl_gap`、`area_gap`、`violations_relative` 分別大不大。
> * 哪個指標惡化 → 把對應的 weight 調大。
> * 改 weight 之後**要重新編譯 + 重 deploy binary**。

### 3.4 SA 退火排程 (`sa.hpp` `SAConfig`，26–34 行)

| 參數 | 預設 | 何時動 |
|------|------|--------|
| `n_iters_per_block` | `200` | 每個 block 在每個溫度 step 裡執行的 iter 數。SA log 看起來「溫度沒下來就跑完時間」→ 降到 `100`；「溫度太快下到 0」→ 升到 `400`。 |
| `K` | `7` | Fast-SA Stage 2 結束的 step 編號。改了會徹底改變退火曲線，沒有強烈理由不要動。 |
| `p_accept_init` | `0.99` | 起始溫度希望接受的 uphill 機率。SA 卡在初始解 → 升到 `0.999`；初期 random walk 過頭 → 降到 `0.95`。 |
| `c_fastsa` | `100.0` | Fast-SA Stage 2 的常數。同 K。 |

### 3.5 Move 機率與設定 (`moves.cpp`)

| 參數 | 行號 | 預設 | 何時動 |
|------|------|------|--------|
| `P_FIXB` | `moves.cpp:310` | `0.05` | **boundary V 一直高**：調到 `0.10–0.15`。 |
| `P_FIXG` | `moves.cpp:311` | `0.05` | **grouping V 一直高**：調到 `0.10–0.15`。 |
| `P_AR` | `moves.cpp:312` | `0.18` | aspect ratio 探索不足（block 都呈方形，沒貼齊）→ 升到 `0.25`。 |
| `P_MIB` | `moves.cpp:313` | `0.05` | MIB V 一直高 → 升到 `0.10`。 |
| `P_ROT` | `moves.cpp:314` | `0.15` | floorplan 看起來都同一方向（沒探索旋轉）→ 升到 `0.20`。 |
| `P_SWP` | `moves.cpp:315` | `0.15` | SA 收斂太慢 → 跟 `P_ROT` 一起升 |
| `tol`（AR move 內 area 抖動容忍） | `moves.cpp:26` | `0.005` | hard 容忍是 1%，留 0.5% 安全 margin。改太大會 area 違規。 |

> **重要**：六個機率加起來必須 = 1.0（剩下的會自動分給 `MoveKind::Move`）。改的時候自己算一下。

### 3.6 主程式 CLI (`main.cpp`)

```bash
floorplanner <input.txt> <output.sol> [options]
  --time SEC              預設 30
  --threads N             預設 hardware_concurrency()
  --seed S                預設 42
  --iters-per-block K     預設 200
  --verbose
```

`my_optimizer.py` 已自動把 `--time / --threads / --seed` 接好，`--iters-per-block` 沒接（要的話加一個 `FLOORPLANNER_IPB` env var 進 wrapper）。

---

## Part 4 — 故障排除對照表

| 看到的症狀 | 第一個要檢查的位置 | 可能的調整 |
|------------|--------------------|------------|
| 全部 case `cost = 10`（infeasible） | `iccad2026_evaluate.py` print 的 `overlap_violations` / `area_violations` / `dimension_violations` 哪個 > 0 | overlap → 看 `cost.hpp` 的 `w_overlap`；area → 看 `moves.cpp` 的 `tol = 0.005`；dimension → `my_optimizer.py::_write_txt` 有沒有把 fixed/preplaced 的 wi/hi/xi/yi 寫對 |
| `area_gap` 很大（floorplan 太鬆） | `cost.hpp::w_area`；`my_optimizer.py` baseline_area 倍率 | 升 `w_area` 到 2.0；或調 baseline 倍率到 1.05 |
| `hpwl_gap` 很大（連線太長） | `cost.hpp::w_hpwl` | 升到 2.0–3.0 |
| `V_rel` 一直在 0.5 以上 | 看 `boundary / grouping / mib` 哪個是主因；對應升 `P_FIXB/G` 與 `w_bound/group/mib` | 例如 boundary 主因 → `P_FIXB = 0.10`、`w_bound = 120` |
| Boundary block 明顯沒貼邊 | `moves.cpp::apply_fixb`（第 204 行起）；`make_initial()` 的 priority sort | 把 corner / edge constraint 的 priority 從 1/2 提到 0.5 / 1.5（如果 priority enum 改，記得 sort comparator 也要改） |
| 同 cluster 的 block 散開 | `moves.cpp::apply_fixg`；`make_initial()` 對同 group 成員的 sibling 偏好 | 升 `P_FIXG`；`make_initial()` 中加大「同 group sibling」的偏好機率 |
| MIB group 的 block 大小不一 | `moves.cpp::apply_mib`；`apply_rotate` 對 MIB group 的同步邏輯（45–67 行） | 通常是 `apply_rotate` 漏寫 MIB 同步；`MibSync` 機率太低 |
| floorplan 一直又高又瘦 | B\*-tree topology 不平衡 | 在 `cost.hpp` 加 `w_outline` 做 aspect ratio penalty（公式：`|bbox_w/bbox_h − 1|`），cost.cpp 加上對應加項；或改 `make_initial` 的 walk 邏輯讓左右 child 機率不對稱 |
| SA 跑完 cost 還在很高處 | 時間預算太短 / 移動效率太差 | 升 `FLOORPLANNER_TIME`；或減 `n_iters_per_block` 讓更多次降溫 |
| 多 thread 結果幾乎一樣 | RNG seed plumbing 有 bug | 看 `parallel.cpp` 給每個 thread 的 seed 是不是真的不同 |
| 想看 SA 進度 | 把 `SAConfig.verbose = true`，或在 `my_optimizer.py` 加 `--verbose` flag | 注意 verbose 多 thread output 會交錯 |
| 想保留中間檔案 | `export FLOORPLANNER_KEEP=1` | 跑完去 `/tmp/my_optimizer_*/` 看 `case_NNN.txt`、`case_NNN.sol` |

---

## Part 5 — 編譯與部署 SOP

每次改完 C++ 都要：

```bash
# 1. 同步檔案進 WSL build dir（如果是在 Windows 開發）
cp 'C:/Gozzz/3_Junior/EDA專題/code/cost.hpp'      /home/pop/2026_EDA_contest/include/
cp 'C:/Gozzz/3_Junior/EDA專題/code/moves.hpp'     /home/pop/2026_EDA_contest/include/
cp 'C:/Gozzz/3_Junior/EDA專題/code/moves.cpp'     /home/pop/2026_EDA_contest/src/
cp 'C:/Gozzz/3_Junior/EDA專題/code/parallel.cpp'  /home/pop/2026_EDA_contest/src/
# (改其他檔案就照樣 cp)

# 2. 重新編譯
cd /home/pop/2026_EDA_contest && make clean && make -j4

# 3. Smoke test（必須通過 contest_cost ≈ 1.0）
make check

# 4. 部署 binary 到 contest dir
cp floorplanner /home/pop/IntelLabs_Floorset/FloorSet/iccad2026contest/

# 5. 重跑 evaluator
cd /home/pop/IntelLabs_Floorset/FloorSet/iccad2026contest
source /home/pop/IntelLabs_Floorset/FloorSet/venv/bin/activate
python iccad2026_evaluate.py --evaluate my_optimizer.py --test-id 0 1 2 3 4 --save-solutions

# 6. 視覺化看結果
python my_visualize.py --solutions my_optimizer_solutions.json --all --save-dir ./vis_output --no-show
```

只動 `my_optimizer.py` 不需要重編，直接跑步驟 5。

---

## 附錄 A — Constraint priority 對應表

| Constraint | hard / soft | 在 cost.cpp 計算位置 | 對應 SA weight |
|------------|-------------|----------------------|----------------|
| Overlap | **hard**（infeasible） | `check_hard_constraints()` line 257 | `w_overlap` |
| Soft block area 1% 容忍 | **hard** | `check_hard_constraints()` line 224 | `w_softarea` |
| Fixed-shape 維度不可變 | **hard** | `check_hard_constraints()` line 235 | `w_overlap` (借用) |
| Preplaced 位置不可變 | **hard** | `check_hard_constraints()` line 245 | `w_overlap` (借用) |
| Boundary（貼邊） | soft | `evaluate()` line 161 | `w_bound` |
| Grouping（cluster） | soft | `evaluate()` line 138 | `w_group` |
| MIB（同 group 同形） | soft | `evaluate()` line 145 | `w_mib` |

## 附錄 B — Boundary code 編碼對照（v9 PDF 重要陷阱）

* **官方 (Python)** 用 bitmask：1=L, 2=R, 4=T, 8=B；corner 是相加（5=TL, 9=BL, 6=TR, 10=BR）。
* **C++ enum** 用順序 0..7：L/R/B/T/BL/BR/TL/TR。
* 轉換在 `my_optimizer.py:76-86` 的 `_BOUNDARY_BITMASK_TO_ENUM` 表。**改 C++ enum 必須同步改這張表**。
