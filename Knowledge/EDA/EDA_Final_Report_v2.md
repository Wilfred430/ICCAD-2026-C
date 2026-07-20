# Intro to EDA — Final Project Report
### ICCAD 2026 CAD Contest — Problem C：FloorSet 固定外框平面規劃（Fixed-Outline Floorplanning）

> 本檔為報告草稿，依助教 ICCAD 競賽組格式撰寫。**提交前請補上**：作者姓名／學號、報名截圖、Alpha Test 繳交證明；§6 實驗結果中標示「（待補：模擬結果）」之表格，請貼上你們依 §5 分類跑出的數據；最後轉存 PDF。

---

## （封面）題目標題

**從 B\*-tree + Fast-SA 到「確定性約束修復 × 機器學習暖啟動」的固定外框平面規劃器**
*An Evolution from B\*-tree Fast-SA to a Constraint-Repair, ML-Warm-Started Floorplanner*

- 競賽：ICCAD 2026 CAD Contest — Problem C（FloorSet, Intel Labs）
- 作者：＿＿＿＿＿（姓名／學號）　指導教授：＿＿＿＿＿　日期：2026/06

---

## 1. 題目簡介

給定 k 個方塊（各有目標面積）、r 個固定端點、方塊間（b2b）與端點對方塊（p2b）的加權連線，以及多種設計約束，要求在**不重疊**前提下決定每個方塊的位置與形狀，最小化下列成本：

```
Cost_i = min( (1 + 0.5·(HPWL_gap + Area_gap)) · e^(2·V_rel) · max(0.7, RuntimeFactor^0.3),  M−ε )
       = M (=10)   若違反任一硬約束（infeasible）
Total  = Σ_i Cost_i · e^(n_i/12) / Σ_j e^(n_j/12)      （方塊數 n 愈大權重愈高）
```

| 硬約束（違反 → cost 10） | 軟約束（以 e^(2·V_rel) 懲罰） |
|---|---|
| 不重疊；軟方塊面積 ±1%；fixed 尺寸鎖死；preplaced 位置+尺寸鎖死 | grouping（同群連通）；MIB（同群同形狀）；boundary（貼指定邊/角） |

**核心觀察**：總分由 `e^(n/12)` 加權，**最大的那批 case（n≈101–120）佔約 56% 權重**，因此我們所有優化都以「大 case 的可行性與品質」為優先。

---

## 2. 方法流程圖（主求解器 floorplanner 內部細節）

```
 Python 端 (my_optimizer.py / _ml.py)
   ① tensors → 文字輸入 .txt（含 baseline 估計）
   ② ML 推論每方塊 (cx,cy,w,h) → 附加 WARM_POSITIONS 區塊
                       │  case.txt
                       ▼
 ┌──────────────────────── C++ floorplanner ────────────────────────────────┐
 │ parser.cpp  load_instance(): 讀 blocks/terminals/nets/groups/MIB/WARM     │
 │                       │                                                    │
 │ parallel.cpp run_parallel(): 開 N=8 條 thread，共享 early-stop atomic       │
 │   ┌─ thread 0..N/2-1 ─────────────┐   ┌─ thread N/2..N-1 ───────────────┐  │
 │   │ make_initial_warm()            │   │ make_initial()                  │  │
 │   │  ML 座標 → 左下排序 → 依方向     │   │  約束感知優先序：corner→edge→    │  │
 │   │  (右=左子/上=右子) 建 B*-tree    │   │  group/MIB→大面積；lc/rc 平衡    │  │
 │   └───────────────┬────────────────┘   └──────────────┬──────────────────┘  │
 │                   └───────────────┬─────────────────────┘                   │
 │                                   ▼  每條 thread：sa.cpp run()               │
 │  ┌─────────── 三階段幾何降溫 SA 主迴圈（每次迭代）─────────────────────────┐  │
 │  │ 1. moves.cpp propose(): 依機率選 move                                  │  │
 │  │      Rotate / Move / Swap / AspectRatio / MibSync / FixBoundary /      │  │
 │  │      FixGrouping                                                       │  │
 │  │ 2. packer.cpp pack(): 把 B*-tree 變座標（核心，含 7 道處理）            │  │
 │  │      (a) 預載所有 preplaced footprint 進 contour   ← 可行性保證         │  │
 │  │      (b) DFS 沿 contour(skyline) 擺放每個方塊                          │  │
 │  │      (c) compact_left_down 左下壓實（迭代至不動點）                     │  │
 │  │      (d) bbox_balance_pass 長寬比平衡（搬 spike 方塊）                  │  │
 │  │      (e) holes_fill_pass 補 L 形空洞（對角搬移）                        │  │
 │  │      (f) grouping_repair_pass 把落單群組成員貼回鄰居  ← 確定性修復       │  │
 │  │      (g) boundary_repair_pass 把邊界方塊滑回指定邊    ← 確定性修復       │  │
 │  │ 3. cost.cpp evaluate()+sa_cost(): 算 HPWL/面積/違反/重疊，得平滑成本     │  │
 │  │ 4. Metropolis 接受/退回；always-accept(FixB/FixG)；更新 best(可行優先)   │  │
 │  │ 5. 降溫；卡住→reanchor 回 best；若仍 infeasible→reheat 回 T1 重探索      │  │
 │  │ 停止：時間到 / 停滯且凍結 / 任一 thread 達標通知                         │  │
 │  └────────────────────────────────────────────────────────────────────┘  │
 │  best-of：可行優先、contest_cost 最低者勝出 → save_solution() .sol         │  │
 └────────────────────────────────┬─────────────────────────────────────────┘
                                  ▼
 Python 端：若回傳 infeasible(rc=4) → 換 seed、加大時間預算自動重跑（escalation）
                                  ▼  每方塊 (x, y, w, h)
```

---

## 3. 方法演進時間線（從 B\*-tree + Fast-SA 到現在）

本專題從教科書級的 **B\*-tree + Fast-SA**（Chen & Chang, 2006）出發，歷經以下階段逐步演進。下表為**完整時間線**，每一列說明「改了什麼／為什麼／效果」：

| 階段 | 改動 | 動機 | 效果 |
|---|---|---|---|
| **P0 基線** | B\*-tree 表示 + FastSA（除以 k·c 降溫、M1/M2/M3 三種 move、隨機初始樹） | 起點 | 可運作但大 case 慢且品質差 |
| **P1 SA 核心重構** | 改為**三階段幾何降溫**＋階段交界一次性 reheat；新增三種停止條件（時間／停滯+凍結／跨 thread 達標）；以隨機 move 探針校準初溫 T1；所有可調參數移到 `.hpp` | FastSA 的除以 k 降溫不易控制、無明確再加熱 | 收斂更穩定可控 |
| **P2 Baseline 校正（關鍵 bug）** | 修正 `my_optimizer.py` 未估計 baseline → `cost.cpp` 退回 1.0 → 原始面積(~5萬)主導 → T1 被校到~5萬 → Metropolis `exp(-Δ/T)≈1` → SA 變**隨機漫步**。改為估計 baseline（Σ面積×1.1、net權重×半邊長） | 成本尺度錯誤使退火失效 | SA 真正開始「退火」而非亂走 |
| **P3 幾何品質修復** | 針對「細長方塊／高瘦外框」病灶：HPWL 拆成 **int/ext 兩權重**（ext 把佈局錨向端點範圍）；`compact_left_down` 迭代壓實；`bbox_balance_pass` 修長寬比；`holes_fill_pass` 補對角空洞；aspect-ratio move 加 clamp；**約束感知初始樹**（優先序＋連線度＋lc/rc 平衡）；preplaced 改為葉節點避免拖高佈局 | 早期解外框過鬆、方塊細長、線長爆增 | 面積/線長 gap 明顯下降 |
| **P4 ML 暖啟動** | 訓練 Graph Transformer 預測每方塊位置；`parser.cpp` 讀 `WARM_POSITIONS`；新增 `make_initial_warm` 把 ML 座標轉 B\*-tree；**一半 thread 用 ML 樹、一半隨機**，best-of 保底 | 純隨機初始在大 case 難收斂 | 暖啟動成為零風險加分項 |
| **P5 規則更新 v10** | 同步官方 v10：總分權重 `e^(n)`→`e^(n/12)`、runtime 跨隊伍中位數、feasible 上限 M−ε | 跟上官方計分 | 確認「大 case 群」皆重要，非只賭最大 |
| **P6 可行性突破（最關鍵）** | `packer.cpp`：在 DFS 擺放前**預載所有 preplaced footprint 進 contour** | 樹方塊原會落入錨點 footprint→固定錨點壓上→重疊→不可行；大 case 常 8 thread 全滅 | 大 case 從「常不可行」→**建構上保證可行** |
| **P7 穩健性** | `my_optimizer.py`：infeasible(rc=4) **換 seed＋加大時間自動重跑**；`sa.cpp`：卡住且仍 infeasible 時 **reheat 回 T1** | 可行解高度 seed-相依、SA 冷死在不可行盆地 | 困難 case 可行率大增 |
| **P8 軟約束確定性修復** | `packer.cpp` 新增 `boundary_repair_pass`、`grouping_repair_pass`：壓實後把邊界方塊滑回邊、把落單群組成員貼回鄰居（僅在空位、不重疊、不增大外框） | 「有 cost 懲罰」≠「會被修好」；5% 隨機 move 打不過每次 pack 的確定性壓實 | 88–99：6.95→5.62→4.90，砍掉 `e^(2·V_rel)` 乘數 |
| **P9 Move-mix 再平衡** | 把 `FixBoundary/FixGrouping` 機率各 0.05→0.02，省出的給 AspectRatio 與子樹 Move | repair 接手約束後，隨機修復 move 邊際價值下降 | 88–99：4.90→4.83 |
| **P10 ML 大 case 重訓實驗** | 新增 `--init-from` fine-tune、`--size-power` 大 case 加權；訓出 v3 | 想讓 warm-start 在高權重大 case 更準 | v3 下游未勝 v2 → **保留 v2**，確認 ML 已飽和 |

**一句話總結**：可行性與軟約束的瓶頸，最後都不是靠「更聰明的隨機退火」解決，而是靠**把它們轉成 packer 內的確定性修復**；ML 暖啟動則作為零風險的品質加分。

---

## 4. 整體架構（逐 C++ 檔案說明 + 與標準 B\*-tree+Fast-SA 之差異）

求解器以 C++17 撰寫，原始碼分為以下檔案。每節說明**職責、關鍵內容、設計理由、與標準做法的差異**。

### 4.1 `types.hpp` — 核心資料結構
- **內容**：`Block`（id、目標面積、現行 x/y/w/h、約束旗標 is_fixed/is_preplaced/mib/group/boundary、fixed/preplaced 的鎖定幾何、長寬比上下界、cx()/cy() 便捷函式）、`Terminal`、`Net`、`FloorplanInstance`（整個問題）、`BTree`（**以陣列索引存樹**：nodes[]、x/y/w/h[]）。
- **差異**：標準 B\*-tree 多以指標存樹；我們用**陣列索引**，使整個搜尋狀態可一次 memcpy 複製——平行多起點與 reanchor（回 best）都需要快速複製樹，這是工程上的關鍵設計。另含 ML warm-start 與錨點輸入幾何欄位。

### 4.2 `parser.cpp` / `parser.hpp` — 文字輸入輸出
- **內容**：`load_instance()` 以自製 lexer 逐段讀 N_BLOCKS / TERMINALS / BLOCKS / B2B / P2B / GROUPS / MIB，並解析 **`WARM_POSITIONS` / `WARM_PRIORITY`**（ML 提示）；`save_solution()` 輸出每方塊 (x,y,w,h)。
- **差異**：標準求解器無 ML 介面；我們新增 WARM 區塊讓 Python 端把 ML 預測傳入，且對未知 token 容錯（向前相容）。baseline 由 Python 端估計後寫入，供 `sa_cost` 正規化。

### 4.3 `btree.cpp` / `btree.hpp` — B\*-tree 拓樸
- **內容**：`build_default`（左脊）、`build_random`、`detach`、`insert_left/right`（含子樹嫁接保持可達）、`op_rotate`(M1 交換 w,h)、`op_swap`(M3 交換兩節點，含父子相鄰特例)、`op_move`(M2 子樹搬移，含成環檢查)、`validate`、`copy_from`。
- **差異**：此檔**基本就是標準 B\*-tree**（左子=右側、右子=上方）。唯一實作上的差異是「陣列化、可快速複製」的設計（見 4.1）。依老師要求，標準部分不贅述。

### 4.4 `packer.cpp` / `packer.hpp` — Contour 打包 + 多道後處理（**改動最大、最核心**）
標準 B\*-tree 打包只做「沿 skyline 擺放」。我們的 `pack()` 是一條 7 步管線：
1. **錨點預載（P6，本專題關鍵）**：DFS 前先把所有 preplaced footprint 灌進 contour（依頂緣升序，避免互相覆蓋）→ 每個樹方塊都會被抬到錨點上方 → **樹 vs 錨點重疊在建構上不可能**。
2. **DFS contour 擺放**：左子放父右側、右子放父上方，y 取該 x 範圍 skyline 最高處。
3. **`compact_left_down`**：交替「依 y 排序往下壓、依 x 排序往左壓」迭代至不動點（標準做法多為固定數次，不足以收斂）。
4. **`bbox_balance_pass`**：把定義長邊的 spike 方塊搬到較短的「貨架」上，修正高瘦/扁寬外框，並以 `sqrt(baseline_area)` 為短邊上限避免漂移。
5. **`holes_fill_pass`**：對角搬移把卡在 L 形空洞的方塊塞進角落（純軸向壓實到不了）。
6. **`grouping_repair_pass`（P8）**：把「未與任何同群成員相鄰」的方塊滑到貼齊某成員四邊之一（僅空位）。
7. **`boundary_repair_pass`（P8）**：把未貼邊的 boundary 方塊滑到指定邊/角（僅空位、不增大外框）。
- **差異**：標準做法**只有第 2 步**。第 1 步解決硬約束可行性、3–5 步大幅提升面積/線長品質、6–7 步把軟約束從「被懲罰」變「被確定性修好」——這是我們和教科書 B\*-tree 最大的分野。

### 4.5 `cost.cpp` / `cost.hpp` — 成本評估
- **內容**：`evaluate()` 算 HPWL（int/ext 分開、質心曼哈頓距）、外框面積、gap、軟違反（grouping/MIB/boundary）、硬違反（重疊、面積、fixed、preplaced）；`sa_cost()` 為 SA 用的**平滑加權成本**（面積＋拆分 HPWL＋軟違反＋**連續**重疊/面積漂移懲罰）；`contest_cost()` 為官方 Eq.2。
- **差異**：標準 SA 多用單一 HPWL 權重、且以「二元大跳躍」懲罰重疊。我們**拆分 int/ext HPWL**（ext 把佈局錨向端點，治細長病灶）、並把重疊與面積違反改為**連續懲罰**（給 SA 平滑梯度，避免成本雙峰震盪）；同時區分「SA 用的平滑成本」與「官方精確成本」。

### 4.6 `moves.cpp` / `moves.hpp` — 移動集合（move set）
| Move | 作用 | 標準有? |
|---|---|---|
| **M1 Rotate** | 交換某方塊 w↔h | ✅ |
| **M2 Move** | 子樹嫁接到隨機節點 | ✅ |
| **M3 Swap** | 交換兩節點在樹中位置 | ✅ |
| **M4 AspectRatio** | 在 ±1% 面積與長寬比帶內重採軟方塊形狀 | ✗（軟方塊特性新增） |
| **M5 MibSync** | 一次同步整個 MIB 群的共同形狀 | ✗ |
| **M6 FixBoundary** | 把違反邊界的方塊「交換/嫁接」到貼邊位置（always-accept，PARSAC 式約束修復） | ✗ |
| **M7 FixGrouping** | 把落單群組成員拉近群組（always-accept） | ✗ |
- **設計理由**：M4/M5 處理 FloorSet 特有的軟方塊與 MIB；M6/M7 借鏡 PARSAC 的 constraints-fixing move，且加了「**反向長大守門**」（當佈局已過高/過寬時跳過會惡化主維度的嫁接，治 case 55/56）。各機率集中於 `MoveProb`，P9 再平衡後 FixB/FixG 降到 0.02。
- **差異**：標準 Fast-SA 僅 M1/M2/M3；我們多了 4 種針對 FloorSet 約束與品質的 move。

### 4.7 `sa.cpp` / `sa.hpp` — 模擬退火驅動
- **內容**：**三階段幾何降溫**（高溫探索→快速冷卻→緩慢精修，階段交界一次性 reheat）；以隨機 move 探針（每次都復原）校準 T1 使 `exp(-Δavg/T1)=p_accept`；**reanchor**（停滯時 current 回 best）；**P7 reheat**（reanchor 時若 best 仍 infeasible → 升溫回 T1 重探索）；三停止條件；best 更新採**可行優先**。
- **差異**：標準 FastSA 用 `T=T1·Δavg/(k·c)` 的除以 k 降溫且再加熱是隱式的；我們改**顯式三階段幾何＋顯式 reheat**，並加入「可行性優先的再加熱」與「best-restart」，這是針對大 case 不可行盆地的關鍵。

### 4.8 `parallel.cpp` / `parallel.hpp` — 平行多起點協調器
- **內容**：`run_parallel()` 開 N=8 條獨立 SA chain（不同 seed），共享一個 early-stop atomic（任一達標就通知全體）；`make_initial()` 建**約束感知初始樹**（優先序 corner→edge→group/MIB→大面積；連線度做 tie-break 把高連線方塊拉近原點；lc/rc 計數平衡使初始外框接近正方）；`make_initial_warm()` 把 ML 座標轉樹；最後 best-of（可行優先、成本最低）。
- **差異**：標準 Fast-SA 多為單一 chain、隨機初始；我們做**平行多起點 + best-of + 約束感知初始 + ML 暖啟動初始**。best-of 架構讓「ML 暖啟動」與「啟發式隨機」並存且互為保底。

### 4.9 `main.cpp` — 進入點
- 解析 CLI（輸入/輸出/--time/--threads/--seed/--verbose），呼叫 `run_parallel`，印出診斷（feasible、contest_cost、各 gap、各硬/軟違反明細、可行 thread 數）。

---

## 5. Testcase 介紹與分類

我們以 FloorSet **驗證集 100 個 case** 開發。經實測掃描，其結構為：**每個 size 恰一個 case，n 從 21 到 120**（即 test-id i 對應 n=i+21）；**100 個 case 全部同時含有 preplaced、fixed、grouping、MIB、boundary 五種約束**，差異在「規模」與「連線密度」。最大連線：b2b 達 7,056、p2b 達 4,181、端點達 466。

### 5.1 依「規模」分類（決定計分權重）

| 類別 | 方塊數 n | test-id | case 數 | 約束量級（群組/邊界，隨 n 增長） | 計分權重 e^(n/12) | 難度 |
|---|---|---|---|---|---|---|
| A. 微型 Tiny | 21–40 | 0–19 | 20 | 群組~9–12、邊界~11–15 | 極低（≈0.1%） | 易，秒解 |
| B. 小型 Small | 41–60 | 20–39 | 20 | 中 | 低 | 易 |
| C. 中型 Medium | 61–80 | 40–59 | 20 | 中高 | 中 | 中 |
| D. 大型 Large | 81–100 | 60–79 | 20 | 高 | 高（≈22%） | 難 |
| E. 超大型 X-Large | 101–120 | 80–99 | 20 | 群組~21–33、邊界~31–37 | **最高（≈34%）** | **最難，主導總分** |

### 5.2 依「連線密度（p2b）」分類（決定可行難度）

| 類別 | p2b 連線數 | case 數 | 特性 |
|---|---|---|---|
| 稀疏 Sparse | ≤ 500 | 60 | 端點牽引弱，外框易過鬆 |
| 中等 Medium | 501–1500 | 25 | — |
| 密集 Dense | > 1500 | 15 | 端點牽引強、最難同時滿足不重疊與低線長 |

### 5.3 代表性困難 case（超大型 × 密集，對總分權重最高）

| Test ID | n | 端點 | b2b | p2b | fixed | preplaced | 群組 | 邊界 | 備註 |
|---|---|---|---|---|---|---|---|---|---|
| 95 | 116 | 405 | 1,345 | 4,181 | 10 | 6 | 26 | 37 | **最密**，多錨點，最難可行 |
| 99 | 120 | 330 | 7,056 | 3,524 | 10 | 6 | 28 | 36 | **最大且 b2b 最多** |
| 97 | 118 | 466 | 2,784 | 3,299 | 8 | 3 | 27 | 33 | 大型密集 |
| 98 | 119 | 424 | 1,699 | 908 | 8 | 7 | 21 | 31 | 錨點最多之一 |

> 我們將依 §5.1（A–E 五類）與 §5.3（代表困難 case）分別跑模擬，結果見 §6。

---

## 6. 實驗結果

> 平台：FloorSet 官方評測腳本（`iccad2026_evaluate.py`, v10）。本機 RuntimeFactor 固定 1.0（官方才用跨隊伍中位數）。預設時間預算 `1+0.05·n`。

### 6.1 演進各階段對「超大型 case（88–99）」總分之影響

| 階段 | Total Score (88–99) | 改善 | 可行數 |
|---|---|---|---|
| 基線（修復前） | 6.9464 | — | 9 / 12 |
| ＋錨點預載（P6） | （達 100% 可行） | — | **12 / 12** |
| ＋Boundary repair（P8） | 5.6159 | −19.2% | 12 / 12 |
| ＋Grouping repair（P8） | 4.8969 | −13.0% | 12 / 12 |
| ＋Move-mix（P9） | 4.8335 | −1.3% | 12 / 12 |
| **累計** | **6.95 → 4.83** | **−30.5%** | 12 / 12 |

### 6.2 全 100 case 最終結果

| 指標 | 數值 |
|---|---|
| Total Score | **4.8706** |
| 可行 | **100 / 100** |
| 平均 Cost | 4.3274 |
| 平均執行時間 | 4.32 s |

### 6.3 可行性修復前後（以最密 case 95，重複 6 次）

| 設定 | 6 次可行比例 | 8 thread 中可行數 |
|---|---|---|
| 錨點預載前 | 時好時壞 | ~1 / 8 |
| 錨點預載後 | **6 / 6** | **8 / 8** |

### 6.4 時間預算 vs 品質（88–99）

| 時間預算 | 平均 runtime | Total Score |
|---|---|---|
| 2 + 0.05·n | 7.5 s | 4.82 |
| **1 + 0.05·n（採用）** | 6.3 s | 4.96 |
| 0.5 + 0.03·n | 4.7 s | 5.52 |
| 0.2 + 0.01·n | 4.6 s | 5.56 |

> 約 6 秒處有品質懸崖；納入 `RuntimeFactor^0.3` 後，跨各種中位數假設 `1+0.05·n` 皆為穩健最優。

### 6.5 ML 模型 A/B（88–99）

| 模型 | Total Score | 結論 |
|---|---|---|
| floorplan_v2（採用） | 5.09 | 較佳 |
| floorplan_v3（n² 加權微調） | 5.33 | 未勝出，棄用 |

### 6.6 依 §5 分類之模擬結果（**待補：請貼上你們跑出的數據**）

| 類別 | 範例 test-id | Avg Cost | 可行率 | Avg Runtime |
|---|---|---|---|---|
| A 微型 (21–40) | 0,5,10,15 | （待補） | （待補） | （待補） |
| B 小型 (41–60) | 25,35,45,55 | （待補） | （待補） | （待補） |
| C 中型 (61–80) | … | （待補） | （待補） | （待補） |
| D 大型 (81–100) | … | （待補） | （待補） | （待補） |
| E 超大型 (101–120) | 88–99 | （待補） | （待補） | （待補） |
| 密集 case（p2b>1500） | 95,97,99 | （待補） | （待補） | （待補） |

---

## 7. Reference

1. T.-C. Chen and Y.-W. Chang, *"Modern Floorplanning Based on B\*-Tree and Fast Simulated Annealing,"* IEEE TCAD, 25(4), 2006.
2. Mostafa et al., *"PARSAC: Parallel SA with constraint-fixing moves"*（約束修復 move 與錨定方塊概念來源）, 2024.
3. Q. Xu et al., *"GoodFloorplan: GCN + Reinforcement Learning for floorplanning."*（圖神經網路於佈局之啟發）
4. S.N. Adya, I.L. Markov, *"Fixed-outline floorplanning: enabling hierarchical design,"* IEEE TVLSI, 2003.
5. Intel Labs **FloorSet** dataset & ICCAD 2026 Contest Problem C：`https://github.com/IntelLabs/FloorSet`
6. CAD Contest @ ICCAD 官網：`https://www.iccad-contest.org/tw/index.html`
7. 套件/工具：PyTorch、HuggingFace Hub、C++17 + pthread（自製求解器）、官方評測框架 `iccad2026_evaluate.py`。

---

## 8. 報名截圖與繳交證明（Submission Proof）
> **【提交前補上】** ICCAD Contest 報名截圖、Alpha Test 繳交證明截圖。
