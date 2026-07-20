# Intro to EDA — Final Project Report
### ICCAD 2026 CAD Contest — Problem C：FloorSet 固定外框平面規劃（Fixed-Outline Floorplanning）

> 本檔為報告草稿，依助教給定之 ICCAD 競賽組格式撰寫。**提交前請補上**：作者姓名／學號、報名截圖、Alpha Test 繳交證明，並轉存為 PDF。

---

## （封面）題目標題

**基於 B\*-tree 模擬退火、確定性約束修復與機器學習暖啟動的固定外框平面規劃器**

*A Fixed-Outline Floorplanner combining B\*-tree Simulated Annealing, Deterministic Constraint Repair, and Machine-Learning Warm-Start*

- 競賽：ICCAD 2026 CAD Contest — Problem C (FloorSet, Intel Labs)
- 作者：＿＿＿＿＿（姓名／學號）
- 指導教授：＿＿＿＿＿
- 日期：2026/06

---

## 1. 題目簡介（Project Introduction）

本題目要求設計一個**固定外框平面規劃器**：給定一組電路方塊（blocks）、外部端點（terminals）、方塊間與端點間的連線（nets），以及多種設計約束，在不重疊的前提下決定每個方塊的位置與形狀，使一個綜合「面積、線長、約束違反、執行時間」的成本函數最小化。

**輸入**：每個 test case 含 N 個方塊（含各自的目標面積、約束旗標）、外部端點座標、block-to-block（b2b）與 pin-to-block（p2b）加權連線、以及群組／MIB／邊界等約束。

**約束分為兩類：**

| 類別 | 約束 | 說明 |
|---|---|---|
| 硬約束（違反 → cost = M = 10） | 不重疊 | 任何兩方塊不可重疊 |
| | 軟方塊面積 | 軟方塊 w×h 須在目標面積 ±1% 內 |
| | 固定形狀（fixed） | 指定 (w, h) 不可變動 |
| | 預置（preplaced） | 指定 (x, y, w, h) 完全鎖死 |
| 軟約束（以 V_rel 懲罰） | 群組（grouping） | 同群成員須連通（互相貼合） |
| | MIB | 同 MIB 群成員須共用相同形狀 |
| | 邊界（boundary） | 指定方塊須貼齊外框某一邊／角 |

**成本函數（官方 v10）：**

```
Cost_i = ( 1 + 0.5·(HPWL_gap + Area_gap) ) × exp( 2·V_rel ) × max(0.7, RuntimeFactor^0.3)
       = M (=10)  若不可行(infeasible)
```
- `HPWL_gap`、`Area_gap`：線長／外框面積相對於最佳基準的相對差距
- `V_rel ∈ [0,1]`：軟約束相對違反量；以**指數**方式放大成本（影響最大）
- `RuntimeFactor = 本解執行時間 / 全體參賽者在該 case 的中位數`（本機評測時固定為 1.0）

**總分（exponential weighting）：**
```
Total = Σ_i Cost_i · e^((n_i − n_max)/12)  /  Σ_j e^((n_j − n_max)/12)
```
方塊數 n 愈大的 case 權重愈高 → **「最大的那一批 case」主導總分**，這個觀察貫穿我們整個優化策略。

**目標**：在可行（feasible）的前提下，讓大 case 的品質（HPWL/Area gap）與軟約束違反盡量低，同時保持有競爭力的執行時間。

---

## 2. 方法流程圖（Method Flowchart）

```
                    ┌─────────────────────────────────────────────┐
                    │            單一 Test Case (tensors)           │
                    │  blocks / terminals / nets / 約束             │
                    └───────────────────────┬─────────────────────┘
                                            │
                   ┌────────────────────────┴────────────────────────┐
                   │  Python 前處理 (my_optimizer_ml.py)              │
                   │  ① 轉成求解器文字輸入 (.txt)                     │
                   │  ② ML 預測每方塊位置 → 附加 WARM_POSITIONS 區塊   │
                   └────────────────────────┬────────────────────────┘
                                            │ case.txt
                                            ▼
        ┌───────────────────────── C++ 求解器 (floorplanner) ─────────────────────────┐
        │                                                                              │
        │   8 條平行 SA thread（multi-start，最後 best-of 取最佳）                       │
        │   ┌─────────────────────────────┐   ┌─────────────────────────────────────┐ │
        │   │  一半 thread：ML 暖啟動樹     │   │  一半 thread：啟發式隨機初始樹        │ │
        │   │  (make_initial_warm)         │   │  (make_initial，保底＋多樣性)        │ │
        │   └───────────────┬─────────────┘   └──────────────────┬──────────────────┘ │
        │                   └──────────────┬──────────────────────┘                   │
        │                                  ▼                                           │
        │        ┌──────── 每次迭代：propose move → pack → evaluate → accept ────────┐  │
        │        │  三階段幾何降溫 Fast SA；卡住且 infeasible 時 reheat 重探索          │  │
        │        │  Contour Packer（每次 pack 內含）：                                 │  │
        │        │    1) 預載 preplaced 錨點 footprint  2) DFS contour 擺放            │  │
        │        │    3) 左下壓實  4) 長寬比平衡  5) 補洞                              │  │
        │        │    6) Grouping repair  7) Boundary repair（確定性）                │  │
        │        └────────────────────────────────────────────────────────────────┘  │
        │                                  │ best B*-tree                              │
        └──────────────────────────────────┼──────────────────────────────────────────┘
                                            ▼
                           ┌────────────────────────────────────┐
                           │  feasible？                         │
                           │   是 → 輸出解 (.sol)                 │
                           │   否 (rc=4) → 自動加大時間預算重跑    │
                           │            (feasibility escalation)  │
                           └────────────────┬───────────────────┘
                                            ▼
                                    每方塊 (x, y, w, h)
```

---

## 3. 方法介紹（Methods）

### 3.1 整體架構

我們的求解器以 **B\*-tree 表示法 + 平行多起點 Fast Simulated Annealing（Fast-SA）** 為核心，並在三個層面加以強化：

1. **Contour Packer 內建多道確定性「約束修復」pass**——把原本只靠 SA 隨機碰運氣才能滿足的約束（錨點不重疊、邊界貼齊、群組連通）改為**每次 pack 都確定性地修好**。
2. **機器學習暖啟動（ML warm-start）**——以 Graph Transformer 預測每方塊位置，將其轉成 B\*-tree 當作部分 thread 的起始解。
3. **可行性觸發的時間升級（feasibility escalation）**——只在少數困難 case 自動加時間，整體仍保持快速。

### 3.2 B\*-tree 表示法與 Contour Packer

**B\*-tree** 是一種以二元樹編碼緊湊矩形佈局的表示法（Chang 2006）：
- 左子節點 = 放在父節點**右邊**
- 右子節點 = 放在父節點**上方**

**Contour Packer** 以 DFS 走訪 B\*-tree，維護一條「天際線（skyline）輪廓」，依序把每個方塊放在輪廓上方最低可行位置；對 B\*-tree 而言，樹上方塊之間**天生不重疊**。打包後再做數道後處理：左下壓實（compaction）、長寬比平衡（balance）、L 形補洞（holes-fill）。

**問題與創新——錨點重疊（本專題的關鍵突破之一）**

預置（preplaced）方塊位置固定。原本求解器在 DFS「輪到錨點時」才把它加入輪廓，導致**更早擺放的樹方塊落入錨點 footprint**，固定錨點壓在上面 → **重疊 → 不可行（cost 10）**。在大型密集 case 上，8 條 thread 常常全部失敗。

> **解法：錨點 footprint 預載（pre-seed）。** 在擺放任何樹方塊「之前」，先把所有 preplaced footprint 灌進輪廓。如此每個樹方塊經過輪廓查詢時都會自動被抬到錨點上方 → **樹方塊與錨點重疊在建構上不可能發生**。

```
Pseudo-code: Contour Packer（加入錨點預載）
  contour ← {(0,0)}
  for each preplaced block p (依頂緣高度升序):       # ← 關鍵：先預載
      raise_contour(contour, [p.x, p.x+p.w], p.y+p.h)
  DFS over B*-tree:
      if block is preplaced: x,y ← 固定值 (不更新輪廓，已預載)
      else:                  x ← 由父節點推得; y ← contour 高度; 更新輪廓
  compact_left_down(); balance(); holes_fill()
  grouping_repair(); boundary_repair()              # ← 見 3.4
```

### 3.3 平行多起點 Fast Simulated Annealing

- **三階段幾何降溫**：高溫探索 → 快速降溫 → 緩慢精修，並於階段交界一次性 reheat。
- **移動集合（move set）**：Rotate、Move（子樹嫁接）、Swap、Aspect-Ratio（重採樣軟方塊長寬）、MIB-Sync、FixBoundary、FixGrouping。
- **平行多起點 + best-of**：同時跑 8 條獨立 SA chain（不同 seed），最後取「可行優先、成本最低」者。此架構讓 ML 暖啟動成為**零風險的加分項**——好就贏，壞就被其他 thread 蓋過。
- **卡住且不可行時 reheat**：原本降溫到底就停。我們改為「若仍 infeasible 就升溫回最高溫重新探索」，把剩餘時間花在跳出不可行盆地。

### 3.4 確定性約束修復 Pass（本專題核心貢獻）

我們觀察到一個通則：**「有 cost 懲罰」≠「會被修好」**。軟約束雖有懲罰項，但要靠 5% 機率的隨機 move 去滿足，而每次 pack 的確定性壓實又不斷把它們破壞 → 隨機打不過確定性。**解法：用確定性 pass 直接修好。**

| Pass | 修什麼 | 做法（確定性，壓實後執行） |
|---|---|---|
| **Boundary repair** | 邊界方塊未貼邊 | 把每個邊界方塊滑到其指定邊；僅在目標格子為空時移動（不重疊、不增大外框） |
| **Grouping repair** | 群組成員落單 | 把「未與任何同群成員相鄰」的方塊滑到貼齊某成員的四邊之一（若有空位） |

兩者皆 **不破壞硬約束、不增大外框面積**，代價僅是少量內部空白——換取砍掉 `exp(2·V_rel)` 這個指數乘數，極為划算。

### 3.5 機器學習暖啟動（ML Warm-Start）

**動機**：純隨機初始的 SA 在大型 case 上難以快速找到好佈局。我們以監督式學習，讓模型直接預測「每個方塊大概該放哪」，作為 SA 的良好起點。

- **模型**：Graph Transformer（PreLN，4 層、4 heads、hidden=128，約 0.57M 參數）。輸入為每方塊 16 維特徵（面積、約束旗標、連線度數、邊界碼、形狀提示等）＋端點座標，經自注意力後輸出每方塊 (cx, cy, w, h)。
- **訓練資料**：FloorSet Lite v2，約 100 萬個 (問題, 最佳解) 配對。
- **損失**：位置 smooth-L1 ＋ 形狀 smooth-L1 ＋ 面積一致性，並以每 case 外框尺寸正規化（避免大佈局主導損失）。
- **暖啟動轉換**：將 ML 預測的座標依「左下優先、方向（右→左子、上→右子）」轉成一棵合法 B\*-tree，供一半 thread 使用；若樹不合法則自動退回啟發式初始樹（保底）。
- **產線模型**：`floorplan_v2`（驗證集 pos 誤差約 2.8%）。我們另以「大 case 加權」微調（fine-tune）出 v3，但下游實測 v3 並未勝過 v2（見 5.5），故 **v2 為最終採用模型**。

### 3.6 時間預算與可行性升級

- **時間公式**：`budget = 1 + 0.05·n` 秒（n = 方塊數），小 case 快、大 case 給足時間。
- **Feasibility escalation**：若求解器回報不可行（rc=4），Python 端自動以更大預算、**不同 seed** 重跑該 case（獨立抽樣，提高可行機率）。因不可行＝cost 10，這筆時間極值得；且只花在少數困難 case 上。實作確定性修復後，此機制已很少被觸發。

---

## 4. Testcase 介紹（Testcase Table）

我們以 FloorSet Lite 官方驗證集共 **100 個 test case** 進行開發與評測。各 case 規模與約束差異甚大：

| 屬性 | 範圍／說明 |
|---|---|
| 方塊數 n | 約 10 ～ 120（大 case 主導總分） |
| 外部端點數 | 數個 ～ 約 400 |
| b2b 連線 | 數十 ～ 約 1,700 |
| p2b 連線 | 數十 ～ 約 4,000（最密的 case 高達數千） |
| 硬約束 | 不重疊、軟方塊面積 ±1%、fixed 形狀、preplaced 位置 |
| 軟約束 | grouping、MIB、boundary（每個 case 數量不一） |

**代表性困難 case（最大且最密的一批，對總分權重最高）：**

| Test ID | 方塊數 n | 端點 | b2b | p2b | preplaced | 特性 |
|---|---|---|---|---|---|---|
| 95 | 116 | 405 | 1,345 | 4,181 | 6 | 超密連線＋多錨點，最難可行 |
| 98 | 119 | 424 | 1,699 | 908 | — | 大型、外框易過鬆 |
| 93 | 114 | 318 | 1,299 | 652 | — | 大型密集 |
| 88–99 | 109–120 | — | — | — | — | 「最大的 12 個 case」，主導總分 |

---

## 5. 實驗結果（Experimental Results）

> 評測平台：本機 FloorSet 官方評測腳本（`iccad2026_evaluate.py`，v10）。本機 RuntimeFactor 固定為 1.0（官方才以跨參賽者中位數計算）。預設時間預算 `1+0.05·n`。

### 5.1 確定性修復的逐步效果（最大的 12 個 case：88–99）

各項改動對「主導總分的大 case」之 Total Score 影響（愈低愈好）：

| 階段 | Total Score (88–99) | 相對改善 | 可行數 |
|---|---|---|---|
| 基準（修復前） | 6.9464 | — | 9 / 12（3 個不可行） |
| ＋ 錨點預載修復 | （達成 100% 可行） | — | **12 / 12** |
| ＋ Boundary repair | **5.6159** | −19.2% | 12 / 12 |
| ＋ Grouping repair | **4.8969** | −13.0% | 12 / 12 |
| ＋ Move-mix 再平衡 | **4.8335** | −1.3% | 12 / 12 |
| **累計** | **6.95 → 4.83** | **−30.5%** | 12 / 12 |

### 5.2 全 100 個 case 之最終結果

| 指標 | 數值 |
|---|---|
| Total Score | **4.8706** |
| 可行（Feasible） | **100 / 100** |
| 平均 Cost | 4.3274 |
| 平均執行時間 | 4.32 s |

相較開發初期「大型 case 經常不可行（cost 10）」，最終達成**全部可行、且大 case 品質大幅提升**。

### 5.3 可行性修復前後對照（錨點預載的效果）

以最密的 case 95（n=116）為例，重複跑 6 次、統計 8 條 thread 中可行的數量：

| 設定 | 6 次跑可行的比例 | 8 thread 中可行數 |
|---|---|---|
| 修復前 | 時好時壞 | 約 1 / 8（隨機運氣） |
| 錨點預載後 | **6 / 6 皆可行** | **8 / 8** |

### 5.4 時間預算 vs. 品質權衡（88–99）

固定其他設定，掃描不同時間預算公式：

| 時間預算公式 | 平均 runtime | Total Score (88–99) |
|---|---|---|
| 0.2 + 0.08·n | 8.7 s | 4.77 |
| 2 + 0.05·n | 7.5 s | 4.82 |
| **1 + 0.05·n（採用）** | **6.3 s** | **4.96** |
| 0.5 + 0.03·n | 4.7 s | 5.52 |
| 0.3 + 0.02·n | 4.7 s | 5.56 |
| 0.2 + 0.01·n | 4.6 s | 5.56 |

**觀察**：約 6 秒處存在品質懸崖，低於它分數急升（變差）；且大型 case 受限於下限約 4.6 秒。將官方 `RuntimeFactor^0.3` 納入計算後，跨各種「中位數假設」分析顯示 `1+0.05·n` 皆為穩健最優——**追求極端快速反而得不償失**（損失的品質大於省下的時間懲罰），故採用之。

### 5.5 機器學習模型 A/B（88–99）

比較產線模型 v2 與「大 case 加權微調」模型 v3：

| 模型 | Total Score (88–99) | 結論 |
|---|---|---|
| floorplan_v2（採用） | **5.09** | 較佳 |
| floorplan_v3（n² 加權微調） | 5.33 | 未勝出，棄用 |

**結論**：ML 暖啟動的品質在 v2 已趨飽和；剩餘的成本來自 packing/SA 的結構性鬆散，而非 ML 預測準度。故 ML 部分定案於 v2。

### 5.6 移動集合再平衡（Move-mix）

導入確定性修復後，原本負責約束滿足的隨機 move（FixBoundary/FixGrouping）邊際價值下降；我們將其機率（各 0.05）下調至 0.02，省出的預算移給品質型 move（Aspect-Ratio、子樹 Move）。88–99 由 4.897 → 4.834（小幅但無風險的改善）。

---

## 6. Reference

1. Y.-C. Chang, Y.-W. Chang, et al. *"B\*-Trees: A New Representation for Non-Slicing Floorplans,"* 及 *"Modern Floorplanning Based on B\*-Tree and Fast Simulated Annealing,"* IEEE TCAD, 2006.
2. Mostafa et al., *"PARSAC: Parallel Simulated-Annealing with Constraint-fixing moves for analog placement,"* 2024.（錨定方塊與約束修復 move 的概念來源）
3. Q. Xu et al., *"GoodFloorplan: Graph Convolutional Network + Reinforcement Learning for floorplanning."*（圖神經網路用於佈局的啟發來源）
4. Intel Labs, **FloorSet** dataset & ICCAD 2026 Contest Problem C. GitHub: `https://github.com/IntelLabs/FloorSet`
5. ICCAD CAD Contest 官網：`https://www.iccad-contest.org/tw/index.html`
6. 使用套件／工具：PyTorch、HuggingFace Hub（資料下載）、C++17 + pthread（自製求解器 `floorplanner`）、官方評測框架 `iccad2026_evaluate.py`。

---

## 7. 報名截圖與繳交證明（Submission Proof）

> **【提交前請補上】**
> - ICCAD Contest 報名截圖
> - Alpha Test 繳交證明截圖
> - （如選擇等待 Alpha Test 結果，請依規定向大助登記）

---

### 附註：開發里程碑（供撰寫心得參考，正式報告可精簡）
1. 分析 B\*-tree + Fast-SA 基線、三階段降溫重構。
2. 建置 ML pipeline（資料／模型／訓練／推論）並接入 C++ 暖啟動。
3. 診斷並修復大型 case 的錨點重疊不可行問題（錨點預載）。
4. 新增 boundary／grouping 確定性修復 pass，大幅降低軟約束違反。
5. Feasibility escalation、SA reheat、move-mix 再平衡。
6. 時間預算與 ML 模型之系統性實驗，定案產線設定。
