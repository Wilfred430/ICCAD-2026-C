# 奪冠策略：生成式拓樸 + 獎勵微調（Final Submission 衝刺）

> **寫給**：團隊成員 + 未來的 AI 助手（Claude / ChatGPT）
> **目的**：把「結構上能拿第一的方法」固化成可執行計畫，並清楚切分
> 「人要做的探索/驗證」與「Claude 能直接動手寫的程式」。
> **日期**：2026-06-30（alpha test 已過，現在衝 beta → final）
> **前置閱讀**：[EVALUATION.md](EVALUATION.md)、[ALGORITHM_GUIDE.md](ALGORITHM_GUIDE.md)、
> [CLAUDE.md](CLAUDE.md)

---

## 0. TL;DR（30 秒版）

1. **純 SA 在大 case（n≈90–120）上數學上贏不了**——$10^6$ 次評估在 $10^{250}$
   的拓樸空間裡是盲人。而總分被 $e^n$ 加權，大 case 決定一切。
2. **現有 `ml/` 的 formulation 是壞的**：用 smooth-L1/MSE 回歸座標 →
   多峰解的平均 = 無效解（mode collapse）；而且**完全沒用到資料裡的
   `tree_sol`**（near-optimal 拓樸）。見 [ml/data.py:302](ml/data.py)。
3. **能奪冠的方法**：把那 1M 筆 `tree_sol` 當成「**可超越的示範**」，
   訓練**生成式拓樸模型**（採樣，不回歸），再用**真實 contest Cost
   做獎勵微調**超越 baseline（讓 Cost < 1），最後**推論時採樣多個拓樸 +
   連續幾何精修 + 合法化**，算力全砸在大 case。
4. **兩條腿走路**：電靜力法（目前最佳 2.966）當保底 backbone；
   生成式拓樸當衝頂引擎；兩者共用同一個「pack → 幾何精修 → legalize」後端。

---

## 1. 為什麼現在的方法贏不了（嚴謹診斷）

### 1.1 目標函數拆解：分數到底在哪裡

$$\text{Cost} = \underbrace{(1 + 0.5(\text{HPWL\_gap} + \text{Area\_gap}))}_{\text{可正可負}} \cdot \underbrace{e^{2 V_{rel}}}_{[1,\,7.39]} \cdot \underbrace{\max(0.7, RT^{0.3})}_{\geq 0.7}$$

三個推論，每個都改變策略：

- **推論 1：Cost 可以 < 1。** gap 是 signed `(actual − baseline)/baseline`，
  baseline 是 *near-optimal* 不是 optimal。**贏過** baseline → gap 為負 →
  Cost 壓到 0.7 以下。**目標不是逼近參考解，是超越它。**
- **推論 2：$e^{2V_{rel}}$ 是乘法地獄。** 任一 soft 違規（grouping/boundary/MIB）
  指數放大。$V_{rel}=0$ 是生死線，不是加分。約束處理必須是**架構級保證**。
- **推論 3：大 case 決定一切。** 分數按 $e^n$ 加權，n=120 比 n=21 重
  $e^{99}\approx 10^{42}$ 倍。總分幾乎完全由 n≈90–120 決定。
  > ⚠️ **這條必須先驗證**（見 §7）。`train.py` 註解寫的是「v10 contest」，
  > 確認最新 spec 的加權公式到底是 $\sum e^n\text{Cost}$、normalized weighted
  > mean、還是別的。即使比 $e^n$ 溫和，「大 case 又重又難」結論不變。

### 1.2 三個鑑別器（用來量每個方法的天花板）

| 鑑別器 | 純 SA | 電靜力/analytic | MSE warm-start（現況） | 生成式+獎勵微調 |
|---|---|---|---|---|
| **大 n 搜尋空間牆** | ❌ 硬牆，資訊論鎖死 | ✅ O(n) 連續優化，可擴展 | ✅ O(n) 推論 | ✅ O(n) 推論 |
| **多峰解 mode collapse** | N/A | N/A | ❌ 致命（見 1.3） | ✅ 採樣單一具體解 |
| **能否超越 baseline** | 受時間限制 | 受 legalize 限制 | ❌ 上限=示範平均 | ✅ 獎勵微調可超越 |

### 1.3 現有 `ml/` 的致命傷（對照實際程式碼）

1. **沒用 `tree_sol`。** [ml/data.py:302](ml/data.py) 明確寫
   `# tree_sol = t[4][ci] -- unused`。7-tuple 是
   `[0]blocks [1]b2b [2]p2b [3]pins_pos [4]tree_sol [5]fp_sol(w,h,x,y) [6]metrics`。
   你手上有 near-optimal 的**拓樸本身**，卻只拿座標 `fp_sol` 來回歸。
2. **MSE/smooth-L1 回歸 → mode collapse。** [ml/train.py](ml/train.py) 的
   `compute_loss` 對 (cx,cy,w,h) 做 smooth_l1。Floorplan 解高度多峰對稱
   （鏡射、等價 block 互換、旋轉皆等價）。回歸學的是 $\mathbb{E}[\text{layout}\mid\text{input}]$
   = 多個有效解的平均 = **幾乎必然無效**（block 疊一起）。
   $$\text{MSE 最優} = \text{好解的平均} = \text{爛解}$$
   這就是 warm-start 多半沒在幫忙的數學原因——不是訓練不夠，是 formulation 錯。
3. **warm-start 只是 soft hint。** [ml/predict.py:226](ml/predict.py) 產出
   position + priority ordering 餵給 [make_initial()](src/parallel.cpp)，
   但 SA 的 Stage-1 高溫會在數百 iter 內走掉這個起點（見 [SA_TUNING_GUIDE.md](SA_TUNING_GUIDE.md)）。
   → 精修必須改成**低溫 exploitation**，不能用現有高溫 schedule。

> ✅ **已做對的事**：`train.py` 已有 `--size-power` 對大 case 加權，
> 代表團隊已意識到 $e^n$ 問題。這個機制在新架構裡要保留。

---

## 2. 奪冠架構：四階段流水線

```
階段 0  監督預訓練（imitation，打底到 ≈ baseline）
  生成式模型（autoregressive over B*-tree 建構序列，或 sequence-pair）
  輸入: block feats + net hypergraph + constraints + terminals
  輸出: near-optimal「拓樸」(+ soft block 的 aspect ratio)
  資料: 全量 1M 的 tree_sol（teacher forcing）  ← 用起 data.py:302 丟掉的東西
  → gap≈0, V_rel≈0

階段 1  獎勵微調（突破 baseline，做到 Cost < 1）
  用真實 contest Cost 當 reward
  方法候選: policy gradient / best-of-N 蒸餾 / decision transformer
  關鍵: Cost 在訓練時可計算（你有 evaluator）→ 能超越示範品質

階段 2  推論時搜尋（把學到的多峰變成高品質多起點）
  採樣 K 個拓樸 → 各自 pack（packer.cpp）
   → 固定拓樸下的連續幾何精修（analytic HPWL/area + 1% area-slack sizing
                              + rotation，參 Huang 2023）
   → 約束合法化（參 CSF 2025 的 legalization）
   → 低溫短 SA polish（只 exploitation；不可用現有高溫 schedule）
  取 best feasible

階段 3  算力分配（呼應推論 3）
  推論 ~100ms 幾乎免費 → K 與精修預算全砸在 n≥90 的大 case
  小 case (n<50) 純 SA / 電靜力已夠好
```

每階段都打在要害：階段 0 破解大 n 搜尋牆 + 解 mode collapse + 留在 B*-tree；
階段 1 突破 baseline；階段 2 把多峰分布變多起點（等價 AlphaZero 的
inference-time search）；階段 3 把資源放在唯一決定總分的地方。

---

## 3. 為什麼這個有結構性優勢（別人沒有的籌碼）

1. **`tree_sol` 在資料裡** → 可直接預測**拓樸**而非座標，免去
   position→tree 的 inverse mapping（最髒的部分），完全留在現有
   [packer.cpp](src/packer.cpp) 引擎。**多數隊伍不會發現這點。**
2. **Cost 訓練時可計算** → 能做獎勵微調**超越示範**（AlphaGo→AlphaZero 進程）。
   imitation-only 上限 = demonstrator；唯有 reward 微調穩定壓進 Cost<1。
3. **測試集 = 訓練集同分布**（主辦明說）→ 監督式學習理想情境，泛化風險低。
   多數人把 1M 當 benchmark，你把它當武器。

---

## 4. 風險分級與兩條腿策略

| 方案 | 天花板 | 執行風險 | 定位 |
|---|---|---|---|
| 純 SA 調參 | 前 10% | 低 | 不可能第一（資訊論鎖死大 n） |
| 電靜力 + Huang rotation/sizing + CSF legalize | 前 3 有機會 | 中 | **保底 backbone** |
| 生成式拓樸 + 獎勵微調 + 推論搜尋 | **唯一能穩定 Cost<1 → 第一** | 高 | **衝頂引擎** |

**最大風險是時程，不是演算法。** 階段 0–1 是 2–3 人週硬工。
故策略上**兩條腿**：電靜力保底 + 生成式衝頂，**共用同一後端**
（pack → 幾何精修 → legalize），把整合風險降到最低。

---

## 5. 分工

### 5.1 你的部分（探索與驗證）— 用 Google AI Search + Connected Papers

**先驗證地基（最優先，10 分鐘）：**
```
ICCAD 2026 CAD contest problem C FloorSet-Lite scoring formula
exponential weighting e^n cost aggregation evaluation metric v10
```
→ 確認 §7 的兩個假設（加權公式、spec 版本 v9/v10）。

**核心方法探索（Google AI Search）：**
```
[生成式拓樸] generative model B*-tree sequence-pair floorplan topology
prediction autoregressive transformer supervised 2023 2024 2025 arxiv

[獎勵微調] offline reinforcement learning decision transformer combinatorial
optimization placement exceed demonstrations reward fine-tuning EDA 2024 2025

[mode collapse 背書] why regression fails multimodal structured prediction
mode averaging generative model layout combinatorial

[學到的初始解+低溫精修] learned warm start initialization simulated annealing
low temperature refinement neural network placement floorplan 2024

[analytic legalize 後端] fixed-outline floorplanning legalization constraint
graph electrostatic conjugate subgradient analytic 2023 2024 2025
```

**Connected Papers 種子（依優先序，找上下游）：**
1. **MdpoPlanner**（ASP-DAC 2026, DOI 11420532）— 最接近生成式拓樸路線
2. **PARSAC**（arXiv 2405.05495）— constraint-aware SA + 後端精修鄰居
3. **WireMask-BBO**（arXiv 2306.16844）— 把既有解當初始再精修（階段 2）
4. **Huang 2023 Electrostatics rotation/aspect-ratio**（DOI 10323841）—
   電靜力 backbone + DREAMPlace/ePlace 上游

**探索時要回答的問題（把答案貼回本文件 §6.4）：**
- 生成式拓樸最該用哪種表示？（autoregressive B*-tree 建構 vs sequence-pair
  vs CBL）哪個有 open-source 可參考？
- 獎勵微調用哪種方法 ROI 最高？（REINFORCE / GRPO / decision transformer /
  best-of-N 蒸餾）
- legalization 後端：沿用現有 constraint-graph（[electro_submission/legalize.py](electro_submission/legalize.py)）
  還是換 CSF 的 conjugate subgradient？

### 5.2 Claude 的部分（可立即開工，不被研究阻塞）

這些**現在就能做**，不需要等探索結果：

- [ ] **T1｜驗證現有 warm-start 是否有效**（基準線）
  跑 v1/v2/v3 weights vs baseline（無 ML）在 100 validation case 的 Total
  Score。預期：差異不顯著（印證 mode collapse）。產出對照表。
- [ ] **T2｜讓 `data.py` 載入 `tree_sol`**
  把 [ml/data.py:302](ml/data.py) 的 `tree_sol = t[4][ci]` 啟用，
  在 `_unpack_case` 回傳，並寫一個小 script 印出 tree_sol 的張量格式
  （shape / dtype / 編碼方式），這是階段 0 的前提。
- [ ] **T3｜寫 tree_sol → B\*-tree 的 decode/encode 工具 + round-trip 測試**
  確認能在 Python 端把 tree_sol 還原成 (parent, lc, rc) 並用 packer 重現
  near-optimal 座標（對照 fp_sol，誤差應 ≈ 0）。這驗證資料正確性。
- [ ] **T4｜把幾何精修後端獨立出來**（兩條腿共用）
  從 [electro_submission/](electro_submission/) 抽出「固定拓樸/固定相對位置下
  的連續 (w,h,x,y) 精修 + legalize」成獨立模組，讓 SA 解與 ML 解都能呼叫。
- [ ] **T5｜給 SA 加低溫 refine 模式**
  [src/sa.cpp](src/sa.cpp) 加一個 `--refine-from-init` 路徑：跳過 Stage-1
  高溫，直接低溫 exploitation，供「好起點精修」用。

> 開工順序建議：**T1 → T2 → T3**（驗證 + 解鎖資料），再 **T4/T5**（後端）。
> 階段 0 的模型訓練要等 T2/T3 完成 + §6.4 的表示法決定後才動。

---

## 6. 實作細節（給 Claude 的規格）

### 6.1 階段 0：生成式拓樸模型 I/O 與 loss（草案，待 §5.1 探索定案）

```
輸入（沿用 data.py 既有 + 補充）:
  blocks_feat [N,16]  （現成，見 data.py BLOCK_FEAT_DIM）
  b2b / p2b graph     （現成）
  terminals [T,2]     （現成）
輸出（新）:
  拓樸: 每步預測「下一個放哪個 block、接到哪個 parent 的 L/R child」
        （autoregressive），或 sequence-pair 的兩個排列
  + soft block 的 (w,h) 比例
loss（取代現有 smooth-L1）:
  L = 拓樸的 teacher-forcing 交叉熵（對 tree_sol）
    + λ_shape · soft block aspect ratio 的回歸
  ※ 不再對絕對座標做 MSE → 避開 mode collapse
評估:
  decode → packer → 算真實 contest cost（不是看 loss 數字）
```

### 6.2 階段 1：獎勵微調

- reward = 負的 contest Cost（feasible）；infeasible 給大負值。
- 起點 = 階段 0 的權重。用低 lr 微調（參 `train.py --init-from` 既有機制）。
- 具體演算法待 §5.1 探索（REINFORCE / GRPO / decision transformer / best-of-N）。

### 6.3 階段 2：推論時採樣 + 精修後端

- 從階段 1 模型採樣 K 個拓樸（K 隨 n 放大，大 case 多採）。
- 每個 → packer → T4 的幾何精修 + legalize → 真實 cost。
- 取 best feasible。可選：再丟給 T5 的低溫 SA polish。

### 6.4 待解問題（被 §5.1 探索阻塞，研究完把答案填這）

- [ ] 拓樸表示法最終選擇：______
- [ ] tree_sol 的實際編碼格式（待 T2 印出後確認）：______
- [ ] 獎勵微調演算法：______
- [ ] legalization 後端選擇：______

---

## 7. 需要先確認的事實（地基，別建在沙上）

1. **評分加權公式**：到底是不是 $\sum e^n \text{Cost}$？normalized？v9 還是 v10？
   （[train.py](ml/train.py) 註解出現「v10 contest」，但 [CLAUDE.md](CLAUDE.md)
   寫 v9。釐清以哪份 spec 為準。）
2. **時間預算 / RuntimeFactor 規則**在最新 spec 是否與 [ALGORITHM_GUIDE.md §3.1](ALGORITHM_GUIDE.md)
   一致（`8+1.0*n`、$RT^{0.3}$ 下限 0.7）。
3. **FloorSet-Lite vs FloorSet-Prime**：比賽 C 是只考 Lite，還是兩者都要？
   （影響是否要處理 rectilinear（非矩形）partition。）

> 這三點任何一點變動都會改架構，**研究的第一步就是確認它們**。

---

## 附錄：檔案對照（這套策略會動到哪）

| 階段/任務 | 檔案 | 動作 |
|---|---|---|
| T2 解鎖資料 | [ml/data.py](ml/data.py) | 啟用 tree_sol（302 行） |
| T3 拓樸工具 | `ml/topology.py`（新） | tree_sol ↔ B*-tree decode/encode + 測試 |
| 階段 0 模型 | [ml/model.py](ml/model.py) | 改成生成式輸出頭（拓樸 + aspect ratio） |
| 階段 0 訓練 | [ml/train.py](ml/train.py) | 換 loss（交叉熵），保留 size-power |
| 階段 1 微調 | `ml/finetune.py`（新） | 真實 Cost 獎勵微調 |
| 階段 2 推論 | [ml/predict.py](ml/predict.py) | 採樣 K 拓樸（非單一 argmax） |
| T4 幾何後端 | [electro_submission/](electro_submission/) → `refine_backend.py`（新） | 抽共用精修+legalize |
| T5 低溫精修 | [src/sa.cpp](src/sa.cpp) | 加 `--refine-from-init` 路徑 |
| 整合 | [my_optimizer_ml.py](my_optimizer_ml.py) | 串生成式 pipeline |

---

**一句話**：能奪冠的不是「哪個演算法調得好」，而是「誰把那 1M 筆
near-optimal `tree_sol` 當成可超越的示範來用」。你的程式碼現在第 302 行正把
它丟掉——撿起來，就是別人結構上沒有的東西。
