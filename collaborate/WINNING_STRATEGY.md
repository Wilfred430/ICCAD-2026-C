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
   的拓樸空間裡是盲人。而總分被 $e^{n/12}$ 加權（2026-07-01 已對照 spec 原文
   確認，見 §7），大 case 仍決定性地重要，但沒有早期版本以為的那麼極端。
2. **現有 `ml/` 的 formulation 是壞的**：用 smooth-L1/MSE 回歸座標 →
   多峰解的平均 = 無效解（mode collapse）；而且**完全沒用到資料裡的
   `tree_sol`**（near-optimal 拓樸）。見 [ml/data.py:302](ml/data.py)。
3. **能奪冠的方法**：把那 1M 筆 `tree_sol` 當成「**可超越的示範**」，
   訓練**生成式拓樸模型**（採樣，不回歸），再用**真實 contest Cost
   做獎勵微調**超越 baseline（讓 Cost < 1），最後**推論時採樣多個拓樸 +
   連續幾何精修 + 合法化**，算力全砸在大 case。
4. **兩條腿走路**：電靜力法（目前最佳 2.966）當保底 backbone；
   生成式拓樸當衝頂引擎；兩者共用同一個「pack → 幾何精修 → legalize」後端。

> [!info] **2026-07-14 現況更新（上面 4 點是 6/30 寫的，部分數字已過時，保留原文
> 不動，這段補充現況）**：
> - **生成式拓樸這條腿已經走到頭**：Total Score 13.77→3.3185，已確認到達
>   B\*-tree/contour 表示法本身的結構性密度天花板（跟本文件當初設想的「獎勵
>   微調 + 推論採樣衝頂」不同，實測發現post-hoc/模型微調的投報比很低，真正的
>   瓶頸是離散打包表示法本身，不是拓樸生成品質）。詳見
>   `Obsidian/ICCAD_code/6_ML_Generative_BTree.md` §6.6–6.16。
> - **電靜力法這條腿分數已經不是 2.966**：原始基準（Neutral RT）2.9007，
>   跟 Antigravity（Gemini 3.5 Flash）合作優化後目前約 2.47–2.53（Neutral RT，
>   −13%~−15%），MIB 違規已歸零，是目前分數最好的路線。**這條線的程式碼在
>   `collaborate/electro_optimized/`，且截至本次更新仍在被即時修改中**，
>   完整過程見 `Obsidian/ICCAD_code/8_Winning_Strategy_and_Roadmap.md`
>   §8.7–8.18。
> - **當初設想的「兩條腿共用同一個 pack→精修→legalize 後端」沒有實現**：
>   兩條線最終各自獨立發展（B\*-tree/contour vs 電靜力連續佈局），沒有合併成
>   單一後端——這點跟本文件最初的規劃不同，是否要重新評估「兩線整合」是
>   還沒決定的策略問題，需要跟 pop 討論分工。

---

## 1. 為什麼現在的方法贏不了（嚴謹診斷）

### 1.1 目標函數拆解：分數到底在哪裡

$$\text{Cost} = \underbrace{(1 + 0.5(\text{HPWL\_gap} + \text{Area\_gap}))}_{\text{可正可負}} \cdot \underbrace{e^{2 V_{rel}}}_{[1,\,7.39]} \cdot \underbrace{\max(0.7, RT^{0.3})}_{\geq 0.7}$$

三個推論，每個都改變策略：

- **推論 1（2026-07-08 再訂正）：Q 被 clamp 在 ≥1，贏過 baseline 完全沒獎勵。**
  官方 `iccad2026_evaluate.py::compute_cost` 第 322 行是
  `quality = 1 + 0.5·(max(0, hpwl_gap) + max(0, area_gap))`——**每個 gap 都被
  `max(0, ·)` 夾住**。這代表：(a) 就算你的解比 baseline 還好（gap 為負），
  也被夾成 0，Q 一律 ≥1，贏 baseline 一點分都不加；(b) 你只能「追平」
  baseline（把 gap 壓到 ≤0 就飽和），不可能靠品質把 Q 壓到 1 以下。
  加上 `P≥1`、`R≥0.7`，**feasible 的理論最低 Cost 就是 1×1×0.7 = 0.7，
  這是真地板**。此前本文件（及 FABLE_BRIEF）誤寫「Cost 可以 <1 / 贏過
  baseline 讓 gap 為負」——那是漏看了 clamp，已訂正。baseline 本身是資料集
  自帶的 ground-truth 解（`_extract_baseline()` 直接讀 label，已用
  `metrics[0]/[6]/[7]` 對帳確認）。**戰場結論不變甚至更清楚：品質只需
  追平 baseline（Q→1），勝負全在 $V_{rel}=0$ 與壓低 runtime。**
- **推論 2：$e^{2V_{rel}}$ 是乘法地獄。** 任一 soft 違規（grouping/boundary/MIB）
  指數放大。$V_{rel}=0$ 是生死線，不是加分。約束處理必須是**架構級保證**。
- **推論 3：大 case 決定一切，但沒有原先想的那麼極端。** 分數按
  $e^{n/12}$ 加權（2026-07-01 已對照 spec 原文 Total Score 公式確認，非
  純 $e^n$），n=120 比 n=21 重 $e^{(120-21)/12} = e^{8.25}\approx 3820$ 倍。
  大 case 仍應優先，但中段 case（n≈60–90）也有實質分數貢獻，不能完全
  放棄。
  > ✅ **已驗證**（原 §7 的待確認項目，見該節更新）。舊版本此處誤寫
  > $e^{99}\approx 10^{42}$，已訂正——那是拿本地 `iccad2026_evaluate.py`
  > 裡跟 spec 不符的 `compute_total_score()`(用了純 `e^n`)反推出來的
  > 錯誤結論，spec 原文才是權威。

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

### 5.2 Claude 的部分 —— 行動清單（2026-07-01 全面更新，整合 Fable 驗證後建議）

**✅ 已完成**（原 T1-T3 範疇，已超前完成並跑出真實數字）：
- [x] `tree_sol` 已解密並接進 `ml/data.py`（direction bit 語義對照
  `src/packer.cpp` 確認，round-trip 對 `fp_sol` 命中 20–77%，殘差是官方
  後製壓縮，不影響拓樸標籤本身可用）。
- [x] 生成式 B\*-tree 模型建置完成：`ml/model_tree.py`（三個 Pointer
  Network：block-selection / parent-pointer / direction）、
  `ml/train_tree.py`、`ml/pack_tree.py`（含不合法結構的確定性修復）、
  `ml/contest_cost.py`（真實 contest cost 公式移植）、
  `ml/run_pipeline.py`（一條龍 train→sample→pack→score→存檔）。
- [x] GPU 150k 筆訓練：`val_ptr_acc` 87.4%，16 樣本全 feasible，
  best Cost≈5.35（診斷：佔位方形 shape + repair pass 未接）。
- [x] `collaborate/` 修復 submodule 指標問題並推上 GitHub（Wilfred430/ICCAD-2026-C）。
- [x] Fable 計畫書驗證：`e^(n/12)`（非 e^n）、baseline=ground truth、
  RuntimeFactor=跨隊伍逐 case median（非固定參考解、非自己比）——已對照
  spec PDF 原文確認，見 §7。

**🔥 第一優先（高槓桿，現在就做，直接對應 Fable 診斷）**：
- [x] **T6｜soft block shape 優化取代佔位方形**——**已完成並實測**
  （2026-07-08）。用全域 aspect ratio 掃描（含正方形選項，保證不會變差）
  找每個 case 最省 Cost 的長寬比。100-case 結果見 T7。
- [x] **T7｜接上四道 packer repair pass**——**已完成並實測**
  （2026-07-08~09）。`ml/pack_tree.py` 補齊 `bbox_balance_pass` /
  `holes_fill_pass` / `grouping_repair_pass` / `boundary_repair_pass`
  （照抄 `src/packer.cpp`，見 [[ICCAD_code/6_ML_Generative_BTree|6.6 節]]
  完整數字）。**100-case 驗證（全部四道通道）：Total Score（e^(n/12) 加權）
  從只有 `compact_left_down` 的 13.77 降到 5.13，降幅 62.7%**——純粹是把
  C++ 那邊本來就有的修復通道移植過去，沒動任何模型權重。`grouping_repair`/
  `boundary_repair` 讓 area_gap 從 +25% 漲回 +63%（拉去貼群組/邊界重新打開
  一些空隙）但 Cost 仍大降，因為 $\exp(2V_{rel})$ 的指數懲罰降幅更大。這也
  推翻了上一輪「contour 打包有結構性密度天花板」的悲觀結論——缺的只是修復
  管線沒補齊。
- [x] **T8｜三軟約束 by-construction 實作**（讓約束「建構即滿足」而非
  先擺再修）——**2026-07-09 完成 grouping 主線，boundary 部分完成**：
  - ✅ **MIB → 共享形狀變數**：已完成，`eval_full.py::dims_with_aspect`
    強制 MIB 群組 soft 成員跟隨群組 fixed 成員的形狀，$V_{mib}\equiv0$
    by construction（100-case 實測歸零）。
  - ✅ **Grouping → super-block 收縮**：**已完成**（`pack_tree.py::
    _collapse_clusters`/`_shelf_pack`）。打包**前**把整個 group 用
    next-fit-decreasing-height 貨架式打包收成一個剛性 super-block（貨架
    排序保證連通，$V_{group}\equiv0$ by construction），塞進 B*-tree DFS
    當一個節點，事後展開回個別座標。單獨貢獻 Total Score −3.8%
    （v1+分組=3.400，打敗了 v2 模型微調的 3.439——質變比模型量變更有效）。
    除錯過程抓到三個微妙的硬約束 bug，全部修復並驗證（見
    [[ICCAD_code/6_ML_Generative_BTree|6.12 節]]）。
  - **✅/⏳ Boundary**：LEFT(1)/BOTTOM(8)/左下角(9) 已擴展支援
    by-construction（這幾種代碼只看位置不看寬高，跟放大後的 anchor bbox
    天然相容），但驗證後**貢獻趨近於零**——因為這幾種代碼在舊 post-hoc
    通道裡本來就「保證可滿足」，沒有多餘空間可贏。**RIGHT(2)/TOP(4) 仍
    未攻下**：post-hoc 目前用強力修復（沿牆掃描 + push_past 推出界外）
    把違規壓到 ~12，area_gap 代價高但**實測確認是正確定價**（$\exp(2V_{rel})$
    指數項威力大於面積損失）。要同時拿到低 $V_{rel}$ 又低面積，理論上仍
    需要 RIGHT/TOP 的 by-construction（要解「anchor 遠端要對齊整包 bbox
    遠端」的座標平移子問題），但工程成本高、報酬不確定，尚未著手。

> [!結論] **2026-07-09 最終定案：13.77 → 3.3185（−75.9%），100/100
> feasible，經三輪獨立驗證數字穩定**（見 [[ICCAD_code/6_ML_Generative_BTree|6.10–6.15 節]]）。
> 連續兩個新招式（保約束壓實、LEFT/BOTTOM/BL 邊界擴展）貢獻都趨近於零，
> 兩次獨立訊號指向同一結論：**這條路線容易拿的分已經拿完**。跟 pop 電靜力法
> 2.84 的差距從 session 開始的 4.85 倍縮小到 **1.17 倍**。下一步是策略決策：
> 投入 RIGHT/TOP 邊界 by-construction 這個不確定報酬的工程，還是把時間用在
> 跟 pop 討論兩條線（生成式拓樸提案 vs 電靜力法後端）分工/整合。

**30 天：保底骨架**
- [ ] **T4｜把幾何精修後端獨立出來**（兩條腿共用）：從
  [electro_submission/](electro_submission/) 抽出「固定拓樸下的連續
  (w,h,x,y) 精修 + legalize」成獨立模組。
- [ ] **T5｜給 SA 加低溫 refine 模式**：`src/sa.cpp` 加
  `--refine-from-init`，跳過 Stage-1 高溫直接低溫 exploitation。
- [ ] 在 100 筆 validation 上驗證 T8 的 by-construction 版本：**目標
  feasible 率 100%、`V_rel=0`**，逐 size 記錄。

**60 天：速度與品質收斂**
- [ ] 從 1M 訓練集萃取 block-pair 相鄰先驗 + soft block 長寬比分佈，
  當 SA warm-start / move 偏置，縮短收斂時間。
- [ ] **runtime 策略更正**：RuntimeFactor 目標不可知（跨隊伍 median，
  移動標靶，見 §7），**不要瞄準某個精確數字**，改成「盡量壓低每個
  case 的絕對耗時」+ 硬性 wall-clock 早停。
- [ ] Fable 方案 B：生成模型出拓樸 candidate（top-k）→ 餵進 T8 的
  by-construction 後端 + T6 的 shape 優化 + T7 的 repair，取代單純
  sampling 後才 repair 的現有流程。

**90 天：微調與加固**
- [ ] Stage 1 獎勵微調（reward = −contest Cost，見 §6.2），起點用 T6-T8
  跑順後的模型。
- [ ] n=90–120 全面壓力測試，建立 fallback（自估 Cost 超門檻 → 退回
  純 SA / 電靜力法）。
- [ ] Approach A（SA）vs C（電靜力）正式比較，決定最終送出哪個或如何
  組合（取逐 case 較優者）。

**❓ 仍待你確認（不影響上面的工程排程，可以並行）**：
- FloorSet-Lite vs Prime 範圍（見 §7 第 3 點）——是否要處理 rectilinear
  partition，若確認只考 Lite 則現有架構不用改。

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

1. ✅ **評分加權公式**——**已解決（2026-07-01）**。使用者直接提供 spec 原文
   截圖：`Total Score = Σ Cost[i]·e^(n_i/12) / Σ e^(n_j/12)`。**不是**
   $\sum e^n \text{Cost}$（本地 `iccad2026_evaluate.py::compute_total_score()`
   用 `math.exp(n - max_n)`，即純 $e^n$，跟 spec 不符——那支腳本本身有
   bug/簡化，不能當權威，spec PDF 才是）。n=120 比 n=21 重約
   $e^{8.25}\approx 3820$ 倍。全文所有 $e^n$/$e^{99}$ 已訂正為
   $e^{n/12}$/$\approx 3820$倍，見 §0、§1.1。
2. ✅ **RuntimeFactor 規則**——**已解決（2026-07-01）**。Spec 原文：
   `RuntimeFactor = Your Runtime / Median Runtime of All Submissions`，
   footnote 補充：「computed **independently for each test design**,
   using that **individual test case's** median runtime as the sole
   reference point」——這是**跨隊伍、逐 test case 各自獨立計算的中位數**，
   不是「跟自己的 100 個 case 比」，也**無法在本地離線精準得知或瞄準**
   （取決於所有參賽者當下的表現，是個移動標靶）。
   `iccad2026_evaluate.py` 本地 `--evaluate` 模式因為拿不到其他隊伍的
   runtime 資料，退而求其次用「自己 100 個 case 的中位數」當本地近似
   替代品——這只是離線練習用的代理指標，**不是真正的評分機制**，不要
   誤把本地算出的 RT 數字當成正式排名會用的值。實務結論：**沒有能精準
   瞄準的數字目標，唯一能做的是盡量壓低每個 case 的絕對耗時。**
3. **FloorSet-Lite vs FloorSet-Prime**：比賽 C 是只考 Lite，還是兩者都要？
   （影響是否要處理 rectilinear（非矩形）partition。）**尚待確認。**

> 第 3 點仍待確認會改架構；前兩點已用 spec 原文截圖 + 原始碼交叉驗證定案。

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
