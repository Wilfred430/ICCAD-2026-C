# 下一步路線圖：分數再降、時間再壓（2026-07-07）

本文件整理 electro pipeline 的下一步方向。所有建議都建立在**本 repo 已實測驗證的事實**上，
並對照近年 EDA / 演算法 / AI 文獻（連結見文末）。閱讀前提：`EXPERIMENTS.md`（技術嘗試全記錄）、
`CLAUDE.md`（注意其中計分描述是 v9，已過時）。

---

## 0. 現況與已確立的事實

**現況**（真評測器、full-100）：

| 指標 | 數值 |
|---|---|
| Total Score（v10 加權） | **2.8414**（compaction ON，commit `0c665b0`）；OFF = 2.9660 |
| feasibility | 100/100 |
| 每 case 運算時間 | ~5–9 s（n=21→120，seeds=1，CPU） |

**v10 計分要點**（決定所有優先序，勿用 v9 直覺）：
- `quality = 1 + 0.5·(max(0,hpwl_gap) + max(0,area_gap))` — **贏過 baseline 不加分**（clamp）。
- `× exp(2·V_rel)`（boundary/grouping/MIB 軟違規）`× max(0.7, RT^0.3)`。
- `RT = 我方時間 / 全體參賽者該 case 的中位數`：**快最多賺 30%（封頂），慢無上限被罰**。
- Total 以 `e^{n/12}` 加權：n=120 約 34%，n<111 合計仍約 28% — 小 case 不可棄。

**本輪 sprint 實測確立（都有腳本可重跑）**：
1. 排列（相對位置）已達 GT 的 0.74–0.88 一致度（`diag_ml_vs_gt.py`）→ **缺的不是擺哪，是密度**。
2. **area_gap 是主要失血**（util ~0.55–0.6 vs GT ~0.965）；legalizer 只微調（位移 ≤1.2%）。
3. 迴圈內軟塑形（`la` 梯度共優化）**是承重牆但已調滿**：關掉變差（0.63→0.93），放寬 AR_CAP 4→16 也變差（`validate_familyB.sh`）。
4. C++ B\*-tree repack **已否決**：它比 electro 更鬆（area_gap 0.90 vs 0.61）、cost 全輸（`validate_A.sh`）。
5. **壓縮+塑形 pass 有效且嚴格加法**（`shape_compact.py`，額外候選 + cost-aware 排名）：subset 2/6 case 大贏、0 退步；但 **4/6 被排名拒絕**，原因是壓縮破壞 grouping/boundary（V_rel 升）→ 這是下一個最大槓桿（見 S1）。
6. ML 座標回歸有 mode-averaging 天花板（輸出塌成一坨）；僅在 hard-basin case 以 multi-start jitter 形式有窄價值。

**已否決路線**（勿重試，理由見 EXPERIMENTS.md / memory）：
迭代式 RL（GoodFloorplan、HyperGCN+DRQN — 優化被 clamp 的 HPWL、不處理約束群）、
diffusion placer（分鐘級 → RT 罰爆）、C++ repack、AR_CAP>4、單獨的座標回歸 ML。

---

## 1. 分數改善（依投報率排序）

### S1 — 群組/邊界感知壓縮（最高優先，直接接續現有程式）

**問題**：現在的壓縮是「約束盲」的 —— tid40 強制壓縮可把 area_gap 0.96→0.62，但把 cluster
成員拉散、boundary block 拉離牆邊（V_rel 0.24→0.43），`exp(2·V_rel)` 吃掉全部收益，
排名只好拒絕。6 case 中 4 個因此拿不到壓縮紅利。

**做法**（`shape_compact.py` 內改，~100 行）：
- **cluster 當剛體**：同 cluster 的成員合成一個 super-block 一起壓（相對位置不變）→ V_grouping 不可能變差。這是 floorplanning 處理 alignment/abutment 約束的標準做法（constraint-graph 中把群組視為單節點；JigsawPlanner 同精神）。
- **boundary block 釘牆**：left-boundary block 只允許沿 y 壓、x 保持貼 x_min；top-boundary 反向同理。壓縮後 bbox 縮小時，重新 snap 到新邊界。
- MIB 已保護（不塑形），維持。

**預期**：救回被拒的 4/6 型 case。tid40 若 V_rel 保住，cost 2.74→~2.0。full-100 估 2.84 → **~2.6–2.7**。

### S2 — SDS 完整版 slack 分配（升級塑形品質）

現在的塑形是貪婪的 fill-right/fill-up。[SDS（ISPD'12/TCAD'13）](https://dl.acm.org/doi/10.1145/2160916.2160956)
給的是**最佳**解法：固定拓樸與外框寬上界，全域分配 slack、只塑形 critical path 上的 soft block，
高度單調下降、收斂到最優。另一個模式值得試：**以 pin-bbox 寬度為固定 W₀**（GT die 外框的強代理），
最小化 H —— 比自由左下壓更貼近 GT 外形。

### S3 — place ↔ compact 迭代迴圈

把壓縮後的緊layout **回灌**當 `place()` 的 init（`init_centers`），target_util 調高再跑短一輪
（200–300 iters）→ 再壓縮。解析式全域重新優化 HPWL/V 項、壓縮收密度，2–3 輪。
這是 multilevel/multi-stage placement（mPL、ePlace 系）的標準策略，且所有零件都已存在。

### S4 — 小 case 精確/組合式打包（n ≤ 40）

文獻明確：[B&B 窮舉在 <30 個矩形有效](https://www.sciencedirect.com/science/article/abs/pii/S0305054806001985)；
[soft rectangle packing 有精確/近似演算法](https://www.researchgate.net/publication/264352300_Exact_and_approximation_algorithms_for_a_soft_rectangle_packing_problem)（面積固定、長寬比連續 —— 正是我們的 soft block）。
v10 下小 case 合計 ~28% 權重，而我們小 case cost 仍在 1.87–2.0。
做法：n≤30–40 時另跑一個 skyline/BLF+B&B 或 soft-packing 候選，與 electro 候選同台由現有排名選優
（**portfolio racing**，嚴格加法，零風險）。目標把小 case 壓向 ~1.2–1.5。

### S5 — 收尾連續精修（quasi-Newton / 直接優化真 cost 代理）

拓樸定案後，對 (x, y, w, h) 做一次連續精修（L-BFGS 或投影梯度），目標函數直接用
`0.5·area_gap_proxy + hpwl_proxy + 2·V_rel_proxy`（可微版），~150 行（CLAUDE.md 原 roadmap 第 3 點，Ji 2021）。
放在壓縮之後，把「幾乎貼齊」的邊精確貼齊。

### S6 — 逐 n 段參數掃描

`TARGET_UTIL / EXT_WL / ITERS / COMPACT_AR` 依 n ∈ {21–40, 41–70, 71–100, 101–120} 分段調
（現在全域一組）。用 `score_subset.sh` 紀律：subset 篩選 → full-100 確認。

### S7 — ML（只做有據的）

- 保留現況：ML init 僅配 multi-start（hard-basin 救援）。
- 若重訓：改用 **不變量表示**（[MacroDiff](https://ieeexplore.ieee.org/iel8/11132383/11132091/11132593.pdf) 的
  「擴散線長關係而非絕對座標」思想）避開 mode-averaging；或 [chipdiffusion（ICML'25）](https://arxiv.org/abs/2407.12282) 的合成資料預訓練。
- **不要**再投資座標回歸；**不要**上 diffusion 全流程（分鐘級 → RT 罰）。

---

## 2. 運算時間（依投報率排序）

**策略框架**：快的收益封頂在 30%（RT ≤ 0.3 倍中位數即滿），慢的懲罰無上限。
我們無法看到對手中位數 → 合理目標是**穩壓在估計中位數之下**（例如 2–3 s/case），
而不是無限追快。**絕不可用 area_gap 換時間**（0.7 地板讓極端快沒有額外回報）。

### T1 — 早停（最便宜，先做）

`place()` 現在固定 600 iters。加 loss-plateau 偵測（例如連續 50 iters 相對改善 < 1e-4 → 停）。
小 case 遠早於 600 收斂 → 估計小 case 省 30–50% 時間、零品質損失。~15 行。

### T2 — iters 隨 n 排程

`ITERS = f(n)`（例如 300 + 2.5·n），配合 T1 當保險。與 S6 一起掃。

### T3 — 降低 per-iter 開銷（最大單一槓桿）

n≤120 的張量極小，600 iters 的時間主要是 **Python/dispatch 開銷**，不是運算。選項：
- **`torch.compile(mode="reduce-overhead")`**：官方文件明示[小張量、launch-overhead 主導時收益最大](https://docs.nvidia.com/dl-cuda-graph/torch-cuda-graph/best-practices.html)；
  關鍵是 **static shape** —— 把所有 case pad 到固定 N=128，一次編譯、100 case 重用（首次編譯成本攤提掉）。
  這正是 [DREAMPlace](https://research.nvidia.com/publication/2019-06_dreamplace-deep-learning-toolkit-enabled-gpu-acceleration-modern-vlsi-placement)
  「placement=訓練一個網路」的思路（GPU 上 40× vs RePlAce；我們是 CPU 小問題，合理預期 2–5×）。
  注意 requirements.txt 已含 torch，**零新依賴**。
- 退路：把迴圈熱路徑改 numpy 手寫梯度（消 autograd 開銷），或融合現有逐項 loss 成單一 kernel 式表達。

### T4 — 壓縮 pass 向量化

`shape_compact.py` 目前是 O(n²) Python 迴圈（+8% 牆鐘時間）。numpy broadcasting 改寫可到 <1%。
S1 動這個檔時順手做。

### T5 — GPU seed-batching（條件性）

只有在 multi-start 回歸主力（例如 runtime 便宜、要衝品質）時才值得：把 N 個 seed 疊成 batch 維
一次算（DREAMPlace 式）。目前 seeds=1，先擱置。

### T6 — 先 profile 再動手

動 T3 前先量：600-iter 迴圈 / legalize / repair / 壓縮 各占多少。避免優化錯段（我們在 eDensity
時踩過「優化非瓶頸段」的坑）。

---

## 3. 建議執行順序（兩週節奏）

| 週 | 項目 | 驗收 |
|---|---|---|
| W1 | **S1** 群組/邊界感知壓縮 | subset 被拒 case 轉為採納；full-100 < 2.7 |
| W1 | **T1+T2** 早停 + iters 排程 | 平均 case 時間 −30%，full-100 分數不退 |
| W2 | **S3** place↔compact 迭代（或 S2 SDS 完整版，二選一先做） | full-100 再 −0.1 以上 |
| W2 | **T6→T3** profile + torch.compile 實驗 | 平均 case < 3 s |
| 之後 | S4 小 case portfolio、S5 收尾精修、S6 分段掃描 | 逐項 A/B |

紀律不變：每項先 subset（15 case）廉價驗證 → full-100 確認 → 記入 `EXPERIMENTS.md` → commit。
所有新 pass 一律做成**額外候選**（cost-aware 排名擇優），維持嚴格加法、100/100 feasible。

---

## 4. 文獻對照表

| 主題 | 文獻 | 我們的用法 |
|---|---|---|
| Slack 塑形 | [SDS, ISPD'12](https://dl.acm.org/doi/10.1145/2160916.2160956) / [TCAD'13](https://ieeexplore.ieee.org/document/6416107/) | S2：最優塑形取代貪婪 fill |
| 白縫消除 | [JigsawPlanner, ICCAD'24](https://dl.acm.org/db/conf/iccad/iccad2024.html) | S1：約束感知壓縮的參考 |
| 迴圈內塑形 | [ICCAD'23 靜電法長寬比](https://dl.acm.org/doi/10.1145/3676536.3676818)（同會議系列）/ [PeF, TCAD'22](https://arxiv.org/pdf/2210.03293) | 已實作且已調滿（家族二驗證） |
| 解析式引擎 | [ePlace](https://cseweb.ucsd.edu/~jlu/papers/eplace-dac14/paper.pdf) / [ePlace-MS](https://cseweb.ucsd.edu/~jlu/papers/eplace-ms-tcad14/paper.pdf) / RePlAce | 現有引擎地基 |
| GPU/開銷 | [DREAMPlace](https://research.nvidia.com/publication/2019-06_dreamplace-deep-learning-toolkit-enabled-gpu-acceleration-modern-vlsi-placement) / [torch.compile+CUDA graphs](https://docs.nvidia.com/dl-cuda-graph/torch-cuda-graph/best-practices.html) | T3/T5 |
| 精確打包 | [B&B strip packing](https://www.sciencedirect.com/science/article/abs/pii/S0305054806001985) / [soft rectangle exact](https://www.researchgate.net/publication/264352300_Exact_and_approximation_algorithms_for_a_soft_rectangle_packing_problem) | S4 小 case portfolio |
| 生成式（僅取想法） | [MacroDiff LBR](https://ieeexplore.ieee.org/iel8/11132383/11132091/11132593.pdf) / [chipdiffusion, ICML'25](https://arxiv.org/abs/2407.12282) | S7 不變量表示；全流程已否決 |
| 建構式（僅取想法） | [MdpoPlanner, ASP-DAC'26](https://arxiv.org/abs/2510.15897) 系 / MaskPlace | position/wire-mask 思想（若日後做建構式候選） |
