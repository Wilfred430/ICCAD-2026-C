# 研究請求：如何把 ICCAD 2026 FloorSet-Lite 的 Cost 逼近理論下限 0.7

## 背景與目標

我們在做 ICCAD 2026 CAD Contest Problem C（FloorSet-Lite，VLSI 平面規劃）。評分公式（v10，
2026-06-03 版，已用真評測器原始碼逐項核對，不是憑印象）：

```
Cost = (1 + 0.5·(max(0, HPWL_gap) + max(0, Area_gap))) · exp(2·V_rel) · max(0.7, RT^0.3)
       若不可行（重疊/超出 1% 面積誤差/固定形狀或 preplaced 位置偏移），Cost = 10
```

- `HPWL_gap`、`Area_gap` 是相對 baseline（資料集本身的近似最優解）的訊號 gap，**已被 clamp 在
  0 以下**——贏過 baseline 不加分，所以 gap 只要壓到 ~0 就好，不必也不能靠「贏過 baseline」拿更多分。
- `V_rel ∈ [0,1]` 是 boundary / grouping / MIB 三種軟約束違規的加權平均，=0 時 `exp(2·V_rel)=1`。
- `RT = 我方 runtime / 全場該 case 的中位數 runtime`，**只有變快才封頂在 30%（RT^0.3 下限 0.7），
  變慢沒有上限**。
- Total Score = 對 100 個驗證 case（n=21~120）以 `e^(n/12)` 加權平均，**大 case 權重極高**
  （n=120 是 n=21 的 ~3820 倍）。

**理論下限 Cost=0.7 的條件**：`feasible=1` 且 `Area_gap≈0` 且 `HPWL_gap≈0`（即 quality→1）
且 `V_rel=0`（`exp(2·V_rel)=1`）且 `RT` 快到讓 `RT^0.3` 撞到 0.7 這個地板。**我們要問的是：
有沒有具體、可落地的技術路徑能同時達成這四個條件，尤其是把 Area_gap 壓到接近 0（也就是
密度逼近資料集 GT 的 utilization ~0.965）同時維持 V_rel=0 和低 runtime。**

---

## 目前的兩條技術路線與已驗證的具體數字

### 路線 A：離散拓樸（B\*-tree）+ 確定性 contour packer

用 Transformer 自回歸模型，在 1M 訓練集的 `tree_sol`（GT 的 B\*-tree 邊表）上做監督模仿訓練，
生成 B\*-tree 拓樸，再用 contour-based packer（left/bottom 貼齊規則）打包成座標。

**目前最佳 100-case 驗證：Total Score 3.3185**，100% feasible，平均 area_gap +124%
（也就是 util 只有 GT 的 ~45%）。

**⚠️ 關鍵的已證實負面結果**：我們把 **GT（真實最優解，不是 ML 預測）反推成 B\*-tree**
（貪婪最近槽位抽取，best-of-3 插入序，跨 11 個 case、n=21–120），再用標準 contour packer
重建——即「拓樸預測 100% 準確」的品質**上限**：

| 指標（重建/GT） | 平均 | 範圍 |
|---|---|---|
| 面積比 | **1.403**（= area_gap ~0.40） | 1.12–1.69 |
| HPWL 比 | 1.212 | 1.04–1.65 |
| 排列一致度 | 0.93（保住了） | 0.87–0.98 |
| 重疊 | 0.00%（合法性確實免費） | — |
| 再接後製壓縮/塑形 | 1.282（area_gap ~0.28） | 1.06–1.54 |

**判決：合法性由 B\*-tree/contour 保證是「免費」的，但緊密度不是**——GT 是互相咬合的緊密
拼磚，不是 left/bottom contour packing 能表示的可行解子集。**這代表：只要還是用 B\*-tree +
contour packer 這個表示法，不管拓樸預測得多準，area_gap 都有一個結構性下限（~0.28–0.40），
達不到 Cost≈0.7 需要的 area_gap≈0。** 這個路線已經被我們自己的實驗判死刑，不需要再研究
如何在這個表示法內部優化——**除非能找到一種不受 left/bottom contour 規則限制、但仍保留
「合法性 by construction」優點的打包/表示方式**（見下方「特別想知道的問題」）。

### 路線 C：解析式/電靜力連續佈局（ePlace/DREAMPlace 風格）+ legalize + 修復

連續座標 + 梯度下降（Adam）優化 wirelength + 密度散開 + 約束項，legalize 到零重疊，再接
grouping/boundary 修復。

**目前最佳 100-case 驗證：Total Score 2.7215～2.8414**，100% feasible，**平均 runtime
僅 1.91–2 秒/case**（CPU，單 seed）。這是目前兩條線裡分數最好、速度也最快的。

**現況的主要失血是 area_gap**（util ~0.55–0.60 vs GT ~0.965），這是目前離 Cost=0.7 最近
的路線，但還有明顯差距。

---

## 已經嘗試過、有效並已採用的技巧（不需要重新提出）

- **module-area-growing**：soft block 在優化迴圈內逐步放大到目標面積（area-exact），
  是目前最大單一貢獻的技巧。
- **fixed-outline containment**：把佈局約束在目標外框內做梯度優化。
- **multi-start（多個隨機種子取最佳）**。
- **ML warm-start init + jitter**：用 ML 預測當多起點初始化之一（純預測本身比隨機初始化差，
  是 mode-averaging 塌縮的座標回歸模型，但預測+jitter 當多起點的其中一個有幫助）。
- **external（pin）wirelength 權重調整**：加大外部端點的線長拉力，減少負座標漂移。
- **cluster 剛體壓縮（S1）**：同一 grouping 群組的成員合成一個 super-block 一起壓縮
  （保持相對位置，V_grouping 不會變差），boundary 方塊只允許沿自己貼的牆邊滑動；
  壓縮後的候選跟「不壓縮」「單純壓縮不管約束」同台由 cost 排名擇優（portfolio racing，
  嚴格加法式，不會讓任何 case 變差）。
- **Portfolio / dual-candidate racing 設計模式**：任何新的候選手法都當「額外候選」而非
  取代舊有路徑，由 `exp(2·V_rel)`-aware 的 cost 排名逐 case 擇優——這個設計原則本身很重要，
  請在建議新技巧時比照這個模式（永不讓任何 case 變差）。

---

## 已經嘗試過、明確失敗或被否決的方向（請不要重複建議）

- **平滑/加權平均（WA/LSE）wirelength 模型**：比目前的模型差。
- **Naive Nesterov/SGD 優化器**：比 Adam 差。
- **eDensity FFT 密度場（ePlace/DREAMPlace 式的 Poisson 解，Neumann 邊界）單獨使用**：
  密度場本身正確、佈局密度可以逼近 GT 的 0.85-0.98，**但既有的 legalizer（constraint-graph
  longest-path compaction）不尊重固定外框**——移除重疊時會把方塊推出外框，最終密度崩潰
  比不用 eDensity 還差。**這代表問題不在密度場本身，在缺一個真正尊重 fixed-outline 的
  legalizer**（見下方問題）。
- **Bounded push-apart legalizer 原型**：能保持密度，但不收斂，在低重疊輸入下仍會 deadlock
  在 1-7% 殘留重疊，不可行。
- **強制非負座標（canvas clamp / 牆懲罰）**：過度約束會傷分數；發現負座標本身其實是
  「密度不夠、佈局比目標外框大，溢出到負象限」的症狀，強制拉正只是把溢出換個方向擠出去，
  不解決根本問題。
- **CPU fork-parallelism**：worker 之間搶佔 OpenMP 執行緒，反而更慢。
- **C++ B\*-tree/contour 重新打包**（見上方，已用 GT 完美拓樸實測，area_gap 結構性下限
  ~0.28-0.40）。
- **迭代式 RL**（GoodFloorplan、HyperGCN+DRQN 這類）：優化的是被 clamp 的 HPWL，不太處理
  約束群（grouping/boundary），評估後放棄。
- **完整 diffusion placer 流程**（多步生成整個 pipeline）：分鐘級推論時間 → RT 懲罰爆炸
  （RT 沒有封頂），放棄。
- **放寬長寬比上限（AR_CAP 4→16）**：變差。
- **獨立的座標回歸 ML（逐塊直接回歸 (x,y)）**：多個合法解的平均在數學上是最優回歸解，
  但平均出來的座標通常本身就不合法（mode-averaging 塌縮），這是架構性天花板，加資料/調參
  救不了。

---

## 已規劃但尚未執行或才剛起步的方向（供參考，不代表已經驗證有效）

- **S2 — SDS 完整版最優塑形**（Sequence-pair/soft-block shaping 的最優 slack 分配算法，
  取代現在的貪婪 fill-right/fill-up 塑形）。
- **S3 — place↔compact 迭代迴圈**（壓縮後的緊 layout 回灌當下一輪解析優化的初始化，
  2-3 輪，類似 multilevel placement）。
- **S4 — 小 case（n≤40）精確/組合式打包**（branch & bound 窮舉，soft rectangle packing
  的精確演算法，跟解析法同台 racing）。
- **S5 — 收尾連續精修**（拓樸/佈局定案後，對 (x,y,w,h) 做一次 quasi-Newton 或投影梯度精修，
  直接優化 cost 的可微代理函數）。
- **M1 — 建構式自回歸 placement**（HGNN/Transformer 編碼 netlist，自回歸 decoder 逐塊生成，
  每步輸出「哪一塊、放哪個自由格點、長寬比 bin」，用幾何合法性 mask 保證零重疊，純監督
  模仿訓練，不用 RL；**設計上刻意避免 B\*-tree/contour 的規則限制，用自由 (x,y) 格點+mask
  而非 left/bottom 貼齊規則**，理論上能繞開路線 A 證實的密度天花板）。目前只有設計文件，
  沒有訓練過的可用權重。
- **M2 — few-step 生成式（flow matching + consistency distillation）**：一次生成整張佈局
  的 (cx,cy,長寬比)，用 flow matching 訓練後蒸餾到 1-4 步推論；合法性趨近 0 而非結構性
  保證，legalizer 負擔比 M1 大。

---

## 我們想請你（Gemini 深度研究）具體回答的問題

請針對以下問題做literature-grounded 的深度研究，優先看**近 5 年**的 EDA/placement/生成式
AI 文獻，給出**具體可落地**的技術方案（含論文引用），而不是泛泛的方向建議：

1. **有沒有一種打包/legalize 方法，能像 fixed-outline eDensity electrostatic placement
   一樣達到接近 GT 的高密度（util → 0.965），但又像 B\*-tree/contour packer 一樣結構性
   保證零重疊、不需要迭代式的 legalize-then-check？**（也就是「合法性免費 + 緊密度也免費」
   兩者兼得的打包表示法，不是 M1/M2 那種還需要 mask/guided sampling 但重疊只是趨近 0 的方案）
   特別想知道：Abacus 式逐行 legalization、或近年 fixed-outline soft-module placement
   的精確/近似演算法，是否有能同時保證零重疊又不犧牲密度的版本？

2. **eDensity FFT 密度場配上什麼樣的 legalizer，才能在收斂到高密度後不崩潰？** 我們現在的
   legalizer 是 constraint-graph longest-path compaction，不尊重 fixed outline。近年
   DREAMPlace 系列或其他工作，是否有專門處理「固定外框 + 高密度收斂」的 legalization
   演算法可以參考／移植？

3. **有沒有近期（2023-2026）的建構式/自回歸 floorplanning 生成模型，是在自由連續座標
   （而非離散格點或 B\*-tree token）上做逐步條件化生成，同時保證合法性？** 我們已知的
   ChiPFormer（ICML'23，offline RL policy transfer）、MaskPlace、MdpoPlanner
   （ASP-DAC'26）、LayoutFlow（ECCV'24 flow matching）、DiffPlace（guided sampling
   壓重疊）——除了這些，還有沒有更新、更貼合我們場景（soft block 面積固定+長寬比連續、
   grouping/boundary/MIB 約束群、runtime 要壓在秒級以內）的工作？

4. **在保證 V_rel=0（grouping/boundary/MIB 三種約束都滿足）的前提下逼近 GT 密度，
   有沒有把約束當「剛體/單節點」處理的更進階演算法？**（我們的 S1 已經做了 cluster
   剛體壓縮，但仍有 case 被 cost 排名拒絕，代表壓縮跟約束滿足還是有結構性衝突）
   JigsawPlanner（ICCAD'24）之外還有沒有值得參考的白縫消除/約束感知壓縮工作？

5. **在 runtime 必須維持在 1-3 秒/case（CPU）的預算下**，有沒有比目前的
   `torch.compile(mode="reduce-overhead")` + static shape padding 更有效的推論加速手段，
   能讓一個小型 Transformer/analytical placement 模型在 CPU 上跑到次秒級？

請針對每個問題給出：(a) 具體技術/論文名稱與連結、(b) 為什麼它可能解決我們的問題（對照
上面已驗證失敗的方向，說明差異在哪）、(c) 概略的實作複雜度/工時評估。**不需要**重新推薦
我們已經嘗試過或已經否決的方向（見上面兩個清單）。
