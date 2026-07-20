# Brief for Antigravity: 剩餘空間主要在 HPWL/Area gap，不在 soft violation（2026-07-17）

## 背景

`electro_optimized/` 目前定案在 Neutral RT 2.4081（100/100 feasible，V_mib=0）。
繼續在 soft-repair（boundary_snap / grouping_repair）上微調已經進入報酬遞減：

- 放寬 grouping swap 的 `mib_id[j]!=0` 限制（原理：V_mib 只檢查 (w,h) 形狀，
  跟位置無關，swap 不改變 w/h，所以排除 MIB 方塊當交換對象是過度保守）：
  Total Score 2.4081→**2.4071**（小勝，已在獨立 scratch 目錄驗證兩次決定性一致）。
- 同樣的邏輯套用到 `boundary_snap` 的 wide-swap 反而變差（2.4081→2.4084）——
  不能一概而論，只在 grouping swap 有效。
- 拉高 `boundary_snap` 的疊代上限（`passes` 3→6）完全無變化——wall-scan
  本來就已經在 3 輪內收斂，不是瓶頸。

這三個結果加起來的訊號很一致：**soft-repair 這條線已經接近極限，繼續在這裡挖
邊際效益很低**。

## 重大發現：真正的剩餘空間在 HPWL/Area gap

把 Cost 公式 `(1+0.5·gap) × exp(2·V_rel) × RT^0.3` 拆成兩項，用 `e^(n/12)`
加權平均（跟 Total Score 的加權方式一致）：

| 項目 | 加權平均值 | 若壓到 0，Total Score 會變成 |
|---|---|---|
| `exp(2·V_rel)`（soft violation 項） | 1.3132 | ≈1.84（gap 不變） |
| `(1+0.5·gap)`（HPWL/Area gap 項） | **1.8386** | ≈1.31（V_rel 不變） |

（兩者相乘 ≈2.415，跟實際 Total Score 2.4081 吻合，這個拆解是對的，不是我瞎猜。）

**HPWL/Area gap 項的理論改善空間（−45%）比 soft violation 項（−24%）大將近一
倍**，而且 `analytical_place.py`（真正決定擺放品質的電靜力場優化本身）從稍早
加了 MIB shape guiding loss 之後就沒再被碰過——今晚幾乎所有精力都花在
soft-repair 的後處理上。

**個案佐證**（e^(n/12) 加權下最貴的幾個案例，soft violation 其實都不嚴重）：

| n | V_rel | HPWL_gap | Area_gap | cost |
|---|---|---|---|---|
| 104 | 0.073（幾乎沒違規） | 168% | 285% | 3.777 |
| 23 | 0.148 | 188% | 272% | 4.435 |
| 110 | 0.226 | 177% | 183% | 4.400 |

n=104 這種案例，soft violation 已經幾乎解決了，cost 還是很高，純粹是因為擺放
跟 baseline 差太多——soft-repair 這條路對這種案例已經無能為力。

## 快速探測：`ELECTRO_ITERS` 是個有潛力但不穩定的槓桿

拿 4 個代表案例測 `ELECTRO_ITERS` 600→1500（用
`ml/case_report_electro.py` 的底層 API 單獨跑，未跑滿 100 案）：

| n | iters=600 cost | iters=1500 cost |
|---|---|---|
| 21 | 4.445 | 5.124（變差） |
| 23 | 4.435 | **1.819**（大幅變好，−59%） |
| 27 | 5.541 | 4.730（變好） |
| 110 | 4.400 | 4.852（變差） |

**不是「迭代越多越好」**——n=23 大幅改善，但 n=21、n=110 反而變差。猜測是電
靜力場沒有適當的收斂判斷/學習率衰減，跑久了在某些案例上發散或跑過頭（過度
優化到不好的區域）。

## 建議

**優先度高於任何進一步的 soft-repair 微調：**

1. **把「不同 iters 預算」包成 portfolio 候選**，比照現有 `ELECTRO_SEEDS` 多重
   啟動的架構（`electro_optimizer.py::solve()` 已經有 seed-based 的 proxy
   ranking 機制），逐案挑最好的，而不是直接改全局 `ELECTRO_ITERS` 預設值
   （那樣會讓 n=21、n=110 這類案例變差）。
2. **或者**：檢查電靜力場優化（`analytical_place.py`）有沒有適當的收斂/提早
   停止判斷（例如 loss 平台就停，而不是固定 iteration 數）。如果能讓它
   「該多跑就多跑、該提早停就停」，理論上可以同時吃到 n=23 那種大幅改善，又
   不會讓 n=21/n=110 變差，不需要額外的 portfolio 候選人。
3. 這兩個方向都值得先花時間調查，比繼續在 `boundary_snap`/`grouping_repair`
   上挖邊際效益（已知只剩不到 0.1% 的空間）優先度高很多。

## 順便：小的已驗證改善（可以隨手併入，風險低）

`grouping_repair` 的 zero-overlap swap 目前排除 `mib_id[j]!=0`：

```python
if is_pre[j]:
    continue
if mib_id is not None and int(mib_id[j]) != 0:
    continue          # <- 這行可以刪掉
if clust_id[j] != 0:
    continue
```

刪掉那行（保留 `clust_id[j]!=0` 的排除,因為 grouping 連通性才跟位置有關）。
已在獨立 scratch 目錄驗證兩次，決定性一致：Total Score 2.4081→2.4071，
Vgrp 332→326，100/100 feasible 維持。**注意：同樣的刪除不要套用到
`boundary_snap` 的 wide-swap 那邊**——已測試過，套在那邊反而變差
（2.4081→2.4084），原因可能是 boundary_snap 的 swap 沒有排除 `clust_id[j]`，
放寬 mib_id 後會有更多機會擾動到同時屬於 MIB 又屬於 cluster 的方塊。
