# Brief for Antigravity（跟使用者）：多起點平行搜尋是目前最大的槓桿，但是真正的速度換品質（2026-07-17）

## 背景

掃完既有 loss 權重（`ELECTRO_EXT_WL`、`ELECTRO_TARGET_UTIL`、`ELECTRO_LAM_OUT`、
`ELECTRO_LR`、`ELECTRO_WL_WEIGHT`(新)、`ELECTRO_OV1`、`ELECTRO_BB1`、
`ELECTRO_AREA_GROW`、`ELECTRO_GROW_END`——**共 10 個，全部確認已在局部最佳**）
之後，這條「調權重」的路線正式挖乾。換了一個全新維度：`ELECTRO_SEEDS`
（多起點隨機初始化，目前預設 **1**，代表現在完全沒有用多起點）。

## 發現：多起點是目前找到最大的槓桿，但是真正的 trade-off

在已套用的 `ELECTRO_GRP_WEIGHT=0.4` + `ELECTRO_MIB_SHAPE=0.05` 基礎上（
`ELECTRO_ITERS_PORTFOLIO=0`，先關掉那個確保乾淨比較），搭配
`ELECTRO_PARALLEL=1`（多個起點平行跑在不同 CPU 核心，本機測試環境有 32
核心可用）掃 `ELECTRO_SEEDS`：

| `ELECTRO_SEEDS` | Total Score（Neutral RT） | 平均 runtime/case |
|---|---|---|
| 1（現行預設） | 2.1731 | ~2.3s |
| 2 | 2.0036 | 3.19s（~1.4x） |
| 3 | **1.9653**（決定性一致，兩次重跑相同） | 4.60s（~2x） |
| 5 | 1.9240 | 8.21s（~3.6x） |

**這是目前找到最大的單一改善槓桿**，比 `grp_weight`（−7.4%）、`mib_shape`
（−2.5%）都大很多。但**跟前兩個不同，這次真的是拿 runtime 換品質**，不是
免費的——平行執行有明顯開銷（fork/pool 啟動成本、GPU/CPU 資源競爭），不是
理想中的「多核心幾乎零額外時間」，3 個起點還是讓平均 runtime 從 2.3s 拉到
4.6s（約 2 倍）。

## 效益/成本比（相對 seeds=1 基準）

- seeds=2：改善 0.1695、多花 0.89s／案 → **效率最好**，如果只能接受一點點
  變慢，這個最划算
- seeds=3：改善 0.2078、多花 2.3s／案
- seeds=5：改善 0.2491、多花 5.91s／案 → 邊際效益明顯遞減

## 為什麼這次我沒有直接套用

前兩個改動（`grp_weight`、`mib_shape`）是純粹修正一個「本來就沒調好」的
權重常數，不增加任何 runtime，風險低、可逆、向後相容，所以我評估後直接
套用了。**這次不一樣**：使用者明確表達過「不希望用速度換品質」，而多起點
搜尋的本質就是「花更多運算換更好的解」，直接踩在使用者劃的那條線上。這種
需要在 cost 和 runtime 之間做取捨的決策，應該由你們（或使用者）判斷可接受
的代價，不該我單方面套用。

## 建議

1. 如果團隊評估後覺得 seeds=2（~1.4x runtime，改善明顯）是可接受的
   trade-off，可以考慮套用（`os.environ.setdefault("ELECTRO_SEEDS", "2")`
   + `os.environ.setdefault("ELECTRO_PARALLEL", "1")`）。
2. **更正**：原本以為平行開銷（3 起點從 2.3s 變 4.6s）可能是 fork/pool
   啟動成本的問題、可以優化掉。但讀了 `electro_optimizer.py` 自己的註解
   （第 105-108 行）才發現這是**已知、被原作者記錄下來的架構限制**：
   「CPU 上 forked worker 會搶佔 OpenMP 執行緒資源，真正的 seed-batching
   加速要在 GPU 上做（作者自己標記為 TODO，還沒做）」。也就是說這不是我
   能快速修好的小 bug，是需要改用 GPU 批次處理才能真正解決的架構性問題
   ——不建議往這個方向投入時間找「快速修復」，如果要做，是一個真正的
   GPU batching 專案，工程量遠大於這幾輪的權重微調。
3. 跟現有的 `ELECTRO_ITERS_PORTFOLIO` 疊加會讓 runtime 代價更高（兩個
   機制的 runtime 成本會相乘，不是相加），我還沒測疊加後的實際數字，
   如果你們要衝最低 cost（不計較 runtime），可以測 seeds=3 + iters
   portfolio 一起開的組合，理論上應該比單獨任一個都更好。
4. **另一個可能更划算的方向（真正零成本，但工程量較大）——具體設計**：
   目前的隨機初始化（`analytical_place.py` 第 205-206 行
   `cx = torch.rand(N, ...); cy = torch.rand(N, ...)`）變異度很高，這也是
   為什麼多起點挑最好的有用。想法：用 b2b 連通圖做幾輪簡單的鄰居平均
   （Jacobi-style relaxation，圖繪製領域的經典「warm start」手法），取代
   純隨機起點：

   ```python
   # 在現有 init 區塊（190-208 行）之後，torch.rand 初始化完，requires_grad_
   # 之前插入：對非 is_pre 的方塊做 K 輪（例如 20-30 輪）鄰居平均——
   # 每輪把每個方塊往它所有 b2b 鄰居的加權平均位置移動一部分（例如 0.3 的
   # 步長），is_pre 方塊維持在 pre_cx/pre_cy 不動，充當錨點讓其他方塊有東西
   # 可以「靠攏」。
   ```

   **注意**：`eb`（b2b 連通圖）目前在第 228-229 行才抽取，**晚於**
   init 區塊（190-208 行），需要把 `eb = _valid(b2b_connectivity)` 那行
   往前移到 init 之前，或在 init 前重複抽取一份簡化版。這是會影響檔案
   結構、需要仔細測試的改動（要確認沒有破壞 `is_pre`/`is_fixed` 方塊的
   處理、沒有連通邊的孤立方塊要有 fallback，跟現有的 213 行後的 MIB
   分組邏輯順序也要對齊），比純粹調權重常數的風險高很多，**我評估後
   沒有自己動手改**（這是你們正在開發中的檔案，貿然重排結構風險較高）。
   如果要做，建議先在獨立分支/scratch 驗證，跟現有 `torch.rand` 做
   portfolio 比較（同樣零額外 runtime，只是換初始化方式），確認沒有
   regression 再考慮換成預設。

## 追加（2026-07-18）：grp_weight × bnd_weight 聯合網格搜尋，發現權重與自適應門檻的耦合陷阱

使用者要求之後的探索不能增加 runtime，於是系統性網格搜尋（3×3，固定 600
迭代、零額外 runtime）`ELECTRO_GRP_WEIGHT` × `ELECTRO_BND_WEIGHT`，找到
`(0.45, 0.9)` = 2.1318（比現行 `(0.4, 1.0)` = 2.1731 好 1.9%，決定性一致，
附近點確認是真正尖峰，不是雜訊）。

**但套進完整的自適應收斂管線重新驗證，Total Score 反而變成 2.1077，比
官方預設在完整管線下的 2.0987 還差**（兩次決定性一致）。原因：
`ELECTRO_ADAPTIVE_SCORE_THRESH=2.0` 是針對現行權重的 proxy cost 分布校準
的，換權重會改變哪些案例被判定「需要延伸到 1200 迭代」，抵銷掉權重本身
在固定迭代數下的優勢。**沒有套用**——這在真正相關的指標（完整管線冷啟動
分數）上其實是退步的。

**重要教訓給你們**：以後調整 `grp_weight`/`bnd_weight`/`mib_shape` 這類
loss 權重時，**務必用完整管線（含 adaptive 機制，不要 `ELECTRO_ITERS_
PORTFOLIO=0`）重新驗證**，不能只看隔離測試的數字就下結論——兩者的排名
可能不一致，因為權重會反過來影響 adaptive 門檻判斷「哪些案例需要延伸」
的結果。

也追加測試了 `ELECTRO_ADAPTIVE_SCORE_THRESH=2.5`（拉高門檻、觸發更少
延伸、理論上更快）：Total Score=2.1302，比現行預設（2.0987）差，確認
2.0 這個現行門檻本身是合理的，沒有找到「更快且不會變差」的方向。

## 追加（2026-07-18 晚間）：Jacobi 暖啟動已實作命中，但建議把「疊加」改成可選「取代」

看到你們實作了 Jacobi 圖形暖啟動初始化（完全命中上面第 4 點的具體設計！），
獨立驗收確認 Total Score=1.9359 完全吻合，真的是很棒的改善（−7.8%）。

但使用者反映「感覺 runtime 變長超多」，實測確認是真的：平均 runtime 從
2.83s 拉到 4.44s（我的量測）到 5.483s（你們報告的量測），漲了 57-94%。
原因是 `ELECTRO_JACOBI_PORTFOLIO=1`（預設開啟）讓 Jacobi 變成**額外**候選
——每案同時跑 Random 跟 Jacobi 兩種初始化的完整 600 迭代，等於把每案的
初始化搜尋量翻倍，跟 `ELECTRO_SEEDS` 多起點是同一種代價。

**我測了一個問題**：你們報告裡自己寫「Jacobi 大幅降低了佈局變異度」——
如果這是真的，Jacobi 是不是不需要 Random 當備援？在獨立 scratch 目錄
驗證（加了 `ELECTRO_JACOBI_ONLY` 測試開關，讓 Jacobi**取代**而非疊加
Random，並補了 `run_start_jacobi_diag` 讓自適應延伸機制正確接上 Jacobi
路徑）：

| 設定 | Total Score | 平均 Runtime |
|---|---|---|
| 官方預設（Random + 自適應延伸） | 2.0987 | 2.83s |
| **Jacobi 取代 Random**（新發現） | **2.0749** | **3.05s** |
| 現行版本（Random+Jacobi 都跑） | 1.9359 | 4.44-5.48s |

隔離測試（固定 600 迭代，不含自適應延伸）也確認：Jacobi-only=2.0962/2.06s
比 Random-only=2.1731/2.14s **同時**更好且更快，兩次決定性一致——Jacobi
初始化本身的品質提升是真的，不是雜訊，也不需要 Random 陪跑就能拿到。

**建議**：把 `ELECTRO_JACOBI_PORTFOLIO` 的「疊加」邏輯改成可選的「取代」
模式，例如新增 `ELECTRO_JACOBI_MODE=replace|portfolio|off` 三態開關：
- `replace`：Jacobi 取代 Random 當唯一初始化（2.0749/3.05s，runtime 只多
  8%，我認為這應該當新的預設，最符合「不要明顯增加 runtime」的要求）
- `portfolio`：現行的兩者都跑（1.9359/4.44-5.48s，留給不在乎 runtime、
  只想衝最低 cost 的情境）
- `off`：純 Random（回到 2.0987/2.83s 的舊行為）

具體實作：把我在 scratch 目錄補的 `run_start_jacobi_diag`（
`electro_parallel.py`）跟 `electro_optimizer.py` 裡判斷 `jacobi_only` 走
`jacobi_precomputed`（避免 Jacobi 的 600 迭代跑兩次，這是我自己測試時
踩到的一個小心陷阱，注意重複利用 diag 呼叫回傳的 layout，不要另外再呼叫
`run_start_jacobi` 一次）的邏輯直接搬過去即可，兩個函式都已經測過決定性
一致。

## 已排除的捷徑：拆分總運算量不能免費拿到多起點的好處

測了「2 個起點各跑 300 迭代」（總 gradient-descent 運算量跟「1 個起點跑
600 迭代」一樣，不增加 runtime）vs 現行的「1 個起點跑 600 迭代」：結果
幾乎打平（2.1708 vs 2.1731，runtime 也差不多甚至略快）。**證實多起點的
好處需要每個起點都跑滿完整迭代才有效**，不能靠「重新分配同樣的運算預算」
免費拿到——這排除了一個可能的捷徑，確認 §8.27 的 seeds trade-off 是真實
的、無法迴避的成本，不是重新分配就能免費取得的。
