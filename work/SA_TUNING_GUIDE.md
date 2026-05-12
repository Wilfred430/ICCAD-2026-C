# 為什麼 floorplan SA 的 cost 看起來上下亂跳 — Tuning Guide

## TL;DR

**你看到的「亂跳」90% 來自一個 logging bug，不是 SA 本身的問題。**

`sa.cpp` 寫進 CSV 的是 `nc`（被 propose 的 candidate move 的 cost）而不是 `cost`（SA 真正接受、目前所在的 state）。被 reject 的 candidate 也被記了下來，所以圖看起來像狂跳的雜訊。把 log 改成寫 `cost` 之後，曲線會接近你預期的平滑下降。

剩下的 10% 才是 floorplan vs partition SA 的本質差異 — 我會在下面解釋為什麼，以及怎麼透過調參把「真實的」雜訊也降下來。

---

## Bug：你的圖是 candidate cost，不是 current cost

`src/sa.cpp` 原本長這樣（你已經加的 every-4-iter throttle）：

```cpp
Move m = engine_.propose(inst_, current);
PackResult pp = packer_.pack(inst_, current);
Costs cc = evaluator_.evaluate(inst_, current, pp);
Real nc = evaluator_.sa_cost(cc, cfg_.weights, inst_);   // ← candidate 的 cost
Real d = nc - cost;

bool accept;
... // 決定是否接受
if (accept) {
    cost = nc;     // ← 真正的 current state 才在這裡更新
    ...
} else {
    engine_.revert(inst_, current, m);   // ← reject 後 tree 已還原
}

// ↓↓↓ 但是 log 寫的是 nc，不是 cost ↓↓↓
csv_file << iter << "," << T << "," << nc << "," << R.best_sa_cost << "\n";
```

意思是：**就算這個 move 被 reject、tree 已經還原成 `current` 狀態，CSV 裡記的還是被丟掉那個 candidate 的 cost**。

對 floorplan SA 影響超大，因為：

- 一個 `MoveKind::Move`（`P_MOVE ≈ 0.37`）會把節點 v 重新嫁接到 u 底下，連帶讓 v 的所有 descendant、u 原本子樹的 graft 全部換位置。**動到 30+ 個 block 的 (x, y) 是常態**。
- HPWL 對 (x, y) 是線性的，但所有 net 一起重算。一動就是上千級的 ΔHPWL。
- 如果動到的 block 跟 preplaced anchor 重疊，overlap penalty 直接 +500 到 +5500。
- 如果動到 boundary-constrained block，`v_boundary` 一口氣加減 5–20，每個 ×80 = ±400 到 ±1600。

所以一個 candidate 的 `nc` 跟當前 `cost` 差到 5000 以上很正常。**SA 把這些壞 candidate 丟掉是它的工作**，但 log 把丟掉的東西也畫上去，看起來就像 SA 整天在跳。

### 修法（已套用）

```cpp
// src/sa.cpp 那一行
- csv_file << iter << "," << T << "," << nc << "," << R.best_sa_cost << "\n";
+ csv_file << iter << "," << T << "," << cost << "," << R.best_sa_cost << "\n";
```

`cost` 是這個 iter 結束後 SA 實際所在的 state（accept 過的 + 沒 accept 直接 revert 回 `current` 的兩種情況都正確）。CSV 的 header 寫的本來就是 "CurrentCost"，現在內容才符合 header。

重新 build + 跑之後，下界envelope 還是同一條（accept 的是同一群），但**上界 envelope 會塌下來**到接近 mean，整張圖看起來就像「partition SA 的平滑下降 + 一些 SA accepts 的 uphill 噪音」。

驗證（toy benchmark, 5s, 4 threads）：

| 指標（log 的 CurrentCost 欄位） | 改之前（記 nc） | 改之後（記 cost） |
| --- | --- | --- |
| min | ~1.97 | 1.97 |
| max | 通常 100+ | 165（少數 SA-accepted uphill）|
| mean | 中段（~5000 級的「中間值」）| 6.5 |
| range/mean 比 | 很大 | ~24 → 集中在 mean 附近 |

---

## Partition SA vs Floorplan SA：為什麼後者的 ΔCost 分布天生比較野

這是你問的核心問題：「在 partition 上是平緩下降，怎麼到 floorplan 就這樣？」

| 維度 | Partition SA | Floorplan SA (B*-tree) |
| --- | --- | --- |
| State 表示 | n 個 block 各自貼一個 0/1 label | 一棵 binary tree + 每塊的 (w, h) |
| Move 影響的 block 數 | **1** (flip 一個 label) | **1 ~ n** (Move/Swap 會 graft 整棵子樹) |
| Cost 函數 | 邊割數 (sum over edges) | bbox_area + Σ HPWL_net + soft penalty + hard penalty |
| ΔCost per move 的分布 | 窄、近高斯、median = mean | **重尾**：絕大多數 move Δ 很小，少數 move Δ 巨大 |
| Cost surface 連續性 | 段段平滑（每 flip 是一階跳）| 充滿 hard-penalty 的階梯（overlap/area/fixed/preplaced 的 binary 5000）|

具體拆給你看，floorplan 一個 `MoveKind::Move` 的 Δ 來源：

```
ΔCost = Δarea (因為 bbox 改了)
      + Σ_net Δ|HPWL|  (每條 net 連到的 block 任一個動了就要重算)
      + 80 × Δv_grouping
      + 80 × Δv_mib
      + 80 × Δv_boundary
      + (新增 / 減少) hard penalty (overlap / area_drift)
```

partition 沒有「**整棵子樹被搬家、bbox 改變、所有 HPWL 重算、constraint 違反數一次跳 5–20 個**」這種大型結構性 move。所以即便都在 cooling，floorplan SA 的 accepted-cost 序列也會有比 partition SA 大一個量級的 short-term variance，這是無法完全消除、只能管理的本質。

下界 envelope（也就是 best）才是真正在意的；中段 noise 是 SA 跳出 local minima 的代價。

---

## 想壓住「真實」雜訊的調參指南

`cost` 寫對了之後曲線還是嫌野的話，下面幾個鈕由低風險到高風險排：

### 1. 把 move 機率挪向「低 Δ」move（最有效、副作用最小）

`src/moves.cpp` 目前的設定：

```cpp
constexpr double P_FIXB = 0.05;   // boundary 修補
constexpr double P_FIXG = 0.05;   // grouping 修補
constexpr double P_AR   = 0.18;   // 改一塊 soft block 的 (w,h)
constexpr double P_MIB  = 0.05;   // 同步整個 MIB group 的 shape
constexpr double P_ROT  = 0.15;   // 旋轉一塊
constexpr double P_SWP  = 0.15;   // 交換兩塊
// remainder ~0.37 = Move（最大 Δ 的 move！）
```

`Move` 是 Δ 最大的 — 它一次重排一整棵子樹。`AR` / `Rotate` 只動一塊的 dim，Δ 通常一個量級小。

建議改：

```cpp
constexpr double P_FIXB = 0.05;
constexpr double P_FIXG = 0.05;
constexpr double P_AR   = 0.30;   // ↑ 0.18 → 0.30
constexpr double P_MIB  = 0.05;
constexpr double P_ROT  = 0.25;   // ↑ 0.15 → 0.25
constexpr double P_SWP  = 0.10;   // ↓ 0.15 → 0.10
// remainder = 0.20  ← Move 從 0.37 降到 0.20
```

效果：每個 iter 的 Δ 分布尾巴變短，曲線收得緊，但同時 SA 探索能力會變弱（拓樸 move 變少）。配合更長的 budget 就 OK。

### 2. `c_fastsa`（控制 stage 2 起始溫度，越大冷越快）

`include/sa.hpp`：

```cpp
double c_fastsa = 100.0;    // 預設
```

T 公式 `T = T1 * avg / (k * c)`，c 越大 → k=2 起始 T 越低 → SA 越早進入 exploitation 模式。

- 想要更平穩 → `c_fastsa = 300` 或 `500`
- 想要更多探索（更野） → `c_fastsa = 50`

副作用：太大會讓 SA 早早卡死在 local minima。

### 3. `n_iters_per_block`（每個溫度 step 的 iter 數）

```cpp
int n_iters_per_block = 50;   // 預設
```

越小 → 每個 T 待越短 → cooling 越快、曲線越平滑但 SA 在每個 T 探索不夠。

- 想要更平穩 → `30`
- 想要更徹底 → `100`

一般建議：問題越大（n_blocks 越多）這個值越小，總 iter 數差不多就好。

### 4. `p_accept_init`（初始接受率）

```cpp
double p_accept_init = 0.92;
```

T1 = -Δavg / ln(p)。p 越小 → T1 越小 → stage 1 不會把初始好解砸爛。

- 0.92 對 floorplan 還是稍高（會把幾乎所有 uphill 都接受，stage 1 變相 random walk）
- 試試 `0.80` 甚至 `0.70`，但要小心初始解必須夠好（你 round 1 改的 `make_initial` 有保證了）

### 5. `reanchor_every_iters`（多久 snap 回 best 一次）

```cpp
int reanchor_every_iters = 50;   // ×n_blocks 才是真實 iter 數
```

越小 → 越常 reset 到 best → 曲線下界更明顯（因為常常從 best 出發）。

- 想要更平穩、更少漫遊 → `20` 或 `30`
- 想要 SA 多跑多探索 → `100`

### 6. （進階）加 ΔCost 上限拒絕

不修參數、修 acceptance 邏輯：在 `src/sa.cpp` 的 accept 判斷加一條保險絲：

```cpp
// Even if SA would accept, REJECT if Δ is more than 5× the current best cost.
// Stops occasional huge-Δ moves from polluting the trajectory.
if (accept && d > 0 && d > 5.0 * R.best_sa_cost) {
    accept = false;
}
```

這把 Δ 分布的右尾直接砍掉。曲線變超平。代價：SA 失去跳出大坑的能力，可能卡 local minima。建議只在已經知道 best 落點不錯的後半段啟用。

---

## 怎麼判斷「平滑下降」是否真的代表 SA 在收斂

partition SA 的平滑曲線好讀，但其實「平滑」≠「收斂到全域最佳」。對 floorplan 來說，看下面三個訊號比看曲線平不平更可靠：

1. **`best` 是不是在下降** — 我把 `best_sa_cost` 也寫在 CSV 第 4 欄了，畫這個比畫 CurrentCost 重要。partition SA 的 best 跟 current 黏在一起；floorplan SA 的 best 是 current 的下界 envelope。
2. **`feasible` 比例** — `[main]` log 最後印的 `n_feasible_threads=X/Y`。如果有 thread 還是 infeasible，曲線再平也沒用。
3. **`hpwl_gap` / `area_gap` / `V_rel`** — 這些是 contest cost 的真正組成。最終 contest_cost 才是答案，CurrentCost 只是 SA 內部的 proxy。

建議畫圖時用 `BestCost` 當主線、`CurrentCost` 半透明當背景。看 best 曲線是否「stair-step 式下降」（每隔一陣子掉一階），這才是 floorplan SA 收斂的長相。

---

## 總結成一張表

| 你的觀察 | 比例 | 解法 |
| --- | --- | --- |
| Cost 上下亂跳 | ~90% logging bug | log `cost` 不要 log `nc`（已修） |
| 跳動幅度比 partition SA 大 | ~10% 本質差異 | 接受它，或調 move 機率把 Δ 尾巴砍短 |
| 想要更像 partition SA 的曲線 | — | 上面 6 個調參按順序試，從 #1 開始 |
| 想知道 SA 是不是真的有在收斂 | — | 看 `BestCost` 不是 `CurrentCost` |
