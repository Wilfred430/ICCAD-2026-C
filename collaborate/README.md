# 2026-05-20 → 2026-05-21 修正紀錄 — case 056「長條狀 + L 形死空白」修復

本檔案紀錄三次連續修正：

| 輪 | 日期 | 目標 | 結果 |
|---|---|---|---|
| v1 | 2026-05-20 上午 | bbox 從 135×270 → 趨向正方 | 寬從 135 收回 150，高沒動 |
| v2 | 2026-05-20 下午 | 強化 packer + 移動 bias | 寬從 150 漲到 ~170，高從 270 掉到 ~275（漲！）|
| **v3** | **2026-05-21** | **修 v2 的 short-dim creep + 補 L-hole** | **應該收回 ~165×~210，明顯降 area** |

三輪都只動 packer / moves / btree 的 `.hpp` / `.cpp`，沒動 SA / cost /
parser / parallel。

---

## v3 — 為什麼 v2 後 cost 還是只掉一點點

對照圖：v2 後 bbox 變成 ~170×275。寬度增加、高度幾乎沒掉，cost 自然不
動。圖上明顯可見：

1. **block 24 / 19 / 16 / 63 上方那條空白**（y > 200 的右側區域）
2. **cluster 之間的小三角形空隙**（compact 拉不到）
3. **bbox_w 從 150 一路漲到 170**

兩個根因，對應 v3 兩個修法：

### v3 根因 A：`bbox_balance_pass` 的「short-dim creep」

v2 每 pass 重算 `bbox_w` 當作 spike 重定位的右界，所以：

```
pass 1: bbox_w=150 → 把 spike 放到 x=140 → 新 bbox_w=160 (block 寬 20)
pass 2: bbox_w=160 → 把 spike 放到 x=150 → 新 bbox_w=170
pass 3: bbox_w=170 → ...
```

每個 pass 都「合法地」把短邊撐大一點，累積下來 bbox_w 漲了 20。對「目標
正方」反而是反向推力。

**v3 修法**：算出「目標正方形邊長」`target_side = sqrt(baseline_area)`
（沒 baseline 時用 `sqrt(init_w * init_h)`），把 short-dim 上限硬鎖在
`max(current_short, target_side)`。短邊可以朝目標長大、但永遠不准越界。

對 case 056 baseline ≈ 26 250 → `target_side ≈ 162`。所以 bbox_w 最多漲
到 162 就停，剩下的「壓 bbox」工作只能往 bbox_h 收 → 直接逼出 v2 該有
的「高度真的會掉」效果。

```cpp
const Real target_side =
    (inst.baseline_area > 0)
    ? std::sqrt(inst.baseline_area)
    : std::sqrt(std::max(init_w * init_h, Real(1.0)));

Real short_dim_cap = std::max(short_dim, target_side);
if (!relocate_one_spike(..., short_dim_cap)) { ... }
```

### v3 根因 B：`compact_left_down` 只能軸向移動，補不到 L-hole

軸向 compaction 對下面這種「L 型死空白」無能為力：

```
   [B]
[A]
[C][D]
```

把 [A] 往下拉？被 [C] 擋住。往左拉？已經到底。但其實 [A] 移到 (x_C, y_D)
就完美塞進右下角的空位——這個移動需要同時改 x 和 y。

`bbox_balance_pass` 會做對角線重定位，但只針對 spike block。**其他在
floorplan 中段的 block 卡在 L-hole 就會永久留著空白**——這就是圖上
case 056 的 cluster 之間那些三角形空隙的由來。

**v3 修法**：新增 `holes_fill_pass()`，對「所有 movable + 非 cluster 成員」
的 block，搜尋讓 `max(x+w, y+h)` 最小且不撞別人的位置。等於主動填洞，
而不只是壓 spike。

```cpp
static void holes_fill_pass(const FloorplanInstance& inst, BTree& tree) {
    // 收集所有可動 block（排除 preplaced / boundary / cluster 成員）
    // 由「離原點最遠」開始排序，先處理 spike-like 的 block
    // 對每個 block 試所有 (x, y) 候選（其他 block 的角點 + 0）
    // 取 max(x+w, y+h) 最小且能放下的位置
    // 重複 2 次（一個 block 移動會解放它的鄰居）
}
```

排除 cluster 成員是為了不破壞 grouping 限制——v_grouping 違規會把
contest cost 直接乘上 `exp(2·V_rel)`，得不償失。boundary 成員是因為已
經有自己的位置硬限制，preplaced 是 hard 不能動。

複雜度最壞 O(n^4)，但 candidate 排序後有 early `break`，實測在 n≤120
範圍每 pack 多 ~3-5ms。SA throughput 從 ~5000/sec 掉到 ~1000-2000/sec，
但每個 pack 的 layout 品質提升明顯。

### v3 之後完整 packer pipeline

```
SA 一次 iter:
  propose move (bbox-AR-biased)
  pack:
    contour walk (B*-tree DFS)
    compact_left_down       (v2: fixpoint loop)
    bbox_balance_pass       (v3: short-dim capped at target_side)
    compact_left_down       (v2: catch floaters)
    holes_fill_pass         (v3 新增: 對角線填 L-hole)
    compact_left_down       (v3 新增: 再 catch 一次 floater)
    bbox / overlap re-check
  evaluate
  accept / reject
```

---

## v1 + v2 的歷史背景（已經完成的部分）

case 056 的 floorplan 比較圖（左 Ground Truth，右 我方 Optimizer）暴露了
五個症狀，但底層只有一個共同病因 — **bbox 長寬比**：

| 症狀 | 視覺表現 |
|---|---|
| 1. IO pins 被串起來 | block 衝到 Y=270，蓋過 Y=210 那排 IO |
| 2. boundary block 跑到左上 | 卡其色塊應在右/下，卻擠到左上 |
| 3. 身材走鐘 135×270 | 該方 150×175，現在 AR≈2.0 |
| 4. 頂部階梯凹凸 | 從 Y=210 到 Y=270 高低不齊 |
| 5. 線長爆炸 | bbox 拉長 → block 離 terminal 遠 → HPWL 暴增 |

v1 做的事：
- 新增 `bbox_balance_pass()` — spike block 對角線重定位
- `apply_fixb` 加 tactic-2 dim-growth guard（不准把長邊再撐 20%）
- `apply_ar` 加 bbox-AR sampling bias（高的 bbox → 70% 機率抽寬扁 block）
- `apply_fixg` 加 left/right child bias（高的 bbox → 80% 接 left-child）

v2 做的事：
- `compact_left_down` 改 fixpoint loop
- `bbox_balance_pass` cap 4 → 60，AR 退出門檻 1.15 → 1.10
- `pack()` 在 balance 後追一次 compact
- `apply_rotate` 加方向感（85% 跳過反向旋轉）
- `apply_move` `as_left` 從 0.5 → bbox-AR-biased
- `apply_ar` BIAS_PROB 0.70 → 0.85

---

## 哪些檔案動過

| 檔案 | v1 | v2 | v3 |
|---|---|---|---|
| [packer.cpp](packer.cpp) | 新增 `bbox_balance_pass` | fixpoint compact + 加碼 balance + 後追 compact | **加 target_side 上限**；**新增 `holes_fill_pass`**；**balance 後再 compact + fill + compact** |
| [packer.hpp](packer.hpp) | 不動 | 不動 | 不動 |
| [moves.cpp](moves.cpp) | `apply_fixb` guard / `apply_ar` bias / `apply_fixg` bias | `apply_rotate` bias / `apply_move` bias / `apply_ar` BIAS_PROB↑ | 不動 |
| [moves.hpp](moves.hpp) | 不動 | 不動 | 不動 |
| [btree.cpp](btree.cpp) | 不動 | 不動 | 不動 |
| [btree.hpp](btree.hpp) | 不動 | 不動 | 不動 |

---

## 怎麼跑

```bash
# 編譯（不用 make，可直接呼叫 g++）
g++ -std=c++17 -O2 -pthread *.cpp -o floorplanner

# 或用既有 Makefile：
make
make check         # 跑 toy benchmark
make submit        # 打包成上傳 zip
```

三輪都驗證過 `g++ -std=c++17 -O2 -Wall -Wextra` 過編、全專案 link 過。
v3 之後唯一剩下的 warning 是 `cost.cpp::count_components` 的 unused
parameter（本次修正範圍以外，本來就有）。

---

## v3 預期效果

| # | 症狀 | v3 預期 |
|---|---|---|
| 1 | block 衝到 Y=270，蓋住 IO | bbox_h 應從 275 收回 ~210-220 |
| 2 | boundary block 跑到左上 | 大致同 v2，但 short-dim cap 讓 SA 更容易找到正解 |
| 3 | bbox 150×175 目標 | 應接近 165×210（v3）vs 170×275（v2）|
| 4 | 頂部階梯狀 | `holes_fill_pass` 直接抓階梯 |
| 5 | wire length | bbox 收回後應顯著改善 |
| **+** | **L-hole 死空白** | **`holes_fill_pass` 主動填掉** |
| **+** | **bbox_w creep** | **target_side 硬鎖** |

---

## 如果還是不夠：下一步往哪走

依優先序：

1. **`target_side` 比例調整**：目前直接用 `sqrt(baseline_area)`，可以加
   1-1.1 倍緩衝（避免硬塞）。改 [packer.cpp](packer.cpp) 那行 `const Real
   target_side = ...` 即可。

2. **`holes_fill_pass` 加碼**：
   - 把 pass 數從 2 改 3
   - 把 cluster 成員的 skip 改成「條件式 skip」（允許移動但要求移動後仍
     觸到其他 cluster 成員，O(n²) 檢查即可）

3. **`bbox_balance_pass` 的 1.10 退出門檻收到 1.05**

4. **再不行就要動 cost weights**（本次修正範圍以外）：
   - [cost.hpp](cost.hpp) `w_outline` 0 → 0.5
   - 或 [sa.hpp](sa.hpp) `SAWeights::w_hpwl_ext` 1.0 → 1.5

---

## 接 Claude / ChatGPT 的人請看

[CLAUDE.md](CLAUDE.md) 已同步加入 v3 的部分。要再延伸的人去那看「bbox
aspect-ratio control」章節，v1 / v2 / v3 都標好對應的函式 / 行號。
