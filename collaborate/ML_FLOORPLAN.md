# ML Floorplanner — Graph-Transformer Warm-Start for ICCAD 2026

這份文件說明 `ml/` 目錄底下的機器學習 pipeline：怎麼**訓練**、**部署**、跟現有 C++ SA solver 怎麼**整合**，以及**每個參數的意義**。

---

## 0. 概念與動機

現有的 B*-tree + Fast-SA 流程**從亂數初始樹起跑**，要花 60-90 秒收斂。其他隊伍用 ML 跑得更快、分數更好——主要原因是：

1. **SA 大部分時間都在「從亂跳變正常」**——前 10–20 秒就是把離譜的 placement 收拾乾淨
2. **ML 可以一次給一個「8 成正確」的解**作為 warm start，SA 只負責**最後 2 成的精修**
3. FloorSet 已經提供 **百萬等級** 的 `(constraint_graph → ground_truth_positions)` 配對，是**監督學習現成的訓練資料**

我們不重新發明輪子，而是借鑑 PARSAC + GoodFloorplan 兩篇論文：
- **PARSAC** 的 anchored-block 機制 → 我們保留現有 packer
- **GoodFloorplan** 的 GCN+RL → 我們用更穩定的 supervised Graph-Transformer（不打 RL，因為訓練更難）
- **diffusion / quasi-Newton refinement**（後續工作）→ 暫不做

### 整體 pipeline

```
   FloorSet case tensors                         ┌─ ml/predict.py ─┐
   (blocks, nets, terminals,                     │  Transformer    │
    constraints, target_positions)               │  inference      │
        │                                        └────────┬────────┘
        ▼                                                 │
   my_optimizer_ml.py ────── append WARM_POSITIONS ───────┘
   _write_txt() │
                ▼
   case_NNN.txt    (含 BLOCKS / NETS / GROUPS / MIB / WARM_POSITIONS)
                │
                ▼
   ┌──────────────────────────┐
   │  C++ floorplanner SA     │ ← make_initial 用 WARM_PRIORITY
   │  (現有 binary)            │   排序 block 插入，SA 從更好的
   └──────────┬───────────────┘   初始樹起跑
              ▼
   case_NNN.sol → (x, y, w, h) positions
```

---

## 1. 安裝

```bash
cd /home/pop/2026_EDA_contest

# Python 依賴（torch 已經是 contest framework 的依賴，不用裝；其他都是標準庫）
# 若要 GPU 加速，請確認 torch.cuda.is_available() = True

# 建立權重輸出目錄
mkdir -p ml/weights
```

---

## 2. 訓練

### 2.1 最小指令

```bash
# 用 FloorSet Lite 訓練 5 epoch、batch 16、學習率 1e-3
python -m ml.train \
    --data /home/pop/IntelLabs_Floorset/FloorSet/LiteTensorDataTest \
    --out  ml/weights/floorplan_v1.pt \
    --epochs 5
```

### 2.2 完整參數

| Flag | 預設 | 說明 |
|------|------|------|
| `--data` | (必填) | FloorSet 資料根目錄（例如 `LiteTensorDataTest`）。底下要有 `config_*/litedata_*.pth` 跟 `litelabel_*.pth` |
| `--out` | `ml/weights/floorplan_v1.pt` | 訓練好的 checkpoint 存哪 |
| `--epochs` | 5 | epoch 數。5–10 通常已足夠 |
| `--batch` | 16 | batch size。GPU 上可以開到 64 |
| `--lr` | 1e-3 | AdamW 學習率。Cosine annealing 自動降到 0 |
| `--hidden` | 128 | Transformer hidden dim |
| `--layers` | 4 | Transformer encoder 層數 |
| `--heads` | 4 | multi-head attention 頭數 |
| `--max-blocks` | 128 | 最大 block 數（contest 最大 120，留 padding） |
| `--max-terms` | 512 | 最大 terminal 數 |
| `--val-frac` | 0.05 | 從訓練集切出多少做 validation |
| `--workers` | 2 | DataLoader 平行度 |
| `--device` | 自動偵測 cuda / cpu | 強制裝置 |

### 2.3 訓練效能參考

| 機器 | dataset 規模 | epoch 時間 | 5 epoch 總時間 |
|---|---|---|---|
| RTX 3090 GPU | 10K cases | ~30 s | ~3 min |
| RTX 3090 GPU | 1M cases  | ~50 min | ~4 hr |
| CPU (8 core)  | 10K cases | ~10 min | ~1 hr |
| CPU (8 core)  | 1M cases  | 太慢，不建議 | — |

**建議**：先用 LiteTensorDataTest（small subset）跑通 pipeline，再決定要不要下載 LiteTensor_v2 整套訓練。

### 2.4 訓練輸出

每個 epoch 結束會印類似：

```
[epoch 1/5] 32.1s  train_loss=0.4521 (pos=0.30 dim=0.18 area=0.21) | val_loss=0.4102 pos=0.27
[train] saved ml/weights/floorplan_v1.pt (val_loss=0.4102)
```

`val_loss` 是 normalised L1 error；< 0.2 算是不錯（位置誤差約 bbox 的 20%）；< 0.1 表示模型已經學會大致 placement，可以接到 SA 上效果不錯。

---

## 3. 部署 / 推論

### 3.1 用法

把現有 `iccad2026_evaluate.py` 改成載入 `my_optimizer_ml.py`：

```bash
cd /home/pop/IntelLabs_Floorset/FloorSet/iccad2026contest

# 把 ML 檔案部署過去
cp -r /home/pop/2026_EDA_contest/ml             ./
cp    /home/pop/2026_EDA_contest/my_optimizer_ml.py  ./

# 跑測試（注意 --evaluate 指向 my_optimizer_ml.py）
source /home/pop/IntelLabs_Floorset/FloorSet/venv/bin/activate
python iccad2026_evaluate.py --evaluate my_optimizer_ml.py --test-id 56 --save-solutions
```

### 3.2 環境變數

`my_optimizer_ml.py` 多了 3 個 env var：

| Env var | 預設 | 說明 |
|---|---|---|
| `FLOORPLANNER_ML_WEIGHTS` | `ml/weights/floorplan_v1.pt` | checkpoint 路徑。**如果檔案不存在，ML 自動關閉、退回純 SA**——不會 crash |
| `FLOORPLANNER_ML_DEVICE`  | `cpu` | 推論裝置。`cpu` 對 n ≤ 128 已經很快（~50 ms） |
| `FLOORPLANNER_ML_VERBOSE` | `0` | `1` 印 ML predict 過程的 diagnostic |

加上 baseline 原本的 `FLOORPLANNER_BIN / THREADS / TIME / SEED / KEEP`。

### 3.3 跑通沒 ML weights 的情況

故意把 weight 拿走，confirm fall-back 正常：

```bash
mv ml/weights/floorplan_v1.pt ml/weights/floorplan_v1.pt.bak
python iccad2026_evaluate.py --evaluate my_optimizer_ml.py --test-id 56
# stderr 會印: [my_optimizer_ml] ML predictor: disabled
# 之後跑的就是純 SA pipeline，**結果跟 my_optimizer.py 完全一樣**
mv ml/weights/floorplan_v1.pt.bak ml/weights/floorplan_v1.pt
```

---

## 4. C++ 端 warm-start hook（**選擇性**）

`my_optimizer_ml.py` 會把預測結果**附在 `.txt` 檔尾**，加一個 `WARM_POSITIONS` section：

```
WARM_POSITIONS 77
0 75.3 100.2 18.5 13.1
1 23.1  45.0 22.0 22.0
...
WARM_PRIORITY 77
12 5 19 ...  ← 預測 (cy, cx) 排序後的 block id list
```

**目前的 C++ parser 會「禮貌地忽略」這兩個 section**（parser.cpp 的 `else if` 鏈走完都不 match，就跳過），所以加上 ML 也不會破壞 baseline 行為。

要讓 SA 真的用上這個 hint，需要在 `src/parser.cpp` 跟 `src/parallel.cpp::make_initial` 加 hook（**這是我隊友的工作**，不需要你動）。簡述就是：

```cpp
// parser.cpp
else if (tok == "WARM_POSITIONS") {
    int n; rd(n);
    inst.warm_positions.resize(n);
    for (int i = 0; i < n; ++i) {
        int id; Real cx, cy, w, h;
        rd(id); rd(cx); rd(cy); rd(w); rd(h);
        inst.warm_positions[id] = {cx, cy, w, h};
    }
}
else if (tok == "WARM_PRIORITY") {
    int n; rd(n);
    inst.warm_priority.resize(n);
    for (int i = 0; i < n; ++i) rd(inst.warm_priority[i]);
}
```

```cpp
// parallel.cpp::make_initial
if (!inst.warm_priority.empty()) {
    // 用 ML 給的 order 取代原本的 constraint-priority sort
    order = inst.warm_priority;
} else {
    // 原本的 sort
    ...
}

if (!inst.warm_positions.empty()) {
    // 用 ML 給的 (w, h) 設定初始尺寸
    for (int i = 0; i < n; ++i) {
        if (inst.blocks[i].is_fixed || inst.blocks[i].is_preplaced) continue;
        t.w[i] = inst.warm_positions[i][2];
        t.h[i] = inst.warm_positions[i][3];
    }
}
```

跟 SA 配合後，**SA 從 ML 給的近似解開始 polish**，理論上：
- 收斂時間從 80 秒降到 20–30 秒
- 收斂品質（contest cost）跟現在差不多或更好（因為起點更接近全域最佳）

---

## 5. 各檔案職責

### 5.1 `ml/data.py`

`FloorSetLiteDataset` — 從 `config_*/litedata_*.pth + litelabel_*.pth` 一筆筆讀，轉成統一 padding 的 batch 餵給模型。

關鍵函式：
- `case_to_features(blocks, b2b, p2b, geometry)` → `[N, 16]` 特徵向量
- `case_to_targets(geometry)` → `[N, 4]` ground truth (cx, cy, w, h)

每個 block 的 16 維特徵：

| index | 名稱 | 意義 |
|---|---|---|
| 0 | `F_AREA` | area_target 原始值 |
| 1 | `F_AREA_LOG` | log(area+1)，數值穩定性 |
| 2 | `F_IS_FIXED` | 0/1 |
| 3 | `F_IS_PREPLACED` | 0/1 |
| 4 | `F_HAS_MIB` | 0/1 |
| 5 | `F_HAS_CLUSTER` | 0/1 |
| 6–9 | `F_BND_*` | boundary bitmask 拆成 L/R/T/B 四個 binary |
| 10 | `F_DEG_B2B` | log(1+block 連到的 b2b net 數) |
| 11 | `F_DEG_P2B` | log(1+block 連到的 p2b net 數) |
| 12–13 | `F_W_HINT, F_H_HINT` | 預設 0；fixed/preplaced 填入 ground truth dim |
| 14–15 | `F_X_HINT, F_Y_HINT` | 預設 0；preplaced 填入 ground truth pos |

### 5.2 `ml/model.py`

`FloorplanTransformer` — 對 `{blocks ∪ terminals}` 全 self-attention，輸出 `(cx, cy, w, h)` per block。

關鍵設計選擇：
- **Pre-LN Transformer**：訓練更穩，較不需要 learning rate warmup
- **Type embeddings**：給每個 token 一個 learnable bias，模型才能區分「這是 block」vs「這是 terminal」
- **softplus(dim)**：保證輸出的 w/h > 0；避免預測負數讓 packer 報錯
- **全 attention（O(N²)）**：n ≤ 200 完全沒問題；GCN 那種 sparse message passing 反而學不會「跨 cluster 協調」

### 5.3 `ml/train.py`

Supervised training loop。

Loss 設計：
```
L = L_pos + 0.5 · L_dim + 0.2 · L_area_consistency
```
- `L_pos`：smooth L1 on (cx, cy)，**用每個 case 自己的 bbox 做 scale-normalisation**，避免大 case 主導 gradient
- `L_dim`：smooth L1 on (w, h)，**只算 soft block**（fixed/preplaced 是輸入直接給的）
- `L_area`：`|w·h - area_target| / area_target`，確保 area 限制（1% 容忍）

### 5.4 `ml/predict.py`

`Predictor` 類別。process 啟動時 load 一次 model，之後每個 case 呼叫 `.predict(...)`：
1. 從 case tensors 重建 `[N, 16]` 特徵
2. 模型 forward → (cx, cy, w, h) per block
3. **後處理**：
   - fixed/preplaced 強制覆寫 dim（以 target_positions 為準）
   - preplaced 強制覆寫 pos
   - soft block 做 area-snap：`scale = √(area_target/predicted_area)`，乘到 (w, h) 上，保證 1% area 容忍
4. 算 `priority` = blocks 依 (cy, cx) 排序後的 id list

如果 model file 不存在或 forward 失敗 → 返回 `None`，my_optimizer_ml 自動 fall back。

### 5.5 `my_optimizer_ml.py`

繼承 `MyOptimizer`，在 `solve()` 流程中 inject ML：
1. baseline `_write_txt`（產生 BLOCKS / NETS / GROUPS / MIB 等 section）
2. `_append_warm_positions`：附加 `WARM_POSITIONS` + `WARM_PRIORITY`
3. 呼叫 C++ floorplanner subprocess（**完全沒改現有 .cpp**）
4. `_parse_sol` 解析結果

**所有 fall-back 路徑**（ML 失效 / weight 不存在 / 異常）都直接走 baseline，**永遠不會比 my_optimizer.py 差**。

---

## 6. 參考論文

| 論文 | 你的資料夾 | 我們借鑑了什麼 |
|---|---|---|
| **FloorSet** (Intel Labs 2024) | ✓ | 訓練資料格式跟 1M 數量級的可用 ground truth |
| **GoodFloorplan** (Cheng et al. 2022) | ✓ | GCN/Transformer 預測 placement 的可行性；我們用 supervised 取代他們的 RL |
| **PARSAC** (Yi et al. 2024) | ✓ | anchored block 設計、constraint-aware moves；保留在 C++ SA 端 |
| **B*-tree FastSA** (Chen & Chang 2006) | ✓ | 三段式退火；我們的 SA 還是用這套 |
| **A New Algorithm for Floorplan Design** (Wong et al.) | ✓ | 經典 sequence-pair；我們不採用，但提供概念對照 |

---

## 7. 還沒做但可以做的後續工作

按 ROI 從高到低：

1. **C++ 端讀 WARM_POSITIONS / WARM_PRIORITY**（§4）—— 預計可讓 SA 收斂時間 ↓ 50%
2. **訓練資料擴大到 LiteTensor_v2 全集**（1M cases，~4hr GPU）—— val_loss 預計 ↓ 30%
3. **加上 PARSAC §3.5 的 inter-population migration**—— 跨 thread 共享 best，可能再 ↓ 5%
4. **Quasi-Newton geometry refinement**（Ji 2021）—— SA 收斂後對 (w, h, x, y) 做 LBFGS 連續優化，預計 ↓ 3–5% contest cost
5. **Diffusion-based post-processing**（CLAUDE.md Phase 3）—— 研究價值高但實作週期 ≥ 2 週
6. **改用 Graph Attention with edge bias**（用 net adjacency 當 attention bias）—— 對大 case 可能更準

---

## 8. 快速 troubleshooting

| 症狀 | 可能原因 | 怎麼修 |
|---|---|---|
| `[my_optimizer_ml] ML predictor: disabled` | 找不到 weight 檔 | 確認 `ml/weights/floorplan_v1.pt` 存在；或設 `FLOORPLANNER_ML_WEIGHTS` |
| `ImportError: No module named 'ml'` | 沒把 `ml/` 整個目錄 cp 過去 | `cp -r /home/pop/2026_EDA_contest/ml /home/pop/IntelLabs_Floorset/FloorSet/iccad2026contest/` |
| 訓練 loss 一直 = NaN | 學習率太大或 dataset 有壞 case | `--lr 1e-4`；或讀 dataset 時加 `try/except` skip 壞 case |
| 推論時間 > 1 秒 | 模型在 CPU 但 case 太大 | `FLOORPLANNER_ML_DEVICE=cuda`；或減 `--hidden / --layers` 重訓 |
| `predictor` 預測位置全是 0 | 模型剛初始化，沒訓練 / weight 被覆寫 | 看 train.py 印的 `[train] saved` 訊息確認有存 checkpoint |
| C++ solver 因為 WARM_POSITIONS section 出錯 | parser.cpp 沒有加 §4 的 hook 但 strict-mode 開啟 | parser.cpp 預設會忽略未知 section；如果開了 strict，加 §4 patch |

---

## 9. 跟我的分工

| 工作 | 屬於 |
|---|---|
| ML 模型訓練、推論、weights | 我 |
| `my_optimizer_ml.py` Python 端整合 | 我 |
| C++ 端 WARM_POSITIONS reader（§4） | 我隊友 |
| `src/cost.hpp/cpp`、`src/sa.hpp/cpp` 演算法優化 | 我 |
| `src/moves.cpp`、`src/packer.cpp`、`src/parallel.cpp` | 我隊友 |

—— 我們**不會改到同一個檔案**，merge 衝突風險低。
