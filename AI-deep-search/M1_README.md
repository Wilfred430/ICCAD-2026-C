# M1 — 建構式自回歸 placement（監督模仿 + 合法性遮罩）

`electro/NEXT_STEPS.md` 第 5.1 節的實作。目標：ML 一次 rollout 直接輸出**零重疊、
接近 GT** 的排版，legalizer 只需微調；推論次秒級（n≤120 次小 transformer forward）。

## 架構（v1）

```
netlist tensors ──► prep_case()：canonical order（preplaced 先、面積降冪、度數降冪）
                    每塊 26 維 token（靜態 17 + 動態 9：placed 幾何、wire-pull…）
                              │
                    M1Net：4 層 TransformerEncoder (d=192)，MAX_N=128 靜態形狀
                              │
              ┌── position head：32×32 格點 cell logits（塊的左下角）
              └── aspect head：9 個 log-aspect bins（soft 塊；面積永遠精確）
推論 rollout：每步 → aspect → 精確幾何合法性遮罩(擋重疊/出框) → masked argmax
              → snap 到最近鄰接邊（找回被量化吃掉的咬合）→ 放置 → 下一塊
```

- **訓練 = 純監督模仿（teacher forcing）**：把 1M GT 拆成「部分排版(GT 位置) → 下一塊的
  GT cell + GT aspect bin」步樣本。CE loss，無 RL，無 reward engineering。
- **M3-probe 教訓已內建**：格點是自由 (x,y) 槽位（MaskPlace 式），不是 left/bottom
  contour 規則 —— 後者被證明無法重現 GT 的咬合拼磚（area +40%）。
- **preplaced** 直接放在 target 位置當上下文；**fixed** 尺寸鎖定只預測位置；
  **MIB** 群組共用第一個成員的 aspect bin（同形狀 by construction）。
- 遮罩在**推論**時套（訓練不用：GT cell 在 GT 上下文必合法）；死角時畫布放大 10% 重試,
  仍失敗回傳 None → 呼叫端 fallback（嚴格加法契約）。

## 檔案

| 檔案 | 內容 |
|---|---|
| `ml/m1_common.py` | 順序/格點/bin/遮罩/snap/token —— **train 與 infer 共用,改了就要重訓** |
| `ml/m1_model.py` | M1Net |
| `ml/m1_dataset.py` | (case, step) teacher-forcing Dataset(直接讀 1M .th 檔) |
| `ml/m1_train.py` | 訓練 CLI |
| `ml/m1_infer.py` | M1Predictor rollout |
| `electro/electro_optimizer.py::_m1_candidate` | 接進 solve() 當額外候選 |
| `electro/smoke_m1.sh` | 端到端冒煙(真評測器) |

## 如何訓練

```bash
cd ~/2026_EDA_contest
PY=/home/pop/IntelLabs_Floorset/FloorSet/venv/bin/python   # 要有 torch 的 venv

# 1) 冒煙（分鐘級，只驗管線）
$PY -m ml.m1_train --data-root ~/IntelLabs_Floorset/FloorSet/floorset_train_data \
    --max-cases 300 --epochs 2 --out ml/weights/m1_smoke.pt

# 2) CPU 中量（小時級；已驗證 3000 case 可跑）
$PY -m ml.m1_train --data-root ~/IntelLabs_Floorset/FloorSet/floorset_train_data \
    --max-cases 3000 --epochs 2 --bs 64 --out ml/weights/m1_v1.pt

# 3) 正式（GPU，建議規模；lab 機器/colab）
$PY -m ml.m1_train --data-root <train_data> \
    --max-cases 100000 --epochs 3 --bs 256 --lr 3e-4 --device cuda \
    --out ml/weights/m1_v1.pt          # --resume 可接續
```

**看什麼指標**（每 epoch 印 train/val）：
- `pos`：cell CE loss（隨機 = ln(1024) ≈ 6.93）。
- `acc`：cell 全中率;`near`：**±1 cell 內**命中率 —— 這才是實效指標,snap pass 能吸收
  一格內的誤差。目標:`near` > 0.5 開始有實用價值,> 0.7 應可穩定貢獻。
- `asp_acc`：aspect bin 命中率(隨機 ≈ 0.11)。
- val 用資料集前 `--val-cases` 個 case(train 從其後取樣,不重疊)。

**擴大規模的順序**:max-cases 3k → 30k → 100k+;d_model 192 → 256;epochs 2 → 3-5。
1M 全用不必要,~10 萬 case × 60 步 = 600 萬步樣本已很充足。

## 如何使用

```bash
# 單獨評測 M1 候選路徑（真評測器；weights 預設找 ml/weights/m1_v1.pt）
ELECTRO_M1=1 python iccad2026_evaluate.py --evaluate electro/electro_optimizer.py --test-id 0

# 指定權重
ELECTRO_M1=1 ELECTRO_M1_WEIGHTS=/path/to/m1_v1.pt python iccad2026_evaluate.py ...

# 快速冒煙（tid 任選）
bash electro/smoke_m1.sh 40
```

整合方式:`solve()` 裡 M1 rollout 產生一個**額外候選**(以及它的 compact+shape 變體,
若 `ELECTRO_COMPACT=1`),與 electro 的解同台被 `exp(2·V_rel)`-aware 排名挑選 ——
**只在淨值更好時才被採用,永遠不會讓結果變差**。權重不存在/rollout 失敗 → 自動 fallback。
預設 `ELECTRO_M1=0`(未驗證的權重不進提交路徑);等 full-100 驗證贏了再翻預設(比照
ELECTRO_COMPACT 的流程:subset 篩選 → full-100 確認 → 翻 `os.environ.setdefault`)。

## 已驗證(2026-07-07)

- 管線端到端通:1,008,000 case 偵測、步樣本、訓練、存檔、rollout、評測器整合。
- 冒煙權重(60 case,near≈0.02,近隨機)下:rollout 無死角、零重疊、
  tid0 cost 維持 1.871 —— **爛候選被排名正確拒絕,嚴格加法契約成立**。

## 風險與下一步

1. **逐步誤差累積**(大 n):若 near 高但 rollout 品質差 → 加「scheduled sampling」
   (訓練時部分步驟用模型自己的預測當上下文)。
2. **量化誤差**:32×32 格對大 die 一格 ~5 單位;若成瓶頸 → G=48/64 或加 sub-cell offset head。
3. **canvas 失配**:訓練用 GT bbox、推論用 total_area/0.96 估計;若敏感 → 推論時多 util
   候選(0.90/0.96/1.0)各 rollout 一次,一起進排名(便宜,反正每 rollout 次秒級)。
4. **V_rel**:boundary/grouping 已進特徵但無硬保證 → rollout 後接現有
   grouping_repair/boundary_snap(已在 pipeline)。
