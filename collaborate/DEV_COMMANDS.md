# 開發常用指令（生成式 B*-tree + electro 兩條路線）

本文件整理 `ml/` 底下這條生成式 B*-tree pipeline，以及 `electro_optimized/`
電靜力法路線，開發時實際會用到的指令，讓你可以自己重跑、自己觀察，不用每次
都靠 AI 代跑。全部指令都假設工作目錄是 `collaborate/`（即 `cd collaborate`
之後再執行）。第 1–5 節是生成式 B*-tree 路線，第 6 節是 electro 路線。

Python 直譯器：本機是 `/c/Python313/python`（Git Bash）或對應的
`C:\Python313\python.exe`（PowerShell）。以下範例用 `python` 代稱，實際執行時
換成你環境裡正確的路徑。

---

## 1. 訓練 / 微調拓樸模型（`ml/train_tree.py`）

從頭訓練或從既有 checkpoint 暖啟動微調 `TreeGenerator`（block-selection +
parent-pointer + direction 三個 pointer network）。

```bash
python -m ml.train_tree \
  --data "d:/ICCAD-2026-C/ICCAD-C-FloorSet-official" \
  --out ml/weights/tree_v3.pt \
  --limit-cases 300000 --epochs 4 --batch 64 --lr 3e-4 \
  --size-power 2.0 --device cuda --workers 0
```

常用參數：

| 參數 | 說明 | 目前用過的值 |
|---|---|---|
| `--data` | FloorSet 資料根目錄（含 `floorset_lite/`） | 固定路徑 |
| `--out` | 輸出 checkpoint 路徑 | `ml/weights/tree_v{N}.pt` |
| `--init-from` | 從既有 checkpoint 暖啟動（微調而非從頭訓練） | `ml/weights/tree_v1.pt` |
| `--limit-cases` | 用訓練集前 N 筆（1M 資料集很大，通常不用全部） | 150000（v1）/ 300000（v2） |
| `--epochs` | 訓練輪數 | 3（v1）/ 4（v2） |
| `--size-power` | 大 case 加權（loss 乘上 `n**size_power`），因為 Total Score 由大 case 主導 | 2.0（v2 才加，v1 沒有） |
| `--device` | `cuda` 或 `cpu` | `cuda`（GPU 訓練，數小時） |
| `--hidden` / `--ctx-layers` / `--dec-layers` / `--heads` | 模型容量 | 目前 checkpoint 是 hidden=256、6+6 層、8 heads、6.8M 參數 |

訓練過程會印每個 epoch 的 `train_loss` / `val_loss` / `val_b_acc`
（block-selection 準確率）/ `val_p_acc`（parent-pointer 準確率），並在每個
epoch 結束後存檔（若 val_loss 有進步）。

**目前已有的 checkpoint**：
- `ml/weights/tree_v1.pt`：150k 筆 × 3 epoch，`val_ptr_acc` 0.874。
- `ml/weights/tree_v2.pt`：從 v1 暖啟動 + 300k 筆 × 4 epoch + 大 case 加權，
  `val_ptr_acc` 0.879（**目前最佳、預設應該用這個**）。

---

## 2. 快速單一 case 展示（`ml/run_pipeline.py`）

一鍵訓練（若 checkpoint 不存在）+ 對單一 case 取樣多個拓樸 + 打包 + 印出
排名表 + 寫出 `.sol` 格式的最佳解。適合快速肉眼檢查一個 case 的行為，不是
用來衡量整體分數的工具（那是下面的 `eval_full.py`）。

```bash
python -m ml.run_pipeline                        # 預設 case 0，8 個樣本
python -m ml.run_pipeline --case 5 --samples 16  # 換一個 case、取更多樣本
python -m ml.run_pipeline --retrain --train-cases 20000 --epochs 5  # 真正重訓
```

---

## 3. 完整 100-case 品質驗證（`ml/eval_full.py`）

**這是衡量「現在的 pipeline 到底多好」的主工具**，跑官方 cost 公式（含
`max(0,·)` clamp）在全部 100 個驗證 case 上，比較「BEFORE：純方形 soft
block」vs「AFTER：長寬比 sweep + push_past on/off portfolio + HPWL 微調」，
印出 feasible 率、平均 cost、以及 spec 的 `e^(n/12)` 加權 Total Score。

```bash
python -m ml.eval_full --weights ml/weights/tree_v2.pt --samples 4 --limit 100 --device cuda
```

| 參數 | 說明 | 常用值 |
|---|---|---|
| `--weights` | 拓樸模型 checkpoint | `ml/weights/tree_v2.pt`（目前最佳） |
| `--samples` | 每個 case 取樣幾個拓樸（取 cost 最低的） | 4（正式驗證）/ 1-2（debug 用快速跑） |
| `--limit` | 只跑前 N 個 case（debug 用，正式驗證要 100） | 100 |
| `--device` | `cuda`（GPU 生成拓樸，較快）或 `cpu` | 兩者皆可，打包本身是純 CPU |

**耗時參考**（實測）：`samples=4, limit=100` 在 GPU 上約 **80-90 分鐘**
（~50秒/case）；`samples=2, device=cpu` 約 **40 分鐘**（~25秒/case，適合
debug 用的壓力測試，跑得動但不追求最高品質）。

輸出範例：
```
metric                           BEFORE (square) AFTER (shape-opt)
feasible cases                            100/100            100/100
mean Cost (unweighted)                    4.0799            3.1921
Total Score  e^(n/12) weighted            4.1695            3.3185
```

---

## 4. V_rel（軟約束）來源診斷（`ml/diag_vrel.py`）

當 Total Score 卡住不降時，用這個工具先搞清楚「違規主要是 grouping／MIB／
boundary 哪一種」，再決定要修哪裡——不要在不知道主導項是誰的情況下亂改。

```bash
python -m ml.diag_vrel --weights ml/weights/tree_v2.pt --samples 4 --limit 100
```

輸出每種違規的總數、幾個 case 有這種違規，以及 V_rel 最糟的 12 個 case
明細表（`V_rel / Vgrp / Vmib / Vbnd / Nsoft` 逐項列出）。

---

## 5. 逐 case 硬/軟約束 Excel 報告（`ml/case_report.py`）

**你最需要的工具**：每次執行都會重新產生一份 `.xlsx`，讓你自己打開肉眼檢查
每個 case 犯了什麼約束、彙總全部 100 case 的違規次數、runtime、HPWL、area。

```bash
python -m ml.case_report --weights ml/weights/tree_v2.pt --samples 4 --limit 100 --out case_report.xlsx --device cuda
```

輸出的 `case_report.xlsx` 有兩個工作表：

- **Per-Case**：每個 case 一列——case 名稱、n、runtime、4 個硬約束旗標
  （overlap / area / fixed / preplaced，有違規會標紅底）、3 個軟約束計數
  （V_grouping / V_mib / V_boundary，有違規會標紅底）、N_soft、V_relative、
  HPWL_int/ext/gap%、area_gap%、cost。
- **Summary**：feasible 率、每種硬約束違規的 case 數、三種軟約束的**加總**
  次數、總/平均 runtime、平均 HPWL_gap%/area_gap%/cost、`e^(n/12)` 加權
  Total Score。

耗時跟 `eval_full.py` 的 AFTER 欄位同一量級（因為底層是同一條 pipeline）。

---

## 常見疑難排解（生成式 B*-tree 路線）

- **想快速確認程式碼改動沒有壞掉，不想等 80 分鐘**：用
  `--samples 1 --limit 5 --device cpu` 先跑 5 個 case 幾秒鐘看有沒有
  `feasible=False`／crash，通過了再跑 `--samples 2 --limit 100 --device cpu`
  （約 40 分鐘）確認全部 100 case 沒有 edge case，最後才跑
  `--samples 4 --device cuda` 的正式驗證。這是本 session 每次改動
  `pack_tree.py` 都遵守的流程，任何硬約束 bug（overlap/violation）都是這樣
  抓到的，不是靠代碼審查看出來的。
- **GPU 記憶體/其他訓練佔用中**：`eval_full.py`/`case_report.py` 的拓樸生成
  用 GPU（`model.generate`），但打包+修復本身是純 CPU，所以 `--device cpu`
  一樣能跑（只是生成步驟變慢），適合 GPU 被訓練佔用時用來做 CPU 壓力測試。
- **想知道某個改動是不是真的有貢獻**：固定其他變數只換一個（例如同樣的
  `pack_tree.py` 但分別用 `--weights tree_v1.pt` 和 `tree_v2.pt` 各跑一次
  完整 100-case），才能做歸因，不要一次換兩個變數。

---

## 6. electro 路線的驗證工具（`electro_optimized/`）

electro 是 pop 的電靜力法佈局器，2026-07-14 起跟 Antigravity（Gemini 3.5
Flash）協作優化，程式碼在 `collaborate/electro_optimized/`。跟生成式路線
不同，electro 本身跑很快（~2s/case），不需要先跑 CPU 壓力測試再上 GPU 正式
驗證那一套，可以直接跑滿 100 case。

### 6.1 官方框架驗證（含真實 RT，會有量測雜訊——見下方警告）

```bash
cd "d:/ICCAD-2026-C/ICCAD-C-FloorSet-official/iccad2026contest"
python iccad2026_evaluate.py --evaluate "d:/ICCAD-2026-C/collaborate/electro_optimized/electro_optimizer.py"
# 只測單一 case：加 --test-id N（N=0 對應 config_21，N=i 對應 config_(21+i)）
```

> [!warning] **這個指令的 RT（執行時間）是真實牆鐘量測，不是固定值**。
> `RT^0.3` 對變慢沒有封頂，同一份 100% 決定性的程式碼（座標完全相同）連續
> 跑會有 **5-15% 的 Total Score 擺動**，尤其是機器上同時有其他程式在跑
> （例如你自己還開著 Antigravity）。**判斷「這個修法本身有沒有用」不要只跑
> 一次就下結論**，要嘛跑 3 次以上取平均，要嘛改用 6.2 的乾淨版本。詳見
> `CLAUDE.md` gotcha #6a-2。

### 6.2 乾淨版本（Neutral RT，排除雜訊，逐 case Excel 報告）

```bash
python -m ml.case_report_electro --electro-dir "d:/ICCAD-2026-C/collaborate/electro_optimized" --limit 100 --out case_report_electro.xlsx
```

底層用 `ml/contest_cost.py::evaluate()` 的 `runtime_factor=1.0` 預設值
（= neutral RT），**同一份程式碼理論上應該每次都跑出完全一樣的數字**（因為
electro 本身的優化是決定性的，只有 RT 量測會抖動，而這個工具不依賴 RT 量測）。
跟 `ml/case_report.py`（生成式路線用的）格式完全相同，Per-Case 工作表第 102
列有 avg runtime/avg cost/total soft violations 的總結列，可以直接跟
`case_report.xlsx` 並排比較兩條路線。

`--electro-dir` 也可以指到其他 electro 副本（例如
`C:\Users\wende\AppData\Local\Temp\electro_probe\electro` 這種
`git worktree` 唯讀複本），方便同時比較不同版本。

### 6.3 環境變數開關（都是 strictly-additive 的 portfolio 設計，預設不影響原行為）

| 環境變數 | 說明 | 目前建議 |
|---|---|---|
| `ELECTRO_CLAMP` / `ELECTRO_NONNEG` | 保證輸出留在第一象限（提交必要） | 預設 `1`，不要關 |
| `ELECTRO_GROUPING_PUSHPAST` | grouping_repair 的 push-past 候選（找不到空位時退而求其次） | **視 electro_optimized/ 當下版本而定，務必自己重新驗證，不要沿用任何時間點的舊結論**（2026-07-14 晚上這個開關對/錯的結論就翻過兩次，見 Obsidian §8.15/§8.16） |
| `ELECTRO_SEEDS` | 多重起點數量 | 預設 `1`（更多起點分數更好但更慢，RT penalty 通常讓 1 划算） |

### 6.4 疑難排解

- **同一設定連續跑兩次數字不一樣**：先確認是不是用了 6.1（含真實 RT）而不是
  6.2（neutral RT）。如果連 6.2 都不一樣，代表 `electro_optimized/` 底層
  程式碼在兩次執行之間被改過（例如 Antigravity 正在同時編輯），不是隨機性，
  先確認對方是否還在改動，不要對著一個活靶做精細比較。
- **想知道某個改動是不是真的有貢獻**：跟生成式路線一樣，先用 6.2 的乾淨版本
  做 A/B（不要只看 6.1 單次數字），確認有真實改善再考慮要不要設成
  submission-time 預設值（`electro_optimizer.py` 開頭的
  `os.environ.setdefault(...)` 那幾行——這些才是框架真正呼叫 `solve()`
  時會用到的值，不是你手動加的環境變數）。
