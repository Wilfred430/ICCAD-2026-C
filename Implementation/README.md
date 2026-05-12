# FloorSet ICCAD 2026 Contest — Team Submission

PARSAC-style B*-tree + Fast-SA floorplanner for the ICCAD 2026 Contest C
(FloorSet-Lite). Implementation: ~2100 lines of C++17, plus a thin Python
wrapper that plugs into the official contest framework.

## 👉 第一次來？讀這個

**[`START_HERE.md`](START_HERE.md)** — 從零到提交的完整 8 步驟懶人包。
照著做就會有一份能交件的 submission。

## 已經熟了？常用指令

```bash
make static                                      # 編譯靜態 C++ 執行檔
bash tools/run_smoke_test.sh                     # 自動化煙霧測試 (比對 C++ 與 Python Cost)
make sync                                        # 同步 binary 與 wrapper 到官方目錄
make eval                                        # 一鍵執行官方評估 (100 cases)
make viz                                         # 執行視覺化並將圖表抓回 plots/
```

## 架構

```
官方框架 iccad2026_evaluate.py
        │ importlib
        ▼
my_optimizer.py  ← Python wrapper（~300 行）
        │ subprocess
        ▼
floorplanner  ← C++ binary（~2100 行）
   - PARSAC anchored-blocks B*-tree
   - Fast-SA, multi-thread, multi-seed
   - v9 cost: centroid-Manhattan HPWL + bbox area + V_relative
```

要交件的就是 **`my_optimizer.py` + `floorplanner`** 兩個檔，`make submit`
會打包成 `submit/floorplanner_submission.zip`。

## 文件導覽

| 檔案 | 用途 |
|---|---|
| **`START_HERE.md`**  | **🔰 從 0 到提交的完整步驟（懶人包）** |
| `README.md`          | 你正在讀的檔，專案總覽 |
| `SUBMISSION.md`      | 整合層細節：constraint 編碼、為什麼這樣接、deployment |
| `EVALUATION.md`      | 方法評估、優缺點、ML 擴充建議 |
| `CLAUDE.md`          | 給 AI 助理 / 新隊友的內部 handover 筆記 |

## 專案結構

```
floorplanner/
├── START_HERE.md          ← 從這裡開始
├── README.md              ← 本檔
├── SUBMISSION.md          ← 整合協定
├── EVALUATION.md          ← 方法評估
├── CLAUDE.md              ← 內部筆記
│
├── Makefile               ← make static / sync / eval / viz / submit
├── my_optimizer.py        ← 要交件的 Python wrapper
│
├── include/               ← C++ headers
├── src/                   ← C++ implementations
├── tools/                 ← run_smoke_test.sh, verify_solution.py
├── benchmarks/            ← toy.txt, toy.sol
├── logs/                  ← activity_log.md (開發紀錄)
├── plots/                 ← 由 make viz 產生的擺放圖
└── submit/                ← 由 make submit 產生的 zip
```

## 效能調優與環境變數

您可以透過設定環境變數來榨乾 32 核 CPU 的效能，不需重新編譯：

| 變數 | 說明 | 推薦值 (您的機器) |
|---|---|---|
| `FLOORPLANNER_THREADS` | 同時跑幾個種子並行運算 | `30` (由 `nproc` 決定) |
| `FLOORPLANNER_TIME` | 每個案例的時間預算 (秒) | `'5+0.5*n'` |
| `FLOORPLANNER_SEED` | 隨機種子 (用於重現結果) | `42` |

**使用範例：**
```bash
FLOORPLANNER_THREADS=30 FLOORPLANNER_TIME='10+0.8*n' make eval
```

## Build & test

```bash
make static                   # 推薦：static-linked binary
bash tools/run_smoke_test.sh  # 比對 C++ 與 Python Cost
make eval-quick               # 快速抽樣評估 (25 cases)
make report                   # 生成 v9 詳細評分報告
make viz                      # 生成佈局視覺化圖片 (plots/)
```

獨立 Python verifier (`verify_solution.py`) 重算 v9 cost；C++ 跟它應該
一致到小數點 3 位以內。如果不一致表示我們在優化錯誤的目標——立刻修。

## 關鍵 v9 spec 提醒

- HPWL = **centroid-to-centroid Manhattan**（不是 bbox half-perimeter）
- Fixed / preplaced 是 **HARD constraint**（v9 在 4/19 改的）
- Soft block 面積容差 = **1%**
- Boundary 是 **bitmask**（1=L, 2=R, 4=T, 8=B），不是循序 enum
- Cost = `(1 + 0.5·(HPWL_gap + Area_gap)) · exp(2·V_rel) · max(0.7, RT^0.3)`
- Total score 對 case size **指數加權**（n=120 比 n=21 重 8×10⁴² 倍）

完整細節跟 source code 引用見 `CLAUDE.md` 的「v9 gotchas」段。

## 下一步該做什麼

如果你還沒提交過任何東西 → 讀 **`START_HERE.md`**

如果已經提交但分數想再往上 → 讀 **`EVALUATION.md`** 的 Phase 2/3
（GNN move proposer + diffusion model post-processing）
