# 懶人包 — 從 code 到提交的完整流程

**這份文件就是「讀這個就好」的入口。** 照順序跑完八個步驟，你就會有一個能跑、能交件的 submission。每一步都列：要打什麼指令、應該看到什麼、失敗了怎麼辦。

---

## 你現在的位置

| | |
|---|---|
| ✅ | C++ solver code 完整（`include/`、`src/`、`Makefile`） |
| ✅ | Python wrapper 寫好（`my_optimizer.py`，把 C++ 接到官方框架）|
| ✅ | toy benchmark 跑得通 |
| ❌ | 還沒在官方比賽框架（`iccad2026contest/`）裡跑過 |
| ❌ | 還沒包成 submission zip |
| ❌ | 還沒在 100 個 validation case 上看過分數 |

接下來八個步驟總時間約 1–2 小時（其中下載 dataset 佔大部分）。

---

## 整體架構（一張圖看懂）

```
官方框架 iccad2026_evaluate.py  ← 我們不動它
        │ 它會 importlib 載入下面這個檔
        ▼
my_optimizer.py  ← Python wrapper（已經寫好）
        │ subprocess 呼叫下面這個
        ▼
floorplanner  ← 我們的 C++ binary（make 出來的）
```

要交件的就是 **`my_optimizer.py` + `floorplanner` 兩個檔**。`make submit` 會自動打包。

---

## Step 1 — 環境準備 [10 分鐘]

最快確認你的環境齊不齊：

```bash
bash scripts/check_env.sh
```

✅ 看到 `All required tools found` 就跳到 Step 2。

❌ 有缺的話照下面手動裝：

需要的東西：

| 東西 | 怎麼裝 (Ubuntu/Debian) | 怎麼裝 (macOS) |
|---|---|---|
| `g++` ≥ 7（要 C++17） | `sudo apt install build-essential` | `xcode-select --install` |
| `make` | 通常跟 g++ 一起 | 同上 |
| `python3` ≥ 3.8 | `sudo apt install python3 python3-venv` | 內建 |
| `git`、`unzip` | `sudo apt install git unzip` | 內建 |

**Windows 使用者**：請用 WSL2（Windows Subsystem for Linux），或在實驗室的 Linux 機器上跑。我們的 Makefile 跟比賽框架都假設是類 Unix 環境。

驗證你的環境：

```bash
g++ --version       # 預期：g++ (Ubuntu ...) 7.5+ 或更新
python3 --version   # 預期：Python 3.8+
make --version
```

✅ 三個都印得出版本就過關。

---

## Step 2 — Build C++ binary [2 分鐘]

```bash
cd <你放 floorplanner 專案的資料夾>
make clean
make static    # 推薦用 static link，提交時最不會出包
```

預期看到：

```
g++ -std=c++17 -O3 ... -c src/btree.cpp -o src/btree.o
g++ -std=c++17 -O3 ... -c src/packer.cpp -o src/packer.o
... (8 個 cpp 檔)
g++ -std=c++17 -O3 ... -static ... -o floorplanner src/...o
[OK] floorplanner is statically linked.
```

✅ 最後一行有 "is statically linked" 就成功了。

驗證：

```bash
ls -la floorplanner       # 應該看到一個約 3-5 MB 的執行檔
file floorplanner         # 應該包含 "statically linked"
```

❌ **如果失敗**：
- `g++: command not found` → 回 Step 1 裝 build-essential
- 看到 "static linking is not supported" → 改用 `make`（dynamic link 也能跑，只是不能跨機器）

---

## Step 3 — 本機自動化煙霧測試 [3 分鐘]

為了確保您的 C++ 引擎計分邏輯與官方 Python 腳本完全一致，我們提供了一個自動化比對腳本。

```bash
bash tools/run_smoke_test.sh
```

**預期看到：**

```
[1/3] 執行 C++ Floorplanner (5s)...
[main] best thread=0 feasible=1 contest_cost=1.0023
[2/3] 執行 Python 驗證工具...
feasible:   True
contest_cost (rf=1): 1.0023
[3/3] 比對結果...
----------------------------------------------------
C++    Contest Cost: 1.0023
Python Contest Cost: 1.0023
絕對誤差: 0
----------------------------------------------------
✅ 測試通過！C++ 與 Python 邏輯一致。
```

✅ **看到「✅ 測試通過！」**：代表您的 C++ 與 Python 算的是同一個 v9 cost，沒有誤差。

❌ **如果看到「❌ 警告：誤差過大」**：代表 C++ `cost.cpp` 與 Python `verify_solution.py` 對公式的理解有差（差距 > 0.001）。請先修正到一致再繼續，否則後續開發的分數將沒有參考價值。

---

## Step 4 — 設定官方比賽環境 (Conda 版本) [5 分鐘]

您已經在本機有了 `ICCAD-C-FloorSet-official` 目錄，現在使用 Conda 建立環境：

```bash
# 1. 建立 Conda 環境 (推薦 Python 3.10)
conda create -n iccad2026 python=3.10 -y

# 2. 啟用環境
conda activate iccad2026

# 3. 安裝相依套件 (在官方目錄下執行)
cd ../ICCAD-C-FloorSet-official
pip install -r iccad2026contest/requirements.txt
```

✅ **注意**：之後每次開始開發前，請確保已執行 `conda activate iccad2026`。

接著確保 100 個 validation case 已下載（第一次執行評估時會自動抓取）。

---

## Step 5 — 把我們的實作同步到官方目錄 [1 分鐘]

我們已經在 Makefile 中寫好了同步指令，會自動把 `my_optimizer.py` 與 `floorplanner` 複製到正確位置：

```bash
cd ../Implementation
make sync
```

✅ 看到 `[OK] Files synced` 就代表檔案已成功覆蓋官方目錄中的舊版。

---

## Step 6 — 驗證介面格式 [1 分鐘]

在進行大規模評估前，必須先確保您的 `my_optimizer.py` 輸出的格式完全符合比賽要求。我們已經將此步驟自動化：

```bash
make validate
```

**預期看到：**

```
Validating: my_optimizer.py
--------------------------------------------------
  ✓ File exists
  ✓ Valid Python syntax
  ✓ Module loads successfully
  ✓ Contains optimizer class: MyOptimizer
  ✓ Returns correct format
  ✓ Sample runtime: 1.234s
--------------------------------------------------
Result: PASSED
```

✅ **看到 `PASSED`**：代表您的程式介面完全正確。

❌ **如果失敗**：
- `No optimizer class found` → 請檢查 `my_optimizer.py` 內的 Class 名稱。
- `solver binary not found` → `floorplanner` 同步失敗，請重新執行 `make sync`。

---

## Step 7 — 執行全量評估 (多核心加速版) [30 分鐘]

由於您有 32 個核心，我們建議開啟多執行緒並配合通知系統：

```bash
export FLOORPLANNER_THREADS=30
make eval; remind "全量評估完成！"
```

## Step 8 — 快速抽樣評估 (節省時間) [8 分鐘]

如果您正在微調參數，不想等一小時，可以使用抽樣 25 筆案例的快速版本（內建自動通知）：

```bash
export FLOORPLANNER_THREADS=30
make eval-quick; remind "全量評估完成！"
```

## Step 9 — 量化表現與分析

跑完 eval 後，使用此指令獲取符合 v9 規格的詳細評分報告：

```bash
make report
```

這會印出每個案例的 Cost 與最終的 **Total Score**。

---

## Step 8 — 包成最終 submission [1 分鐘]

```bash
cd <你的 floorplanner 專案>
make static                  # 確保是 static linked
make submit                  # 重新打包
ls -la submit/floorplanner_submission.zip
```

✅ 你會拿到一個 `submit/floorplanner_submission.zip`，這就是要交件的東西。

**怎麼交**：根據比賽主辦單位的指示——可能是上傳到 leaderboard 網站、提交 GitHub repo、寄信，等等。這份指示我們沒拿到，要看比賽組委會的最新公告。如果不確定，問老師或主辦方。

---

## ✅ 最終 checklist

提交前最後檢查：

- [ ] `make static` 跑完看到 `[OK] floorplanner is statically linked.`
- [ ] `make check` 在 toy benchmark 上 `feasible=1, contest_cost ≈ 1.0`
- [ ] `tools/verify_solution.py` 跟 C++ 給出相同的 cost
- [ ] `--validate` 印 `PASSED`
- [ ] `--evaluate ... --test-id 0` 印 `Feasible: 1`，cost < 5
- [ ] 前 10 個 case 全 feasible
- [ ] 100-case 全跑 `Feasible: ≥95`，`Total Score < 2.0`
- [ ] `solutions.json` 存好（可重新評分用）
- [ ] `floorplanner_submission.zip` 打包好
- [ ] 確認過比賽主辦單位的提交流程

每項打勾，就可以交件。

---

## 🚨 常見問題與排錯

### 1. `Total Score: 10.0` / `Feasible: 0`（infeasible）

最常見的三個原因，依優先順序排查：

**(a) 解有 overlap**（最常見）
- 在 my_optimizer.py 跑時加 `FLOORPLANNER_KEEP=1` 環境變數，保留中間檔
- 找出 case 對應的 `.txt` 跟 `.sol`（在 `/tmp/my_optimizer_*/case_NNN.{txt,sol}`）
- 用 `tools/verify_solution.py` 重算，會印 `overlap=N`
- 通常表示 SA 沒收斂到 feasible solution → 加大 `FLOORPLANNER_TIME`

**(b) Fixed/preplaced 尺寸沒鎖住**
- v9 把這兩種改成 hard constraint，dimension 不對就 cost=10
- 檢查 my_optimizer.py 寫出的 `.txt` 中，fixed/preplaced 的 `w_in h_in x_in y_in` 欄位是否正確
- 然後檢查 C++ packer 有沒有真的鎖住（看 src/packer.cpp）

**(c) Soft block 面積偏差超過 1%**
- v9 規定 `|w*h - target| / target ≤ 0.01`
- 我們的 SA 有 `area_within_1pct` 的 hard check，正常不會發生
- 如果發生，可能是 SA 在 area aspect-ratio move 時超出容忍

### 2. `Feasible: <90` 但個別 case OK

通常是 time budget 不夠。試：
```bash
FLOORPLANNER_TIME='10+0.8*n' python iccad2026_evaluate.py --evaluate my_optimizer.py
```

或者增加 thread 數：
```bash
FLOORPLANNER_THREADS=12 python iccad2026_evaluate.py --evaluate my_optimizer.py
```

### 3. `solver binary not found`

my_optimizer.py 找不到 `floorplanner` 執行檔。三個檢查：
```bash
# 是否在同一個資料夾？
ls ~/FloorSet/iccad2026contest/floorplanner

# 是否可執行？
ls -la ~/FloorSet/iccad2026contest/floorplanner    # 要看到 -rwxr-xr-x

# 直接跑得起來嗎？
~/FloorSet/iccad2026contest/floorplanner --help
```

或用環境變數明確指定：
```bash
FLOORPLANNER_BIN=/abs/path/to/floorplanner python iccad2026_evaluate.py --evaluate my_optimizer.py
```

### 4. 在 server 上跑，但本機 build 的 binary 跑不起來

通常是 glibc 版本不一致。回到本機重新 `make static`，再傳到 server。如果還不行，問 server 管理員 glibc 版本，在相容的環境（例如同一個 Ubuntu 版本的 docker）裡 build。

### 5. `--validate` 通過但 `--evaluate` 失敗

`--validate` 只是用 5 個 dummy block 試呼叫 `solve()`，不檢查實際的 100 個 case。失敗訊息看 stderr 通常會告訴你哪個 case 出問題。

### 6. 跑很慢，跑不完 100 case

合理的時間是 30-90 分鐘。如果超過：
- 把 `FLOORPLANNER_TIME` 設小一點（預設 `5+0.5*n`，可改 `3+0.3*n`）
- 增加 thread 數
- 先用 `--test-id 99`（最大的 case）測單一 case 的時間，再外推

### 7. 拿到 Total Score 但想再優化

- 看 `my_optimizer_results.json`，找出 cost 最高的幾個 case
- 對那幾個 case 個別跑 `--test-id N --verbose`，看是 hpwl_gap 大還是 area_gap 大還是 V_relative 大
- 對應調整 SA 參數（在 `include/sa.hpp` 的 SAConfig 預設值，或開 CLI flag）

完整的「下一步要往哪走」見 `EVALUATION.md`。

---

## Step 9 — 視覺化與開發紀錄

### 視覺化結果
如果您想看擺放結果圖，執行：

```bash
make viz
```

這會執行官方視覺化工具，並將產生的圖表自動複製到 `Implementation/plots/` 資料夾下供您查看。

### 維護開發日誌 (Log)
為了節省 Token 並幫助 AI 記憶，請養成在 **`logs/activity_log.md`** 紀錄每日進展的習慣。
這能讓您在下次開啟新對話時，快速讓 AI 掌握當前的開發狀態。

| 檔案 | 什麼時候讀 |
|---|---|
| **本檔** | 第一次來、想知道從 0 到提交的步驟 |
| `README.md` | 專案總覽、code 架構 |
| `SUBMISSION.md` | 想知道為什麼這樣接、constraint 編碼細節 |
| `EVALUATION.md` | 跑完 baseline 想知道下一步怎麼提升分數 |
| `CLAUDE.md` | 給 AI 助理或新隊友看的內部開發筆記 |

如果你只想趕在 5/26 alpha-test 前交出第一版，**讀完本檔就夠了**，其他可以晚點再翻。
