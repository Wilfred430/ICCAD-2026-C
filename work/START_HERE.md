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

## Step 3 — 本機 smoke test [3 分鐘]

```bash
make check
```

預期看到：

```
./floorplanner benchmarks/toy.txt benchmarks/toy.sol --time 5 --threads 4 --verbose
[load] benchmarks/toy.txt: n_blocks=6 n_terminals=4 ...
[SA] thread 0: ... cost=...
...
[main] best thread=X feasible=1 contest_cost=1.00...
```

✅ `feasible=1` 而且 `contest_cost ≈ 1.00` 就成功了。

驗證 cost 真的對（用獨立的 Python 重算一次）：

```bash
python3 tools/verify_solution.py benchmarks/toy.txt benchmarks/toy.sol
```

預期看到 `feasible: True` 跟 `contest_cost (rf=1): 1.00...`，跟 C++ 那邊**對得上**（差距 < 0.001）。

✅ 兩邊一致 → 你的 C++ 跟 Python 算的是同一個 v9 cost，沒走偏。

❌ **如果不一致**：C++ cost.cpp 跟 Python verify_solution.py 對 v9 公式的理解有差。先改到一致再繼續，否則接下來在 100-case 跑出來的分數沒意義。

---

## Step 4 — 拉官方比賽框架 + 下載 dataset [15-30 分鐘]

```bash
# 找一個工作目錄
cd ~
git clone https://github.com/IntelLabs/FloorSet.git
cd FloorSet
```

**先確認系統有 pip 跟 venv 支援**（Ubuntu/Debian 常常沒預裝，跳過這步會導致 venv 裡沒有 pip）：

```bash
sudo apt install python3-pip python3-venv -y
```

**建 venv 並啟動**：

```bash
python3 -m venv venv --clear    # --clear 確保乾淨狀態
source venv/bin/activate         # bash/zsh
# 如果你用 csh/tcsh: source venv/bin/activate.csh

# 驗證 pip 是 venv 裡的，不是系統的
which pip    # 預期看到：.../FloorSet/venv/bin/pip
```

❌ **如果 `which pip` 印的不是 venv 路徑**，或看到 `pip not found`：
```bash
# 用 python -m pip 代替
python -m pip install --upgrade pip
```

**裝相依套件**：

```bash
pip install -r iccad2026contest/requirements.txt
# 包含：torch>=2.0, numpy>=1.24, shapely>=2.0, matplotlib, tqdm, requests
```

✅ `pip` 跑完沒紅字就過關。

> **注意**：venv 啟動後，`python` 指令就是 python3，後面所有步驟都用 `python`（不用 `python3`）。如果忘記啟動 venv，`python` 就會找不到。

接著下載 100 個 validation case（dataloader 第一次用會自動從 Hugging Face 抓，約 15 MB）：

```bash
cd iccad2026contest
python -c "
from iccad2026_evaluate import get_validation_dataloader
dl = get_validation_dataloader()
print(f'OK: {len(dl)} validation cases loaded')
"
```

✅ 看到 `OK: 100 validation cases loaded` 就成功了。

❌ **如果失敗**：
- `ModuleNotFoundError: No module named 'numpy'`（或其他套件）→ venv 沒啟動，或套件裝到系統而不是 venv。先 `deactivate`，刪掉舊 venv（`rm -rf venv`），重新照上面步驟建。
- `pip not found` 但 venv 已啟動 → `sudo apt install python3-pip -y`，然後重建 venv
- HuggingFace 連不上 → 公司/學校防火牆，換網路或設 proxy
- `torch` 裝不起來 → 試 `pip install torch --index-url https://download.pytorch.org/whl/cpu`（CPU 版較小）
- `shapely` 裝失敗 → `sudo apt install libgeos-dev` 後重試

---

## Step 5 — 把我們的 optimizer 放進去 [1 分鐘]

回到 floorplanner 專案目錄，包一份 submission 出來：

```bash
cd <你的 floorplanner 專案>
make submit
```

會產出 `submit/floorplanner_submission.zip`（約 50-100 KB），裡面只有兩個檔：`my_optimizer.py` + `floorplanner`。

放到比賽資料夾：

```bash
cp submit/floorplanner_submission/my_optimizer.py ~/FloorSet/iccad2026contest/
cp submit/floorplanner_submission/floorplanner    ~/FloorSet/iccad2026contest/
chmod +x ~/FloorSet/iccad2026contest/floorplanner
```

✅ `ls ~/FloorSet/iccad2026contest/` 應該看到：
```
my_optimizer.py
floorplanner       ← 可執行
iccad2026_evaluate.py
optimizer_template.py
README.md
... 其他官方檔案
```

---

## Step 6 — 驗證格式 [1 分鐘]

```bash
cd ~/FloorSet
source venv/bin/activate    # 確認在 venv 裡（prompt 前面應該有 (venv)）
cd iccad2026contest

python iccad2026_evaluate.py --validate my_optimizer.py
```

> **⚠️ 每次開新 terminal 都要重新 `source venv/bin/activate`**，否則 numpy 等套件找不到。

預期看到：

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

✅ 看到 `PASSED` 就過關。

❌ **如果失敗**：
- `No optimizer class found` → my_optimizer.py 沒被正確放進去，或檔內 class 名稱錯
- `solver binary not found` → `floorplanner` 沒放好或沒 `chmod +x`
- `Sample runtime` 超過 10 秒 → C++ 可能在 5 個 block 的小 case 上花太久，把 `FLOORPLANNER_TIME='5+0.5*n'` 設小一點測試

---

## Step 7 — 跑單一 case + 跑全部 [30-60 分鐘]

### 先跑 1 個 case，看看分數
```bash
python iccad2026_evaluate.py --evaluate my_optimizer.py --test-id 0 --verbose
```

預期看到（重點欄位）：

```
[my_optimizer] case 0: n=21 budget=15.5s
... (solver 跑大概 15 秒)
EVALUATION RESULTS: my_optimizer
====================================================
Total Score: 1.XXXX        ← 越接近 1.0 越好
Tests: 1
Feasible: 1                ← 必須是 1（feasible）
```

✅ `Feasible: 1` 而且 `Total Score < 5` 就好了，先繼續跑全部。

❌ 看到 `Total Score: 10.0`（infeasible）→ 跳到下面「常見問題」第 1 條。

### 再跑前 10 個 case 確認穩定
```bash
for i in 0 1 2 3 4 5 6 7 8 9; do
    python iccad2026_evaluate.py --evaluate my_optimizer.py --test-id $i 2>&1 | grep -E "Total Score|Feasible"
done
```

預期：每個 case 都印 `Feasible: 1` 而且 `Total Score < 5`。

如果有任何一個 infeasible，先 debug 那個 case，不要急著跑全部 100 個。

### 跑全部 100 個 case
```bash
python iccad2026_evaluate.py --evaluate my_optimizer.py --save-solutions
```

預期跑 30-60 分鐘（依機器跟你設的 time budget 而定）。中間會看到 tqdm 進度條。

跑完會印：

```
EVALUATION RESULTS: my_optimizer
====================================================
Total Score: 1.XX
Tests: 100
Feasible: ≥95            ← 越多越好，最好 100
Avg Cost: 1.XX
Avg Runtime: XX.XXs

Results saved to my_optimizer_results.json
Solutions saved to my_optimizer_solutions.json
```

✅ `Total Score < 2.0` 是合理的初步目標。<1.5 算不錯。<1.2 已經很好。

❌ **如果 Feasible 數量不到 90**：時間預算太緊或某幾個 case 有 bug，跳到「常見問題」第 2 條。

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

## 文件導覽（其他 .md 是什麼）

| 檔案 | 什麼時候讀 |
|---|---|
| **本檔** | 第一次來、想知道從 0 到提交的步驟 |
| `README.md` | 專案總覽、code 架構 |
| `SUBMISSION.md` | 想知道為什麼這樣接、constraint 編碼細節 |
| `EVALUATION.md` | 跑完 baseline 想知道下一步怎麼提升分數 |
| `CLAUDE.md` | 給 AI 助理或新隊友看的內部開發筆記 |

如果你只想趕在 5/26 alpha-test 前交出第一版，**讀完本檔就夠了**，其他可以晚點再翻。