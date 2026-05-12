# FloorSet ICCAD 2026 Contest — Team Submission

PARSAC-style B*-tree + Fast-SA floorplanner for the ICCAD 2026 Contest C
(FloorSet-Lite). Implementation: ~2100 lines of C++17, plus a thin Python
wrapper that plugs into the official contest framework.

## 👉 第一次來？讀這個

**[`START_HERE.md`](START_HERE.md)** — 從零到提交的完整 8 步驟懶人包。
照著做就會有一份能交件的 submission。

## 已經熟了？常用指令

```bash
make static                                      # build portable C++ binary
make check                                        # smoke test on toy benchmark
make submit                                       # package my_optimizer.py + binary

# In FloorSet/iccad2026contest/ (after copying my_optimizer.py + floorplanner there):
python iccad2026_evaluate.py --validate my_optimizer.py
python iccad2026_evaluate.py --evaluate my_optimizer.py --save-solutions
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
├── Makefile               ← make / make static / make check / make submit
├── my_optimizer.py        ← 要交件的 Python wrapper
│
├── include/               ← C++ headers
│   ├── types.hpp           Block, Net, FloorplanInstance, BoundaryEdge
│   ├── btree.hpp           B*-tree (indexed array)
│   ├── packer.hpp          contour packer
│   ├── cost.hpp            v9 cost function
│   ├── moves.hpp           SA neighbourhood moves
│   ├── sa.hpp              Fast-SA driver
│   ├── parser.hpp          text format I/O
│   └── parallel.hpp        multi-seed driver
│
├── src/                   ← C++ implementations (~2100 lines total)
│   ├── btree.cpp
│   ├── packer.cpp          contour DFS + anchored-block support
│   ├── cost.cpp            HPWL (centroid), area, V_grouping/V_mib/V_boundary
│   ├── moves.cpp           M1 Rotate, M2 Move, M3 Swap, M4 AspectRatio,
│   │                        M5 MibSync, M6 FixBoundary
│   ├── sa.cpp              3-stage Fast-SA, T1 calibrated to p_accept=0.99
│   ├── parallel.cpp        N std::thread chains
│   ├── parser.cpp          text format reader/writer
│   └── main.cpp            CLI
│
├── tools/
│   ├── floorset_to_txt.py  pkl → our text format (standalone helper)
│   └── verify_solution.py  pure-Python v9 cost reimpl. (cross-check)
│
├── benchmarks/
│   ├── toy.txt             6-block synthetic (preplaced + fixed + MIB +
│   │                        grouping + boundary corner — exercises every
│   │                        code path)
│   └── toy.sol             reference output
│
└── submit/                ← created by `make submit`
    └── floorplanner_submission.zip
```

## Build & test

```bash
make static                   # 推薦：static-linked binary，跨機器最穩
make check                    # toy benchmark, 應該看到 contest_cost ≈ 1.00
python3 tools/verify_solution.py benchmarks/toy.txt benchmarks/toy.sol
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
