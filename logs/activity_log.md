# ICCAD 2026 EDA 開發日誌

這份文件記錄了開發過程中的重要進展、環境變動與決策，用於幫助 AI 助理快速恢復上下文並節省 Token。

## 2026-05-05 (今日進展)

### 1. 環境初始化與修復
- **WSL 設置**：成功在 WSL2 (Ubuntu) 環境中啟用開發。
- **目錄結構修復**：修正了 `Implementation` 目錄結構。將原始碼移至 `src/`，標頭檔移至 `include/`，測試案例移至 `benchmarks/`。
- **編譯成功**：成功使用 `make static` 編譯出靜態連結的 `floorplanner` 執行檔。

### 2. 自動化與整合
- **硬體優化**：偵測到 32 核心強大性能，已在文件與指令中加入 `FLOORPLANNER_THREADS=30` 的建議，可顯著提升 SA 收斂機率。
- **環境遷移**：將環境管理工具從 `venv` 遷移至 **Conda** (環境名稱：`iccad2026`)。
- **煙霧測試 (Smoke Test)**：新增了 `tools/run_smoke_test.sh`，可自動比對 C++ 核心與 Python 驗證工具的 Cost 誤差。目前驗證誤差為 0，邏輯一致。
- **Makefile 強化**：
    - 新增 `make sync`：自動將執行檔與 Wrapper 同步至官方框架。
    - 新增 `make eval`：一鍵執行 100 個案例的官方評估。
    - 新增 `make viz`：自動執行視覺化並將圖表抓回 `Implementation/plots/`。
    - **Conda 相容性**：更新 `OFFICIAL_PYTHON` 邏輯，完美支援已啟用的 Conda 環境。
- **文件更新**：更新了 `START_HERE.md` 與 `README.md`，將手動步驟替換為自動化指令，並加入多核心加速說明。

### 4. 演算法重大升級 (SDB-AP & CA-SA 實作)
- **自適應外框懲罰 (SDB-AP)**：
    - 在 `src/cost.cpp` 引入長寬比懲罰（Aspect Ratio Penalty）與溢出懲罰（Overflow Penalty）。
    - 實作了 **「自適應懲罰塑造 (Adaptive Penalty Shaping)」**：若佈局持續超標，`w_outline` 權重會自動以 1.4 倍率遞增，強制 SA 收斂至合法解。
- **約束感知模擬退火 (CA-SA)**：
    - 在 `src/moves.cpp` 實作動態移動機率調度。隨溫度下降，自動將 `P_FIX`（修復邊界機率）從 0.0005 提升至最高 10% 以上。
    - **溢出感知移動 (Overflow-Aware Move)**：當偵測到佈局溢出時，演算法有 70% 機率優先選中「違規區塊」進行位置重排，顯著加快了可行解（Feasible Solution）的發現速度。
- **環境兼容性修復**：
    - **WSL 連結修復**：解決了 Windows 下 MinGW 報錯 `ld 116` 的問題。目前策略為：在 WSL 下編譯 Linux ELF，並透過 `my_optimizer.py` 的自動路徑轉換邏輯，在 Windows Python 中調用 `wsl floorplanner`。
    - **Makefile 優化**：將 `OFFICIAL_PYTHON` 鎖定為 `conda run -n iccad2026_c python`，解決了環境變數與 `shapely` 套件缺失問題。

### 5. 當前狀態
- **核心功能**：已具備 SOTA（State-of-the-art）傳統演算法核心。
- **評分表現**：初步測試 `case 4` 已成功達成 `is_feasible: true`。
- **待辦事項**：
    - [ ] 等待 `make eval-quick` 完成 25 個案例的評估。
    - [ ] 執行 `make report` 分析加權總分與 HPWL 表現。
    - [ ] 若可行性已穩定，開始降低 `w_outline` 初始值以換取更好的 HPWL (連線長度)。
    - [ ] 考慮實作軟塊尺寸的「離散化採樣 (Discretization)」以減少面積誤差。
