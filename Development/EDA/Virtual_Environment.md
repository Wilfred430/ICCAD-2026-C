# EDA 開發虛擬環境建議

## 環境管理工具對比

| 工具 | 底層實作 | 優點 | 缺點 |
| :--- | :--- | :--- | :--- |
| **Conda** | Python | 官方標準、最穩定 | 解析依賴關係較慢 |
| **Mamba** | C++ | 極速解析、並行下載、資源佔用低 | 需要額外安裝 |

> [!TIP]
> 兩者共用相同的環境結構與套件倉庫。若您的環境非常複雜，建議使用 Mamba 以節省等待時間。

## EDA 開發常用套件建議

### 1. 資料處理與科學計算
*   **numpy / pandas**: 處理電路網表（Netlist）與分析數據。
*   **scipy**: 用於數值優化演算法。

### 2. 佈局與幾何處理
*   **gdspy**: 讀寫 GDSII 檔案。
*   **klayout**: 提供 Python API 進行佈局驗證與查看。
*   **pyverilog**: 解析 Verilog 代碼。

### 3. 機器學習（ICCAD 競賽常見）
*   **pytorch / tensorflow**: 用於 ML-based EDA 演算法（如預測佈局熱點）。

## ICCAD 2026 FloorSet 專屬配置 (Conda/Mamba)

如果您使用的是 **Anaconda** 或 **Miniconda**，請使用 `conda` 指令；如果您安裝的是 **Miniforge**，則可使用更快的 `mamba`。

針對 Problem C 競賽，請執行以下步驟建立環境：

```powershell
# 1. 建立並啟用環境 (建議 Python 3.10)
# 若 mamba 無法辨識，請將 mamba 改為 conda
conda create -n iccad2026_c python=3.10 -y
conda activate iccad2026_c

# 2. 安裝 PyTorch (根據是否有 GPU 選擇版本)
# 若有 NVIDIA GPU:
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y
# 若僅使用 CPU:
conda install pytorch torchvision cpuonly -c pytorch -y

# 3. 安裝官方指定依賴項
pip install matplotlib==3.9.0 numpy>=1.24.0 Requests==2.32.4 Shapely==2.0.5 tqdm==4.66.4
```

> [!NOTE]
> 若想在 Conda 中使用 mamba，請先執行 `conda install mamba -n base -c conda-forge`。

### 開發注意事項
*   **存儲空間**：請確保 D 槽有至少 **35GB** 以上剩餘空間以存儲 FloorSet 數據集。
*   **效能優化**：在 Windows 上開發時，強烈建議在 **WSL2** 的 Linux 分割區中執行訓練，以獲得更快的 I/O 速度（數據集包含數百萬個 Tensor 文件）。
*   **幾何報錯**：若 `validate.py` 報錯，請檢查 `Shapely` 是否正確安裝。

## 相關節點
- [[Development/EDA/VM_Environment]]
- [[README]]
