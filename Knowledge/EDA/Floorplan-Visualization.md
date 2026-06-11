# Floorplan 視覺化工具說明

本文介紹如何使用從 `Implementation` 搬運到 `work` 的視覺化與評估工具。

## 工具列表

1. **`my_visualize.py`**: 
   - 用於對比 Ground Truth 與 Optimizer 的結果。
   - 支援互動式顯示與存檔。
2. **`tools/visualize_floorplan.py`**:
   - 官方風格的視覺化工具。
   - 支援批次處理（`--batch`）產生的結果。
3. **`Makefile` 整合指令**:
   - `make eval`: 執行完整 100 案評估並同步。
   - `make viz`: 根據評估結果生成佈局圖。
   - `make report`: 計算加權總分。

## 視覺化圖像解析：黑線是什麼？

在產生的 Layout 圖片中，可以看到一個粗黑線構成的矩形：

- **來源**：輸入 `.txt` 檔案中的 `OUTLINE` 關鍵字定義。
- **意義**：這是 **Fixed-Outline**（固定輪廓）約束。
- **觀察點**：
  - 所有 Block 理想上應位於此黑框內。
  - 若有 Block 超出黑框，表示目前的 Floorplan 方案違反了邊界約束，將導致懲罰分數。
  - 綠色圓點（Terminals/Pins）通常分佈在 Outline 的邊界或內部。

## 自動化流程

若要從頭開始評估並查看圖片，請執行：

```bash
make eval-quick  # 快速評估 25 案
make viz         # 生成圖片到 plots/ 目錄
```

相關知識節點：
- [[EDA/自動化工作流]]
