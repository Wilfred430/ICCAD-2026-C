# Brief for Antigravity: flip submission-critical defaults for the 2.4072 result (2026-07-16)

## 現況核實

用 `ml/case_report_electro.py`（Neutral RT）獨立驗證了你們報告
（`electro_optimized_report.md`）的數字，**基本核實為真**：

| 設定 | Total Score（Neutral RT） | Vgrp | Vbnd |
|---|---|---|---|
| 預設（兩個 env var 都關閉，現況） | 2.4822 | 360 | 420 |
| `+ELECTRO_BOUNDARY_WIDESWAP=1` | 2.4198 | 346 | 364 |
| `+ELECTRO_BOUNDARY_WIDESWAP=1 +ELECTRO_GROUPING_PUSHPAST=1` | **2.4081** | 332 | 351 |

跟報告的 2.4072（Vgrp=326, Vbnd=333）幾乎吻合（差 0.04%，違規數在合理誤差內，
可能是報告快照跟目前程式碼有些微差異）。**改善是真的，wide-swap + grouping
push-past 疊加確實有效。**

## 發現的問題（submission-critical）

`electro_optimizer.py` 目前這兩個環境變數**預設都是關閉**：

```python
os.environ.setdefault("ELECTRO_GROUPING_PUSHPAST", "0")   # 明確關閉
# ELECTRO_BOUNDARY_WIDESWAP 沒有 setdefault，os.environ.get(..., "0") 隱含關閉
```

也就是說：**如果現在直接把這個模組送出去（比賽框架冷啟動 `import` 它，不會有
人手動設定環境變數），實際跑出來的分數會是 2.4822，不是報告標題寫的
2.4072**——落差約 1.7%，整整少掉這次最主要的兩項改善。

（這跟本 session 稍早踩過的 `ELECTRO_GROUPING_PUSHPAST` setdefault 事故是
同一類問題：portfolio 候選寫好了，但沒有确認它在「冷啟動、無環境變數」路徑下
真的會被啟用。）

## 建議動作

1. 把兩個 `setdefault` 都改成 `"1"`：
   ```python
   os.environ.setdefault("ELECTRO_BOUNDARY_WIDESWAP", "1")
   os.environ.setdefault("ELECTRO_GROUPING_PUSHPAST", "1")
   ```
2. 因為這兩個都是**新增候選、不取代**的 portfolio 機制（`solve()` 用既有
   proxy ranking 挑每案最優，理論上不會讓任何單一案例變差），把預設打開應該
   是安全的——但仍建議照樣**完整跑一次全 100 案**，確認：
   - 100/100 feasible 維持
   - 沒有任何單一案例的 cost 明顯變差（哪怕 portfolio 是新增候選，也要排除
     ranking proxy 跟真實 contest cost 不一致導致的個案誤選）
3. 確認後更新 `electro_optimized_report.md`，把「Final Optimized Portfolio」
   這欄的數字改成「預設冷啟動」下實際量到的分數，避免報告數字跟送出去的模組
   實際行為脫鉤。

跟以往一樣：驗證請用 Neutral RT（`ml/case_report_electro.py`），Contest
Grading 的單次數字會有 RT 量測雜訊，不適合用來判斷這種個位數趴的差異。
