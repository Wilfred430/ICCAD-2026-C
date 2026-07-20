# 任務簡報:修復 electro pipeline 的 V_mib / V_boundary 軟約束違規

## 背景

這是 ICCAD 2026 Contest C(FloorSet-Lite 平面規劃)專案。隊友 pop 寫的電靜力法佈局器
(`electro_optimizer.py` + `analytical_place.py` + `legalize.py` + `soft_repair.py`)
是目前團隊唯一已驗證可用的主力路線,100-case 驗證 **Total Score = 2.8233(100/100
feasible)**,平均 runtime 1.76s/case。

Cost 公式(官方,已在 `iccad2026_evaluate.py`/spec PDF 驗證):
```
Cost = (1 + 0.5·(max(0,HPWL_gap) + max(0,Area_gap))) · exp(2·V_rel) · max(0.7, RT^0.3)
V_rel = min(1, (V_grouping + V_mib + V_boundary) / N_soft)
```
`exp(2·V_rel)` 是指數懲罰,V_rel 每降一點,對所有含軟約束的 case 都是全面性的分數改善。

## 現況診斷(100-case 逐 case Excel 報告已產生,`case_report_electro.xlsx`)

全部 100 case 加總:
- **V_grouping = 435**
- **V_mib = 56** ← 目標之一
- **V_boundary = 534** ← 目標之二(數字最大,優先看這個)

對照組:我們自己另一條生成式 B\*-tree 路線,用 by-construction 手法把 **V_mib 做到 0**、
V_boundary 壓到 334——證明這兩項理論上可以做得更好,electro 目前的 post-hoc 修復
(`soft_repair.py::grouping_repair`/`boundary_snap`)沒有做到位。

## 今晚已經試過、失敗的方向(重要,不要重複)

**MIB 硬性鎖死(已測試、已否決、已復原,不要重做這個版本)**:

在 `analytical_place.py::place()` 內部的 `shapes()` closure 裡,MIB 群組的 soft 成員
形狀是這樣算的(約第 290-300 行附近,依版本行號可能略有不同):

```python
w_soft = (sqrt_area_sg * torch.exp(0.5 * la_b))[inv] * sg_scale
h_soft = (sqrt_area_sg * torch.exp(-0.5 * la_b))[inv] * sg_scale
w = torch.where(is_soft, w_soft, torch.where(is_fixed | is_pre, tw, w_soft))
h = torch.where(is_soft, h_soft, torch.where(is_fixed | is_pre, th, h_soft))
```

`sqrt_area_sg`(群組平均面積開根號)是共享的,但 `la_b`(對數長寬比)是**每個 shape-group
獨立優化的自由變數**,沒有被綁定去匹配群組裡任何 fixed/preplaced 錨點的精確長寬比——
這就是 V_mib > 0 的根因:群組成員面積平均相近,但長寬比不保證跟錨點一致。

**已測試的修法(失敗)**:偵測每個 MIB 群組是否有 fixed/preplaced 錨點,若有,直接把該
群組全部成員的 `w_soft, h_soft` **強制覆寫**成錨點的精確 `(tw, th)`,完全繞過
`exp(la)` 公式。

**結果**:單一測資(config_21,7 個 MIB 成員共用一個群組)驗證,**V_mib 確實變 0**,
但 **area_gap 從合理範圍暴增到 183.9%、hpwl_gap 148.3%,整體 Cost 從 1.8485 惡化到
5.335**(超過 3 倍)。已復原,不影響現有的 2.8233 分基準。

**失敗原因分析**:electro 是連續梯度優化,每個 soft 方塊本來能自由選長寬比去貼合鄰居、
填補空隙。把整個 MIB 群組(可能好幾個方塊)全部鎖死成同一個剛性形狀,一次拿掉太多自由度,
求解器再也找不到緊密排列——密度崩壞的代價遠大於修好 V_mib 省下的懲罰。

## 建議嘗試的方向(還沒做,這是你的任務)

**軟性引導,而非硬性鎖死**:不要用 `torch.where` 強制覆寫,改成在 loss function 裡加一個
**懲罰項**,讓群組內每個 soft 成員的 `la_b`(對數長寬比)**趨近**錨點的對數長寬比,但不
100% 鎖死——保留一部分自由度讓優化器還能在「貼近錨點形狀」和「找到緊密排列」之間找平衡。

具體做法建議:
1. 對每個有錨點的 MIB 群組,算出錨點的目標對數長寬比 `la_target = log(anchor_w/anchor_h) / ...`(注意跟現有 `la_b` 的參數化方式對齊,可能需要看 `la` 變數本身的定義,不是看 `shapes()` 算出來的 w/h 反推)。
2. 在 `place()` 的主 loss 函式裡加一項:`lam_mib_shape * (la[group] - la_target)^2`,權重 `lam_mib_shape` 用環境變數控制、可調(比照這個檔案裡其他 `ELECTRO_*` 環境變數的風格)。
3. **一樣先在單一測資(test-id 0,對應 config_21)驗證**:V_mib 有沒有降、area_gap/hpwl_gap 有沒有跟著爆掉。如果權重太大會重現今晚的災難,權重太小則沒有效果——這是需要掃參數的地方。
4. 通過單一測資驗證後,再跑完整 100-case(`iccad2026_evaluate.py --evaluate <你的檔案> `,不帶 `--test-id` 就是跑全部 100 筆),確認 Total Score 有沒有真的比 2.8233 低,且 100/100 保持 feasible。

## V_boundary(534)——已經診斷 + 已經試過一版,結果是「單一 case 大勝、100-case 整體退步」

**診斷結果(用既有的 `case_report_electro.xlsx` 資料,零額外運算)**:534 次違規分佈在
**98/100 個 case**,跟 case 大小無關(n≤50/50-90/>90 三組平均都在 4.9-6.0 之間),前 15
個最糟的 case 只佔總數的 36.5%——**這是普遍性、系統性問題,不是少數異常值**。

**已經試過的修法(有效但整體是負分,不要照原樣重做)**:`soft_repair.py::boundary_snap`
的 `_slot_along_y`/`_slot_along_x` 已經有沿牆掃描找空位的邏輯(比預期成熟),但**沒有
push-past 後備機制**——找不到候選空位就完全放棄,方塊不動,違規永遠留著。移植我們
`pack_tree.py::_boundary_repair_pass` 的 push-past(找不到空位就推過當前邊界,保證接觸但
可能撐大 bbox):

- **單一測資(config_117,n=117,原本全場最糟的 19 次違規)**:V_boundary **19→8**,
  cost 3.237→3.023(框架測到 2.5868)。看起來是大勝。
- **完整 100-case 驗證(重要!)**:**Total Score 卻從 2.8233 惡化到 3.0668**——整體退步。
  已復原,沒有留在程式碼裡。

**原因**:push-past 對「真的卡住、沒有它就永遠違規」的 case 有幫助,但對「沿牆掃描本來
就找得到空位、這次只是沒剛好找到」的 case,push-past 會不必要地撐大 bbox——淨效果由
`exp(2·V_rel)` 省下的懲罰 vs 面積代價的拉鋸決定。**跟我們自己路線 §6.9 的 push_past
on/off portfolio 是同一個教訓**:不能全面套用,必須逐 case A/B 選擇。

**建議的下一步(這是具體、明確、有現成方法論可抄的任務,不是要你重新想方向)**:
把 push-past 做成 **on/off 兩個候選**——每個 case 用 `boundary_snap` 原版(off)跑一次、
加了 push-past 的版本(on)跑一次,兩邊都送進 `contest_cost.py`/真評測器算實際 Cost,
逐 case 取較低者。實作位置建議在 `electro_parallel.py::run_start`(呼叫
`boundary_snap` 的地方),不是在 `soft_repair.py` 內部直接改——讓 `boundary_snap`
保留一個 `push_past: bool` 參數(預設 False,原行為不變),呼叫端跑兩次比較。
**一樣先在 config_117(test-id 96)這種原本違規多的 case 上驗證有沒有選對,再跑滿
100 筆確認 Total Score 真的 < 2.8233。**

## 重要提醒

1. **每次改動都先驗證單一小測資,再跑完整 100-case**——今晚每個改動都是先用
   `iccad2026_evaluate.py --evaluate <file> --test-id 0` 快速驗證,通過才跑全部
   100 筆(約 3-5 分鐘)。這個紀律今晚幫我們抓到好幾次「看起來合理但實際上更差」的
   改動,不要跳過。
2. **改動請在獨立複本上做**,不要動到 pop 的正式 repo/分支,除非使用者明確同意要
   合併回去。
3. Cost 公式的 `max(0,·)` clamp:贏過 baseline 不會額外加分,所以不用執著把 area_gap/
   hpwl_gap 壓到負值,重點是 V_rel 趨近 0 且不要讓 area_gap/hpwl_gap 大幅惡化。
4. 官方重疊/邊界判定容差是 `1e-6`(不是更嚴格的 `1e-7`)——如果你重寫或複製了
   cost 計算邏輯,這個容差要對,否則會有假性 infeasible 的問題(今晚才修過這個 bug)。
