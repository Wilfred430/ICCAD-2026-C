# ICCAD 2026 Problem C(FloorSet-Lite)穩健方案:單卡 RTX 5060、單 case Cost 0.75–0.8 的可執行路線

## TL;DR
- **最穩健的路線是「約束 by-construction 的 B\*-tree + 極快平行 CA-SA(PARSAC 風格)+ 從 1M 資料萃取的統計先驗做 warm-start」,並輔以 Stockmeyer/slack 型 shape 優化**;此路線同時打中三條件,且不依賴端到端生成模型的泛化,能在隱藏 test set 上保底達到 0.75–0.8。
- **經官方 spec 驗證後,0.75–0.8 目標的真正瓶頸不是品質而是「速度(R 要壓到 0.7 地板,runtime ≤ 全體提交 median 的 ~31%)」與「零軟違規(P=1)」**;品質(Q)有 +0.14~+0.28 的 gap 容忍空間,不需贏過 baseline,只要「接近 near-optimal ground truth 且夠快」。
- **最高槓桿的第一步是把現有生成式 B\*-tree 模型的 soft-block 長寬從佔位正方形換成 shape-curve/slack 優化,並接上四道確定性 repair pass**;這一步預期把該路線從 Cost≈5.35 拉到 ~1.x,之後速度與零違規才是決勝點。

## Key Findings

### 1. 官方評分機制(已由 contest spec C_20260522.pdf 逐條驗證,常數確認)
- Cost 公式確認為 `Cost = min[ (1 + 0.5·(HPWL_gap + Area_gap_bbox)) · exp(2.0·V_rel) · max(0.7, RT^0.3), 10 − 1e−6 ]`,不可行則 Cost = 10。即 Q=1+0.5·(gaps)、P=exp(2·V_rel)、R=max(0.7, RT^0.3),α=0.5、β=2.0、γ=0.3。spec 明列可行解 Cost 典型落在 [0.7, 7.4];違規乘子表:V_rel=0→×1.00、0.25→×1.65、0.5→×2.72、1.0→×7.39。
- **RT 的分母是「所有提交在該 test case 的 median runtime」,不是固定參考解 runtime**(spec 原文:「RuntimeFactor = Your Runtime / Median Runtime of All Submissions」;footnote:「computed independently for each test design, using that individual test case's median runtime as the sole reference point」)。速度效益封頂在 −30%(R 地板 0.7,對應 RT≈0.31),但慢的懲罰不封頂。
- **baseline(HPWL_baseline, Area_baseline_bbox)是資料集隨附的 optimal-by-construction ground-truth 值**(spec:「Baseline values … for each test case are provided in the dataset」),不是 B\*-tree SA 的解。這與使用者原本假設(baseline 是標準 B\*-tree SA)有出入,是必須親自驗證的關鍵點:若 baseline 真的是 near-optimal,則 HPWL_gap/Area_gap 幾乎必為正、Q<1 幾乎不可能,0.75–0.8 只能靠壓低 R + 零違規達成。
- V_rel = (V_group + V_boundary + V_mib) / N_soft,N_soft = |B_boundary| + Σ(|G_p|−1) + Σ(|M_q|−1),∈[0,1]。V_group = Σ(連通分量數−1)、V_mib = Σ(相異 (w,h) 數−1)、V_boundary = 未貼指定邊/角的 block 數。
- 總分 = Σ_{i=21}^{120} Cost[i]·e^(n_i/12)/Z(確認),大 case 指數加權,故 n≈90–120 決定勝負;spec:「A perfect solution achieving baseline metrics on all test cases with median runtime would have a Total Score close to 1.0」。
- **重要更新:fixed-shape 與 preplaced 已從軟約束改為硬約束**(spec 明載此更動:「Previously treated as a soft constraint; updated to hard constraint」),違反即不可行(Cost=10)。soft block 面積須 |w·h − a|/a ≤ 0.01;解必須嚴格 overlap-free(可共邊不可交疊)、fixed-outline。
- boundary bitmask 由 repo README 確認:LEFT=1, RIGHT=2, TOP=4, BOTTOM=8,角落為兩邊 OR(TOP-LEFT=5, TOP-RIGHT=6, BOTTOM-LEFT=9, BOTTOM-RIGHT=10)。
- spec 自述:傳統重度平行 SA(引用 PARSAC/distributed SA)在 60 partitions 就常需 >10 分鐘且仍有 ≥10% wirelength/area gap;而其內部 diffusion 模型可在 sub-minute 得高擬真解。**anti-cheating:逆向工程資料生成器會取消資格。**

### 2. 重新框定:0.75–0.8 vs 0.7 的策略差異
把 R 壓到地板 0.7(runtime ≤ ~31% median)後:
- 命中 **0.7** 需 Q·P = 1.0,即零違規 **且** 品質 gap 總和 ≈ 0(完全追平 near-optimal ground truth)。
- 命中 **0.8** 只需 Q·P ≤ 0.8/0.7 = 1.143,即零違規 + gap 總和 ≤ +0.286;命中 **0.75** 需 Q ≤ 1.071,gap 總和 ≤ +0.14。
- 換言之,**放寬到 0.75–0.8 後,品質戰場基本可以「投降」——不追求追平最優,只要在 near-optimal 的 14–28% 誤差內即可**。省下的工程與算力,應全部投入「保證零軟違規」與「保證 runtime 進入 median 前 ~31%」。這是與 0.7 目標最根本的差異:0.7 迫使你打品質極限賽,0.75–0.8 讓你打「穩定 + 快」的工程賽,後者對單卡消費級硬體友善得多。
- 風險反轉點:若大量隊伍都用快速前饋/短 pipeline,median runtime 會下降,0.7 地板(≤31% median)會愈難守;此時 R 浮上 0.7 以上,品質(Q)與零違規(P)的邊際價值上升。需在 validation 上持續監控自估 runtime 分佈。

### 3. 三種軟約束的 by-construction(建構即滿足)技術盤點
零違規是硬門檻(P 對 V_rel 指數放大:V_rel=0.25 就 ×1.65、=0.5 ×2.72)。文獻明確支持「建構即滿足」而非「先擺再修」:
- **Boundary**:B\*-tree 有嚴格可行性條件與 repair-free 建構法。Lin & Chang(NTU,《Module placement with boundary constraints using B\*-trees》,IEE Proc. Circuits Devices Syst.)證明「每次擾動都保證 boundary-feasible 的 B\*-tree」——左邊界 block 必在根的左脊(left spine)、其他邊界對應特定分支、角落取兩邊界交集(對應 bitmask OR),並提供 O(n) 的 infeasible→feasible 轉換與線性時間 packing。Slicing 版有 Young, Wong & Yang(《Slicing Floorplans with Boundary Constraints》,IEEE TCAD 18(9):1385–1389, 1999)的 Polish expression 充要條件。PARSAC(Hesham Mostafa, Uday Mallappa, Mikhail Galkin, Mariano Phielipp, Somdeb Majumdar,皆 Intel Labs;arXiv:2405.05495v3, 2024-08-01)則用「constraints-fixing moves(必接受)+ anchored blocks」,並證明純軟懲罰 SA 會卡在「hard-constraint 局部極小」(其 GitHub repo 確認:「The SA engine in PARSAC is especially suited to problems with boundary constraints … and to problems with preplaced blocks where certain blocks have to be placed at specific x-y locations」)。
- **Grouping(共享邊、單一連通分量)**:最穩健是 **super-block 收縮法**——把每個 group 收縮成一個超級 block,對超級 block 與非群組 block 跑 B\*-tree;群組內部用一棵子 B\*-tree 排成互相 abut 的緊湊塊,天然形成單一連通分量,V_group≡0。文獻依據:MB\*-tree 的 clustering/declustering 兩階段(Lee et al.),Young & Wong 的 slicing clustering constraint,Ma et al. 的 corner block list abutment constraint(能精確控制任意數目 block 的相對相鄰,SP/BSG 難以做到)。
- **MIB(同組共享 (w,h))**:把整個 MIB 群組的形狀當**單一共享形狀變數**——選一次 (w,h),所有 instance 套用,V_mib≡0 by-construction;在 shape 優化時,MIB 群組只有一個自由度。

### 4. 30% runtime 預算的架構
- **PARSAC(Intel 自己的相關工作,極可能是 baseline 近親)**:CA-SA + 大規模平行 + C++ SA 核心,B\*-tree 表示,原生支援 boundary/preplaced/grouping。這是「約束感知 + 快」的現成骨架與思路來源,也是保底方案的天然起點(repo Apache-2.0,但已 archived)。
- **解析式快解**:PeF(Ximeng Li, Keyu Peng, Fuxing Huang, Wenxing Zhu,《PeF: Poisson's Equation-Based Large-Scale Fixed-Outline Floorplanning》,IEEE TCAD 42(6):2002–2015, 2023;arXiv:2210.03293)——「the widths of soft modules appear as variables in the energy function and can be optimized」,global floorplanning + legalization 兩階段;Per-RMAP(Yu, Censor, Jiang & Luo,ISEDA-2023 / TCAD 2024,arXiv:2406.03165)以 feasibility-seeking + superiorization,原文宣稱「legal floorplanning results 166 times faster than the branch-and-bound (B&B) method while incurring only a 5% wirelength increase … 15% improved runtime compared with PeF」。這類方法快,但對 boundary/grouping 硬約束的 by-construction 保證不如 B\*-tree。
- **建議 pipeline**:輕量生成/先驗出初始拓樸(一次前饋或短自回歸)→ 極短窗口 SA / greedy swap 局部精修 → shape 解析優化(Stockmeyer/slack)→ 確定性 repair 保零違規。runtime 預算切成大約「初始 10% + 局部精修 60% + shape+repair 30%」,並設硬性 wall-clock 早停,確保進入 median 前段。

### 5. 1M 資料在單卡 8GB 上的利用(ML 當先驗、傳統當骨幹)
- **穩健替代:統計先驗而非端到端生成**——從 1M tree_sol/最優解萃取:(a) block-pair「該不該相鄰」的 edge classifier;(b) soft block 長寬比分佈;(c) boundary block 典型位置模式。把先驗餵給 SA/解析式當初始解與搜尋偏置。這比端到端更抗泛化失敗(隱藏 test set)。兩階段「learned prior + classical refinement」有實證:GraphPlanner(Yiting Liu 等 9 人,《GraphPlanner: Floorplanning with Graph Neural Network》,ACM TODAES 28(2), Art. 21, 2022,doi:10.1145/3555804)用 variational GCN 產初始佈局後接 RePlAce/DREAMPlace;IncreMacro(ISPD/TCAD 2024)用 constraint-graph LP 對初始 macro 佈局做 refinement + 合法化。
- **現有生成式 B\*-tree 模型(parent-pointer 87%、採樣 100% feasible、但 Cost≈5.35)**:診斷 5.35 的三大主因與修法——(i) soft block 用佔位正方形→bbox area 爆炸→Area_gap 巨大;改用 shape-curve/slack 優化可大幅降 Q;(ii) 四道 repair 未接→殘留軟違規→P 被指數放大;接上後 P→1;(iii) 未做 Cost-as-reward 微調。前兩項是確定性工程,槓桿最大,預期把該路線拉到 ~1.x。
- **RL 微調(單卡可行)**:cross-entropy 預訓後用 REINFORCE / self-critical(Bello et al. 2016《Neural Combinatorial Optimization with Reinforcement Learning》;Graph Pointer Network + hierarchical RL,Ma et al. 2019 對 constrained TSP 用分層 reward 穩定訓練)在單卡對 B\*-tree 生成做 reward = −Cost 的 policy-gradient 微調;reward shaping 同時含 HPWL/Area/violation。屬高上限但高風險,列為第三優先。

### 6. shape(長寬比)優化的既有技術
- **Slicing 拓樸**:Stockmeyer shape curve(《Optimal Orientations of Cells in Slicing Floorplan Designs》,Information and Control 1983;Shi 改進為 O(n log n) 最優)一次算出最優長寬比組合;適合 super-block 內部或 slicing 骨架。
- **非 slicing / B\*-tree**:slack-based sizing(Adya & Markov,《Fixed-outline floorplanning: enabling hierarchical design》,IEEE Trans. VLSI Syst. 11(6):1120–1135, 2003,原文:「Our proposed moves are based on the notion of floorplan slack … optimization of aspect ratios of soft blocks are explicitly addressed」,含 PackSoftBlocks);Chaomin Luo et al. 兩階段非線性(convex 全域 + sizing 合法化,《A nonlinear optimization methodology for VLSI fixed-outline floorplanning》,J. Combinatorial Optimization, Springer, 2008)原文「wirelength is decreased by about 16.8% and 8.6% on average, compared with two previous fixed-outline floorplanners on soft modules, which are both proved to be better than Parquet」;Lagrangian relaxation 處理非 slicing soft module。給定拓樸後這些都是快速、確定性的品質來源,直接吃進 Q 的容忍空間。

## Details

### 主推方案 A(保底、強烈建議先落地):約束 by-construction 的 B\*-tree + 極快 CA-SA + ML 統計先驗
**核心機制**
1. **硬約束先鎖死**:preplaced 直接固定 (x,y,w,h);fixed-shape 鎖 (w,h)。因為它們現在是硬約束,鎖死即零風險。
2. **Grouping super-block 收縮**:每個 group → 一顆內部子 B\*-tree 排成 abut 緊湊塊 → 當單一超級 block 進主 B\*-tree。V_group≡0。
3. **MIB 共享形狀變數**:每個 MIB 群組一個 (w,h) 自由度,所有 instance 綁定。V_mib≡0。
4. **Boundary 用 Lin & Chang 可行性條件**:每次擾動只在 boundary-feasible 的 B\*-tree 空間內移動(左脊放左邊界、對應分支放其他邊界,角落取交集),搭配 PARSAC 的 constraints-fixing moves 作雙保險。V_boundary≡0。
5. **shape 優化**:給定拓樸,用 slack-based sizing / Stockmeyer(super-block 內)在 1% 面積容忍內選長寬比,壓 Area_gap 與 HPWL_gap。
6. **ML 先驗 warm-start**:用 1M 萃取的 edge-adjacency 先驗與長寬比分佈生成初始 B\*-tree 與偏置 SA move,縮短收斂 → 直接降 runtime → 壓 R 到地板。
7. **確定性 repair + 硬性早停**:收斂後跑確定性檢查(overlap、outline、三軟約束、面積 1% 容忍),wall-clock 早停確保進入 median 前段。

**為何同時打中三條件**:三軟約束 by-construction → P=1;硬約束鎖死 + 確定性 repair → feasible;ML warm-start + 有限迭代 + 早停 → runtime 壓進 ~31% median → R=0.7;shape 優化把 Q 收進 +0.14~+0.28 容忍區。

**最大風險**:(a) boundary + grouping + MIB 疊加時,by-construction 空間可能過窄導致 Q 變差或偶發 infeasible——需在 100 筆 validation 上逐 size 驗證可行率 100%;(b) median runtime 若被其他快速隊伍拉低,R 浮上 0.7 以上。
**驗證方法**:在 100 筆 validation(每 size 各一)上跑,記錄逐 case 的 feasible/V_rel/Q/自估 runtime,重點看 n≥90 的大 case。

### 次推方案 B(升級、有上限):生成式 B\*-tree 拓樸 proposer + 方案 A 的確定性後端
**核心機制**:用現有自回歸 Transformer + pointer network 一次(或短自回歸)產出拓樸候選(取 top-k 幾條),直接餵進方案 A 的 super-block/boundary-feasible packing + shape 優化 + repair。生成只負責「拓樸提案」,所有約束滿足與品質由確定性後端保證。可選 REINFORCE(reward=−Cost)微調。
**為何打中三條件**:一次前饋比 SA 迭代快得多 → R 更容易到地板;拓樸品質由 87% 準確率模型提供好起點 → Q 好;約束仍由後端 by-construction 保證 → P=1、feasible。
**交接介面設計(最穩健)**:模型只輸出「parent-pointer 序列 + 每 block 的左/右 child 關係」,不輸出座標;後端負責 packing、shape、repair。這樣模型泛化失敗時,後端仍能保證合法解,最壞退化成方案 A。
**最大風險**:隱藏 test set 上拓樸泛化失敗 → Q 變差。fallback:偵測到自估 Cost 高於門檻時,自動退回方案 A 的 SA。

### 純 A 路線(完全不用 ML)能否達到 0.8?
**評估:對中小 case(n≤~70)很可能可以;對 n≥90 的大 case 有風險但值得認真投入,因為 0.75–0.8 的品質門檻已放寬。** 關鍵論證:0.75–0.8 不要求贏 baseline,只要零違規 + 快 + gap ≤ +0.28。零違規由 by-construction 保證,與 ML 無關;快由「有限迭代 + 早停 + 約束感知空間縮小」達成;品質只需 near-optimal 的 ~28% 內。PARSAC 已證明 CA-SA 能在複雜約束下產生合法 Pareto 前緣。**風險在 runtime**:純 SA 在大 n 要同時「夠快(≤31% median)」與「gap≤0.28」可能拉扯——這正是 ML warm-start(方案 A 第 6 步)的價值:它不改變保底邏輯,只加速收斂。因此建議「純 A 為保底骨架,ML 先驗為加速器」,而非二選一。

## Recommendations(30/60/90 天,標出高槓桿)

**第一優先(高槓桿,Day 0–15,先做)**
1. **精讀 `iccad2026_evaluate.py` / `cost.py`,親自確認 baseline 到底是 optimal-by-construction 還是 B\*-tree SA**,以及 RT median 的計時邊界(是否含 I/O、每 case 上限)。這決定 Q 是否可能 <1,是整個策略的地基。
2. **把生成式 B\*-tree 模型的 soft-shape 從佔位正方形換成 slack/Stockmeyer 優化 + 接上四道 repair pass**。預期 Cost 5.35 → ~1.x。這是單點槓桿最大的工程,且與方案 A/B 都相容。

**30 天:保底骨架**
3. 實作方案 A 的 by-construction 三軟約束(super-block grouping、MIB 共享形狀、Lin & Chang boundary feasibility)+ 硬約束鎖死 + 確定性 repair。目標:100 筆 validation **feasible 100%、V_rel=0**。
4. 接上多執行緒 CA-SA(PARSAC 風格)+ shape 優化,建立逐 case 的 Cost/runtime 記錄表。

**60 天:速度與品質收斂**
5. 從 1M 萃取 edge-adjacency 先驗與長寬比分佈,做 SA warm-start 與 move 偏置;實作硬性 wall-clock 早停,把大 case runtime 壓進目標。
6. 接上方案 B:生成模型出拓樸 → A 的後端;A/B 取逐 case 較優者(oracle 選擇在 validation 上估上限)。

**90 天:微調與加固**
7. 選擇性做 REINFORCE/self-critical(reward=−Cost)微調生成模型,只在 validation 顯示穩定增益時採用。
8. 全面壓力測試 n=90–120,建立 fallback 自動切換(自估 Cost 超門檻 → 退回純 A)。

**會改變建議的門檻/基準**
- 若確認 baseline = optimal-by-construction:放棄追 Q<1,全力壓 R 與零違規。
- 若 validation 自估 runtime 已穩定 ≤ 25% 的合理 median 估計:把剩餘算力全投品質(shape LP / 更長 SA),因為 R 已到地板。
- 若任何 size 的 feasible 率 < 100%:暫停一切品質優化,回頭修 by-construction/repair——一個 infeasible = Cost 10,在指數加權下足以毀掉大 case。

## Caveats(最不確定、需使用者補實驗驗證的假設)
1. **baseline 定義**:spec(subagent 讀取的 C_20260522.pdf)指 baseline 為資料集 optimal-by-construction ground truth,與使用者原述「標準 B\*-tree SA」矛盾。**必須以 repo 內 `cost.py`/`iccad2026_evaluate.py` 原始碼為準親自確認**,因為它決定 Q<1 是否可能。此為最高不確定點。
2. **median runtime 的動態**:RT 分母是全體提交 median,無法事前得知;~31% 門檻是浮動目標。需以「盡量快 + 早停」策略對沖,並在賽中(若有中期 leaderboard)校準。
3. **Cost 5.35 → 1.x 的量級估計**是基於「placeholder 正方形爆 area + 殘留軟違規被指數放大」的推斷,未經實測;需在修 shape+repair 後用 validation 實測確認落點。
4. **by-construction 空間是否過窄**:boundary+grouping+MIB 同時嚴格滿足時,B\*-tree 可達空間可能顯著縮小,潛在犧牲 Q 或偶發 infeasible。需逐 size 實測可行率與 Q 分佈。
5. **spec 版本**:subagent 取得的是 Google Drive 上的 C_20260522.pdf(GitHub 的 v8 PDF 路徑已 404)。常數與公式內部一致且與 repo README 約束編碼吻合,但仍建議以最新官方 Problems.html 連結為準。
6. **RL 微調在單卡 8GB 的實際收斂性**與 reward 稀疏問題未經你的環境驗證,列為選配而非關鍵路徑。
7. **fixed-shape/preplaced 已改硬約束**:請確認你現 有 pipeline 沒有把它們當軟約束處理,否則會直接產生 Cost=10。

---
**主要來源**:ICCAD 2026 CAD Contest Problem C 官方 spec(C_20260522.pdf);IntelLabs/FloorSet GitHub repo 與 Hugging Face dataset;PARSAC(arXiv:2405.05495);PeF(arXiv:2210.03293 / IEEE TCAD 2023);Per-RMAP(arXiv:2406.03165 / 2304.06698);Young, Wong & Yang《Slicing Floorplans with Boundary Constraints》(IEEE TCAD 1999);Lin & Chang《Module placement with boundary constraints using B\*-trees》;Adya & Markov《Fixed-outline floorplanning: enabling hierarchical design》(IEEE TVLSI 2003);Luo et al.《A nonlinear optimization methodology for VLSI fixed-outline floorplanning》(Springer JOCO 2008);GraphPlanner(ACM TODAES 2022);Stockmeyer《Optimal Orientations of Cells in Slicing Floorplan Designs》(1983);Bello et al. 2016 與 Ma et al. 2019(neural combinatorial optimization)。