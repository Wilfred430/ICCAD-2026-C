# Git 團隊協作與同步流程

在參與 ICCAD-2026-C 競賽時，多位隊員共同維護同一個專案，需要建立一套標準的同步流程以避免程式碼衝突。

## 1. 設定遠端倉庫 (Remote Setup)

首先，需要將同學的主倉庫（Upstream）加入到你的本地環境：

```powershell
# 新增同學的倉庫作為 upstream
git remote add upstream https://github.com/popisgood/2026EDA_contest_problemC.git

# 確認設定
git remote -v
```

## 2. 定期同步流程 (Sync Workflow)

為了確保你的開發是基於最新的團隊進度：

### A. 提交或暫存本地修改
在抓取新程式碼前，請先處理本地未提交的內容：
```powershell
git add .
git commit -m "feat: local infrastructure setup"
# 或者使用 git stash 暫存
```

### B. 抓取並合併最新進度
```powershell
# 抓取遠端所有分支資訊
git fetch upstream

# 合併活躍分支 (例如 SA_core) 到你的本地分支
git merge upstream/SA_core
```

## 3. 衝突解決 (Conflict Resolution)

當你與同學修改了同一個檔案（如 `parallel.cpp`）時，會發生衝突。
1. 開啟 IDE (如 VS Code) 尋找衝突標記 `<<<<<<<`。
2. 決定保留哪一部分或進行手動整合。
3. 完成後提交：
```powershell
git add <衝突檔案>
git commit -m "merge: sync with upstream SA_core"
```

## 4. 推送與貢獻 (Push & PR)

將你的修改推送到自己的 `origin`，再透過 GitHub 發起 Pull Request：
```powershell
git push origin main
```

## 關聯節點
- [[自動化工作流]]
- [[Parallel_Cpp_Optim]]
