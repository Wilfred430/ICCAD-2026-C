#!/bin/bash

# 自動化煙霧測試腳本 (Smoke Test)
# 功能：執行 C++ floorplanner 並使用 Python 工具驗證結果，比對兩者 Cost 是否一致。

# 設定路徑
BIN="./floorplanner"
TOY_TXT="benchmarks/toy.txt"
TOY_SOL="benchmarks/toy.sol"
VERIFY_TOOL="tools/verify_solution.py"

echo "===================================================="
echo "🚀 開始執行自動化煙霧測試 (Smoke Test)"
echo "===================================================="

# 1. 確保執行檔存在
if [ ! -f "$BIN" ]; then
    echo "❌ 找不到 $BIN，正在嘗試編譯..."
    make static
    if [ $? -ne 0 ]; then
        echo "❌ 編譯失敗，請檢查原始碼。"
        exit 1
    fi
fi

# 2. 執行 C++ Solver
echo "[1/3] 執行 C++ Floorplanner (5s)..."
# 執行並補捉輸出，同時隱藏進度條以外的雜訊
CPP_OUT=$($BIN "$TOY_TXT" "$TOY_SOL" --time 5 --threads 4 --verbose 2>&1)
echo "$CPP_OUT" | grep -E "\[main\]|feasible"

# 提取 C++ 的 contest_cost (格式: contest_cost=1.0023)
CPP_COST=$(echo "$CPP_OUT" | grep "contest_cost=" | sed 's/.*contest_cost=//' | awk '{print $1}')

# 3. 執行 Python 驗證工具
echo -e "\n[2/3] 執行 Python 驗證工具..."
PY_OUT=$(python3 "$VERIFY_TOOL" "$TOY_TXT" "$TOY_SOL" 2>&1)
echo "$PY_OUT" | grep -E "feasible|contest_cost"

# 提取 Python 的 contest_cost (格式: contest_cost (rf=1): 1.0023)
PY_COST=$(echo "$PY_OUT" | grep "contest_cost" | awk '{print $NF}')

# 4. 比對結果
echo -e "\n[3/3] 比對結果..."

if [ -z "$CPP_COST" ] || [ -z "$PY_COST" ]; then
    echo "❌ 錯誤：無法從輸出中提取 Cost 數值。請檢查程式是否正確印出結果。"
    exit 1
fi

# 使用 awk 計算誤差
DIFF=$(awk -v c="$CPP_COST" -v p="$PY_COST" 'BEGIN { d = c - p; if (d<0) d=-d; print d }')

echo "----------------------------------------------------"
echo "C++    Contest Cost: $CPP_COST"
echo "Python Contest Cost: $PY_COST"
echo "絕對誤差: $DIFF"
echo "----------------------------------------------------"

# 判定標準：誤差小於 0.001
RESULT=$(awk -v d="$DIFF" 'BEGIN { if (d < 0.001) print "PASS"; else print "FAIL" }')

if [ "$RESULT" == "PASS" ]; then
    echo "✅ 測試通過！C++ 與 Python 邏輯一致。"
    exit 0
else
    echo "❌ 警告：誤差過大 ($DIFF >= 0.001)！"
    echo "請檢查 cost.cpp 的公式實作是否與 Python 版本不同。"
    exit 1
fi
