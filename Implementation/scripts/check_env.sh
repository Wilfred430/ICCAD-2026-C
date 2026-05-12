#!/usr/bin/env bash
# scripts/check_env.sh
#
# Quick environment sanity check. Run this before Step 2 of START_HERE.md.
# Tells you which of g++, make, python3 are missing.

set -u

ok=0
fail=0

check() {
    local name="$1"
    local cmd="$2"
    local min="${3:-}"
    if command -v "$cmd" >/dev/null 2>&1; then
        local ver
        ver=$($cmd --version 2>&1 | head -1)
        printf "  ✅ %-15s %s\n" "$name" "$ver"
        ok=$((ok+1))
    else
        printf "  ❌ %-15s NOT FOUND\n" "$name"
        if [ -n "$min" ]; then
            printf "       (need %s) install hint: %s\n" "$name" "$min"
        fi
        fail=$((fail+1))
    fi
}

echo "Environment check for floorplanner submission:"
echo

check "g++"        g++       "Ubuntu: sudo apt install build-essential | macOS: xcode-select --install"
check "make"       make      "ships with build-essential / xcode CLI tools"
check "python3"    python3   "Ubuntu: sudo apt install python3 python3-venv | macOS: built-in"
check "pip"        pip3      "Ubuntu: sudo apt install python3-pip"
check "git"        git       "Ubuntu: sudo apt install git | macOS: built-in"
check "unzip"      unzip     "Ubuntu: sudo apt install unzip"
check "file"       file      "(optional) for verifying static binary"

echo
if [ $fail -eq 0 ]; then
    echo "All required tools found ($ok/$((ok+fail))). You can run 'make static' next."
    exit 0
else
    echo "Missing $fail tool(s). Install them, then re-run this script."
    exit 1
fi
