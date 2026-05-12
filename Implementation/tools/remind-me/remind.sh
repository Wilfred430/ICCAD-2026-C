#!/bin/bash
# =============================================================================
# Remind Me - Terminal Notification System for WSL
# =============================================================================
# Source this file in your shell to enable the 'remind' command.
#
# Usage:
#   source /mnt/d/ICCAD-2026-C/Implementation/tools/remind-me/remind.sh
#
# Then use any of these patterns:
#
#   make build; remind                  # Notify after command (auto-detect success/fail)
#   make build; remind "Build done!"    # Notify with custom message
#   remind make build                   # Wrap command, auto-notify when done
#   remind "Just a reminder"            # Show a simple notification
#
# Features:
#   - Tracks command duration (when using wrapper mode)
#   - Shows success/failure status with exit code
#   - Runs notification in background (non-blocking)
#   - Pure local execution, zero network usage
# =============================================================================

# --- Internal: resolve Windows path to notify.ps1 ---
_REMIND_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REMIND_WIN_PS1="$(wslpath -w "${_REMIND_SCRIPT_DIR}/notify.ps1" 2>/dev/null)"

# --- Internal: send notification via PowerShell ---
_remind_notify() {
    local title="$1"
    local message="$2"
    local exit_code="${3:-0}"

    powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass \
        -File "$_REMIND_WIN_PS1" \
        -Title "$title" \
        -Message "$message" \
        -ExitCode "$exit_code" \
        &>/dev/null &
    disown 2>/dev/null
}

# --- Internal: format duration ---
_remind_format_duration() {
    local seconds=$1
    if [ "$seconds" -ge 3600 ]; then
        printf "%dh %dm %ds" $((seconds / 3600)) $((seconds % 3600 / 60)) $((seconds % 60))
    elif [ "$seconds" -ge 60 ]; then
        printf "%dm %ds" $((seconds / 60)) $((seconds % 60))
    else
        printf "%ds" "$seconds"
    fi
}

# --- Main function ---
remind() {
    # Capture exit code from the PREVIOUS command (for "cmd; remind" pattern)
    local prev_exit=$?

    # -----------------------------------------------------------
    # Case 1: No arguments -> notify based on previous command
    # Usage: some_command; remind
    # -----------------------------------------------------------
    if [ $# -eq 0 ]; then
        local msg
        if [ $prev_exit -eq 0 ]; then
            msg="Command completed successfully"
        else
            msg="Command failed (exit code: $prev_exit)"
        fi
        _remind_notify "Terminal Done" "$msg" $prev_exit
        return $prev_exit
    fi

    # -----------------------------------------------------------
    # Case 2: Single non-command argument -> custom message
    # Usage: some_command; remind "Custom message"
    # -----------------------------------------------------------
    if [ $# -eq 1 ] && ! command -v "$1" &>/dev/null && [ ! -f "$1" ]; then
        _remind_notify "Terminal Done" "$1" $prev_exit
        return $prev_exit
    fi

    # -----------------------------------------------------------
    # Case 3: Command arguments -> wrap and run, then notify
    # Usage: remind make build
    # -----------------------------------------------------------
    local start_time
    start_time=$(date +%s)

    "$@"
    local cmd_exit=$?

    local end_time
    end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local duration_str
    duration_str=$(_remind_format_duration $duration)

    local cmd_name="$1"
    local msg
    if [ $cmd_exit -eq 0 ]; then
        msg="[$cmd_name] completed in $duration_str"
    else
        msg="[$cmd_name] failed (exit: $cmd_exit) after $duration_str"
    fi

    _remind_notify "Terminal Done" "$msg" $cmd_exit
    return $cmd_exit
}

# Confirmation message when sourced
echo "[remind-me] Notification system loaded. Type 'remind' after any command."
