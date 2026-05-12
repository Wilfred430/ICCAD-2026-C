# Remind Me - Terminal Notification System

A local notification system that shows Windows toast notifications when terminal commands finish.

## Design Principles

- **Pure local execution**: Uses Windows built-in .NET Framework, zero network dependency
- **Non-blocking**: Notifications run in background, never interrupts terminal workflow
- **Zero installation**: No extra PowerShell modules or packages required
- **Safe**: Read-only operation, no side effects on system state

## Quick Start

### 1. Source the script in WSL

```bash
source /mnt/d/ICCAD-2026-C/Implementation/tools/remind-me/remind.sh
```

### 2. Use it

```bash
# Pattern A: Append after any command
make build; remind

# Pattern B: Append with custom message
make build; remind "Build finished"

# Pattern C: Wrap a command (includes duration tracking)
remind make build

# Pattern D: Simple notification
remind "Time to check results"
```

### 3. Permanent setup (optional)

Add to `~/.bashrc`:

```bash
# Load remind-me notification system
source /mnt/d/ICCAD-2026-C/Implementation/tools/remind-me/remind.sh
```

## Usage Patterns

| Pattern | Syntax | Duration Tracking | Exit Code Detection |
|---|---|---|---|
| Post-command | `cmd; remind` | No | Yes (auto) |
| Custom message | `cmd; remind "msg"` | No | Yes (auto) |
| Wrapper | `remind cmd args` | Yes | Yes |
| Simple message | `remind "msg"` | No | No |

## Architecture

```
remind.sh          -> Bash function (sourced in WSL)
    |
    v
powershell.exe     -> Calls Windows PowerShell from WSL
    |
    v
notify.ps1         -> Shows toast via System.Windows.Forms.NotifyIcon
    |
    v
Windows Toast      -> BalloonTip renders as modern toast on Win10/11
```

## File Structure

```
remind-me/
  notify.ps1     PowerShell notification script (Windows side)
  remind.sh      Bash helper functions (WSL side)
  README.md      This file
```

## Troubleshooting

### Notification does not appear

1. Check Windows notification settings: Settings > System > Notifications
2. Ensure "PowerShell" notifications are enabled
3. Test directly:
   ```powershell
   powershell.exe -ExecutionPolicy Bypass -File "D:\ICCAD-2026-C\Implementation\tools\remind-me\notify.ps1"
   ```

### PowerShell execution policy error

Run once in PowerShell (admin):
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
