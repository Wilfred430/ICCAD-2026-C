# =============================================================================
# Remind Me - Windows Toast Notification Script
# =============================================================================
# Purpose: Show a Windows toast notification (bottom-right popup)
# Method:  Uses built-in .NET System.Windows.Forms (no extra modules needed)
# Note:    On Windows 10/11, BalloonTip automatically renders as modern toast
# =============================================================================

param(
    [string]$Title   = "Terminal Done",
    [string]$Message = "Your command has finished!",
    [int]$ExitCode   = 0
)

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

$notify = New-Object System.Windows.Forms.NotifyIcon

# Use built-in system icon
$notify.Icon = [System.Drawing.SystemIcons]::Information

# Set icon style based on exit code
if ($ExitCode -eq 0) {
    $notify.BalloonTipIcon = [System.Windows.Forms.ToolTipIcon]::Info
} else {
    $notify.BalloonTipIcon = [System.Windows.Forms.ToolTipIcon]::Warning
}

$notify.BalloonTipTitle = $Title
$notify.BalloonTipText  = $Message
$notify.Visible = $true
$notify.ShowBalloonTip(10000)

# Keep process alive briefly so notification displays, then clean up
Start-Sleep -Seconds 4
$notify.Dispose()
