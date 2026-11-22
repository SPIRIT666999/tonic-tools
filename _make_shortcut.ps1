$ErrorActionPreference = "Stop"
$dir = Split-Path -Parent $MyInvocation.MyCommand.Path

$pyw = $null
$py = & py -3 -c "import sys;print(sys.executable)" 2>$null
if ($LASTEXITCODE -eq 0 -and $py) {
    $cand = Join-Path (Split-Path -Parent $py) "pythonw.exe"
    if (Test-Path $cand) { $pyw = $cand }
}
if (-not $pyw) {
    $cmd = Get-Command pythonw.exe -ErrorAction SilentlyContinue
    if ($cmd) { $pyw = $cmd.Source }
}

$ws = New-Object -ComObject WScript.Shell
$desktop = $ws.SpecialFolders.Item("Desktop")
$lnkPath = Join-Path $desktop ("Tonika.lnk")
$lnk = $ws.CreateShortcut($lnkPath)
if ($pyw) {
    $lnk.TargetPath = $pyw
    $lnk.Arguments = '"' + (Join-Path $dir "tonic_analyzer.py") + '"'
} else {
    $lnk.TargetPath = (Join-Path $dir "ZAPUSK.bat")
}
$lnk.WorkingDirectory = $dir
$lnk.Save()
Write-Output "OK"
Write-Output $lnkPath
