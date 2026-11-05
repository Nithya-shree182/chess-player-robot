# Windows: start the chess pipeline (arm -> look pose, camera, YOLO, Stockfish) and open its window in the default browser.
# Usage (from the repo folder):  scripts\run_pipeline.ps1 [--dry-run] [--skill 0-20] [--no-webcam]
# The game starts PAUSED: press Play (or Space) in the window. Voice (Robo) is off unless you pass --voice.
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$py = if ($env:CHESS_PY) { $env:CHESS_PY } else { "C:\Users\asus\lerobot-env\Scripts\python.exe" }
$env:PYTHONPATH = "$repo\src"
$env:PYTHONIOENCODING = "utf-8"
$port = if ($env:PORT) { $env:PORT } else { "8770" }
Set-Location $repo
Start-Job -ScriptBlock { param($p) Start-Sleep 8; Start-Process "http://localhost:$p/" } -ArgumentList $port | Out-Null
& $py -m chess_player.ui.server --port $port @args
