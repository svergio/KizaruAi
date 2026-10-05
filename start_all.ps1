# Start all KizaruAi bot services in separate visible ConHost consoles
# (not Windows Terminal tabs). Usage:
#   powershell -ExecutionPolicy Bypass -File .\start_all.ps1
# Optional: -SkipDiscord / -SkipMood / -SkipPic

param(
    [switch]$SkipDiscord,
    [switch]$SkipMood,
    [switch]$SkipPic
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not (Test-Path (Join-Path $Root "app\orchestrator"))) {
    $Root = Split-Path -Parent $Root
}
Set-Location $Root

$Logs = Join-Path $Root "var\logs"
New-Item -ItemType Directory -Force -Path $Logs | Out-Null

# Load .env (KEY=VALUE), do not print secrets
$EnvFile = Join-Path $Root ".env"
if (Test-Path $EnvFile) {
    Get-Content $EnvFile -Encoding UTF8 | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line.StartsWith("#")) { return }
        $eq = $line.IndexOf("=")
        if ($eq -lt 1) { return }
        $key = $line.Substring(0, $eq).Trim()
        $val = $line.Substring($eq + 1).Trim()
        if (($val.StartsWith('"') -and $val.EndsWith('"')) -or ($val.StartsWith("'") -and $val.EndsWith("'"))) {
            $val = $val.Substring(1, $val.Length - 2)
        }
        [Environment]::SetEnvironmentVariable($key, $val, "Process")
    }
}

$Py = (Get-Command python -ErrorAction Stop).Source
$Db = "data/bot.db"

function Start-KizaruWindow {
    param(
        [Parameter(Mandatory)][string]$Title,
        [Parameter(Mandatory)][string]$LogStem,
        [Parameter(Mandatory)][string]$PyArgs
    )
    $outLog = Join-Path $Logs "$LogStem.out.log"
    $errLog = Join-Path $Logs "$LogStem.err.log"
    # Visible console: tee to logs while keeping stdout/stderr on screen
    $inner = @"
`$Host.UI.RawUI.WindowTitle = '$Title'
Set-Location '$Root'
`$env:PYTHONUNBUFFERED = '1'
Write-Host '[kizaru] $Title starting...' -ForegroundColor Cyan
& '$Py' $PyArgs 2>&1 | Tee-Object -FilePath '$outLog' -Append
Write-Host '[kizaru] $Title exited code='`$LASTEXITCODE -ForegroundColor Yellow
Read-Host 'Press Enter to close'
"@
    $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($inner))
    # Force classic ConHost (not Windows Terminal tabs) so each service is its own visible window.
    $proc = Start-Process -FilePath "conhost.exe" -WorkingDirectory $Root `
        -ArgumentList @(
            "powershell.exe",
            "-NoExit",
            "-ExecutionPolicy", "Bypass",
            "-EncodedCommand", $encoded
        ) `
        -WindowStyle Normal -PassThru
    [PSCustomObject]@{
        Title = $Title
        Pid   = $proc.Id
        Log   = $outLog
    }
}

$started = @()
$started += Start-KizaruWindow "kizaru-orchestrator" "orchestrator" "-m app.orchestrator --db $Db"
Start-Sleep -Seconds 1

$workers = @(
    @{ Title = "kizaru-lstm";       Stem = "worker-lstm";       Args = "-m app.workers.lstm checkpoints/kizaru.pt --db $Db" }
    @{ Title = "kizaru-lstm-word";  Stem = "worker-lstm-word";  Args = "-m app.workers.lstm_word checkpoints/kizaru_word.pt --db $Db" }
    @{ Title = "kizaru-transformer"; Stem = "worker-transformer"; Args = "-m app.workers.transformer checkpoints/transformer.pt --db $Db" }
    @{ Title = "kizaru-gpt";        Stem = "worker-gpt";        Args = "-m app.workers.gpt checkpoints/gpt.pt --db $Db" }
    @{ Title = "kizaru-deepseek";   Stem = "worker-deepseek";   Args = "-m app.workers.deepseek checkpoints/deepseek.pt --db $Db" }
    @{ Title = "kizaru-bert-gen";   Stem = "worker-bert-gen";   Args = "-m app.workers.bert_gen checkpoints/bert_gen.pt --db $Db" }
)

foreach ($w in $workers) {
    $started += Start-KizaruWindow $w.Title $w.Stem $w.Args
    Start-Sleep -Milliseconds 400
}

if (-not $SkipMood) {
    $started += Start-KizaruWindow "kizaru-mood" "worker-mood" "-m app.workers.mood checkpoints/mood.pt --db $Db"
}
if (-not $SkipPic) {
    $started += Start-KizaruWindow "kizaru-pic-still" "worker-pic-still" "-m app.workers.still --db $Db"
    $started += Start-KizaruWindow "kizaru-pic-video" "worker-pic-video" "-m app.workers.video --db $Db"
}

$token = [Environment]::GetEnvironmentVariable("DISCORD_TOKEN", "Process")
if (-not $SkipDiscord) {
    if ($token) {
        $started += Start-KizaruWindow "kizaru-discord" "discord" "-m app.discord --db $Db"
        Write-Host "Discord: started (DISCORD_TOKEN present)"
    } else {
        Write-Host "Discord: SKIPPED (DISCORD_TOKEN not set in env/.env)"
    }
}

Write-Host ""
Write-Host "Started $($started.Count) visible windows:"
$started | Format-Table -AutoSize
$pidFile = Join-Path $Logs "start_all_pids.txt"
$started | ForEach-Object { "$($_.Pid)`t$($_.Title)" } | Set-Content -Path $pidFile -Encoding UTF8
Write-Host "PID list: $pidFile"
Write-Host "Logs: $Logs"
