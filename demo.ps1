param(
    [switch]$producers  = $false,
    [switch]$skip_reset = $false,
    [int]$consumers     = 2
)

# Project directory is resolved from this script's own location, so the
# launcher works from any clone location or user account.
$projectPath = $PSScriptRoot
$venv        = Join-Path $projectPath "venv\Scripts\Activate.ps1"
$venv311     = Join-Path $projectPath "venv311\Scripts\Activate.ps1"

function Section($msg) {
    Write-Host ""
    Write-Host "-- $msg" -ForegroundColor Cyan
}

function Abort($msg) {
    Write-Host "ABORT: $msg" -ForegroundColor Red
    exit 1
}

# Returns the parsed /stats JSON from the GNN service, or $null if unreachable.
function Get-GnnStats {
    try {
        $resp = Invoke-WebRequest -UseBasicParsing -Uri "http://localhost:8002/stats" -TimeoutSec 3 -ErrorAction Stop
        return ($resp.Content | ConvertFrom-Json)
    } catch {
        return $null
    }
}

# Healthy means the service answers AND the persistent embedding store loaded.
function Test-GnnReady {
    $s = Get-GnnStats
    return (($null -ne $s) -and ($s.persistent_store_loaded -eq $true))
}

Write-Host ""
Write-Host "=== Mule Account Detection System - Demo Launcher ===" -ForegroundColor Cyan
Write-Host "=== CyberPeace Foundation Research Wing            ===" -ForegroundColor Cyan
Write-Host ""

Set-Location $projectPath

Section "Phase 0/5 - Checking prerequisites"

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Abort "docker not found. Install and start Docker Desktop (see README)."
}

$required = @(
    "ibm_aml_small.csv",
    "xgb_hybrid_model.pkl",
    "scaler_hybrid.pkl",
    "hybrid_model_config.json",
    "best_gnn_model.pt",
    "embeddings_store.parquet",
    "start_gnn.py",
    "warm_redis_cache.py",
    "consumer.py",
    "producer.py",
    "dashboard_v2_light_theme.py"
)
$missing = @()
foreach ($f in $required) {
    if (-not (Test-Path (Join-Path $projectPath $f))) { $missing += $f }
}
if ($missing.Count -gt 0) {
    foreach ($f in $missing) { Write-Host "  MISSING: $f" -ForegroundColor Red }
    Abort "Required files are missing. See README sections 'Datasets' and 'Model artifacts' for how to obtain or regenerate them."
}

foreach ($v in @($venv, $venv311)) {
    if (-not (Test-Path $v)) {
        Abort "Virtual environment not found: $v. Create it as described in the README setup section."
    }
}
Write-Host "  All required files and environments found" -ForegroundColor Green

Section "Phase 1/5 - Verifying Docker containers"
$containers = @("redis", "kafka", "zookeeper")
foreach ($c in $containers) {
    $status = docker inspect --format "{{.State.Status}}" $c 2>$null
    if ($status -ne "running") {
        Write-Host "  $c not running - attempting docker compose up..." -ForegroundColor Yellow
        docker compose up -d | Out-Null
        Start-Sleep -Seconds 10
        $status = docker inspect --format "{{.State.Status}}" $c 2>$null
        if ($status -ne "running") {
            Abort "$c failed to start. Check Docker Desktop is running."
        }
    }
    Write-Host "  $c : running" -ForegroundColor Green
}

if (-not $skip_reset) {
    Section "Phase 2/5 - Resetting environment"

    $redisResult = docker exec redis redis-cli FLUSHALL
    if ($redisResult -ne "OK") { Abort "Redis flush failed: $redisResult" }
    $dbSize = docker exec redis redis-cli DBSIZE
    if ($dbSize -ne "0") { Abort "Redis not empty after flush. Keys: $dbSize" }
    Write-Host "  Redis flushed. Keys: 0" -ForegroundColor Green

    $timestamp = Get-Date -Format "yyyyMMddHHmmss"
    $topicName = "transactions-raw-$timestamp"
    docker exec kafka kafka-topics --create --topic $topicName --bootstrap-server localhost:9092 --partitions 4 --replication-factor 1 | Out-Null
    Start-Sleep -Seconds 2

    $offsetCheck = docker exec kafka kafka-run-class kafka.tools.GetOffsetShell --broker-list localhost:9092 --topic $topicName
    $lines = $offsetCheck -split "`n"
    foreach ($line in $lines) {
        $line = $line.Trim()
        if ($line -ne "" -and $line -notmatch ":0$") {
            Abort "Partition not empty: $line"
        }
    }

    $topicName | Out-File -FilePath (Join-Path $projectPath "active_topic.txt") -Encoding ASCII -NoNewline
    Write-Host "  Kafka topic: $topicName" -ForegroundColor Green
    Write-Host "  Partitions: 4, verified empty" -ForegroundColor Green

} else {
    Section "Phase 2/5 - Skipping reset"
    $topicFile = Join-Path $projectPath "active_topic.txt"
    if (-not (Test-Path $topicFile)) {
        Abort "active_topic.txt not found. Run once without -skip_reset first."
    }
    $topicName = Get-Content $topicFile
    Write-Host "  Active topic: $topicName" -ForegroundColor Green
}

Section "Phase 3/5 - Starting GNN service"

$gnnAlreadyRunning = Test-GnnReady
if ($gnnAlreadyRunning) {
    Write-Host "  GNN service already running and healthy. Skipping startup." -ForegroundColor Green
}

if (-not $gnnAlreadyRunning) {
    # Kill any existing process on port 8002
    $existing = netstat -ano | Select-String ":8002" | Select-String "LISTENING"
    if ($existing) {
        $pidLine = $existing | Select-Object -First 1
        $pid8002 = ($pidLine -split "\s+")[-1]
        Stop-Process -Id $pid8002 -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 2
        Write-Host "  Killed existing process on port 8002" -ForegroundColor Yellow
    }

    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$projectPath'; & '$venv311'; python start_gnn.py" -WindowStyle Normal

    Write-Host "  Waiting for GNN service to initialize..." -ForegroundColor Gray
    $gnnReady = $false
    for ($i = 0; $i -lt 40; $i++) {
        Start-Sleep -Seconds 2
        if (Test-GnnReady) { $gnnReady = $true; break }
        Write-Host "  Loading... $($i*2)s elapsed" -ForegroundColor Gray
    }
    if (-not $gnnReady) {
        $s = Get-GnnStats
        if ($null -ne $s) {
            Abort "GNN service is up but the persistent embedding store did not load. Check embeddings_store.parquet (see README)."
        }
        Abort "GNN service did not become healthy within 80s. Check the GNN service window for errors."
    }
    Write-Host "  GNN service healthy." -ForegroundColor Green
}

Section "Phase 3b/5 - Warming Redis embedding cache"
$statsResp = Get-GnnStats
if ($null -ne $statsResp) { $cachedCount = $statsResp.redis_cached_embeddings } else { $cachedCount = 0 }
if ($cachedCount -gt 100000) {
    Write-Host "  Redis already warmed with $cachedCount embeddings. Skipping." -ForegroundColor Green
} else {
    Write-Host "  Loading embeddings into Redis..." -ForegroundColor Gray
    & powershell -Command "cd '$projectPath'; & '$venv311'; python warm_redis_cache.py"
    if ($LASTEXITCODE -ne 0) { Abort "warm_redis_cache.py failed (exit code $LASTEXITCODE)." }
    Write-Host "  Redis cache warmed." -ForegroundColor Green
}

Section "Phase 4/5 - Starting $consumers consumer(s)"
for ($i = 1; $i -le $consumers; $i++) {
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$projectPath'; & '$venv'; python consumer.py" -WindowStyle Normal
    Start-Sleep -Seconds 3
    Write-Host "  Consumer $i started" -ForegroundColor Green
}
Start-Sleep -Seconds 5

Section "Phase 5/5 - Starting dashboard"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$projectPath'; & '$venv'; streamlit run dashboard_v2_light_theme.py" -WindowStyle Normal
Start-Sleep -Seconds 4
Start-Process "http://localhost:8501"
Write-Host "  Dashboard at http://localhost:8501" -ForegroundColor Green

if ($producers) {
    Section "Starting producer"
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$projectPath'; & '$venv'; python producer.py" -WindowStyle Normal
    Write-Host "  Producer started" -ForegroundColor Green
}

Write-Host ""
Write-Host "=== All systems operational ===" -ForegroundColor Green
Write-Host "  GNN service : http://localhost:8002/health" -ForegroundColor White
Write-Host "  Dashboard   : http://localhost:8501" -ForegroundColor White
Write-Host "  Topic       : $topicName" -ForegroundColor White
Write-Host "  Consumers   : $consumers active" -ForegroundColor White
Write-Host ""
Write-Host "  Start producer  : python producer.py" -ForegroundColor Gray
Write-Host "  With producer   : .\demo.ps1 -producers" -ForegroundColor Gray
Write-Host "  Skip reset      : .\demo.ps1 -skip_reset" -ForegroundColor Gray