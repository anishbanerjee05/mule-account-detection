# demo.ps1
#
# SINGLE COMMAND full demo launcher.
# Resets environment AND starts all services in correct order.
# Run this once before every demo session.
#
# Usage:
#   .\demo.ps1                    # reset + GNN + 2 consumers + dashboard
#   .\demo.ps1 -consumers 4       # use 4 consumers (only on high-RAM machines)
#   .\demo.ps1 -producers         # also auto-start producer
#   .\demo.ps1 -skip_reset        # skip reset, just restart services

param(
    [switch]$producers   = $false,
    [switch]$skip_reset  = $false,
    [int]$consumers      = 2
)

$projectPath = "C:\Users\ASUS\Desktop\mule_detection"
$venv        = "$projectPath\venv\Scripts\Activate.ps1"
$venv311     = "$projectPath\venv311\Scripts\Activate.ps1"

# ── Helper: print section header ─────────────────────────────────────────────
function Section($msg) {
    Write-Host ""
    Write-Host "── $msg" -ForegroundColor Cyan
}

# ── Helper: abort with error ──────────────────────────────────────────────────
function Abort($msg) {
    Write-Host ""
    Write-Host "ABORT: $msg" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║   Mule Account Detection System — Demo Launcher  ║" -ForegroundColor Cyan
Write-Host "║   CyberPeace Foundation Research Wing            ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# ════════════════════════════════════════════════════════════════════════════
# PHASE 1 — VERIFY DOCKER
# ════════════════════════════════════════════════════════════════════════════
Section "Phase 1/5 — Verifying Docker containers"
$containers = @("redis", "kafka", "zookeeper", "postgres")
foreach ($c in $containers) {
    $status = docker inspect --format "{{.State.Status}}" $c 2>$null
    if ($status -ne "running") {
        Write-Host "  $c`: NOT running — attempting docker-compose up..." -ForegroundColor Yellow
        Set-Location $projectPath
        docker-compose up -d | Out-Null
        Start-Sleep -Seconds 10
        $status = docker inspect --format "{{.State.Status}}" $c 2>$null
        if ($status -ne "running") {
            Abort "$c failed to start. Check Docker Desktop is running."
        }
    }
    Write-Host "  $c`: running" -ForegroundColor Green
}

# ════════════════════════════════════════════════════════════════════════════
# PHASE 2 — RESET ENVIRONMENT
# ════════════════════════════════════════════════════════════════════════════
if (-not $skip_reset) {
    Section "Phase 2/5 — Resetting environment (Redis flush + new Kafka topic)"

    # Flush Redis
    $redisResult = docker exec redis redis-cli FLUSHALL
    if ($redisResult -ne "OK") { Abort "Redis flush failed: $redisResult" }
    $dbSize = docker exec redis redis-cli DBSIZE
    if ($dbSize -ne "0") { Abort "Redis not empty after flush: $dbSize keys" }
    Write-Host "  Redis flushed (0 keys)" -ForegroundColor Green

    # Create new timestamped Kafka topic with 4 partitions
    $timestamp = Get-Date -Format "yyyyMMddHHmmss"
    $topicName = "transactions-raw-$timestamp"
    docker exec kafka kafka-topics --create --topic $topicName `
        --bootstrap-server localhost:9092 `
        --partitions 4 `
        --replication-factor 1 | Out-Null
    Start-Sleep -Seconds 2

    # Verify all 4 partitions are empty
    $offsetCheck = docker exec kafka kafka-run-class kafka.tools.GetOffsetShell `
        --broker-list localhost:9092 --topic $topicName
    $lines = $offsetCheck -split "`n"
    foreach ($line in $lines) {
        $line = $line.Trim()
        if ($line -ne "" -and $line -notmatch ":0$") {
            Abort "Partition not empty: $line"
        }
    }

    # Write topic name to shared config
    $topicName | Out-File -FilePath "$projectPath\active_topic.txt" `
        -Encoding ASCII -NoNewline
    Write-Host "  Kafka topic: $topicName (4 partitions, verified empty)" -ForegroundColor Green

} else {
    Section "Phase 2/5 — Skipping reset (using existing topic)"
    $topicName = Get-Content "$projectPath\active_topic.txt"
    Write-Host "  Active topic: $topicName" -ForegroundColor Green
}

# ════════════════════════════════════════════════════════════════════════════
# PHASE 3 — START GNN SERVICE
# ════════════════════════════════════════════════════════════════════════════
Section "Phase 3/5 — Starting GNN service (venv311, port 8002, workers=1)"

# Kill any existing gnn_service process on port 8002
$existing = netstat -ano | Select-String ":8002" | Select-String "LISTENING"
if ($existing) {
    $pid8002 = ($existing -split "\s+")[-1]
    Stop-Process -Id $pid8002 -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
    Write-Host "  Killed existing process on port 8002" -ForegroundColor Yellow
}

$gnnCmd = "cd '$projectPath'; & '$venv311'; " +
          "python -c `"import uvicorn; " +
          "uvicorn.run('gnn_service:app', host='0.0.0.0', port=8002, " +
          "workers=1, timeout_worker_healthcheck=30)`""
Start-Process powershell -ArgumentList "-NoExit", "-Command", $gnnCmd `
    -WindowStyle Normal

# Poll until healthy
Write-Host "  Waiting for GNN service..." -ForegroundColor Gray
$gnnReady = $false
for ($i = 0; $i -lt 40; $i++) {
    Start-Sleep -Seconds 2
    try {
        $resp = Invoke-WebRequest -UseBasicParsing -Uri "http://localhost:8002/health" `
            -TimeoutSec 2 -ErrorAction Stop
        if ($resp.StatusCode -eq 200) { $gnnReady = $true; break }
    } catch {}
    Write-Host "  Still loading... ($($i*2)s elapsed)" -ForegroundColor Gray
}
if (-not $gnnReady) { Abort "GNN service did not become healthy within 80s." }
Write-Host "  GNN service healthy." -ForegroundColor Green

# ════════════════════════════════════════════════════════════════════════════
# PHASE 4 — START CONSUMERS
# ════════════════════════════════════════════════════════════════════════════
Section "Phase 4/5 — Starting $consumers consumer(s)"

for ($i = 1; $i -le $consumers; $i++) {
    $consumerCmd = "cd '$projectPath'; & '$venv'; python consumer.py"
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $consumerCmd `
        -WindowStyle Normal
    Start-Sleep -Seconds 3
    Write-Host "  Consumer $i started" -ForegroundColor Green
}

# Give consumers time to subscribe to Kafka
Start-Sleep -Seconds 5

# ════════════════════════════════════════════════════════════════════════════
# PHASE 5 — START DASHBOARD
# ════════════════════════════════════════════════════════════════════════════
Section "Phase 5/5 — Starting dashboard"

$dashCmd = "cd '$projectPath'; & '$venv'; streamlit run dashboard.py"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $dashCmd `
    -WindowStyle Normal
Start-Sleep -Seconds 4
Start-Process "http://localhost:8501"
Write-Host "  Dashboard started at http://localhost:8501" -ForegroundColor Green

# ── Optional: auto-start producer ────────────────────────────────────────────
if ($producers) {
    Section "Bonus — Starting producer"
    $prodCmd = "cd '$projectPath'; & '$venv'; python producer.py"
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $prodCmd `
        -WindowStyle Normal
    Write-Host "  Producer started" -ForegroundColor Green
}

# ════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════════════════
Write-Host ""
Write-Host "╔══════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║   All systems operational                        ║" -ForegroundColor Green
Write-Host "╚══════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Write-Host "  GNN service:  http://localhost:8002/health" -ForegroundColor White
Write-Host "  Dashboard:    http://localhost:8501" -ForegroundColor White
Write-Host "  Topic:        $topicName" -ForegroundColor White
Write-Host "  Consumers:    $consumers active" -ForegroundColor White
Write-Host ""
Write-Host "  To start producing transactions:" -ForegroundColor Gray
Write-Host "    python producer.py" -ForegroundColor Gray
Write-Host ""
Write-Host "  To start with producer:" -ForegroundColor Gray
Write-Host "    .\demo.ps1 -producers" -ForegroundColor Gray
Write-Host ""
Write-Host "  To restart services without resetting Kafka/Redis:" -ForegroundColor Gray
Write-Host "    .\demo.ps1 -skip_reset" -ForegroundColor Gray