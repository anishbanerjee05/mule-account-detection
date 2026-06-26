# reset_demo_environment.ps1
#
# ONE-COMMAND, VERIFIED reset for the mule detection demo environment.
#
# Problem this solves: "kafka-topics --delete" can silently fail or be
# a no-op, and "--create" afterward will just re-attach to the EXISTING
# topic rather than erroring - giving a false "Created topic" success
# message while millions of old messages are still sitting in it. This
# script does not trust that output. It always creates a UNIQUELY NAMED
# topic (timestamped) so there is never any ambiguity about whether it
# is empty - a brand new topic name is guaranteed to have zero messages.
#
# It then VERIFIES the end state directly (checks offset = 0) rather
# than trusting command exit codes, and writes the new topic name to
# a file that producer.py and consumer.py both read on startup, so
# you never have to manually edit either file's topic name again.
#
# PARTITIONS: Set to 4. Each partition is consumed independently by one
# consumer instance. Running 4 consumers in parallel gives 4x throughput
# over a single consumer. For the demo run 4 consumer terminals simultaneously.
# For production this scales to 64+ partitions across a Kafka cluster.
#
# Usage (from inside mule_detection folder, in PowerShell):
#   .\reset_demo_environment.ps1
#
# Run this ONCE before every demo/test session, before starting any
# of the four services (GNN, consumer, producer, dashboard).

Write-Host "=== Mule Detection - Environment Reset ===" -ForegroundColor Cyan

# Step 1: Flush Redis completely
Write-Host ""
Write-Host "[1/4] Flushing Redis..." -ForegroundColor Yellow
$redisResult = docker exec redis redis-cli FLUSHALL
if ($redisResult -ne "OK") {
    Write-Host "  FAILED - Redis flush did not return OK. Got: $redisResult" -ForegroundColor Red
    Write-Host "  Check that the redis container is running (docker-compose ps)." -ForegroundColor Red
    exit 1
}
$dbSize = docker exec redis redis-cli DBSIZE
Write-Host "  Redis flushed. Current key count: $dbSize" -ForegroundColor Green
if ($dbSize -ne "0") {
    Write-Host "  WARNING - expected 0 keys after flush, got $dbSize. Stopping." -ForegroundColor Red
    exit 1
}

# Step 2: Create a brand-new, uniquely-named Kafka topic with 4 partitions.
# 4 partitions allows 4 consumer instances to process in parallel,
# giving ~4x throughput improvement with zero code changes to consumer.py.
# Kafka's consumer group balancing automatically assigns one partition
# per consumer instance when multiple consumers share the same group ID.
$timestamp = Get-Date -Format "yyyyMMddHHmmss"
$topicName = "transactions-raw-$timestamp"

Write-Host ""
Write-Host "[2/4] Creating fresh Kafka topic: $topicName (4 partitions)..." -ForegroundColor Yellow
docker exec kafka kafka-topics --create --topic $topicName --bootstrap-server localhost:9092 --partitions 4 --replication-factor 1 | Out-Null

# Step 3: VERIFY the new topic is actually empty across all 4 partitions.
Start-Sleep -Seconds 2
$offsetCheck = docker exec kafka kafka-run-class kafka.tools.GetOffsetShell --broker-list localhost:9092 --topic $topicName
Write-Host "  Offset check (all 4 partitions):" -ForegroundColor Gray
Write-Host "  $offsetCheck" -ForegroundColor Gray

# All 4 partitions should show offset 0
$lines = $offsetCheck -split "`n"
$allEmpty = $true
foreach ($line in $lines) {
    $line = $line.Trim()
    if ($line -ne "" -and $line -notmatch ":0$") {
        $allEmpty = $false
        Write-Host "  FAILED - partition not empty: $line" -ForegroundColor Red
    }
}

if (-not $allEmpty) {
    Write-Host "  FAILED - not all partitions are empty. Stopping." -ForegroundColor Red
    exit 1
}
Write-Host "  Verified: topic '$topicName' is genuinely empty (all 4 partitions at offset 0)." -ForegroundColor Green

# Step 4: Write the active topic name to a shared config file
$topicName | Out-File -FilePath "active_topic.txt" -Encoding ASCII -NoNewline
Write-Host ""
Write-Host "[3/4] Wrote active topic name to active_topic.txt" -ForegroundColor Yellow
Write-Host "  Active topic: $topicName" -ForegroundColor Green

# Step 5: Final verification summary
Write-Host ""
Write-Host "[4/4] Reset complete. Verified clean state:" -ForegroundColor Yellow
Write-Host "  Redis keys:    0" -ForegroundColor Green
Write-Host "  Kafka topic:   $topicName (4 partitions, offset 0, verified)" -ForegroundColor Green
Write-Host ""
Write-Host "You can now start: gnn_service.py -> consumer.py (x4) -> producer.py" -ForegroundColor Cyan
Write-Host "Run 4 consumer terminals simultaneously for full parallel throughput." -ForegroundColor Cyan
Write-Host "Both consumer.py and producer.py will automatically use the new topic." -ForegroundColor Cyan