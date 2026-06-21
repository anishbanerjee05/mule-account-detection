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

# Step 2: Create a brand-new, uniquely-named Kafka topic
# Never reuse 'transactions-raw' - a fresh name sidesteps the delete
# ambiguity entirely. No race condition, no trusting Kafka's delete.
$timestamp = Get-Date -Format "yyyyMMddHHmmss"
$topicName = "transactions-raw-$timestamp"

Write-Host ""
Write-Host "[2/4] Creating fresh Kafka topic: $topicName ..." -ForegroundColor Yellow
docker exec kafka kafka-topics --create --topic $topicName --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1 | Out-Null

# Step 3: VERIFY the new topic is actually empty - do not trust the
# create command's exit code alone.
Start-Sleep -Seconds 2
$offsetCheck = docker exec kafka kafka-run-class kafka.tools.GetOffsetShell --broker-list localhost:9092 --topic $topicName
Write-Host "  Offset check: $offsetCheck" -ForegroundColor Gray

if ($offsetCheck -notmatch ":0$") {
    Write-Host "  FAILED - new topic is not empty (offset check did not end in :0)." -ForegroundColor Red
    Write-Host "  This should never happen with a freshly-named topic. Stopping." -ForegroundColor Red
    exit 1
}
Write-Host "  Verified: topic '$topicName' is genuinely empty (offset 0)." -ForegroundColor Green

# Step 4: Write the active topic name to a shared config file
$topicName | Out-File -FilePath "active_topic.txt" -Encoding ASCII -NoNewline
Write-Host ""
Write-Host "[3/4] Wrote active topic name to active_topic.txt" -ForegroundColor Yellow
Write-Host "  Active topic: $topicName" -ForegroundColor Green

# Step 5: Final verification summary
Write-Host ""
Write-Host "[4/4] Reset complete. Verified clean state:" -ForegroundColor Yellow
Write-Host "  Redis keys:    0" -ForegroundColor Green
Write-Host "  Kafka topic:   $topicName (offset 0, verified)" -ForegroundColor Green
Write-Host ""
Write-Host "You can now start: gnn_service.py -> consumer.py -> producer.py" -ForegroundColor Cyan
Write-Host "Both consumer.py and producer.py will automatically use the new topic." -ForegroundColor Cyan