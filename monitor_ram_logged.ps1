while ($true) {
    $free = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB
    $used = ((Get-CimInstance Win32_OperatingSystem).TotalVisibleMemorySize - (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory) / 1MB
    $line = "$(Get-Date -Format 'HH:mm:ss.fff') | Free: $([math]::Round($free,3)) GB | Used: $([math]::Round($used,3)) GB"
    Write-Host $line
    Add-Content -Path "ram_log.txt" -Value $line
    Start-Sleep -Milliseconds 500
}