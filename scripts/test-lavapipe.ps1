<#
.SYNOPSIS
    Test HC4J heterogeneous multi-device collective communication with Intel GPU + Mesa Lavapipe (CPU).

.DESCRIPTION
    Discovers Mesa Lavapipe Vulkan ICD, configures process-scoped environment variables
    (VK_ADD_DRIVER_FILES, HC4J_COLLECTIVE_INCLUDE_CPU), and runs the CollectiveTest suite.

.PARAMETER IcdPath
    Explicit path to lvp_icd.x86_64.json.

.PARAMETER ForceTier
    Optional transport tier to force (auto, p2p, host_dma, staged). Defaults to 'auto'.

.PARAMETER Backend
    Optional collective backend to force (auto, vulkan, wgpu). Defaults to 'auto'.

.EXAMPLE
    .\scripts\test-lavapipe.ps1
    .\scripts\test-lavapipe.ps1 -ForceTier host_dma
#>

[CmdletBinding()]
param(
    [string]$IcdPath = "",
    [ValidateSet("auto", "p2p", "host_dma", "staged")]
    [string]$ForceTier = "auto",
    [ValidateSet("auto", "vulkan", "wgpu")]
    [string]$Backend = "auto"
)

$ErrorActionPreference = "Stop"

Write-Host "=== HC4J Lavapipe Heterogeneous Test Harness ===" -ForegroundColor Cyan

# ── 1. Discover Lavapipe ICD ────────────────────────────────────────────────
$resolvedIcd = $null

$candidates = @()
if ($IcdPath -and (Test-Path $IcdPath)) {
    $candidates += (Resolve-Path $IcdPath).Path
}
if ($env:HC4J_LAVAPIPE_ICD -and (Test-Path $env:HC4J_LAVAPIPE_ICD)) {
    $candidates += (Resolve-Path $env:HC4J_LAVAPIPE_ICD).Path
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repoLocalMesaX64 = Join-Path $repoRoot "tools\mesa\x64\lvp_icd.x86_64.json"
if (Test-Path $repoLocalMesaX64) {
    $candidates += (Resolve-Path $repoLocalMesaX64).Path
}

$repoLocalMesa = Join-Path $repoRoot "tools\mesa\lvp_icd.x86_64.json"
if (Test-Path $repoLocalMesa) {
    $candidates += (Resolve-Path $repoLocalMesa).Path
}

$systemMesaX64 = "C:\mesa\x64\lvp_icd.x86_64.json"
if (Test-Path $systemMesaX64) {
    $candidates += (Resolve-Path $systemMesaX64).Path
}

$systemMesa = "C:\mesa\lvp_icd.x86_64.json"
if (Test-Path $systemMesa) {
    $candidates += (Resolve-Path $systemMesa).Path
}

if ($candidates.Count -gt 0) {
    $resolvedIcd = $candidates[0]
}

# ── 2. Handle missing ICD ───────────────────────────────────────────────────
if (-not $resolvedIcd) {
    Write-Host "`n[!] Mesa Lavapipe Vulkan ICD not found." -ForegroundColor Yellow
    Write-Host "`nTo test heterogeneous multi-device collective (e.g. Intel GPU + Lavapipe CPU):" -ForegroundColor White
    Write-Host "  Step 1: Download latest 'mesa3d-*.7z' from https://github.com/pal1000/mesa-dist-win/releases" -ForegroundColor Gray
    Write-Host "  Step 2: Extract Mesa archive into tools\mesa\ (e.g. tools\mesa\x64\lvp_icd.x86_64.json)" -ForegroundColor Gray
    Write-Host "          (Checked: $repoLocalMesaX64, $repoLocalMesa)" -ForegroundColor DarkGray
    Write-Host "  Step 3: Re-run this script: .\scripts\test-lavapipe.ps1`n" -ForegroundColor Gray
    exit 0
}

Write-Host "[+] Found Lavapipe ICD: $resolvedIcd" -ForegroundColor Green

# ── 3. Configure process-scoped environment variables ───────────────────────
$origDriverFiles = $env:VK_ADD_DRIVER_FILES
$origIncludeCpu = $env:HC4J_COLLECTIVE_INCLUDE_CPU
$origForceTier = $env:HC4J_COLLECTIVE_FORCE_TIER
$origBackend = $env:HC4J_COLLECTIVE_BACKEND

try {
    $env:VK_ADD_DRIVER_FILES = $resolvedIcd
    $env:HC4J_COLLECTIVE_INCLUDE_CPU = "1"
    $env:HC4J_COLLECTIVE_FORCE_TIER = $ForceTier
    $env:HC4J_COLLECTIVE_BACKEND = $Backend

    Write-Host "[+] Configured environment:" -ForegroundColor DarkCyan
    Write-Host "    VK_ADD_DRIVER_FILES         = $env:VK_ADD_DRIVER_FILES" -ForegroundColor DarkGray
    Write-Host "    HC4J_COLLECTIVE_INCLUDE_CPU = $env:HC4J_COLLECTIVE_INCLUDE_CPU" -ForegroundColor DarkGray
    Write-Host "    HC4J_COLLECTIVE_FORCE_TIER  = $env:HC4J_COLLECTIVE_FORCE_TIER" -ForegroundColor DarkGray
    Write-Host "    HC4J_COLLECTIVE_BACKEND     = $env:HC4J_COLLECTIVE_BACKEND" -ForegroundColor DarkGray

    # ── 4. Execute Gradle Test Suite ─────────────────────────────────────────
    Write-Host "`n[>] Running CollectiveTest suite..." -ForegroundColor Cyan
    Push-Location $repoRoot
    try {
        & .\gradlew.bat :core:test --tests hc4j.CollectiveTest --rerun-tasks --info
    } finally {
        Pop-Location
    }
} finally {
    # Restore original environment
    $env:VK_ADD_DRIVER_FILES = $origDriverFiles
    $env:HC4J_COLLECTIVE_INCLUDE_CPU = $origIncludeCpu
    $env:HC4J_COLLECTIVE_FORCE_TIER = $origForceTier
    $env:HC4J_COLLECTIVE_BACKEND = $origBackend
}

Write-Host "`n=== Test Run Finished ===" -ForegroundColor Green
