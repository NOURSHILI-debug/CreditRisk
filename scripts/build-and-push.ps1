# build-and-push.ps1
# Phase 2F: Image Build and Deployment Automation
#
# Builds and pushes FastAPI and Streamlit container images to Amazon ECR.
#
# This script performs build-and-push ONLY. It does NOT create AWS resources,
# deploy ECS services, or modify infrastructure. It displays all actions
# before execution and requires explicit confirmation.
#
# Prerequisites:
#   - Docker Desktop installed and running
#   - AWS CLI configured with credentials
#   - ECR repositories created (credit-risk-fastapi, credit-risk-streamlit)
#
# Usage:
#   .\scripts\build-and-push.ps1 -ImageTag v1.0.0 -Region us-east-1 -AccountId 123456789012 -Confirm

[CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = "High")]
param(
    [Parameter(Mandatory = $true)]
    [string]$ImageTag,

    [Parameter(Mandatory = $true)]
    [string]$Region,

    [Parameter(Mandatory = $true)]
    [string]$AccountId,

    [Parameter(Mandatory = $true)]
    [switch]$Confirm
)

$ErrorActionPreference = "Stop"
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
$repoRoot = Split-Path -Parent $scriptRoot

# ---------------------------------------------------------------------------
# Validate AWS account and region
# ---------------------------------------------------------------------------
Write-Host "=== Validating AWS configuration ===" -ForegroundColor Cyan

$caller = aws sts get-caller-identity --region $Region 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Error "AWS credentials not configured or invalid. Run 'aws login' or configure credentials first."
}

$actualAccount = ($caller | ConvertFrom-Json).Account
if ($actualAccount -ne $AccountId) {
    Write-Error "Configured AWS account ($actualAccount) does not match --AccountId ($AccountId). Aborting to prevent pushing to wrong account."
}

Write-Host "AWS Account: $actualAccount"
Write-Host "AWS Region:  $Region"
Write-Host "Image Tag:   $ImageTag"
Write-Host ""

# ---------------------------------------------------------------------------
# Validate Docker is available
# ---------------------------------------------------------------------------
Write-Host "=== Validating Docker availability ===" -ForegroundColor Cyan
$dockerVersion = docker --version 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Error "Docker is not available. Install and start Docker Desktop, then retry."
}
Write-Host "Docker: $dockerVersion"

# ---------------------------------------------------------------------------
# Display planned actions
# ---------------------------------------------------------------------------
Write-Host "=== Planned Actions ===" -ForegroundColor Cyan
Write-Host "1. Log in to ECR registry: $AccountId.dkr.ecr.$Region.amazonaws.com"
Write-Host "2. Build FastAPI image:   credit-risk-fastapi:$ImageTag  (Dockerfile.fastapi)"
Write-Host "3. Build Streamlit image: credit-risk-streamlit:$ImageTag (Dockerfile.streamlit)"
Write-Host "4. Tag images with account/region/tag"
Write-Host "5. Push 4 image tags to ECR (fastapi:$ImageTag, fastapi:latest, streamlit:$ImageTag, streamlit:latest)"
Write-Host ""
Write-Warning "This will push container images to Amazon ECR. Image storage costs apply."
Write-Warning "No ECS services, task definitions, or ALB resources will be created or modified."
Write-Host ""

# ---------------------------------------------------------------------------
# Require explicit confirmation
# ---------------------------------------------------------------------------
if (-not $Confirm) {
    Write-Error "Use -Confirm to explicitly authorize the build-and-push operation."
}

$confirmChoice = Read-Host "Type 'YES' to proceed with build-and-push"
if ($confirmChoice -ne "YES") {
    Write-Host "Operation cancelled by user."
    exit 0
}

# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "=== Executing Build and Push ===" -ForegroundColor Green

$ecrRegistry = "$AccountId.dkr.ecr.$Region.amazonaws.com"

# 1. ECR login
Write-Host "`n[1/5] Authenticating Docker to ECR..." -ForegroundColor Cyan
$loginCmd = "aws ecr get-login-password --region $Region | docker login --username AWS --password-stdin $ecrRegistry"
Write-Host "  Command: $loginCmd"
& $loginCmd
if ($LASTEXITCODE -ne 0) {
    Write-Error "ECR login failed."
}

# 2. Build FastAPI
Write-Host "`n[2/5] Building FastAPI image..." -ForegroundColor Cyan
$fastapiTag = "$ecrRegistry/credit-risk-fastapi:$ImageTag"
& docker build -t credit-risk-fastapi:$ImageTag -f "$repoRoot/Dockerfile.fastapi" "$repoRoot"
if ($LASTEXITCODE -ne 0) {
    Write-Error "FastAPI image build failed."
}

# 3. Build Streamlit
Write-Host "`n[3/5] Building Streamlit image..." -ForegroundColor Cyan
$streamlitTag = "$ecrRegistry/credit-risk-streamlit:$ImageTag"
& docker build -t credit-risk-streamlit:$ImageTag -f "$repoRoot/Dockerfile.streamlit" "$repoRoot"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Streamlit image build failed."
}

# 4. Tag images
Write-Host "`n[4/5] Tagging images..." -ForegroundColor Cyan
$tags = @(
    @{ src = "credit-risk-fastapi:$ImageTag"; dst = "$ecrRegistry/credit-risk-fastapi:$ImageTag" },
    @{ src = "credit-risk-fastapi:$ImageTag"; dst = "$ecrRegistry/credit-risk-fastapi:latest" },
    @{ src = "credit-risk-streamlit:$ImageTag"; dst = "$ecrRegistry/credit-risk-streamlit:$ImageTag" },
    @{ src = "credit-risk-streamlit:$ImageTag"; dst = "$ecrRegistry/credit-risk-streamlit:latest" }
)

foreach ($t in $tags) {
    Write-Host "  docker tag $($t.src) $($t.dst)"
    & docker tag $t.src $t.dst
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Tagging failed: $($t.src) -> $($t.dst)"
    }
}

# 5. Push images
Write-Host "`n[5/5] Pushing images to ECR..." -ForegroundColor Cyan
foreach ($t in $tags) {
    Write-Host "  docker push $($t.dst)"
    & docker push $t.dst
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Push failed: $($t.dst)"
    }
}

# ---------------------------------------------------------------------------
# Complete
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "=== Build and Push Complete ===" -ForegroundColor Green
Write-Host "Images pushed to ECR:"
foreach ($t in $tags) {
    Write-Host "  $($t.dst)"
}
Write-Host ""
Write-Host "Next step: Register ECS task definitions and update services using deploy-ecs.ps1"
Write-Host "  .\scripts\deploy-ecs.ps1 -ImageTag $ImageTag -Region $Region -AccountId $AccountId -Confirm"
