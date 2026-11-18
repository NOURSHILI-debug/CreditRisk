# deploy-ecs.ps1
# Phase 2F: Image Build and Deployment Automation
#
# Registers ECS task definitions and updates ECS services for FastAPI and
# Streamlit. This script performs deployment configuration updates ONLY.
# It does NOT create ECR repositories, ALB resources, VPC resources, or
# IAM roles. It displays all actions before execution and requires explicit
# confirmation.
#
# Prerequisites:
#   - AWS CLI configured with credentials
#   - ECR repositories created and images pushed
#   - ECS cluster created (with Service Connect for internal networking)
#   - IAM execution and task roles created
#   - CloudWatch log groups created
#
# Usage:
#   .\scripts\deploy-ecs.ps1 -ImageTag v1.0.0 -Region us-east-1 -AccountId 123456789012 -Cluster credit-risk -Confirm

[CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = "High")]
param(
    [Parameter(Mandatory = $true)]
    [string]$ImageTag,

    [Parameter(Mandatory = $true)]
    [string]$Region,

    [Parameter(Mandatory = $true)]
    [string]$AccountId,

    [Parameter(Mandatory = $true)]
    [string]$Cluster,

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
    Write-Error "Configured AWS account ($actualAccount) does not match --AccountId ($AccountId). Aborting to prevent deploying to wrong account."
}

Write-Host "AWS Account: $actualAccount"
Write-Host "AWS Region:  $Region"
Write-Host "Image Tag:   $ImageTag"
Write-Host "ECS Cluster: $Cluster"
Write-Host ""

# ---------------------------------------------------------------------------
# Validate prerequisites
# ---------------------------------------------------------------------------
Write-Host "=== Validating Prerequisites ===" -ForegroundColor Cyan

# Check ECR repositories exist
foreach ($repo in @("credit-risk-fastapi", "credit-risk-streamlit")) {
    $repoCheck = aws ecr describe-repositories --repository-names $repo --region $Region 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "ECR repository '$repo' not found. Create it first: aws ecr create-repository --repository-name $repo --region $Region"
    } else {
        Write-Host "  ECR repository '$repo' exists"
    }
}

# Check ECS cluster exists
$clusterCheck = aws ecs describe-clusters --clusters $Cluster --region $Region 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Warning "ECS cluster '$Cluster' not found. Create it first with Service Connect enabled."
} else {
    Write-Host "  ECS cluster '$Cluster' exists"
}

Write-Host ""

# ---------------------------------------------------------------------------
# Display planned actions
# ---------------------------------------------------------------------------
Write-Host "=== Planned Actions ===" -ForegroundColor Cyan
Write-Host "1. Register FastAPI task definition:  credit-risk-fastapi  (image: $ImageTag)"
Write-Host "2. Register Streamlit task definition: credit-risk-streamlit (image: $ImageTag)"
Write-Host "3. Update ECS service: credit-risk-fastapi  (if service exists)"
Write-Host "4. Update ECS service: credit-risk-streamlit (if service exists)"
Write-Host ""
Write-Warning "This will register task definitions and may update ECS services."
Write-Warning "No ECR repositories, ALB resources, VPC resources, or IAM roles will be created."
Write-Host ""

# ---------------------------------------------------------------------------
# Require explicit confirmation
# ---------------------------------------------------------------------------
if (-not $Confirm) {
    Write-Error "Use -Confirm to explicitly authorize the deployment operation."
}

$confirmChoice = Read-Host "Type 'YES' to proceed with deployment"
if ($confirmChoice -ne "YES") {
    Write-Host "Operation cancelled by user."
    exit 0
}

# ---------------------------------------------------------------------------
# Execute: Register task definitions
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "=== Registering Task Definitions ===" -ForegroundColor Green

$fastapiTd = "$repoRoot/aws/ecs/fastapi-task-definition.json"
$streamlitTd = "$repoRoot/aws/ecs/streamlit-task-definition.json"

# Replace placeholders in task definition JSON
$fastapiTdContent = Get-Content $fastapiTd -Raw
$fastapiTdContent = $fastapiTdContent -replace "ACCOUNT_ID", $AccountId
$fastapiTdContent = $fastapiTdContent -replace "REGION", $Region
$fastapiTdContent = $fastapiTdContent -replace "IMAGE_TAG", $ImageTag
$fastapiTdTemp = "$repoRoot/aws/ecs/fastapi-task-definition.rendered.json"
$fastapiTdContent | Set-Content $fastapiTdTemp

$streamlitTdContent = Get-Content $streamlitTd -Raw
$streamlitTdContent = $streamlitTdContent -replace "ACCOUNT_ID", $AccountId
$streamlitTdContent = $streamlitTdContent -replace "REGION", $Region
$streamlitTdContent = $streamlitTdContent -replace "IMAGE_TAG", $ImageTag
$streamlitTdTemp = "$repoRoot/aws/ecs/streamlit-task-definition.rendered.json"
$streamlitTdContent | Set-Content $streamlitTdTemp

# Register FastAPI task definition
Write-Host "  Registering FastAPI task definition..."
$result = aws ecs register-task-definition --cli-input-json file://$fastapiTdTemp --region $Region 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to register FastAPI task definition.`n$result"
} else {
    Write-Host "  FastAPI task definition registered successfully"
}

# Register Streamlit task definition
Write-Host "  Registering Streamlit task definition..."
$result = aws ecs register-task-definition --cli-input-json file://$streamlitTdTemp --region $Region 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to register Streamlit task definition.`n$result"
} else {
    Write-Host "  Streamlit task definition registered successfully"
}

# Clean up temp files
Remove-Item $fastapiTdTemp -ErrorAction SilentlyContinue
Remove-Item $streamlitTdTemp -ErrorAction SilentlyContinue

# ---------------------------------------------------------------------------
# Execute: Update ECS services (if they exist)
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "=== Updating ECS Services ===" -ForegroundColor Green

foreach ($service in @("credit-risk-fastapi", "credit-risk-streamlit")) {
    $svcCheck = aws ecs describe-services --cluster $Cluster --services $service --region $Region 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  Service '$service' not found in cluster '$Cluster'. Skipping service update (create service manually)."
        continue
    }

    Write-Host "  Updating service '$service'..."
    $result = aws ecs update-service --cluster $Cluster --service $service --task-definition $service --region $Region 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "  Failed to update service '$service': $result"
    } else {
        Write-Host "  Service '$service' updated successfully"
    }
}

# ---------------------------------------------------------------------------
# Complete
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "=== Deployment Complete ===" -ForegroundColor Green
Write-Host "Task definitions registered with image tag: $ImageTag"
Write-Host "ECS services updated (if they existed)."
Write-Host ""
Write-Host "Verify deployment:"
Write-Host "  aws ecs describe-services --cluster $Cluster --services credit-risk-fastapi,credit-risk-streamlit --region $Region"
Write-Host ""
Write-Host "Check CloudWatch logs:"
Write-Host "  aws logs describe-log-groups --log-group-name-prefix /ecs/credit-risk --region $Region"
