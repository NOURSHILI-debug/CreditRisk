# Amazon ECR Repository Configuration — Person 2 Components

## Overview

Amazon Elastic Container Registry (ECR) stores the Docker images for the
FastAPI backend and Streamlit dashboard. These are prepared configurations
only — no repositories have been created and no images have been pushed.

## Repository Naming Strategy

| Component | Repository Name | Image Tag Strategy |
|---|---|---|
| FastAPI | `credit-risk-fastapi` | Git commit SHA + `latest` for development |
| Streamlit | `credit-risk-streamlit` | Git commit SHA + `latest` for development |

Image tags use immutable, uniquely identifiable values (git commit SHA)
in addition to any development tag. Previous release tags are never
overwritten.

## Repository URIs

| Component | Repository URI Pattern |
|---|---|
| FastAPI | `ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com/credit-risk-fastapi` |
| Streamlit | `ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com/credit-risk-streamlit` |

Replace `ACCOUNT_ID` and `REGION` with actual values before use.

## Lifecycle Policy

Recommended ECR lifecycle policy to control storage costs:

| Rule | Description |
|---|---|
| Keep last 10 images per tag prefix | Retain recent images for rollback |
| Untagged images older than 7 days | Delete to save storage |
| Images older than 90 days | Delete regardless of tag |

## Build, Tag, and Push Commands

These commands are prepared but NOT executed. Run them manually after
creating the ECR repositories and configuring AWS credentials.

```powershell
# Set variables
$env:AWS_REGION = "us-east-1"
$env:ACCOUNT_ID = "123456789012"
$env:IMAGE_TAG = "v1.0.0"

# Authenticate Docker to ECR
aws ecr get-login-password --region $env:AWS_REGION | docker login --username AWS --password-stdin $env:ACCOUNT_ID.dkr.ecr.$env:AWS_REGION.amazonaws.com

# Build FastAPI image
docker build -t credit-risk-fastapi:$env:IMAGE_TAG -f Dockerfile.fastapi .
docker tag credit-risk-fastapi:$env:IMAGE_TAG $env:ACCOUNT_ID.dkr.ecr.$env:AWS_REGION.amazonaws.com/credit-risk-fastapi:$env:IMAGE_TAG
docker tag credit-risk-fastapi:$env:IMAGE_TAG $env:ACCOUNT_ID.dkr.ecr.$env:AWS_REGION.amazonaws.com/credit-risk-fastapi:latest

# Build Streamlit image
docker build -t credit-risk-streamlit:$env:IMAGE_TAG -f Dockerfile.streamlit .
docker tag credit-risk-streamlit:$env:IMAGE_TAG $env:ACCOUNT_ID.dkr.ecr.$env:AWS_REGION.amazonaws.com/credit-risk-streamlit:$env:IMAGE_TAG
docker tag credit-risk-streamlit:$env:IMAGE_TAG $env:ACCOUNT_ID.dkr.ecr.$env:AWS_REGION.amazonaws.com/credit-risk-streamlit:latest

# Push images
docker push $env:ACCOUNT_ID.dkr.ecr.$env:AWS_REGION.amazonaws.com/credit-risk-fastapi:$env:IMAGE_TAG
docker push $env:ACCOUNT_ID.dkr.ecr.$env:AWS_REGION.amazonaws.com/credit-risk-fastapi:latest
docker push $env:ACCOUNT_ID.dkr.ecr.$env:AWS_REGION.amazonaws.com/credit-risk-streamlit:$env:IMAGE_TAG
docker push $env:ACCOUNT_ID.dkr.ecr.$env:AWS_REGION.amazonaws.com/credit-risk-streamlit:latest
```

## Repository Creation

Create repositories before pushing (one-time operation):

```powershell
aws ecr create-repository --repository-name credit-risk-fastapi --region us-east-1
aws ecr create-repository --repository-name credit-risk-streamlit --region us-east-1
```

## Image Tagging Guidelines

- Use git commit SHA for production deployments (immutable, traceable)
- Use `latest` only for development/testing
- Never overwrite an existing release tag
- Tag images before pushing; do not re-tag after push

## Access Control

- ECR repository policies should restrict push/pull to authorized IAM roles
- ECS task execution role needs `ecr:GetAuthorizationToken` and
  `ecr:BatchCheckLayerAvailability`, `ecr:GetDownloadUrlForLayer`,
  `ecr:BatchGetImage` to pull images
- Developers need `ecr:PutImage`, `ecr:InitiateLayerUpload`,
  `ecr:UploadLayerPart`, `ecr:CompleteLayerUpload` to push images
