# ECS Task Definition Notes — Person 2 Components

## Overview

Task definitions for deploying the FastAPI backend and Streamlit dashboard
on AWS ECS with Fargate. These are templates — replace `ACCOUNT_ID`, `REGION`,
and `IMAGE_TAG` with actual values before use.

## Files

| File | Purpose |
|---|---|
| `fastapi-task-definition.json` | FastAPI backend task definition |
| `streamlit-task-definition.json` | Streamlit dashboard task definition |

## Resource Allocation

| Component | CPU | Memory | Rationale |
|---|---|---|---|
| FastAPI | 256 (0.25 vCPU) | 512 MB | Lightweight API server; model loading is memory-intensive but CPU-light |
| Streamlit | 512 (0.5 vCPU) | 1024 MB | UI server with embedded LangGraph agent; more memory for agent state |

## Networking

Both task definitions use `awsvpc` network mode (required for Fargate).
Each task gets its own ENI and private IP. Service discovery between
FastAPI and Streamlit is NOT via Docker Compose service DNS — see
`aws/networking/service-connect.md` for the ECS networking approach.

## IAM Roles

| Role | Purpose | Permissions |
|---|---|---|
| `ecsTaskExecutionRole` | ECS agent pulls images, writes logs | `AmazonECSTaskExecutionRolePolicy` + Secrets Manager read |
| `credit-risk-fastapi-task` | FastAPI task permissions | Minimal — only what FastAPI needs (none by default) |
| `credit-risk-streamlit-task` | Streamlit task permissions | Minimal — no S3, Kafka, or SageMaker access |

## Secrets

The FastAPI task definition includes a placeholder secret reference for
`OPENAI_API_KEY` from AWS Secrets Manager. This is a template — create the
secret in Secrets Manager and update the ARN before deployment.

Streamlit does not require secrets in production (no API keys needed
directly; it calls FastAPI over internal networking).

## Health Checks

| Component | Command | Interval | Timeout | Retries | Start Period |
|---|---|---|---|---|---|
| FastAPI | `curl -f http://localhost:8000/health` | 30s | 10s | 3 | 10s |
| Streamlit | `curl -f http://localhost:8501/_stcore/health` | 30s | 10s | 3 | 15s |

FastAPI health distinguishes liveness (HTTP 200 = process alive) from
readiness (JSON `status == "ok"` = model + data loaded). ECS health checks
use liveness only; ALB target group health checks can use either.

## Logging

Both containers use the `awslogs` log driver with separate CloudWatch log
groups:
- `/ecs/credit-risk-fastapi`
- `/ecs/credit-risk-streamlit`

Logs are structured JSON (FastAPI) or standard Python logging (Streamlit).
No API keys or applicant data are logged.

## Placeholders to Replace

Before deploying, replace all occurrences of:

| Placeholder | Description |
|---|---|
| `ACCOUNT_ID` | AWS account ID |
| `REGION` | AWS region (e.g., `us-east-1`) |
| `IMAGE_TAG` | Container image tag (e.g., `v1.0.0` or git commit SHA) |

## Least Privilege

- Streamlit task role has NO AWS service permissions (it only calls FastAPI).
- FastAPI task role has NO AWS service permissions by default.
- Secrets Manager access is granted via the execution role only for reading
  the `OPENAI_API_KEY` secret.
- No S3, Kafka, SageMaker, or Glue permissions are granted to either task.
