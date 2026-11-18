# ECS Service Connect Networking Design — Person 2 Components

## Overview

Docker Compose service DNS (`http://fastapi:8000`) does NOT automatically
work across separate ECS services. ECS tasks run on separate ENIs with
separate IP addresses, and there is no automatic service discovery between
them unless explicitly configured.

## Selected Approach: ECS Service Connect

ECS Service Connect provides service discovery and internal load balancing
within an ECS cluster. Services are discoverable by a `discovery name`
within the namespace, and traffic is routed through an Envoy proxy sidecar.

### Why Service Connect

| Approach | Pros | Cons |
|---|---|---|
| **ECS Service Connect** (selected) | Native ECS integration, service discovery, health checking, traffic encryption, observability | Requires ECS cluster with Service Connect enabled, adds Envoy sidecar |
| Cloud Map + DNS | Native AWS service discovery | More complex setup, no built-in load balancing |
| ALB path-based routing | Simple, familiar | Exposes internal traffic through ALB, extra cost, latency |

### Prerequisites

1. ECS cluster must have Service Connect enabled
2. A Service Connect namespace must exist (e.g., `credit-risk.local`)
3. Both FastAPI and Streamlit services must be in the same cluster/namespace

### Service Connect Configuration

#### FastAPI Service

| Setting | Value |
|---|---|
| Discovery name | `fastapi` |
| Discovery namespace | `credit-risk.local` |
| Client port | 8000 |
| Server port | 8000 |
| Protocol | HTTP |

FastAPI is reachable by Streamlit at:
- `http://fastapi:8000` (within the ECS cluster, via Service Connect)
- `http://fastapi.credit-risk.local:8000` (fully qualified)

#### Streamlit Service

| Setting | Value |
|---|---|
| Discovery name | `streamlit` |
| Discovery namespace | `credit-risk.local` |
| Client port | 8501 |
| Server port | 8501 |
| Protocol | HTTP |

Streamlit calls FastAPI using `CREDIT_RISK_API_URL=http://fastapi:8000`
(resolved via Service Connect).

### Cloud Map Alternative

If Service Connect is not available, Cloud Map service discovery can be
used instead. Services register with Cloud Map and are discoverable via
DNS. This requires additional configuration (health checks, routing
policy) and does not provide built-in load balancing or traffic encryption.

### Security Groups

| Service | Inbound | Outbound |
|---|---|---|
| FastAPI ALB target | 443 from ALB security group | 8000 to Streamlit (not needed), 443 to Secrets Manager (if using VPC endpoint) |
| Streamlit ALB target | 443 from ALB security group | 8000 to FastAPI (via Service Connect) |
| FastAPI task | 8000 from Streamlit task | Outbound as needed |
| Streamlit task | 8501 from ALB | 8000 to FastAPI (via Service Connect) |

### What Is NOT Exposed

- Redpanda/Kafka (Person 1 infrastructure) — not accessible from ECS tasks
- SageMaker endpoints (Person 1 infrastructure) — not accessible from ECS tasks
- S3 buckets (Person 1 infrastructure) — not accessible from ECS tasks unless explicitly configured
- FastAPI direct public access — only reachable through ALB (if configured)

### Prerequisites Checklist

- [ ] ECS cluster created with Service Connect enabled
- [ ] Service Connect namespace created (`credit-risk.local`)
- [ ] ECR repositories created (`credit-risk-fastapi`, `credit-risk-streamlit`)
- [ ] Container images built and pushed to ECR
- [ ] IAM execution and task roles created
- [ ] CloudWatch log groups created
- [ ] Secrets Manager secret created for `OPENAI_API_KEY` (if using OpenAI)
- [ ] ALB created with target groups for FastAPI and Streamlit
- [ ] Security groups configured for least privilege
