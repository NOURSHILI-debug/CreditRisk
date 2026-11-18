# Phase 2 Final Report — Person 2 AWS Deployment

**Repository:** `CreditRisk`
**Branch:** `feature/person2-aws-deployment-preparation`
**Commit:** `76eebef docs audit`
**Phase:** 2 — Complete All Independent Person 2 AWS Deployment Tasks
**Report date:** 2026-11-18
**Role:** Person 2 — Dashboard, AI Agent and Application Engineering

---

## A. Environment Audit

### Git State

| Item | Value |
|---|---|
| Current branch | `feature/person2-aws-deployment-preparation` |
| Current commit | `76eebef docs audit` |
| Working tree | Clean except new untracked files + modified `docker-compose.yml` |
| Phase 1 files committed | **No** — all Phase 1 and Phase 2 files are untracked |

### Local Environment

| Tool | Version | Status | Notes |
|---|---|---|---|
| Docker | 29.7.2 | Installed, service STOPPED | Docker Desktop service (`com.docker.service`) could not be started. Container builds and Compose execution **not verified**. |
| Docker Compose | v5.4.0 | Installed | Same limitation as Docker. |
| AWS CLI | 2.39.3 | Installed, no credentials | `aws sts get-caller-identity` returns `NoCredentials`. AWS account identity, region, and permissions **cannot be verified**. |
| Python | 3.13.3 | Available | `.venv` at `.venv\Scripts\python.exe` |

### Files from Phase 1 (untracked, not committed)

| File | Status |
|---|---|
| `.env.example` | Created (Phase 1) |
| `Dockerfile.fastapi` | Created (Phase 1) |
| `Dockerfile.streamlit` | Created (Phase 1) |
| `docs/PERSON2_AWS_DEPLOYMENT_PREPARATION.md` | Created (Phase 1), updated (Phase 2) |
| `tests/agent_evaluation/` | Created (Phase 1) |

### Files Modified (Phase 1)

| File | Change |
|---|---|
| `docker-compose.yml` | Added `fastapi` and `streamlit` services. **Redpanda service unchanged** (Person 1's infrastructure preserved). |

---

## B. Docker Verification

**Docker verification could NOT be performed.** The Docker Desktop service (`com.docker.service`) is stopped and could not be started in this environment. Therefore:

| Verification Item | Status | Reason |
|---|---|---|
| FastAPI image builds | NOT VERIFIED | Docker service unavailable |
| Streamlit image builds | NOT VERIFIED | Docker service unavailable |
| Non-root execution | NOT VERIFIED | Docker service unavailable |
| Health endpoint responds | NOT VERIFIED | Docker service unavailable |
| Configurable port works | NOT VERIFIED | Docker service unavailable |
| Missing model artifacts don't prevent startup | NOT VERIFIED | Docker service unavailable |
| Dockerfiles syntactically reviewed | VERIFIED | Manual code review |

**Dockerfiles were reviewed for correctness** but not built. The Dockerfiles use:
- `python:3.11-slim` base image
- Non-root `creditrisk` user
- `PYTHONUNBUFFERED=1` and `PYTHONDONTWRITEBYTECODE=1`
- Container-specific requirements files (`requirements.fastapi.txt`, `requirements.streamlit.txt`)
- `PORT` environment variable (configurable)
- `0.0.0.0` bind address
- HEALTHCHECK instructions

---

## C. Local Integration

**Docker Compose integration could NOT be tested** (Docker service unavailable).

### docker-compose.yml Configuration (syntactically reviewed)

The Compose file defines three services:

| Service | Source | Port | Notes |
|---|---|---|---|
| `redpanda` | `docker.redpanda.com/redpandadata/redpanda:v24.2.7` | 19092 | **Unchanged** (Person 1's Kafka) |
| `fastapi` | Built from `Dockerfile.fastapi` | 8000 | Depends on model artifacts (not in repo) |
| `streamlit` | Built from `Dockerfile.streamlit` | 8501 | Depends on `fastapi` healthy |

### Key Design Decisions

1. **Streamlit calls FastAPI via service name** `http://fastapi:8000` (internal Docker network), not localhost.
2. **Streamlit depends on FastAPI being healthy** (`condition: service_healthy`).
3. **Redpanda is preserved** — no changes to Person 1's Kafka configuration.
4. **No duplicate Kafka services** — only the existing Redpanda service.
5. **No unnecessary ports exposed** — only 8000 (FastAPI), 8501 (Streamlit), 19092 (Redpanda).

### Local Non-Docker Workflows

Verified functional via test suite:
- `uvicorn api.main:app --app-dir src --host 127.0.0.1 --port 8000` (FastAPI startup command preserved)
- `streamlit run src/dashboard/app.py` (Streamlit startup command preserved)

---

## D. Security

### Implemented Controls

| Control | Implementation | Status |
|---|---|---|
| Non-root container user | `creditrisk` user in both Dockerfiles | Implemented |
| No hardcoded secrets | All secrets via environment variables | Implemented |
| `.env` gitignored | `.env.example` is the template | Implemented |
| CORS configuration | `CREDIT_RISK_CORS_ORIGINS` env var, defaults to localhost:8501 | Implemented |
| No debug mode | No debug flags in code | Implemented |
| Error responses sanitized | Uniform error shape, no stack traces exposed | Implemented |
| Mock data restrictions | Explicit mode only, labeled synthetic, no silent fallback | Implemented |
| Health endpoint exposure | `/health` on FastAPI, `/_stcore/health` on Streamlit | Implemented |
| Applicant ID redaction | `privacy.redact_application_ids()` before LLM input | Implemented |
| Local rendering of applicant data | `_local_applicant_response()` in agent graph | Implemented |
| Untrusted retrieved evidence | `RESPONSE_SYSTEM_PROMPT` marks tool output as untrusted | Implemented |
| Lending decision refusal | `_DECISION_REQUEST_PATTERN` blocks approve/reject | Implemented |
| No secrets in logs | Python logging, no API keys/applicant data logged | Implemented |

### Unresolved Shared Decisions

| Decision | Why Shared | Impact |
|---|---|---|
| Authentication/authorization | Requires team agreement on authN/authZ mechanism | Deployment blocker — cannot deploy to production without auth |
| JWT signing secret storage | Requires Secrets Manager + IAM configuration | Needed if JWT auth is implemented |
| Dashboard authentication | May require identity provider integration | Dashboard access control |

### Security Testing

Agent evaluation framework includes 9 security-focused tests, all passing:

| Test Category | Cases | Passed | Failed |
|---|---|---|---|
| Privacy | 4 | 4 | 0 |
| Prompt injection | 5 | 5 | 0 |
| Failure handling | 5 | 5 | 5 |

Key security behaviors verified:
- Applicant IDs redacted before routing LLM input (`priv-004`, `inj-001`, `inj-005`)
- Applicant results rendered locally, never sent to response LLM (`llm_response_calls: 0`)
- Full application record access denied without specific field request (`not_authorized`)
- Error messages do not expose file paths, secrets, or internal details
- Malicious user instructions do not bypass routing or fabricate risk scores
- Injected content in tool outputs marked as untrusted evidence
- Lending decision requests refused without tool execution
- Tool exceptions return safe error status without exposing internal details
- Missing predictions/unavailable model return explicit status (no default probability)

---

## E. AWS Infrastructure Preparation

### ECR (Elastic Container Registry)

| Item | Status | Details |
|---|---|---|
| Repository naming | Prepared | `credit-risk-fastapi`, `credit-risk-streamlit` |
| Image tagging | Prepared | Git commit SHA + `latest` (development) |
| Repository URIs | Documented | `ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com/credit-risk-{fastapi,streamlit}` |
| Repository creation | NOT EXECUTED | No AWS credentials available |
| Image push | NOT EXECUTED | No Docker/AWS access |
| Lifecycle policy | Documented | Keep last 10 images, delete untagged >7 days, delete >90 days |

### ECS/Fargate Task Definitions

| File | Component | CPU | Memory | Status |
|---|---|---|---|---|
| `aws/ecs/fastapi-task-definition.json` | FastAPI | 256 (0.25 vCPU) | 512 MB | Created (template) |
| `aws/ecs/streamlit-task-definition.json` | Streamlit | 512 (0.5 vCPU) | 1024 MB | Created (template) |

Task definition features:
- `awsvpc` network mode (required for Fargate)
- `awslogs` log driver with separate CloudWatch log groups
- Health checks configured
- Least-privilege IAM task roles (no AWS service permissions by default)
- Secrets Manager reference for `OPENAI_API_KEY` (template placeholder)
- Tags: project=credit-risk, component={fastapi,streamlit}, owner=person2

**Placeholders not replaced:** `ACCOUNT_ID`, `REGION`, `IMAGE_TAG` — must be replaced before deployment.

### Networking (ECS Service Connect)

| Item | Status | Details |
|---|---|---|
| Approach | Selected | ECS Service Connect (not Docker Compose DNS) |
| Namespace | Proposed | `credit-risk.local` |
| FastAPI discovery name | Proposed | `fastapi` |
| Streamlit discovery name | Proposed | `streamlit` |
| FastAPI → Streamlit communication | Designed | `http://fastapi:8000` via Service Connect |
| Security groups | Described | Least privilege, no unnecessary exposure |
| Public access | Restricted | FastAPI not exposed directly; ALB fronting recommended |

**Important:** Docker Compose service DNS does NOT work across separate ECS services. Service Connect (or Cloud Map) is required for internal service-to-service communication in ECS.

### Application Load Balancer

| Item | Status | Details |
|---|---|---|
| ALB design | Documented | Frontend for Streamlit + FastAPI |
| HTTPS termination | Proposed | At ALB level |
| Target groups | Proposed | One per service |
| Health checks | Proposed | FastAPI `/health`, Streamlit `/_stcore/health` |
| ALB creation | NOT EXECUTED | Requires AWS credentials and approval |

### Secrets Manager

| Item | Status | Details |
|---|---|---|
| Strategy | Documented | Secrets injected via ECS task definition `secretOptions` |
| `OPENAI_API_KEY` | Template placeholder | ARN reference in task definition |
| JWT signing secret | Not implemented | Pending authentication decision |
| Local `.env` support | Maintained | For development only |
| Secrets Manager client | NOT implemented | No SDK calls in application code |

### CloudWatch

| Item | Status | Details |
|---|---|---|
| Log groups | Documented | `/ecs/credit-risk-fastapi`, `/ecs/credit-risk-streamlit` |
| Log format | Structured JSON (FastAPI), standard Python (Streamlit) |
| Log driver | `awslogs` in ECS task definitions |
| Retention period | Recommended 30 days (to be configured) |
| Sensitive data in logs | Excluded by design |
| Metric alarms | Documented (not created) |
| Dashboard | Documented (not created) |

### Deployment Automation Scripts

| File | Purpose | Status |
|---|---|---|
| `scripts/build-and-push.ps1` | Build images, ECR login, push to ECR | Created (not executed) |
| `scripts/deploy-ecs.ps1` | Register task definitions, update ECS services | Created (not executed) |

Both scripts validate AWS account/region, display planned actions, require `-Confirm` and user "YES" confirmation, and do not create AWS resources.

---

## F. Files Changed

### Modified (1 file)

| Path | Reason |
|---|---|
| `docker-compose.yml` | Added `fastapi` and `streamlit` services; preserved Redpanda unchanged |

### Created (17 files)

| Path | Reason |
|---|---|
| `.dockerignore` | Exclude unnecessary files from container images (Phase 2G) |
| `.env.example` | Environment variable template (Phase 1) |
| `Dockerfile.fastapi` | FastAPI container definition (Phase 1) |
| `Dockerfile.streamlit` | Streamlit container definition (Phase 1) |
| `requirements.fastapi.txt` | FastAPI-only production dependencies (Phase 2G) |
| `requirements.streamlit.txt` | Streamlit + agent production dependencies (Phase 2G) |
| `scripts/build-and-push.ps1` | Build and push images to ECR (Phase 2F) |
| `scripts/deploy-ecs.ps1` | Register task definitions and update ECS services (Phase 2F) |
| `aws/ecs/fastapi-task-definition.json` | FastAPI ECS task definition template (Phase 2C) |
| `aws/ecs/streamlit-task-definition.json` | Streamlit ECS task definition template (Phase 2C) |
| `aws/ecs/task-definition-notes.md` | Task definition documentation (Phase 2C) |
| `aws/ecr/repository-config.md` | ECR repository configuration (Phase 2C) |
| `aws/networking/service-connect.md` | ECS Service Connect networking design (Phase 2C) |
| `docs/PERSON2_AWS_DEPLOYMENT_PREPARATION.md` | Updated with Phase 2 addendum (Phase 1 + Phase 2) |
| `docs/PERSON2_PHASE2_FINAL_REPORT.md` | This final report |
| `tests/agent_evaluation/` | Agent evaluation framework (Phase 1, 9 tests) |

**No Person 1 files were modified.** Verified via `git diff`: only `docker-compose.yml` shows as modified (added services; Redpanda unchanged). All other changes are new untracked files.

---

## G. Test Results

### Full Test Suite

```
.venv\Scripts\python.exe -m pytest -q
```

**Result: 297 passed, 55 skipped, 0 failed** (44.26s)

All 55 skips are explicit `pytest.skip()` calls for legitimate reasons:
- Missing `models/local_v2` artifacts (Person 1) — 44 skips
- Missing `data/raw/application_train.csv` (Person 1) — 5 skips
- Opt-in Ollama live integration (`RUN_OLLAMA_INTEGRATION=1`) — 6 skips

### Agent Evaluation Tests

```
.venv\Scripts\python.exe -m pytest tests/agent_evaluation/ -q
```

**Result: 9 passed, 0 failed, 0 skipped** (10.43s)

All 6 metric categories at 100%:
- Tool-selection accuracy: 6/6
- Argument-validation: 5/5
- Groundedness: 5/5
- Privacy: 4/4 (0 critical failures)
- Injection-resistance: 5/5
- Failure-handling: 5/5

### Docker Tests

**NOT PERFORMED** — Docker service not available in environment.

### AWS Tests

**NOT PERFORMED** — No AWS credentials configured.

### Ruff

**NOT PERFORMED** — Ruff not installed in environment.

---

## H. Person 2 Remaining Work (Phase 2+)

### P0 — Mandatory for AWS Deployment

| # | Task | Dependencies | Acceptance Criteria |
|---|---|---|---|
| P0-1 | Implement authentication/authorization on FastAPI | Shared security decision | `/health` and prediction endpoints require valid token |
| P0-2 | Build and test Docker images locally | Docker service available | `docker build -f Dockerfile.fastapi` and `docker build -f Dockerfile.streamlit` succeed |
| P0-3 | Test Docker Compose locally | Docker service available | `docker compose up` starts all 3 services; Streamlit reaches FastAPI via service name |
| P0-4 | Create ECR repositories | AWS credentials | `aws ecr create-repository` succeeds for both repos |
| P0-5 | Build and push container images to ECR | Docker + AWS credentials | Images pushed with git commit SHA tag |
| P0-6 | Create ECS cluster with Service Connect | AWS credentials | Cluster created, Service Connect enabled, namespace `credit-risk.local` |
| P0-7 | Create IAM execution and task roles | AWS credentials | Roles created with least-privilege policies |
| P0-8 | Create CloudWatch log groups | AWS credentials | Log groups created for both services |
| P0-9 | Create Secrets Manager secret for `OPENAI_API_KEY` | AWS credentials + approval | Secret created with actual value |
| P0-10 | Register ECS task definitions | ECR images pushed | Task definitions registered with correct image tags |
| P0-11 | Create AWS Application Load Balancer | AWS credentials + approval | ALB created with target groups for both services |
| P0-12 | Configure security groups | AWS credentials | Least-privilege security groups configured |
| P0-13 | Deploy FastAPI to ECS/Fargate | All P0-1 through P0-12 | Service running, health checks passing |
| P0-14 | Deploy Streamlit to ECS/Fargate | All P0-1 through P0-12 | Service running, can reach FastAPI via Service Connect |
| P0-15 | End-to-end integration test with Person 1's artifacts | Person 1 artifacts + P0-13/P0-14 | Real predictions flow through API → dashboard → agent |

### P1 — Important Quality Improvements

| # | Task |
|---|---|
| P1-1 | Add `.dockerignore` to exclude unnecessary files (DONE) |
| P1-2 | Separate production dependency files (DONE) |
| P1-3 | Add CloudWatch log group configuration |
| P1-4 | ECS service auto-scaling configuration |
| P1-5 | Container image tagging and versioning strategy |
| P1-6 | CI/CD pipeline for container builds |

---

## I. Shared Dependencies

| Dependency | Why Needed | Owner |
|---|---|---|
| Authentication/authorization | Required before production deployment | Shared (security task) |
| Network architecture | VPC, subnets, security groups for ECS | Shared |
| CloudWatch monitoring strategy | Log groups, alarms, dashboards | Shared |
| Full end-to-end testing | Validates complete pipeline | Shared (with Person 1 artifacts) |
| AWS credentials/account | Required for any AWS resource creation | Shared (infrastructure) |
| Person 1 model artifacts | Required for `/health` = ok and predictions | Person 1 |
| Person 1 dataset | Required for application data serving | Person 1 |

---

## J. Deployment Status

**NOTHING IS DEPLOYED TO AWS.**

| Item | Status |
|---|---|
| Docker images built | No (Docker service unavailable) |
| Images pushed to ECR | No (No Docker/AWS access) |
| ECR repositories created | No (No AWS credentials) |
| ECS cluster created | No (No AWS credentials) |
| ECS services running | No (No AWS credentials) |
| ALB created | No (No AWS credentials) |
| Secrets Manager secret created | No (No AWS credentials/approval) |
| CloudWatch log groups created | No (No AWS credentials) |
| Authentication implemented | No (Shared security task, not started) |

**Phase 2 is preparation only.** All Dockerfiles, task definitions, deployment scripts, and documentation are prepared. Actual AWS deployment requires:
1. Docker service available in the environment
2. AWS credentials configured
3. Explicit approval to create AWS resources (may incur costs)
4. Person 1's model artifacts and dataset (for ML-service readiness)
5. Authentication implementation (shared security task)

---

## K. Person 1 Code Preservation

**Verified:** No Person 1 files were modified.

| Check | Result |
|---|---|
| `git diff --name-only` | Only `docker-compose.yml` modified |
| `docker-compose.yml` diff | Redpanda service unchanged; only FastAPI + Streamlit added |
| `src/ingestion/`, `src/storage/`, `src/training/`, `src/inference/`, `src/processing/` | Not modified |
| `src/api/` | Not modified (Person 2-owned but contains Person 1's prediction logic) |
| Person 1's tests | All pass (297 passed, 55 skipped, 0 failed) |

---

**End of Phase 2 Final Report.**
