# Person 2 AWS Deployment Preparation

**Repository:** `CreditRisk`
**Branch:** `feature/person2-aws-deployment-preparation`
**Owner:** Person 2 — Dashboard, AI Agent and Application Engineering
**Phase:** 1 — Prepare Person 2's application components for AWS deployment
**Status:** Prepared for AWS (not deployed)

---

## 1. Target Architecture

The target deployment architecture uses independent containers for each
Person 2 application, orchestrated on AWS ECS/Fargate (proposed, subject to
repository compatibility):

```
                    Client (browser)
                          |
                          v
              +---------------------+
              |  AWS Load Balancer  |
              +---------------------+
                          |
        +-----------------+-----------------+
        |                                 |
        v                                 v
+---------------+                 +---------------+
|  FastAPI      |                 |  Streamlit    |
|  (port 8000)  |                 |  (port 8501)  |
|  ECS/Fargate  |                 |  ECS/Fargate  |
+---------------+                 +---------------+
        |                                 ^
        |  HTTP API (internal)            |
        v                                 |
+-----------------------------------------+
|  LangGraph AI Agent                     |
|  (embedded in Streamlit container)      |
+-----------------------------------------+

Shared (Person 1 — not in scope for this phase):
- Kafka/MSK (event streaming)
- S3 (raw/processed storage)
- Lambda/Glue (data processing)
- SageMaker (ML training/inference)
- Model artifacts (XGBoost, LSTM)
```

**Deployment target:** AWS ECS/Fargate (proposed). Containers are built and
verified locally; no AWS resources are provisioned in this phase.

**Secrets:** AWS Secrets Manager (production) — configuration prepared, not
implemented at the application layer.

**Logs:** CloudWatch (application logs via stdout JSON format).

---

## 2. Component Ownership

| Component | Owner | Status | Notes |
|---|---|---|---|
| FastAPI backend | Person 2 | Prepared | Dockerfile, health check, env config |
| Streamlit dashboard | Person 2 | Prepared | Dockerfile, API URL config, no silent mock fallback |
| LangGraph AI agent | Person 2 | Prepared | Runs inside Streamlit container; no separate HTTP service |
| Docker Compose (local) | Person 2 | Prepared | FastAPI + Streamlit + existing Redpanda |
| S3 storage | Person 1 | Not in scope | Person 1 owns S3 raw/processed |
| Kafka / MSK | Person 1 | Not in scope | Person 1 owns Kafka |
| XGBoost / LSTM models | Person 1 | Not in scope | Person 1 owns model artifacts |
| SageMaker | Person 1 | Not in scope | Person 1 owns SageMaker |

---

## 3. FastAPI Container Preparation

### Entry point
`src/api/main.py` — `create_app()` / `app = create_app()`. Serves on
`0.0.0.0:${PORT}` (default 8000).

### Endpoints (preserved unchanged)
| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Service info |
| GET | `/health` | Model + data availability (liveness + readiness) |
| GET | `/models` | Model metadata + thresholds |
| GET | `/model/evaluation?split=` | Evaluation rows + provenance |
| GET | `/model/prediction-summary?split=` | Saved prediction counts/risk bands |
| GET | `/model/global-importance` | SHAP global importance |
| GET | `/applications` | Paginated, field-allowlisted rows |
| GET | `/applications/{id}` | One allowlisted record (TARGET excluded) |
| POST | `/predict` | Predict by ID or features (+ optional explanation) |
| GET | `/applications/{id}/prediction` | Predict by ID (+ optional explanation) |

### Configuration (environment variables)
| Variable | Default | Purpose |
|---|---|---|
| `CREDIT_RISK_MODEL_DIR` | `models/local_v2` | Model artifact directory |
| `CREDIT_RISK_APPLICATION_DATA` | `data/raw/application_train.csv` | Application dataset |
| `CREDIT_RISK_CORS_ORIGINS` | `http://localhost:8501,http://127.0.0.1:8501` | Allowed CORS origins |
| `CREDIT_RISK_LOG_LEVEL` | `INFO` | Log level |
| `PORT` | `8000` | Server port |

### Dockerfile: `Dockerfile.fastapi`
- Base image: `python:3.11-slim`
- Installs from `requirements.txt`
- Copies only `src/` (not tests, docs, configs)
- Non-root user (`creditrisk`)
- Binds to `0.0.0.0:${PORT}`
- Health check: `GET /health` returns HTTP 200 when process is alive
- `PORT` environment variable configurable

### Health checks
- **Liveness:** Container process is alive — `GET /health` returns HTTP 200 (always, even when degraded)
- **Readiness:** Model + data loaded — `GET /health` returns JSON with `"status": "ok"`; `"degraded"` means model or data is missing

The existing `/health` endpoint already returns 200 with a `status` field.
No change needed to distinguish liveness from readiness — orchestration
can use the HTTP status code for liveness and the JSON `status` field for
readiness.

### Startup behavior
- Lifespan loads model artifacts (`CreditRiskPredictor`) and data
  (`ApplicationRepository`) at startup
- Missing model/data → `/health` reports `"degraded"`; prediction endpoints
  return 503 with explicit status (`model_unavailable` / `unavailable`)
- No silent fallback to mock predictions
- Startup failures are logged via JSON-formatted logging

### Logging
- JSON-formatted stdout logging (`src/api/logging_config.py`)
- Request ID correlation via `ContextVar`
- Uvicorn access logs disabled (reduces noise)

### Error handling
- Uniform error shape (`src/api/errors.py`) — no stack traces exposed
- Canonical prediction statuses: `available`, `no_prediction`, `unavailable`,
  `model_unavailable`, `invalid`
- Probability is always `null` on errors; never defaulted to zero

---

## 4. Streamlit Container Preparation

### Entry point
`src/dashboard/app.py` — `main()`. Serves on `0.0.0.0:${PORT}` (default 8501).

### Pages (preserved unchanged)
1. **Overview** — KPIs, risk distribution, evaluation, model metadata
2. **Application Explorer** — search, pagination, details, prediction + SHAP
3. **Model Performance** — model comparison with provenance gating
4. **AI Assistant** — LangGraph agent chat

### Configuration (environment variables)
| Variable | Default | Purpose |
|---|---|---|
| `CREDIT_RISK_API_URL` | `http://localhost:8000` | Backend API URL |
| `CREDIT_RISK_DATA_MODE` | `api` | `api` or `mock` |
| `CREDIT_RISK_MOCK_SAMPLE_SIZE` | `300` | Mock sample size |
| `CREDIT_RISK_MOCK_SEED` | `42` | Mock seed |
| `PORT` | `8501` | Server port |
| `CREDIT_RISK_LOG_LEVEL` | `INFO` | Log level |

### Dockerfile: `Dockerfile.streamlit`
- Base image: `python:3.11-slim`
- Installs from `requirements.txt`
- Copies only `src/`
- Non-root user (`creditrisk`)
- Binds to `0.0.0.0:${PORT}`
- Health check: Streamlit `_stcore/health` endpoint

### API communication
- `src/dashboard/api_client.py` reads `CREDIT_RISK_API_URL` from environment
- Default is `http://127.0.0.1:8000` (local); in Docker Compose it is
  `http://fastapi:8000` (service name on internal network)
- **Never hardcodes localhost in production mode**
- API errors raise explicit `DataSourceUnavailableError` — no silent crash
- **No silent fallback to mock data** — API errors remain errors; mock mode
  is explicit only

### Mock mode restrictions
- Mock mode is explicit (`CREDIT_RISK_DATA_MODE=mock`) — never automatic
- Mock data is labeled `"source_mode": "mock"` and `"Synthetic mock/demo data (illustrative only)"`
- Mock evaluation returns empty results — no fabricated metrics

---

## 5. LangGraph AI Agent Container Preparation

### Architecture
The LangGraph agent is **embedded in the Streamlit container** — it is not
an independent HTTP service. It is invoked from
`src/dashboard/pages/assistant.py` via `graph.run_agent()`.

There is no separate agent container. Per the project constraints, we do
not invent an agent HTTP service without a justified interface. The agent
runs in-process with Streamlit.

### LLM provider configuration
| Variable | Default | Purpose |
|---|---|---|
| `AGENT_LLM_PROVIDER` | `openai` | `openai` or `ollama` |
| `AGENT_LLM_MODEL` | `gpt-4o-mini` | Model name |
| `OPENAI_API_KEY` | (required for openai) | API key — never hardcoded |
| `OPENAI_BASE_URL` | (empty) | Optional OpenAI base URL |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint |
| `OLLAMA_ALLOW_REMOTE` | `false` | Allow remote Ollama (HTTPS only) |

### Dependencies on local Ollama
The agent supports Ollama but does **not require** it. OpenAI is the default
provider. In AWS ECS/Fargate, Ollama is not available by default; OpenAI or
another provider must be configured via environment variables or Secrets
Manager.

### Unsupported/missing provider
`src/agent/graph.py:_create_chat_model()` raises `AgentConfigurationError`
with a clear message for unsupported providers or missing OpenAI API key.

### Security
- Application IDs redacted before LLM input (`src/agent/privacy.py`)
- Applicant results rendered locally, never sent to response LLM
- Retrieved data marked as untrusted evidence for the response model
- Lending decisions refused (`_DECISION_REQUEST_PATTERN`)
- Mock results labeled as synthetic

### Logging
- Python standard logging (`logger = logging.getLogger(__name__)`)
- No applicant identifiers or API keys logged

---

## 6. Local Docker Compose Instructions

### Prerequisites
- Docker Desktop or Docker Engine + Compose
- No AWS credentials required
- No Person 1 model artifacts required (FastAPI health reports "degraded" without them)

### Commands

```powershell
# From the repository root
cd C:\Users\souro\Desktop\CreditRisk

# Build all services
docker compose build

# Start all services
docker compose up

# Start in detached mode
docker compose up -d

# View logs
docker compose logs -f

# Stop all services
docker compose down

# Rebuild and restart
docker compose up --build
```

### Service URLs (local)
| Service | URL |
|---|---|
| FastAPI | http://localhost:8000 |
| FastAPI docs | http://localhost:8000/docs |
| FastAPI health | http://localhost:8000/health |
| Streamlit | http://localhost:8501 |
| Redpanda (Kafka) | localhost:19092 |

### Internal networking
- Streamlit calls FastAPI at `http://fastapi:8000` (Docker service name)
- FastAPI and Streamlit are on the default Docker Compose network
- Redpanda is on the same network (unchanged from existing setup)

### .env file (local overrides)
Copy `.env.example` to `.env` and adjust values:
```powershell
copy .env.example .env
# Edit .env with your values, then:
docker compose up
```
`.env` is gitignored — never commit it.

---

## 7. Environment Variable Reference

Complete reference of all environment variables used by Person 2 components:

| Variable | Component | Default | Required? | Description |
|---|---|---|---|---|
| `ENVIRONMENT` | All | `local` | No | Deployment environment |
| `LOG_LEVEL` | All | `INFO` | No | Log level |
| `API_HOST` | FastAPI | `0.0.0.0` | No | Bind host |
| `API_PORT` / `PORT` | FastAPI | `8000` | No | Server port |
| `CREDIT_RISK_MODEL_DIR` | FastAPI | `models/local_v2` | No | Model artifact path |
| `CREDIT_RISK_APPLICATION_DATA` | FastAPI | `data/raw/application_train.csv` | No | Dataset path |
| `CREDIT_RISK_CORS_ORIGINS` | FastAPI | localhost:8501 | No | CORS origins |
| `DASHBOARD_HOST` / `PORT` | Streamlit | `0.0.0.0` / `8501` | No | Bind host/port |
| `CREDIT_RISK_API_URL` | Streamlit | `http://localhost:8000` | Yes (production) | Backend API URL |
| `CREDIT_RISK_DATA_MODE` | Streamlit | `api` | No | `api` or `mock` |
| `CREDIT_RISK_MOCK_SAMPLE_SIZE` | Streamlit | `300` | No | Mock sample size |
| `CREDIT_RISK_MOCK_SEED` | Streamlit | `42` | No | Mock seed |
| `AGENT_LLM_PROVIDER` | Agent | `openai` | No | `openai` or `ollama` |
| `AGENT_LLM_MODEL` | Agent | `gpt-4o-mini` | No | Model name |
| `OPENAI_API_KEY` | Agent | (empty) | Yes (if openai) | API key — via Secrets Manager |
| `OPENAI_BASE_URL` | Agent | (empty) | No | Optional OpenAI URL |
| `OLLAMA_BASE_URL` | Agent | `http://localhost:11434` | No | Ollama endpoint |
| `OLLAMA_ALLOW_REMOTE` | Agent | `false` | No | Allow remote Ollama |
| `AWS_REGION` | All | `us-east-1` | No | AWS region |
| `CREDIT_RISK_BUCKET` | All | (empty) | No | S3 bucket (Person 1) |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | All | (empty) | No | Via IAM role or Secrets Manager |

---

## 8. Secrets Management Strategy

### Local development
- Copy `.env.example` to `.env` and fill in values
- `.env` is gitignored — never committed
- No AWS credentials required for local development

### Production (AWS ECS/Fargate)
- Store secrets in **AWS Secrets Manager**: `OPENAI_API_KEY`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`
- Inject secrets into container environment at task definition time
- Application code reads from environment variables only — no SDK calls to Secrets Manager
- No hardcoded credentials anywhere in the repository

### What is NOT done
- No fake secrets manager implementation
- No AWS SDK calls for secrets in application code
- No credentials committed to version control

---

## 9. Health Checks

| Service | Endpoint | Liveness | Readiness |
|---|---|---|---|
| FastAPI | `GET /health` | HTTP 200 (process alive) | JSON `status == "ok"` (model + data loaded) |
| Streamlit | `GET /_stcore/health` | HTTP 200 | HTTP 200 (Streamlit ready) |

FastAPI readiness distinction:
- `status: "ok"` — model and data loaded, predictions available
- `status: "degraded"` — model or data missing; container alive but cannot serve predictions

Docker Compose health checks are configured for both services.

---

## 10. Logging Strategy

| Component | Format | Destination |
|---|---|---|
| FastAPI | JSON (stdout) | CloudWatch (via ECS/Fargate log driver) |
| Streamlit | Python logging (stdout) | CloudWatch |
| LangGraph | Python logging (stdout) | CloudWatch |

FastAPI logging:
- JSON-formatted with timestamp, level, logger, message, request_id
- No applicant identifiers or secrets in log output
- Uvicorn access logs disabled

Agent logging:
- Standard Python logging
- No sensitive data logged

---

## 11. Security Considerations

| Control | Status | Notes |
|---|---|---|
| Non-root container user | Implemented | `creditrisk` user in both Dockerfiles |
| No hardcoded secrets | Implemented | All via environment variables |
| `.env` gitignored | Implemented | `.env.example` is the template |
| CORS configured | Implemented | `CREDIT_RISK_CORS_ORIGINS` |
| No authentication | Missing | Documented as deployment blocker |
| No TLS at application layer | Missing | Handled by AWS ALB/load balancer |
| No debug mode | Implemented | No debug flags in code |
| Error responses sanitized | Implemented | No stack traces exposed |
| Mock data restrictions | Implemented | Explicit mode, labeled synthetic |
| Health endpoint exposure | Implemented | `/health` on FastAPI |

**Authentication:** Missing. The FastAPI service has no authentication or
authorization. This is a known gap documented in `README.md` ("authentication
is required before any real deployment"). It is a shared security task — not
introduced here to avoid breaking existing functionality.

---

## 12. Tests Executed and Exact Results

### FastAPI tests
```
.venv\Scripts\python.exe -m pytest tests/unit/test_api_unit.py tests/unit/test_api_contract.py -q
```
Result: All passed (these tests use synthetic TestClient fixtures, not real artifacts)

### Streamlit tests
```
.venv\Scripts\python.exe -m pytest tests/unit/test_dashboard_data_access.py tests/unit/test_dashboard_overview.py tests/unit/test_application_explorer.py tests/unit/test_mock_provider.py -q
```
Result: All passed

### LangGraph / agent tests
```
.venv\Scripts\python.exe -m pytest tests/unit/test_agent.py tests/unit/test_report_generator.py -q
```
Result: All passed

### Agent evaluation tests (new)
```
.venv\Scripts\python.exe -m pytest tests/agent_evaluation/ -q
```
Result: 9 passed, 0 failed, 0 skipped

### Full test suite
```
.venv\Scripts\python.exe -m pytest -q
```
Result: **297 passed, 55 skipped, 0 failed**

All 55 skips are explicit `pytest.skip()` due to:
- Missing `models/local_v2` artifacts (Person 1)
- Missing `data/raw/application_train.csv` (Person 1)
- Opt-in Ollama live integration (`RUN_OLLAMA_INTEGRATION=1`)

### Docker build
Not executed — Docker is not available in this environment. Dockerfiles are
syntactically reviewed but not built.

### Docker Compose
Not executed — Docker is not available in this environment. Configuration is
syntactically reviewed but not tested.

---

## 13. Current Blockers

| Blocker | Impact | Owner | Resolution |
|---|---|---|---|
| No authentication on FastAPI | Cannot deploy to production without auth | Shared (security task) | Implement authN/authZ before production deployment |
| Model artifacts missing (`models/local_v2/`) | `/health` reports "degraded"; predictions unavailable | Person 1 | Train XGBoost and place artifacts |
| Dataset missing (`data/raw/application_train.csv`) | `/health` reports "degraded"; no application data | Person 1 | Acquire and place dataset |
| Docker not available in environment | Cannot verify container builds locally | Environment | Build in CI/CD or local Docker environment |
| No live LLM provider configured | Agent cannot answer questions in production | Shared | Configure OpenAI API key or Ollama in Secrets Manager |

---

## 14. Dependencies on Person 1

| Dependency | Why Person 2 needs it | Status |
|---|---|---|
| `data/raw/application_train.csv` | FastAPI `ApplicationRepository` needs it to serve applications | Missing (gitignored) |
| `models/local_v2/` artifacts | FastAPI `CreditRiskPredictor` needs model + preprocessor | Missing (gitignored) |
| Model hash / version | For dashboard model metadata display | Not yet available |
| XGBoost prediction API | FastAPI `/predict` endpoint needs a working model | Not yet available |

---

## 15. Remaining AWS Deployment Steps

### Phase 1 (current) — Complete
- [x] Dockerfile.fastapi created
- [x] Dockerfile.streamlit created
- [x] .env.example created
- [x] docker-compose.yml updated (FastAPI + Streamlit + Redpanda)
- [x] Documentation created (this file)
- [x] All tests pass (297 passed, 55 skipped)

### Phase 2 — Next
- [ ] Build and test Docker images locally (requires Docker)
- [ ] Test Docker Compose locally (requires Docker)
- [ ] Configure AWS ECS/Fargate task definitions
- [ ] Set up AWS Secrets Manager for `OPENAI_API_KEY`
- [ ] Configure AWS Application Load Balancer
- [ ] Implement authentication (shared security task)
- [ ] Deploy FastAPI to ECS/Fargate
- [ ] Deploy Streamlit to ECS/Fargate
- [ ] End-to-end integration testing with Person 1's artifacts
- [ ] CloudWatch log group configuration
- [ ] ECS service auto-scaling configuration

---

## 16. Status Summary

| Area | Status |
|---|---|
| FastAPI container | Prepared (Dockerfile + config) — not built/tested (no Docker) |
| Streamlit container | Prepared (Dockerfile + config) — not built/tested (no Docker) |
| LangGraph agent | Prepared (embedded in Streamlit) — no separate container |
| Docker Compose | Prepared — not tested (no Docker) |
| Environment variables | Documented in `.env.example` |
| Secrets management | Prepared (Secrets Manager strategy, no code changes) |
| Health checks | Configured in Dockerfiles and Compose |
| Logging | JSON stdout (CloudWatch-compatible) |
| Security | Non-root user, no hardcoded secrets, CORS, sanitized errors |
| Authentication | **Missing** — deployment blocker |
| AWS deployment | **Not performed** — Phase 1 is preparation only |

**Nothing is deployed to AWS.** This phase prepares the application
components for independent container deployment. Actual AWS deployment
requires Docker, AWS credentials, Person 1's model artifacts, and
authentication implementation.

---

## 17. Phase 2 Addendum — Docker Image Optimization, Deployment Automation, and AWS Infrastructure

### 17.1 Docker Image Optimization

Separate production dependency files were created to avoid installing
unnecessary dependencies in each container:

| File | Container | Contents |
|---|---|---|
| `requirements.txt` | CI / local development | All dependencies (shared) |
| `requirements.fastapi.txt` | FastAPI container | fastapi, uvicorn, pandas, numpy, scikit-learn, xgboost, joblib, scipy, httpx |
| `requirements.streamlit.txt` | Streamlit container | streamlit, pandas, numpy, plotly, httpx, langgraph, langchain-openai, langchain-ollama |

Rationale:
- FastAPI does not need streamlit, plotly, langgraph, or langchain packages.
- Streamlit does not need scikit-learn, xgboost (model loading happens in FastAPI).
- The agent is embedded in Streamlit, so langgraph/langchain deps are in the Streamlit container.

Dockerfiles updated to use container-specific requirements files:
- `Dockerfile.fastapi` → installs from `requirements.fastapi.txt`
- `Dockerfile.streamlit` → installs from `requirements.streamlit.txt`

### 17.2 Docker Ignore File

`.dockerignore` created to exclude unnecessary files from container images:

| Excluded | Reason |
|---|---|
| `.git/`, `.venv/`, `__pycache__/` | Build artifacts and local environments |
| `.env`, `.pytest_cache/`, `htmlcov/` | Secrets, test caches, coverage reports |
| `data/raw/`, `models/`, `*.pkl`, `*.joblib` | Gitignored data and ML artifacts (Person 1) |
| `docs/`, `tests/`, `build/`, `dist/` | Not needed at runtime |
| `.terraform/`, `*.tfstate*`, `local_lake/` | Infrastructure and runtime state |

### 17.3 Deployment Automation Scripts

| File | Purpose |
|---|---|
| `scripts/build-and-push.ps1` | Builds Docker images, authenticates to ECR, pushes images |
| `scripts/deploy-ecs.ps1` | Registers ECS task definitions, updates ECS services |

Both scripts:
- Validate AWS account ID and region before acting
- Display all planned actions before execution
- Require explicit `-Confirm` flag and user confirmation ("YES")
- Do NOT create ECR repositories, ALB resources, VPC resources, or IAM roles
- Do NOT silently create expensive infrastructure

### 17.4 ECS Task Definitions

| File | Component | CPU | Memory |
|---|---|---|---|
| `aws/ecs/fastapi-task-definition.json` | FastAPI | 256 (0.25 vCPU) | 512 MB |
| `aws/ecs/streamlit-task-definition.json` | Streamlit | 512 (0.5 vCPU) | 1024 MB |

Task definitions use:
- `awsvpc` network mode (required for Fargate)
- `awslogs` log driver with separate CloudWatch log groups
- Health checks on `/health` (FastAPI) and `/_stcore/health` (Streamlit)
- Least-privilege IAM task roles (no AWS service permissions by default)
- Secrets Manager reference for `OPENAI_API_KEY` (FastAPI only, template placeholder)

Placeholders to replace before deployment: `ACCOUNT_ID`, `REGION`, `IMAGE_TAG`.

### 17.5 ECR Repository Configuration

| Component | Repository Name |
|---|---|
| FastAPI | `credit-risk-fastapi` |
| Streamlit | `credit-risk-streamlit` |

Image tagging strategy: git commit SHA (immutable) + `latest` (development).
Previous release tags are never overwritten.

Repositories must be created before pushing:
```powershell
aws ecr create-repository --repository-name credit-risk-fastapi --region us-east-1
aws ecr create-repository --repository-name credit-risk-streamlit --region us-east-1
```

### 17.6 ECS Service Connect Networking

Docker Compose service DNS does NOT work across separate ECS services.
The selected approach is **ECS Service Connect** with a Service Connect
namespace (`credit-risk.local`).

Streamlit calls FastAPI at `http://fastapi:8000` (resolved via Service Connect
within the ECS cluster). FastAPI is not exposed directly to the public
internet; both services are behind an Application Load Balancer.

See `aws/networking/service-connect.md` for the complete networking design.

### 17.7 Secrets Management

Production secrets (`OPENAI_API_KEY`, JWT signing secret if authentication
is implemented) are stored in AWS Secrets Manager and injected into ECS
task definitions via `secretOptions`. Application code reads from
environment variables only — no SDK calls to Secrets Manager.

Local development uses `.env` (gitignored, copied from `.env.example`).
No AWS credentials are required for local development.

### 17.8 CloudWatch Logging

| Component | Log Group | Format |
|---|---|---|
| FastAPI | `/ecs/credit-risk-fastapi` | Structured JSON |
| Streamlit | `/ecs/credit-risk-streamlit` | Standard Python logging |

Log drivers configured in ECS task definitions using `awslogs`.
Retention period should be configured per organizational policy (recommended: 30 days).
No API keys or applicant data are logged.

### 17.9 Security Testing Results

Security test coverage was added for the agent evaluation framework:

| Test | Result |
|---|---|
| Applicant ID redaction from routing LLM input | PASS |
| Applicant results not sent to response LLM | PASS |
| `not_authorized` for full record without field request | PASS |
| Error messages do not expose file paths/secrets | PASS |
| Prompt injection in user input handled | PASS |
| Prompt injection in retrieved tool output marked untrusted | PASS |
| Lending decision requests refused | PASS |
| Tool exceptions return safe error status | PASS |
| Missing predictions return explicit status (no default probability) | PASS |
| Unavailable model returns explicit status | PASS |

All agent evaluation tests: **9 passed, 0 failed, 0 skipped**.

### 17.10 Updated Remaining AWS Deployment Steps

### Phase 1 (complete)
- [x] Dockerfile.fastapi created
- [x] Dockerfile.streamlit created
- [x] .env.example created
- [x] docker-compose.yml updated (FastAPI + Streamlit + Redpanda)
- [x] Documentation created (this file)
- [x] All tests pass (297 passed, 55 skipped)
- [x] .dockerignore created
- [x] Separate requirements files created (fastapi, streamlit)
- [x] ECS task definitions created
- [x] ECR repository configuration documented
- [x] ECS Service Connect networking designed
- [x] Deployment automation scripts created
- [x] Security testing added (agent evaluation)

### Phase 2 (next — requires Docker and AWS credentials)
- [ ] Build and test Docker images locally (requires Docker)
- [ ] Test Docker Compose locally (requires Docker)
- [ ] Create ECR repositories (`credit-risk-fastapi`, `credit-risk-streamlit`)
- [ ] Build and push container images to ECR
- [ ] Create ECS cluster with Service Connect enabled
- [ ] Create IAM execution and task roles
- [ ] Create CloudWatch log groups
- [ ] Create Secrets Manager secret for `OPENAI_API_KEY`
- [ ] Register ECS task definitions
- [ ] Create AWS Application Load Balancer with target groups
- [ ] Configure security groups (least privilege)
- [ ] Deploy FastAPI to ECS/Fargate
- [ ] Deploy Streamlit to ECS/Fargate
- [ ] Implement authentication (shared security task)
- [ ] End-to-end integration testing with Person 1's artifacts
- [ ] Configure CloudWatch alarms and dashboards
- [ ] Configure ECS service auto-scaling

---

## 18. Updated Status Summary

| Area | Status |
|---|---|
| FastAPI container | Prepared (Dockerfile + config) — not built/tested (no Docker in environment) |
| Streamlit container | Prepared (Dockerfile + config) — not built/tested (no Docker in environment) |
| LangGraph agent | Prepared (embedded in Streamlit) — no separate container |
| Docker Compose | Prepared — not tested (no Docker in environment) |
| .dockerignore | Created |
| Separate requirements files | Created (fastapi, streamlit) |
| ECS task definitions | Created (templates, placeholders not replaced) |
| ECR repository configuration | Documented |
| ECS Service Connect networking | Designed |
| Deployment automation scripts | Created (build-and-push.ps1, deploy-ecs.ps1) |
| Environment variables | Documented in `.env.example` |
| Secrets management | Prepared (Secrets Manager strategy, no code changes) |
| Health checks | Configured in Dockerfiles and Compose |
| Logging | JSON stdout (CloudWatch-compatible) |
| Security | Non-root user, no hardcoded secrets, CORS, sanitized errors, agent evaluation tests |
| Authentication | **Missing** — deployment blocker (shared security task) |
| AWS deployment | **Not performed** — preparation only |
| Docker image builds | **Not verified** — Docker service not available in environment |
| Docker Compose integration | **Not verified** — Docker service not available in environment |
| AWS infrastructure | **Not provisioned** — no AWS credentials configured |

**Nothing is deployed to AWS.** This phase prepares the application
components for independent container deployment. Actual AWS deployment
requires Docker, AWS credentials, Person 1's model artifacts, and
authentication implementation.
