# Credit Risk Intelligence API

FastAPI service that loads the XGBoost artifacts and application dataset and
serves predictions, applications, model metadata, evaluation rows and SHAP
explanations.

The authoritative endpoint, status, error and schema definitions live in
[`docs/api_contract.md`](api_contract.md). This page is only a quick start.

> **WARNING**
> Responses contain applicant-derived data. Run the service on localhost only.
> Authentication and authorization are required before any real deployment.

## Start the service (PowerShell)

```powershell
.venv\Scripts\python.exe -m uvicorn api.main:app --app-dir src --host 127.0.0.1 --port 8000
```

The first startup loads the application CSV and model artifacts, so readiness
can take several seconds. Check `GET /health` before sending requests.

## Start the dashboard (second PowerShell window)

```powershell
cd C:\path\to\CreditRisk
.venv\Scripts\Activate.ps1
$env:CREDIT_RISK_API_URL = "http://127.0.0.1:8000"
python -m streamlit run src/dashboard/app.py --server.address 127.0.0.1 --server.port 8501
```

Open `http://127.0.0.1:8501`. The dashboard has no mock or offline data mode:
when the API cannot be reached, it displays an explicit connection error.
Applicant lists are paginated and omit `TARGET`.

## Quick checks

```powershell
Invoke-RestMethod http://127.0.0.1:8000/ | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/health | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/applications?fields=SK_ID_CURR,TARGET,AMT_CREDIT`&limit=2 | ConvertTo-Json -Depth 5
Invoke-RestMethod http://127.0.0.1:8000/predict -Method Post -ContentType application/json -Body '{"application_id":120321}' | ConvertTo-Json
```

Interactive documentation is served at `http://127.0.0.1:8000/docs`; the
generated OpenAPI document is committed at [`docs/openapi.json`](openapi.json).

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `CREDIT_RISK_MODEL_DIR` | `models/local_v2` | Model artifact directory. Relative values resolve against the project root. |
| `CREDIT_RISK_APPLICATION_DATA` | `data/raw/application_train.csv` | Application CSV with `SK_ID_CURR`. Relative values resolve against the project root. |
| `CREDIT_RISK_CORS_ORIGINS` | `http://localhost:8501,http://127.0.0.1:8501` | Comma-separated allowed dashboard origins. |
| `CREDIT_RISK_LOG_LEVEL` | `INFO` | Python log level. |
| `CREDIT_RISK_API_URL` | `http://127.0.0.1:8000` | Dashboard base URL for the FastAPI service. |

The service has no authentication. Bind it to loopback only and do not expose
applicant data to a network or public deployment until authentication and
authorization are implemented.
