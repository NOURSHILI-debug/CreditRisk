# Development setup

The dashboard defaults to the existing FastAPI data source. You can choose the
source in the dashboard sidebar for the current session or set the initial mode
in PowerShell before starting Streamlit.

## Real API mode (default)

Start the API in one PowerShell terminal:

```powershell
Set-Location C:\Users\souro\Desktop\CreditRisk
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = (Join-Path (Get-Location).Path 'src')
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Start the dashboard in another terminal:

```powershell
Set-Location C:\Users\souro\Desktop\CreditRisk
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = (Join-Path (Get-Location).Path 'src')
$env:CREDIT_RISK_DATA_MODE = 'api'
$env:CREDIT_RISK_API_URL = 'http://127.0.0.1:8000'
streamlit run src\dashboard\app.py
```

API mode never falls back to synthetic data when the backend is unavailable.

## Synthetic mock mode

The mock provider generates a deterministic Home Credit-shaped sample in
memory. It is not the Kaggle dataset, contains no real applicants, and does not
represent model performance. No generated CSVs are written to the repository.

In a fresh dashboard terminal:

```powershell
Set-Location C:\Users\souro\Desktop\CreditRisk
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = (Join-Path (Get-Location).Path 'src')
$env:CREDIT_RISK_DATA_MODE = 'mock'
$env:CREDIT_RISK_MOCK_SAMPLE_SIZE = '300'
$env:CREDIT_RISK_MOCK_SEED = '42'
$env:AGENT_LLM_PROVIDER = 'ollama'
$env:AGENT_LLM_MODEL = 'llama3.1:8b'
streamlit run src\dashboard\app.py
```

`CREDIT_RISK_DATA_MODE` accepts only `api` or `mock` and defaults to `api`.
`CREDIT_RISK_MOCK_SAMPLE_SIZE` defaults to 300 (allowed range 1–5000), and
`CREDIT_RISK_MOCK_SEED` defaults to 42. The provider caches a generated sample
by its configured size and seed, so reruns do not produce new records.

The sidebar selector switches modes for the current Streamlit session. In mock
mode all dashboard and agent retrieval uses the local synthetic provider;
application details omit `TARGET`. Synthetic predictions use an illustrative
deterministic formula, not a trained XGBoost or LSTM. Explanations are synthetic
illustrative factors, not SHAP. Evaluation results are unavailable in mock
mode because no trained evaluation artifact exists; the provider does not
invent model scores. Mock
historical tables are small, linked examples for application, previous-loan,
bureau, installment, POS cash, and credit-card structures; they are not an
implemented event or pipeline-status integration.

To return the initial mode to real API after restarting Streamlit, set
`$env:CREDIT_RISK_DATA_MODE = 'api'`. You can also select Real API in the
dashboard sidebar. If the API is unavailable in API mode, the dashboard reports
that error and does not switch providers.
