# Credit Risk Intelligence Platform

An end-to-end, cloud-based credit risk intelligence platform combining **data engineering, machine learning, real-time event simulation, interactive visualization, and LLM-powered analytics**.

The project aims to analyze credit applications, estimate repayment difficulty, compare predictive models, and provide explainable insights through an interactive dashboard and an AI-powered assistant.

---

## Project Overview

The platform uses the [Home Credit Default Risk dataset](https://www.kaggle.com/competitions/home-credit-default-risk/data) to develop a complete credit risk analysis pipeline.

It combines two machine learning approaches:

- **XGBoost:** Predicts credit default risk using structured application data.
- **PyTorch LSTM:** Models ordered repayment-history sequences to investigate the additional predictive value of historical payment behavior.

Historical application records will be replayed as simulated streaming events through Kafka, processed and stored using cloud data services, and made available to the prediction and application layers.

The final platform will provide risk statistics, model comparisons, application-level predictions, prediction explanations, and AI-generated analytical reports.

## Architecture

```text
                 Home Credit Dataset
                         |
                         v
                 Kafka Event Streaming
                 (Historical Replay)
                         |
                         v
                Data Ingestion Layer
                         |
                         v
              Validation & Processing
                         |
                         v
                  AWS S3 Storage
                 Raw / Processed
                         |
                +--------+--------+
                |                 |
                v                 v
             XGBoost           PyTorch LSTM
          Tabular Data       Payment Sequences
                |                 |
                +--------+--------+
                         |
                         v
                 Model Evaluation
                         |
                         v
                  Amazon SageMaker
                 (Where Feasible)
                         |
                         v
                Prediction Interface
                         |
                +--------+--------+
                |                 |
                v                 v
             Streamlit         AI Agent
             Dashboard       LangGraph /
                             LangChain
```

*Historical replay represents simulated streaming, not live banking transactions.*

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python 3.11 |
| Dataset | Home Credit Default Risk |
| Event streaming | Apache Kafka |
| Data processing | Pandas, NumPy |
| Cloud storage | Amazon S3 |
| Cloud processing | AWS Lambda / AWS Glue |
| Tabular ML | XGBoost |
| Sequential ML | PyTorch LSTM |
| Cloud ML | Amazon SageMaker |
| Dashboard | Streamlit |
| AI agent | LangGraph / LangChain |
| Explainability | SHAP |
| Testing | Pytest |
| Code quality | Ruff, Black, pre-commit |
| Version control | Git, GitHub |

## Dataset

The project uses the Home Credit Default Risk dataset.

Initial development focuses on:

- `application_train.csv`: Main application dataset containing applicant features and the prediction target.
- `installments_payments.csv`: Historical repayment information used to investigate sequential modeling.

### Prediction Target

The target variable is `TARGET`:

- `1`: Repayment difficulty.
- `0`: No defined repayment difficulty.

### Evaluation Strategy

Both models will be evaluated using the same held-out applicants.

Evaluation metrics:

- ROC-AUC
- PR-AUC
- Precision
- Recall
- F1-score

The modeling process will account for class imbalance, target leakage, feature availability at prediction time, and reproducibility.

## Repository Structure

```text
credit-risk-intelligence-platform/
│
├── .github/
│   └── workflows/
│
├── configs/
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── README.md
│
├── notebooks/
│
├── schemas/
│
├── src/
│   ├── ingestion/
│   ├── processing/
│   ├── storage/
│   ├── training/
│   ├── evaluation/
│   ├── inference/
│   ├── dashboard/
│   └── agent/
│
├── aws/
│   ├── glue/
│   ├── lambda/
│   └── sagemaker/
│
├── artifacts/
│
├── tests/
│   ├── unit/
│   └── integration/
│
├── docs/
│
├── .gitignore
├── .pre-commit-config.yaml
├── .python-version
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── Makefile
└── README.md
```

## Team Responsibilities

The project is developed collaboratively by two contributors, with separate ownership of the main technical components.

### Person 1 — Data Engineering & Machine Learning

Responsible for:

- Dataset acquisition and exploration.
- Kafka producer and consumer.
- Data validation and transformation.
- Raw and processed data storage.
- XGBoost model development.
- PyTorch LSTM sequence modeling.
- Model evaluation and comparison.
- SageMaker training and inference integration.
- Prediction interfaces and model versioning.

### Person 2 — Dashboard, AI Agent & Application

Responsible for:

- Streamlit dashboard development.
- Application explorer and risk visualizations.
- Dashboard data-access layer.
- LangGraph / LangChain agent.
- Tool calling and application data retrieval.
- Prediction explanations.
- Risk summaries and report generation.

### Shared Responsibilities

Both contributors participate in:

- Architecture and interface design.
- Integration and testing.
- Security and cost management.
- Technical documentation.
- Final demonstration and presentation.

## Development Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/credit-risk-intelligence-platform.git

cd credit-risk-intelligence-platform
```

### 2. Create a virtual environment

```bash
py -3.11 -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip

pip install -r requirements.txt

pip install -r requirements-dev.txt
```

### 4. Install pre-commit hooks

```bash
pre-commit install
```

### 5. Run tests

```bash
pytest
```

## Development Roadmap

| Phase | Focus |
|---|---|
| Phase 0 | Repository, branches, development environment and code quality |
| Phase 1 | Dataset exploration, architecture and shared schemas |
| Phase 2 | XGBoost baseline and dashboard prototype |
| Phase 3 | Kafka streaming and cloud data engineering |
| Phase 4 | LSTM, model comparison and SageMaker |
| Phase 5 | AI agent and end-to-end application |
| Phase 6 | Integration, testing, documentation and final demonstration |

## Project Deliverables

1. End-to-end simulated streaming and machine learning pipeline.
2. XGBoost and LSTM model comparison.
3. Cloud-based data storage and processing architecture.
4. Interactive Streamlit dashboard.
5. AI agent connected to actual project data.
6. Technical documentation and architecture diagram.
7. Two-minute project demonstration.
8. Ten-minute final presentation.

## Engineering Principles

- **Reproducibility:** Versioned configurations, dependencies and experiments.
- **Data integrity:** Validate incoming events and processed datasets.
- **Leakage prevention:** Use only information available at prediction time.
- **Modularity:** Separate ingestion, processing, training, evaluation and inference.
- **Explainability:** Provide evidence-based model explanations where available.
- **Cost awareness:** Prefer local development and provision cloud resources only when necessary.
- **Collaboration:** Use feature branches, pull requests and documented interfaces.

## Scope Limitations

The initial version focuses on delivering a working, testable end-to-end platform.

The following are outside the initial scope:

- Hybrid XGBoost–LSTM architectures.
- Multi-agent orchestration.
- Kubernetes deployment.
- Real banking integrations.
- Automated lending decisions.

Advanced features may be considered after the core deliverables are complete.

---

**Project Status:** Initial Development / Phase 0

**Domain:** Credit Risk Analytics · Machine Learning · Data Engineering · Cloud Computing · Generative AI
