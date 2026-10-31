from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

MOCK_MODE = True


def _build_mock_applications() -> pd.DataFrame:
    """Construct a small synthetic application dataset for local dashboard development."""
    rng = np.random.default_rng(42)
    row_count = 120

    sk_ids = np.arange(1001, 1001 + row_count)
    income = np.clip(rng.lognormal(mean=10.3, sigma=0.55, size=row_count) * 1000, 15000, 500000)
    credit = np.clip(rng.lognormal(mean=11.1, sigma=0.5, size=row_count) * 1500, 40000, 2500000)
    annuity = np.clip(credit * rng.uniform(0.12, 0.32, size=row_count), 4500, 180000)
    target = rng.binomial(1, 0.27, size=row_count)
    risk_probability = np.clip(0.08 + 0.72 * target + rng.normal(0.0, 0.12, size=row_count), 0.01, 0.99)
    risk_category = np.where(
        risk_probability >= 0.60,
        "High Risk",
        np.where(risk_probability >= 0.35, "Medium Risk", "Low Risk"),
    )

    income_types = [
        "Working",
        "Commercial associate",
        "Pensioner",
        "State servant",
        "Student",
    ]
    education_types = [
        "Higher education",
        "Secondary / secondary special",
        "Incomplete higher",
        "Lower secondary",
    ]

    data = {
        "SK_ID_CURR": sk_ids,
        "AMT_INCOME_TOTAL": income,
        "AMT_CREDIT": credit,
        "AMT_ANNUITY": annuity,
        "NAME_INCOME_TYPE": rng.choice(income_types, size=row_count),
        "NAME_EDUCATION_TYPE": rng.choice(education_types, size=row_count),
        "TARGET": target,
        "RISK_PROBABILITY": np.round(risk_probability, 3),
        "RISK_CATEGORY": risk_category,
    }

    df = pd.DataFrame(data)
    df["RISK_PROBABILITY"] = df["RISK_PROBABILITY"].astype(float)
    df["TARGET"] = df["TARGET"].astype(int)
    return df


_MOCK_APPLICATIONS = _build_mock_applications()


def get_applications() -> pd.DataFrame:
    """Return the current application dataset used by the dashboard."""
    return _MOCK_APPLICATIONS.copy()


def get_application_by_id(application_id: Any) -> Optional[pd.Series]:
    """Fetch a single application by SK_ID_CURR with safe validation."""
    if application_id is None:
        return None

    try:
        app_id = int(application_id)
    except (TypeError, ValueError):
        return None

    match = _MOCK_APPLICATIONS.loc[_MOCK_APPLICATIONS["SK_ID_CURR"] == app_id]
    if match.empty:
        return None

    return match.iloc[0].copy()


def get_overview_metrics() -> dict[str, Any]:
    """Build summary metrics for the Overview page."""
    df = get_applications()
    default_target_counts = df["TARGET"].value_counts().reindex([0, 1], fill_value=0)
    risk_distribution = df["RISK_CATEGORY"].value_counts().to_dict()

    return {
        "total_applications": int(len(df)),
        "target_1": int(default_target_counts.get(1, 0)),
        "target_0": int(default_target_counts.get(0, 0)),
        "risk_distribution": risk_distribution,
    }


def get_model_metrics() -> dict[str, dict[str, float | str]]:
    """Return illustrative model metrics used to render the comparison page."""
    return {
        "XGBoost": {
            "ROC-AUC": 0.82,
            "PR-AUC": 0.74,
            "Precision": 0.68,
            "Recall": 0.71,
            "F1-score": 0.69,
            "source": "mock",
        },
        "LSTM": {
            "ROC-AUC": 0.79,
            "PR-AUC": 0.72,
            "Precision": 0.66,
            "Recall": 0.70,
            "F1-score": 0.67,
            "source": "mock",
        },
    }
