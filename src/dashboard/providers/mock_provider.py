"""Deterministic, explicitly synthetic Home Credit-shaped demonstration data."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from api.fields import (
    APPLICATION_FIELDS,
    DEFAULT_APPLICATION_FIELDS,
    FILTERABLE_APPLICATION_FIELDS,
    MAX_APPLICATION_LIMIT,
)

LOW_UPPER = 0.30
MEDIUM_UPPER = 0.60
MODEL_NAME = "Synthetic demo predictor"
MODEL_VERSION = "synthetic-demo-v1"


class MockProviderError(ValueError):
    """Invalid request or malformed mock-provider configuration."""


class MockConfigurationError(MockProviderError):
    """Invalid mock sample-size or seed setting."""


@dataclass(frozen=True)
class MockDataset:
    applications: pd.DataFrame
    installments_payments: pd.DataFrame
    bureau: pd.DataFrame
    bureau_balance: pd.DataFrame
    previous_application: pd.DataFrame
    pos_cash_balance: pd.DataFrame
    credit_card_balance: pd.DataFrame


def _integer_setting(name: str, default: int, *, minimum: int, maximum: int | None = None) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as error:
        raise MockConfigurationError(f"{name} must be an integer.") from error
    if value < minimum or (maximum is not None and value > maximum):
        bound = f"between {minimum} and {maximum}" if maximum is not None else f"at least {minimum}"
        raise MockConfigurationError(f"{name} must be {bound}.")
    return value


def sample_size() -> int:
    return _integer_setting("CREDIT_RISK_MOCK_SAMPLE_SIZE", 300, minimum=1, maximum=5000)


def seed() -> int:
    return _integer_setting("CREDIT_RISK_MOCK_SEED", 42, minimum=0)


@lru_cache(maxsize=4)
def _build_dataset(size: int, random_seed: int) -> MockDataset:
    rng = np.random.default_rng(random_seed)
    ids = np.arange(100001, 100001 + size, dtype=np.int64)
    income = np.clip(rng.lognormal(np.log(175_000), 0.48, size), 45_000, 1_500_000)
    credit = np.clip(income * rng.uniform(1.1, 4.8, size), 25_000, 1_800_000)
    annuity = credit * rng.uniform(0.045, 0.14, size)
    goods_price = credit * rng.uniform(0.82, 1.0, size)
    children = rng.choice([0, 1, 2, 3, 4], size=size, p=[0.48, 0.25, 0.18, 0.07, 0.02])
    family_members = children + rng.choice([1, 2], size=size, p=[0.34, 0.66])
    days_birth = -rng.integers(20 * 365, 70 * 365, size=size)
    days_employed = -rng.integers(0, 25 * 365, size=size)
    unemployed = rng.random(size) < 0.06
    days_employed[unemployed] = 365243
    ext_sources = [rng.beta(2.2, 2.0, size=size) for _ in range(3)]
    for values, missing_rate in zip(ext_sources, (0.15, 0.03, 0.17)):
        values[rng.random(size) < missing_rate] = np.nan

    # The historical target is a synthetic sample label. Predictions below are
    # generated independently and never read TARGET.
    target = rng.binomial(1, 0.085, size=size).astype(np.int8)
    applications = pd.DataFrame(
        {
            "SK_ID_CURR": ids,
            "TARGET": target,
            "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Revolving loans"], size=size, p=[0.88, 0.12]),
            "CODE_GENDER": rng.choice(["F", "M", "XNA"], size=size, p=[0.659, 0.340, 0.001]),
            "FLAG_OWN_CAR": rng.choice(["Y", "N"], size=size, p=[0.34, 0.66]),
            "FLAG_OWN_REALTY": rng.choice(["Y", "N"], size=size, p=[0.69, 0.31]),
            "CNT_CHILDREN": children.astype(int),
            "AMT_INCOME_TOTAL": income.round(2),
            "AMT_CREDIT": credit.round(2),
            "AMT_ANNUITY": annuity.round(2),
            "AMT_GOODS_PRICE": goods_price.round(2),
            "NAME_INCOME_TYPE": rng.choice(["Working", "Commercial associate", "Pensioner", "State servant", "Maternity leave"], size=size, p=[0.52, 0.23, 0.17, 0.075, 0.005]),
            "NAME_EDUCATION_TYPE": rng.choice(["Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary", "Academic degree"], size=size, p=[0.70, 0.24, 0.04, 0.018, 0.002]),
            "NAME_FAMILY_STATUS": rng.choice(["Married", "Single / not married", "Civil marriage", "Separated", "Widow"], size=size, p=[0.64, 0.15, 0.08, 0.08, 0.05]),
            "NAME_HOUSING_TYPE": rng.choice(["House / apartment", "With parents", "Rented apartment", "Municipal apartment", "Office apartment", "Co-op apartment"], size=size, p=[0.88, 0.06, 0.035, 0.015, 0.008, 0.002]),
            "DAYS_BIRTH": days_birth.astype(int),
            "DAYS_EMPLOYED": days_employed.astype(int),
            "OCCUPATION_TYPE": rng.choice(["Laborers", "Sales staff", "Core staff", "Managers", "Drivers", "Accountants", None], size=size, p=[0.22, 0.12, 0.10, 0.08, 0.07, 0.05, 0.36]),
            "CNT_FAM_MEMBERS": family_members.astype(int),
            "REGION_POPULATION_RELATIVE": rng.uniform(0.0003, 0.0725, size).round(6),
            "EXT_SOURCE_1": ext_sources[0].round(6),
            "EXT_SOURCE_2": ext_sources[1].round(6),
            "EXT_SOURCE_3": ext_sources[2].round(6),
        }
    )

    previous_rows: list[dict[str, Any]] = []
    installment_rows: list[dict[str, Any]] = []
    bureau_rows: list[dict[str, Any]] = []
    bureau_balance_rows: list[dict[str, Any]] = []
    pos_rows: list[dict[str, Any]] = []
    card_rows: list[dict[str, Any]] = []
    previous_id = 200001
    bureau_id = 500001
    installment_applicants = set(ids[::2].tolist())
    for app_id, app_credit in zip(ids, credit):
        if rng.random() < 0.64:
            loan_count = int(rng.integers(1, 4))
            for _ in range(loan_count):
                prev_id = previous_id
                previous_id += 1
                decision_day = -int(rng.integers(60, 2200))
                prev_credit = float(app_credit * rng.uniform(0.15, 1.2))
                previous_rows.append(
                    {
                        "SK_ID_CURR": int(app_id),
                        "SK_ID_PREV": prev_id,
                        "NAME_CONTRACT_STATUS": str(rng.choice(["Approved", "Refused", "Canceled", "Unused offer"], p=[0.72, 0.19, 0.06, 0.03])),
                        "AMT_APPLICATION": round(prev_credit * float(rng.uniform(0.8, 1.15)), 2),
                        "AMT_CREDIT": round(prev_credit, 2),
                        "AMT_ANNUITY": round(prev_credit * float(rng.uniform(0.04, 0.14)), 2),
                        "DAYS_DECISION": decision_day,
                    }
                )
                if rng.random() < 0.65:
                    month_count = int(rng.integers(2, 8))
                    for month in range(-month_count + 1, 1):
                        pos_rows.append(
                            {
                                "SK_ID_PREV": prev_id,
                                "MONTHS_BALANCE": month,
                                "CNT_INSTALMENT": month_count,
                                "CNT_INSTALMENT_FUTURE": max(0, -month),
                                "NAME_CONTRACT_STATUS": "Active" if month < 0 else "Completed",
                            }
                        )
                if rng.random() < 0.22:
                    for month in range(-4, 1):
                        card_rows.append(
                            {
                                "SK_ID_PREV": prev_id,
                                "MONTHS_BALANCE": month,
                                "AMT_BALANCE": round(float(rng.uniform(0, 0.75) * prev_credit), 2),
                                "AMT_CREDIT_LIMIT_ACTUAL": round(prev_credit * 1.15, 2),
                                "AMT_DRAWINGS_CURRENT": round(float(rng.uniform(0, 0.12) * prev_credit), 2),
                                "AMT_PAYMENT_CURRENT": round(float(rng.uniform(0, 0.1) * prev_credit), 2),
                            }
                        )

        if rng.random() < 0.5:
            for _ in range(int(rng.integers(1, 4))):
                current_bureau_id = bureau_id
                bureau_id += 1
                overdue = int(rng.choice([0, 0, 0, 15, 45, 90]))
                bureau_credit = float(rng.uniform(5_000, 500_000))
                bureau_rows.append(
                    {
                        "SK_ID_CURR": int(app_id),
                        "SK_ID_BUREAU": current_bureau_id,
                        "CREDIT_ACTIVE": str(rng.choice(["Closed", "Active", "Sold", "Bad debt"], p=[0.58, 0.36, 0.05, 0.01])),
                        "CREDIT_CURRENCY": "currency 1",
                        "DAYS_CREDIT": -int(rng.integers(30, 2500)),
                        "CREDIT_DAY_OVERDUE": overdue,
                        "AMT_CREDIT_SUM": round(bureau_credit, 2),
                        "AMT_CREDIT_SUM_DEBT": round(float(rng.uniform(0, 0.9) * bureau_credit), 2),
                        "AMT_CREDIT_SUM_LIMIT": round(float(rng.uniform(0, 0.4) * bureau_credit), 2),
                        "AMT_CREDIT_MAX_OVERDUE": round(float(rng.choice([0, rng.uniform(100, 15000)])), 2),
                    }
                )
                for month in range(-int(rng.integers(1, 8)) + 1, 1):
                    bureau_balance_rows.append(
                        {
                            "SK_ID_BUREAU": current_bureau_id,
                            "MONTHS_BALANCE": month,
                            "STATUS": str(rng.choice(["0", "1", "2", "3", "4", "5", "C", "X"], p=[0.62, 0.08, 0.035, 0.015, 0.01, 0.01, 0.20, 0.03])),
                        }
                    )

        if app_id in installment_applicants:
            app_previous = [row["SK_ID_PREV"] for row in previous_rows if row["SK_ID_CURR"] == app_id]
            for prev_id in app_previous[:2]:
                installment_count = int(rng.integers(3, 9))
                installment_amount = float(rng.uniform(4_000, 35_000))
                for number in range(1, installment_count + 1):
                    installment_day = -30 * (installment_count - number + 1)
                    event = float(rng.random())
                    if event < 0.08:
                        entry_day = np.nan
                        payment = 0.0
                    elif event < 0.22:
                        entry_day = installment_day + int(rng.integers(4, 22))
                        payment = installment_amount * float(rng.uniform(0.35, 0.75))
                    else:
                        entry_day = installment_day + int(rng.integers(-3, 4))
                        payment = installment_amount * float(rng.uniform(0.9, 1.08))
                    installment_rows.append(
                        {
                            "SK_ID_CURR": int(app_id),
                            "SK_ID_PREV": int(prev_id),
                            "NUM_INSTALMENT_VERSION": 1.0,
                            "NUM_INSTALMENT_NUMBER": number,
                            "DAYS_INSTALMENT": installment_day,
                            "DAYS_ENTRY_PAYMENT": entry_day,
                            "AMT_INSTALMENT": round(installment_amount, 2),
                            "AMT_PAYMENT": round(payment, 2),
                        }
                    )

    installments = pd.DataFrame(
        installment_rows,
        columns=["SK_ID_CURR", "SK_ID_PREV", "NUM_INSTALMENT_VERSION", "NUM_INSTALMENT_NUMBER", "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT", "AMT_INSTALMENT", "AMT_PAYMENT"],
    ).sort_values(["SK_ID_CURR", "SK_ID_PREV", "DAYS_INSTALMENT"], kind="stable", ignore_index=True)
    return MockDataset(
        applications=applications,
        installments_payments=installments,
        bureau=pd.DataFrame(bureau_rows, columns=["SK_ID_CURR", "SK_ID_BUREAU", "CREDIT_ACTIVE", "CREDIT_CURRENCY", "DAYS_CREDIT", "CREDIT_DAY_OVERDUE", "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_LIMIT", "AMT_CREDIT_MAX_OVERDUE"]),
        bureau_balance=pd.DataFrame(bureau_balance_rows, columns=["SK_ID_BUREAU", "MONTHS_BALANCE", "STATUS"]),
        previous_application=pd.DataFrame(previous_rows, columns=["SK_ID_CURR", "SK_ID_PREV", "NAME_CONTRACT_STATUS", "AMT_APPLICATION", "AMT_CREDIT", "AMT_ANNUITY", "DAYS_DECISION"]),
        pos_cash_balance=pd.DataFrame(pos_rows, columns=["SK_ID_PREV", "MONTHS_BALANCE", "CNT_INSTALMENT", "CNT_INSTALMENT_FUTURE", "NAME_CONTRACT_STATUS"]),
        credit_card_balance=pd.DataFrame(card_rows, columns=["SK_ID_PREV", "MONTHS_BALANCE", "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL", "AMT_DRAWINGS_CURRENT", "AMT_PAYMENT_CURRENT"]),
    )


def get_dataset() -> MockDataset:
    """Return cached data; callers should treat frames as immutable."""
    return _build_dataset(sample_size(), seed())


def _risk_level(probability: float) -> str:
    if probability < LOW_UPPER:
        return "LOW"
    if probability < MEDIUM_UPPER:
        return "MEDIUM"
    return "HIGH"


def _prediction_probability(row: pd.Series) -> float:
    # A transparent deterministic demonstration formula, independent of TARGET.
    source = row[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].astype(float)
    source_mean = float(source.mean()) if source.notna().any() else 0.5
    credit_ratio = min(float(row["AMT_CREDIT"]) / max(float(row["AMT_INCOME_TOTAL"]), 1.0), 8.0) / 8.0
    score = 0.68 * (1.0 - source_mean) + 0.32 * credit_ratio
    return round(min(max(score, 0.02), 0.98), 4)


def get_health() -> dict[str, Any]:
    dataset = get_dataset()
    return {
        "status": "ok",
        "model_loaded": True,
        "data_available": not dataset.applications.empty,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "model_hash": None,
        "source_mode": "mock",
    }


def get_applications_page(
    *,
    limit: int = 100,
    offset: int = 0,
    fields: list[str] | None = None,
    application_id: int | None = None,
    filter_field: str | None = None,
    filter_value: str | None = None,
) -> tuple[pd.DataFrame, int]:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_APPLICATION_LIMIT:
        raise MockProviderError(f"limit must be between 1 and {MAX_APPLICATION_LIMIT}.")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise MockProviderError("offset must be a non-negative integer.")
    selected = list(DEFAULT_APPLICATION_FIELDS if not fields else fields)
    if any(field not in APPLICATION_FIELDS for field in selected):
        raise MockProviderError("Requested fields include a field outside the application API allowlist.")
    if (filter_field is None) != (filter_value is None):
        raise MockProviderError("filter_field and filter_value must be provided together.")
    if filter_field is not None and filter_field not in FILTERABLE_APPLICATION_FIELDS:
        raise MockProviderError("The requested application filter is not supported.")
    records = get_dataset().applications
    if application_id is not None:
        records = records.loc[records["SK_ID_CURR"].eq(application_id)]
    if filter_field is not None and filter_value is not None:
        records = records.loc[records[filter_field].astype(str).eq(filter_value)]
    total = len(records)
    page = records.iloc[offset : offset + limit].loc[:, selected].copy()
    return page, total


def get_prediction(application_id: int, *, include_explanation: bool) -> dict[str, Any]:
    records = get_dataset().applications
    match = records.loc[records["SK_ID_CURR"].eq(application_id)]
    if match.empty:
        return {
            "status": "no_prediction",
            "application_id": application_id,
            "probability": None,
            "risk_level": None,
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "prediction_timestamp": None,
            "explanation": None,
            "message": "Synthetic mock data has no application with this ID.",
        }
    row = match.iloc[0]
    if (application_id - 100001) % 17 == 0:
        return {
            "status": "no_prediction",
            "application_id": application_id,
            "probability": None,
            "risk_level": None,
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "prediction_timestamp": None,
            "explanation": None,
            "message": "No synthetic demonstration prediction is assigned to this application.",
        }
    probability = _prediction_probability(row)
    explanation = None
    if include_explanation:
        feature_effects = []
        for field, direction in (
            ("EXT_SOURCE_1", "decreases_risk"),
            ("EXT_SOURCE_2", "decreases_risk"),
            ("AMT_CREDIT", "increases_risk"),
        ):
            value = row[field]
            feature_effects.append(
                {
                    "raw_feature": field,
                    "raw_value": None if pd.isna(value) else float(value),
                    "effect": direction,
                }
            )
        explanation = {
            "synthetic": True,
            "method": "illustrative_rules",
            "output_space": "probability",
            "top_features": feature_effects,
            "summary": "Synthetic illustrative factors only; this is not a trained model explanation or SHAP output.",
        }
    return {
        "status": "available",
        "application_id": application_id,
        "probability": probability,
        "risk_level": _risk_level(probability),
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "prediction_timestamp": None,
        "explanation": explanation,
        "message": "Synthetic illustrative prediction; not produced by a trained XGBoost or LSTM model.",
    }


def get_model_info() -> dict[str, Any]:
    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "model_hash": "",
        "training_date": None,
        "library_versions": {},
        "thresholds": {
            "decision_threshold": 0.5,
            "risk_bands": {"low_upper": LOW_UPPER, "medium_upper": MEDIUM_UPPER},
        },
        "source_mode": "mock",
        "provenance": "synthetic illustrative data; no model training occurred",
    }


def get_prediction_summary(split: str = "test") -> dict[str, Any]:
    if split not in {"test", "validation"}:
        raise MockProviderError("split must be either 'test' or 'validation'.")
    records = get_dataset().applications
    predictions = [
        _prediction_probability(row)
        for _, row in records.iterrows()
        if (int(row["SK_ID_CURR"]) - 100001) % 17 != 0
    ]
    levels = [_risk_level(value) for value in predictions]
    positive = sum(value >= 0.5 for value in predictions)
    count = len(predictions)
    return {
        "split": split,
        "prediction_count": count,
        "predicted_outcomes": {
            "no_defined_repayment_difficulty": count - positive,
            "repayment_difficulty": positive,
        },
        "risk_distribution": {level: levels.count(level) for level in ("LOW", "MEDIUM", "HIGH")},
        "risk_bands": {"low_upper": LOW_UPPER, "medium_upper": MEDIUM_UPPER},
        "decision_threshold": 0.5,
        "source": "synthetic_illustrative_predictions",
        "source_mode": "mock",
        "label": "Synthetic illustrative outputs; not trained-model results.",
    }


def get_model_evaluation_results(split: str = "test") -> pd.DataFrame:
    if split not in {"test", "validation"}:
        raise MockProviderError("split must be either 'test' or 'validation'.")
    # The demo provider has no trained evaluation artifact. Returning an empty
    # table prevents illustrative numbers from being mistaken for model scores.
    results = pd.DataFrame(columns=["model", "metric", "value"])
    results.attrs["split"] = split
    results.attrs["provenance"] = {
        "model_name": None,
        "model_version": None,
        "model_hash": None,
        "training_date": None,
        "generated_at": None,
        "threshold": None,
        "n_samples": None,
        "class_counts": None,
        "split_protocol": None,
        "source": "synthetic_no_evaluation_artifact",
        "synthetic": True,
        "lstm_available": False,
        "note": "No trained model evaluation artifact is available in synthetic mock mode.",
    }
    results.attrs["confusion_matrix"] = None
    return results

