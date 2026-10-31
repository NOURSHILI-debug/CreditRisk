from __future__ import annotations

import pandas as pd
from streamlit.testing.v1 import AppTest

from dashboard.pages.overview import (
    _category_distribution,
    _numeric_values,
    _summarize_applications,
    _target_by_category,
    _target_distribution,
)


def test_target_statistics_use_labeled_rows_and_exclude_missing_or_invalid_values() -> None:
    applications = pd.DataFrame({"TARGET": [0, 1, None, "invalid"]})

    summary = _summarize_applications(applications)
    distribution = _target_distribution(applications)

    assert summary == {
        "total": 4,
        "target_0": 1,
        "target_1": 1,
        "difficulty_rate": 0.5,
        "unlabeled": 2,
    }
    assert distribution["Applications"].tolist() == [1, 1]
    assert distribution["Share"].tolist() == [0.5, 0.5]


def test_single_target_class_has_valid_rate_and_zero_count_for_other_class() -> None:
    applications = pd.DataFrame({"TARGET": [1, 1, 1]})

    summary = _summarize_applications(applications)
    distribution = _target_distribution(applications)

    assert summary["difficulty_rate"] == 1.0
    assert summary["target_0"] == 0
    assert summary["target_1"] == 3
    assert distribution["Applications"].tolist() == [0, 3]
    assert distribution["Share"].tolist() == [0.0, 1.0]


def test_missing_target_and_empty_data_are_unavailable_not_zero() -> None:
    for applications in (pd.DataFrame(), pd.DataFrame({"AMT_CREDIT": [10, 20]})):
        summary = _summarize_applications(applications)

        assert summary["target_0"] is None
        assert summary["target_1"] is None
        assert summary["difficulty_rate"] is None
        assert _target_distribution(applications).empty


def test_category_distribution_preserves_missing_category() -> None:
    applications = pd.DataFrame({"NAME_INCOME_TYPE": ["Working", None, "Working"]})

    distribution = _category_distribution(applications, "NAME_INCOME_TYPE")

    assert set(distribution["Category"]) == {"Working", "Missing"}
    assert sorted(distribution["Applications"].tolist()) == [1, 2]
    assert sorted(distribution["Share"].tolist()) == [1 / 3, 2 / 3]


def test_target_by_category_excludes_missing_target_and_keeps_missing_category() -> None:
    applications = pd.DataFrame(
        {
            "NAME_INCOME_TYPE": ["Working", None, "Pensioner"],
            "TARGET": [0, 1, None],
        }
    )

    distribution = _target_by_category(applications, "NAME_INCOME_TYPE")

    assert set(distribution["Category"]) == {"Working", "Missing"}
    assert set(distribution["TARGET"]) == {
        "No defined repayment difficulty (0)",
        "Repayment difficulty (1)",
    }


def test_numeric_distribution_values_drop_invalid_and_non_finite_data() -> None:
    applications = pd.DataFrame({"AMT_CREDIT": ["100", None, "invalid", "inf", "-inf"]})

    values = _numeric_values(applications, "AMT_CREDIT")

    assert values.tolist() == [100]


def test_empty_provider_result_renders_unavailable_target_metrics() -> None:
    app = _render_overview_with_rows([])

    assert not app.exception
    assert [(metric.label, metric.value) for metric in app.metric] == [
        ("Total applications", "0"),
        ("TARGET = 0", "N/A"),
        ("TARGET = 1", "N/A"),
        ("Repayment-difficulty rate", "N/A"),
    ]
    assert "No applications are available from the configured data source." in [
        element.value for element in app.info
    ]


def test_missing_target_column_renders_other_available_statistics() -> None:
    app = _render_overview_with_rows([{"AMT_CREDIT": 50000}, {"AMT_CREDIT": None}])

    assert not app.exception
    assert app.metric[0].value == "2"
    assert [metric.value for metric in app.metric[1:]] == ["N/A", "N/A", "N/A"]


def _render_overview_with_rows(rows: list[dict[str, object]]) -> AppTest:
    serialized_rows = pd.DataFrame(rows).to_json(orient="records")
    app_source = (
        "import json\n"
        "import pandas as pd\n"
        "from dashboard.pages import overview\n"
        f"rows = json.loads({serialized_rows!r})\n"
        "overview.data_access.get_applications = lambda: pd.DataFrame(rows)\n"
        "overview.render_overview()\n"
    )
    return AppTest.from_string(app_source).run()
