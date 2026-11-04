from __future__ import annotations

import os
from typing import Any, Mapping

import httpx


DEFAULT_API_URL = "http://127.0.0.1:8000"
DEFAULT_TIMEOUT_SECONDS = 20.0


class APIClientError(RuntimeError):
    """Base error for dashboard-to-API communication."""


class APIUnavailableError(APIClientError):
    """The API could not be reached or is not ready."""


class APIResponseError(APIClientError):
    """The API returned an unsuccessful response."""

    def __init__(
        self,
        status_code: int,
        message: str,
        payload: Any = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload


class APIClient:
    """Small synchronous client for the dashboard's FastAPI data source."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        client: httpx.Client | None = None,
    ) -> None:
        configured_url = base_url or os.environ.get(
            "CREDIT_RISK_API_URL", DEFAULT_API_URL
        )
        self.base_url = configured_url.rstrip("/")
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("CREDIT_RISK_API_URL must be an HTTP(S) URL.")
        self.timeout = timeout
        self._client = client

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        json: Mapping[str, Any] | None = None,
    ) -> Any:
        url = f"{self.base_url}{path}"
        try:
            if self._client is not None:
                response = self._client.request(
                    method, url, params=params, json=json
                )
            else:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.request(
                        method, url, params=params, json=json
                    )
        except httpx.RequestError as exc:
            raise APIUnavailableError(
                f"Credit-risk API at {self.base_url} is unavailable: {exc}"
            ) from exc

        try:
            payload = response.json()
        except ValueError:
            payload = None

        if not response.is_success:
            if isinstance(payload, dict):
                message = payload.get("message") or payload.get("detail")
            else:
                message = None
            raise APIResponseError(
                response.status_code,
                str(message or f"API request failed with HTTP {response.status_code}."),
                payload,
            )
        if payload is None:
            raise APIClientError(f"The API returned invalid JSON for {path}.")
        return payload

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def list_applications(
        self,
        *,
        limit: int,
        offset: int,
        fields: list[str] | None = None,
        application_id: int | None = None,
        filter_field: str | None = None,
        filter_value: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if fields:
            params["fields"] = ",".join(fields)
        if application_id is not None:
            params["application_id"] = application_id
        if filter_field is not None and filter_value is not None:
            params["filter_field"] = filter_field
            params["filter_value"] = filter_value
        return self._request("GET", "/applications", params=params)

    def get_application(
        self, application_id: int, fields: list[str] | None = None
    ) -> dict[str, Any]:
        params = {"fields": ",".join(fields)} if fields else None
        return self._request(
            "GET", f"/applications/{application_id}", params=params
        )

    def predict(
        self, application_id: int, *, include_explanation: bool = True
    ) -> dict[str, Any]:
        return self._request(
            "GET",
            f"/applications/{application_id}/prediction",
            params={"include_explanation": include_explanation},
        )

    def model_info(self) -> dict[str, Any]:
        return self._request("GET", "/models")

    def evaluation(self, split: str = "test") -> dict[str, Any]:
        return self._request(
            "GET", "/model/evaluation", params={"split": split}
        )

    def prediction_summary(self, split: str = "test") -> dict[str, Any]:
        return self._request(
            "GET", "/model/prediction-summary", params={"split": split}
        )
