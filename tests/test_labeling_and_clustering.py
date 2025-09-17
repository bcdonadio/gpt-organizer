"""Tests covering clustering utilities and LLM-driven labeling."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Sequence, cast

import numpy as np
from numpy.typing import NDArray
import pytest

import GptCategorize.categorize as categorize


def test_cluster_embeddings_uses_dbscan_when_clusters_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """DBSCAN results should be returned when it finds clusters."""

    class DummyDBSCAN:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.args = args
            self.kwargs = kwargs

        def fit_predict(self, vectors: NDArray[np.float64]) -> NDArray[np.int_]:
            return np.array([0, 0, 1])

    monkeypatch.setattr(categorize, "DBSCAN", DummyDBSCAN)

    labels = categorize.cluster_embeddings(np.eye(3), eps_cosine=0.2, min_samples=2)
    assert np.array_equal(labels, np.array([0, 0, 1]))


def test_cluster_embeddings_falls_back_to_kmeans(monkeypatch: pytest.MonkeyPatch) -> None:
    """If DBSCAN finds only noise, the fallback KMeans path is used."""

    class NoiseDBSCAN:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def fit_predict(self, vectors: Sequence[Sequence[float]] | NDArray[np.float64]) -> NDArray[np.int_]:
            return np.full(len(vectors), -1)

    class DummyKMeans:
        def __init__(self, n_clusters: int, **kwargs: object) -> None:
            self.n_clusters = n_clusters

        def fit_predict(self, vectors: Sequence[Sequence[float]] | NDArray[np.float64]) -> NDArray[np.int_]:
            n = len(vectors)
            return np.arange(n) % self.n_clusters

    monkeypatch.setattr(categorize, "DBSCAN", NoiseDBSCAN)
    monkeypatch.setattr(categorize, "KMeans", DummyKMeans)

    labels = categorize.cluster_embeddings(np.eye(4), eps_cosine=0.3, min_samples=2)
    assert set(labels) == {0, 1}


class DummyLLMClient:
    def __init__(self, responses: Sequence[str], errors: Sequence[Exception] | None = None) -> None:
        self.responses: list[str] = list(responses)
        self.errors: list[Exception] = list(errors) if errors is not None else []
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, *args: object, **kwargs: object) -> SimpleNamespace:
        if self.calls < len(self.errors):
            err = self.errors[self.calls]
            self.calls += 1
            raise err
        resp = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=resp))])


class DummyHTTPError(RuntimeError):
    def __init__(self, message: str = "boom") -> None:
        super().__init__(message)
        self.response = SimpleNamespace(status_code=500, headers={"Retry-After": "1"}, text="fail")
        self.request = SimpleNamespace(url="https://api", method="POST", headers={"X": "y"})


def _sample_clusters() -> dict[int, list[categorize.Chat]]:
    return {
        0: [categorize.Chat(id="1", title="First title"), categorize.Chat(id="2", title="Second title")],
        1: [categorize.Chat(id="3", title="Third")],
    }


def test_label_clusters_with_llm_success() -> None:
    """Successful responses should be parsed into a mapping."""

    resp = json.dumps(
        [
            {
                "cluster_id": 0,
                "label": "Alpha",
                "project_folder_slug": "alpha",
                "project_title": "Alpha",
                "confidence": 0.9,
            }
        ]
    )
    client = DummyLLMClient([resp])
    result = categorize.label_clusters_with_llm(cast(Any, client), {0: _sample_clusters()[0]})
    assert result[0]["label"] == "Alpha"


def test_label_clusters_with_llm_retries_before_succeeding(monkeypatch: pytest.MonkeyPatch) -> None:
    """Transient API errors should trigger retries before success."""

    resp = json.dumps(
        [
            {
                "cluster_id": 0,
                "label": "Beta",
                "project_folder_slug": "beta",
                "project_title": "Beta",
                "confidence": 0.8,
            }
        ]
    )
    client = DummyLLMClient([resp], errors=[DummyHTTPError("Connection failed")])

    def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(categorize, "MAX_RETRIES", 2, raising=False)
    monkeypatch.setattr(categorize, "RETRY_DELAY", 0.01, raising=False)
    monkeypatch.setattr(cast(Any, categorize).time, "sleep", no_sleep, raising=False)

    result = categorize.label_clusters_with_llm(cast(Any, client), {0: _sample_clusters()[0]})
    assert result[0]["project_folder_slug"] == "beta"


def test_label_clusters_with_llm_fallback_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """On the final attempt the fallback call should still succeed."""

    resp = json.dumps(
        [
            {
                "cluster_id": 0,
                "label": "Gamma",
                "project_folder_slug": "gamma",
                "project_title": "Gamma",
                "confidence": 0.7,
            }
        ]
    )
    client = DummyLLMClient([resp])
    client.errors = [DummyHTTPError()]  # first call fails

    def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(categorize, "MAX_RETRIES", 1, raising=False)
    monkeypatch.setattr(categorize, "RETRY_DELAY", 0.01, raising=False)
    monkeypatch.setattr(cast(Any, categorize).time, "sleep", no_sleep, raising=False)

    result = categorize.label_clusters_with_llm(cast(Any, client), {0: _sample_clusters()[0]})
    assert result[0]["project_title"] == "Gamma"


def test_label_clusters_with_llm_fallback_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the fallback also fails, the original exception propagates."""

    client = DummyLLMClient([""], errors=[DummyHTTPError(), DummyHTTPError("fallback")])

    def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(categorize, "MAX_RETRIES", 1, raising=False)
    monkeypatch.setattr(categorize, "RETRY_DELAY", 0.01, raising=False)
    monkeypatch.setattr(cast(Any, categorize).time, "sleep", no_sleep, raising=False)

    with pytest.raises(DummyHTTPError):
        categorize.label_clusters_with_llm(cast(Any, client), {0: _sample_clusters()[0]})


def test_label_clusters_with_llm_requires_content() -> None:
    """Missing content should raise an informative error."""

    client = DummyLLMClient([""], errors=[])
    with pytest.raises(RuntimeError):
        categorize.label_clusters_with_llm(cast(Any, client), {0: _sample_clusters()[0]})


def test_label_clusters_with_llm_regex_and_parse_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Responses requiring regex extraction and parse errors are handled."""

    messy = (
        "Noise before "
        + json.dumps(
            [
                {
                    "cluster_id": 0,
                    "label": "Delta",
                    "project_folder_slug": "delta",
                    "project_title": "Delta",
                    "confidence": 0.75,
                },
                {"cluster": "invalid"},
            ]
        )
        + " trailing text"
    )

    client = DummyLLMClient([messy])
    monkeypatch.setattr(categorize, "VERBOSE_LOGGING", True, raising=False)
    result = categorize.label_clusters_with_llm(cast(Any, client), {0: _sample_clusters()[0]})
    assert result[0]["label"] == "Delta"


def test_label_clusters_with_llm_errors_without_json() -> None:
    """If no JSON is found the function should raise a runtime error."""

    client = DummyLLMClient(["No JSON here"], errors=[])
    with pytest.raises(RuntimeError):
        categorize.label_clusters_with_llm(cast(Any, client), {0: _sample_clusters()[0]})
