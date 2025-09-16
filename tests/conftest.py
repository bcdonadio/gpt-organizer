"""Test configuration for the gpt-organizer project.

This module installs light-weight stubs for optional third party
integrations so that unit tests can import ``GptCategorize.categorize``
without requiring the real ``openai`` and ``qdrant_client`` packages to
be installed in the execution environment.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _ensure_openai_stub() -> None:
    if "openai" in sys.modules:
        return

    openai_stub = types.ModuleType("openai")

    class _OpenAI:  # pragma: no cover - trivial container
        def __init__(self, *args, **kwargs) -> None:  # noqa: D401
            """Minimal stub that accepts arbitrary arguments."""

    openai_stub.OpenAI = _OpenAI
    sys.modules["openai"] = openai_stub


def _ensure_qdrant_stub() -> None:
    if "qdrant_client" in sys.modules:
        return

    qdrant_stub = types.ModuleType("qdrant_client")

    class _QdrantClient:  # pragma: no cover - trivial container
        def __init__(self, *args, **kwargs) -> None:  # noqa: D401
            """Minimal stub that accepts arbitrary arguments."""

    qdrant_stub.QdrantClient = _QdrantClient

    http_stub = types.ModuleType("qdrant_client.http")
    models_stub = types.ModuleType("qdrant_client.http.models")

    class _Distance:  # pragma: no cover - trivial container
        COSINE = "cosine"

    class _VectorParams:  # pragma: no cover - trivial container
        def __init__(self, *args, **kwargs) -> None:  # noqa: D401
            """Minimal stub that accepts arbitrary arguments."""

    class _PointStruct:  # pragma: no cover - trivial container
        def __init__(self, *args, **kwargs) -> None:  # noqa: D401
            """Minimal stub that accepts arbitrary arguments."""

    models_stub.Distance = _Distance
    models_stub.VectorParams = _VectorParams
    models_stub.PointStruct = _PointStruct

    http_stub.models = models_stub

    qdrant_stub.http = http_stub

    sys.modules["qdrant_client"] = qdrant_stub
    sys.modules["qdrant_client.http"] = http_stub
    sys.modules["qdrant_client.http.models"] = models_stub


def _ensure_tqdm_stub() -> None:
    if "tqdm" in sys.modules:
        return

    tqdm_stub = types.ModuleType("tqdm")

    def _tqdm(iterable=None, *args, **kwargs):  # pragma: no cover - trivial container
        return iterable

    tqdm_stub.tqdm = _tqdm
    sys.modules["tqdm"] = tqdm_stub


def _ensure_sklearn_stub() -> None:
    if "sklearn" in sys.modules:
        return

    sklearn_stub = types.ModuleType("sklearn")
    cluster_stub = types.ModuleType("sklearn.cluster")
    preprocessing_stub = types.ModuleType("sklearn.preprocessing")

    import numpy as _np

    class _DBSCAN:  # pragma: no cover - minimal stand-in
        def __init__(self, *args, **kwargs) -> None:  # noqa: D401
            """Minimal stub that accepts arbitrary arguments."""

        def fit_predict(self, vectors):
            return _np.zeros(len(vectors), dtype=int)

    class _KMeans:  # pragma: no cover - minimal stand-in
        def __init__(self, n_clusters=8, *args, **kwargs) -> None:  # noqa: D401
            self.n_clusters = max(1, int(n_clusters))

        def fit_predict(self, vectors):
            n = len(vectors)
            if n == 0:
                return _np.array([], dtype=int)
            return _np.arange(n, dtype=int) % self.n_clusters

    def _normalize(vectors, norm="l2"):
        arr = _np.array(vectors, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if norm != "l2":
            raise ValueError("Only l2 norm is supported in the stub")
        norms = _np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return arr / norms

    cluster_stub.DBSCAN = _DBSCAN
    cluster_stub.KMeans = _KMeans
    preprocessing_stub.normalize = _normalize

    sklearn_stub.cluster = cluster_stub
    sklearn_stub.preprocessing = preprocessing_stub

    sys.modules["sklearn"] = sklearn_stub
    sys.modules["sklearn.cluster"] = cluster_stub
    sys.modules["sklearn.preprocessing"] = preprocessing_stub


def pytest_configure() -> None:
    """Install compatibility stubs before pytest starts collecting tests."""

    _ensure_openai_stub()
    _ensure_qdrant_stub()
    _ensure_tqdm_stub()
    _ensure_sklearn_stub()

