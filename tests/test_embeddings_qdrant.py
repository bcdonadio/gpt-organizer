"""Tests covering embedding retries and Qdrant persistence helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import GptCategorize.categorize as categorize


@pytest.fixture(autouse=True)
def patch_tqdm(monkeypatch):
    """Avoid noisy tqdm output during tests."""

    monkeypatch.setattr(categorize, "tqdm", lambda iterable, **kwargs: iterable, raising=False)


def _make_embedding_response(vectors: list[list[float]]):
    return SimpleNamespace(data=[SimpleNamespace(embedding=v) for v in vectors])


def test_embed_chats_with_retry_success(monkeypatch):
    """Embeddings should be gathered across batches when the API succeeds."""

    calls: list[list[str]] = []

    class DummyEmbeddings:
        def create(self, model: str, input: list[str]):
            calls.append(list(input))
            return _make_embedding_response([[float(len(input)), 0.5] for _ in input])

    client = SimpleNamespace(embeddings=DummyEmbeddings())
    arr = categorize.embed_chats_with_retry(client, ["chat-1", "chat-2"], batch_size=1)
    assert arr.shape == (2, 2)
    assert calls == [["chat-1"], ["chat-2"]]


def test_embed_chats_with_retry_handles_transient_failures(monkeypatch):
    """The function retries failed batches with exponential backoff."""

    class DummyError(RuntimeError):
        def __init__(self):
            super().__init__("boom")
            self.response = SimpleNamespace(status_code=500, headers={"Retry-After": "1"})
            self.request = SimpleNamespace(url="https://api", method="POST")

    attempts: list[int] = []

    class FlakyEmbeddings:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, model: str, input: list[str]):
            self.calls += 1
            attempts.append(self.calls)
            if self.calls == 1:
                raise DummyError()
            return _make_embedding_response([[1.0, 0.0] for _ in input])

    monkeypatch.setattr(categorize, "MAX_RETRIES", 3, raising=False)
    monkeypatch.setattr(categorize, "RETRY_DELAY", 0.01, raising=False)
    monkeypatch.setattr(categorize.time, "sleep", lambda s: None, raising=False)

    client = SimpleNamespace(embeddings=FlakyEmbeddings())
    arr = categorize.embed_chats_with_retry(client, ["only"], batch_size=1)
    assert arr.shape == (1, 2)
    assert attempts == [1, 2]


def test_embed_chats_with_retry_exhausts_and_raises(monkeypatch):
    """If all attempts fail the original exception is raised."""

    class FailingEmbeddings:
        def create(self, model: str, input: list[str]):
            raise RuntimeError("nope")

    monkeypatch.setattr(categorize, "MAX_RETRIES", 2, raising=False)
    monkeypatch.setattr(categorize, "RETRY_DELAY", 0.01, raising=False)
    monkeypatch.setattr(categorize.time, "sleep", lambda s: None, raising=False)

    client = SimpleNamespace(embeddings=FailingEmbeddings())
    with pytest.raises(RuntimeError):
        categorize.embed_chats_with_retry(client, ["fail"], batch_size=1)


def test_embed_chats_with_retry_detects_empty_embeddings(monkeypatch):
    """A zero-dimensional embedding array triggers a runtime error."""

    class EmptyEmbeddings:
        def create(self, model: str, input: list[str]):
            return _make_embedding_response([[] for _ in input])

    client = SimpleNamespace(embeddings=EmptyEmbeddings())
    with pytest.raises(RuntimeError):
        categorize.embed_chats_with_retry(client, ["anything"], batch_size=16)


def test_get_qdrant_client_with_timeout(monkeypatch):
    """The Qdrant client factory should forward configuration parameters."""

    constructed = {}

    class DummyQdrant:
        def __init__(self, **kwargs):
            constructed.update(kwargs)

    monkeypatch.setattr(categorize, "QdrantClient", DummyQdrant)
    monkeypatch.setattr(categorize, "QDRANT_URL", "http://qdrant", raising=False)
    monkeypatch.setattr(categorize, "QDRANT_API_KEY", "key", raising=False)
    monkeypatch.setattr(categorize, "QDRANT_TIMEOUT", 42, raising=False)

    client = categorize.get_qdrant_client_with_timeout()
    assert isinstance(client, DummyQdrant)
    assert constructed == {"url": "http://qdrant", "api_key": "key", "timeout": 42}


def test_ensure_qdrant_collection_already_exists():
    """No action is taken when the collection is already present."""

    class Client:
        def __init__(self) -> None:
            self.calls = 0

        def get_collection(self, name: str):
            self.calls += 1
            return {"name": name}

    client = Client()
    categorize.ensure_qdrant_collection(client, "col", 3)
    assert client.calls == 1


def test_ensure_qdrant_collection_creates_when_missing():
    """A missing collection should be created immediately."""

    events: list[str] = []

    class Client:
        def get_collection(self, name: str):
            raise Exception("404 Not found")

        def recreate_collection(self, **kwargs):
            events.append("created")

    categorize.ensure_qdrant_collection(Client(), "col", 3)
    assert events == ["created"]


def test_ensure_qdrant_collection_retry_create_fail(monkeypatch):
    """Creation errors should retry until the maximum is hit."""

    class Client:
        def __init__(self) -> None:
            self.calls = 0

        def get_collection(self, name: str):
            raise Exception("404")

        def recreate_collection(self, **kwargs):
            self.calls += 1
            raise RuntimeError("still missing")

    monkeypatch.setattr(categorize, "MAX_RETRIES", 2, raising=False)
    monkeypatch.setattr(categorize, "RETRY_DELAY", 0.01, raising=False)
    monkeypatch.setattr(categorize.time, "sleep", lambda s: None, raising=False)

    with pytest.raises(RuntimeError):
        categorize.ensure_qdrant_collection(Client(), "col", 3)


def test_ensure_qdrant_collection_other_error_retry(monkeypatch):
    """Non-404 errors retry and eventually succeed."""

    class Client:
        def __init__(self) -> None:
            self.calls = 0

        def get_collection(self, name: str):
            self.calls += 1
            if self.calls < 2:
                raise RuntimeError("temporary")
            return {"ok": True}

    monkeypatch.setattr(categorize, "MAX_RETRIES", 3, raising=False)
    monkeypatch.setattr(categorize, "RETRY_DELAY", 0.01, raising=False)
    monkeypatch.setattr(categorize.time, "sleep", lambda s: None, raising=False)

    categorize.ensure_qdrant_collection(Client(), "col", 3)


def test_ensure_qdrant_collection_other_error_exhaust(monkeypatch):
    """If the non-404 error persists it propagates on the last attempt."""

    class Client:
        def get_collection(self, name: str):
            raise RuntimeError("fatal")

    monkeypatch.setattr(categorize, "MAX_RETRIES", 2, raising=False)
    monkeypatch.setattr(categorize, "RETRY_DELAY", 0.01, raising=False)
    monkeypatch.setattr(categorize.time, "sleep", lambda s: None, raising=False)

    with pytest.raises(RuntimeError):
        categorize.ensure_qdrant_collection(Client(), "col", 3)


def test_upsert_to_qdrant_batched_with_no_points(capsys):
    """No upsert should occur when there are no chats."""

    categorize.upsert_to_qdrant_batched(SimpleNamespace(upsert=lambda **kwargs: None), "col", [], np.zeros((0, 3)))
    out = capsys.readouterr().out
    assert "No points" in out


def test_upsert_to_qdrant_batched_batches_and_succeeds(monkeypatch):
    """Batches are respected and sent to the client."""

    class Client:
        def __init__(self) -> None:
            self.calls: list[int] = []

        def upsert(self, collection_name: str, points):
            self.calls.append(len(list(points)))

    chats = [categorize.Chat(id=str(i), title=f"Chat {i}") for i in range(3)]
    vectors = np.array([[1.0, 0.0], [0.5, 0.5], [0.1, 0.2]], dtype=float)

    monkeypatch.setattr(categorize, "QDRANT_BATCH_SIZE", 2, raising=False)
    monkeypatch.setattr(categorize, "MAX_RETRIES", 2, raising=False)
    monkeypatch.setattr(categorize, "RETRY_DELAY", 0.01, raising=False)
    monkeypatch.setattr(categorize.time, "sleep", lambda s: None, raising=False)

    client = Client()
    categorize.upsert_to_qdrant_batched(client, "col", chats, vectors)
    assert client.calls == [2, 1]


def test_upsert_to_qdrant_batched_retries_then_succeeds(monkeypatch):
    """Temporary upsert errors should retry before succeeding."""

    class Client:
        def __init__(self) -> None:
            self.calls = 0
            self.record: list[str] = []

        def upsert(self, collection_name: str, points):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("retry")
            self.record.append(collection_name)

    chats = [categorize.Chat(id="1", title="A"), categorize.Chat(id="2", title="B")]
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)

    monkeypatch.setattr(categorize, "MAX_RETRIES", 2, raising=False)
    monkeypatch.setattr(categorize, "RETRY_DELAY", 0.01, raising=False)
    monkeypatch.setattr(categorize.time, "sleep", lambda s: None, raising=False)

    client = Client()
    categorize.upsert_to_qdrant_batched(client, "col", chats, vectors)
    assert client.record == ["col"]


def test_upsert_to_qdrant_batched_exhausts(monkeypatch):
    """Persistent upsert failures should propagate."""

    class Client:
        def upsert(self, collection_name: str, points):
            raise RuntimeError("fail")

    chats = [categorize.Chat(id="1", title="A"), categorize.Chat(id="2", title="B")]
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)

    monkeypatch.setattr(categorize, "MAX_RETRIES", 2, raising=False)
    monkeypatch.setattr(categorize, "RETRY_DELAY", 0.01, raising=False)
    monkeypatch.setattr(categorize.time, "sleep", lambda s: None, raising=False)

    with pytest.raises(RuntimeError):
        categorize.upsert_to_qdrant_batched(Client(), "col", chats, vectors)


def test_upsert_to_qdrant_alias(monkeypatch):
    """Legacy wrapper should delegate to the batched implementation."""

    called: list[tuple] = []

    def fake_batched(client, name, chats, vectors):
        called.append((client, name, chats, vectors))

    monkeypatch.setattr(categorize, "upsert_to_qdrant_batched", fake_batched)
    marker = object()
    categorize.upsert_to_qdrant(marker, "col", [], np.zeros((0, 3)))
    assert called and called[0][0] is marker
