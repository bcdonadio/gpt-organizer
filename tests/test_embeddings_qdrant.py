"""Tests covering embedding retries and Qdrant persistence helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Iterator, Sequence, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray
import pytest

import GptCategorize.categorize as categorize

CREATED_PROGRESS: list[Any] = []

FloatArray: TypeAlias = NDArray[np.float64]


@pytest.fixture(autouse=True)
def patch_tqdm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid noisy tqdm output during tests."""

    CREATED_PROGRESS.clear()

    class DummyProgress:
        def __init__(self, iterable: Iterable[Any] | None = None, **kwargs: object) -> None:
            self._iterable = iterable
            self.total: object | None = kwargs.get("total")
            self.count = 0
            self.closed = False

        def update(self, amount: int = 1) -> None:
            self.count += amount

        def close(self) -> None:  # pragma: no cover - nothing to clean up
            self.closed = True

        def __iter__(self) -> Iterator[Any]:
            if self._iterable is None:
                return iter(())
            return iter(self._iterable)

    def fake_tqdm(iterable: Iterable[Any] | None = None, **kwargs: object) -> DummyProgress:
        prog = DummyProgress(iterable, **kwargs)
        CREATED_PROGRESS.append(prog)
        return prog

    monkeypatch.setattr(categorize, "tqdm", fake_tqdm, raising=False)


def _make_embedding_response(vectors: Sequence[Sequence[float]]) -> SimpleNamespace:
    return SimpleNamespace(data=[SimpleNamespace(embedding=list(v)) for v in vectors])


def test_categorize_chats_reports_progress(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Progress bars should count chats as they are embedded and upserted."""

    chats = [
        categorize.Chat(id="chat-a", title="Alpha", prompt_excerpt="First message"),
        categorize.Chat(id="chat-b", title="Beta", prompt_excerpt="Second message"),
    ]

    monkeypatch.setattr(categorize, "load_chats_from_conversations_json", lambda _path: list(chats))

    def fake_embed(_: Any, texts: Sequence[str], batch_size: int = 96) -> np.ndarray:
        assert len(texts) == len(chats)
        return np.ones((len(texts), 2), dtype=np.float32)

    def fake_cluster(vectors: np.ndarray, *, eps_cosine: float, min_samples: int, min_cluster_size: int) -> np.ndarray:
        assert vectors.shape[0] == len(chats)
        return np.zeros(vectors.shape[0], dtype=int)

    def fake_label(_: Any, clusters: dict[int, list[categorize.Chat]]) -> dict[int, dict[str, object]]:
        return {
            0: {
                "label": "Alpha Beta",
                "project_folder_slug": "alpha-beta",
                "project_title": "Alpha Beta",
                "confidence_model": 0.9,
            }
        }

    monkeypatch.setattr(categorize, "get_embedding_client", lambda: SimpleNamespace())
    monkeypatch.setattr(categorize, "embed_chats_with_retry", fake_embed)
    monkeypatch.setattr(categorize, "cluster_embeddings", fake_cluster)
    monkeypatch.setattr(categorize, "cluster_text_cohesion", lambda *_args, **_kwargs: 0.8)
    monkeypatch.setattr(categorize, "temporal_cohesion", lambda *_args, **_kwargs: 0.7)
    monkeypatch.setattr(categorize, "get_inference_client", lambda: SimpleNamespace())
    monkeypatch.setattr(categorize, "label_clusters_with_llm", fake_label)

    out_path = tmp_path / "plan.json"
    code = categorize.categorize_chats("ignored.json", out=str(out_path), no_qdrant=True)
    assert code == 0

    matching = [prog for prog in CREATED_PROGRESS if prog.total == len(chats)]
    assert matching, "Expected a progress bar sized to chat count"
    tracker = matching[0]
    assert tracker.count == len(chats)
    assert tracker.closed is True


def test_embed_chats_with_retry_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Embeddings should be gathered across batches when the API succeeds."""

    calls: list[list[str]] = []

    class DummyEmbeddings:
        def create(self, model: str, input: list[str]) -> SimpleNamespace:
            calls.append(list(input))
            return _make_embedding_response([[float(len(input)), 0.5] for _ in input])

    client = SimpleNamespace(embeddings=DummyEmbeddings())
    arr = categorize.embed_chats_with_retry(cast(Any, client), ["chat-1", "chat-2"], batch_size=1)
    assert arr.shape == (2, 2)
    assert calls == [["chat-1"], ["chat-2"]]


def test_embed_chats_with_retry_handles_transient_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """The function retries failed batches with exponential backoff."""

    class DummyError(RuntimeError):
        def __init__(self) -> None:
            super().__init__("boom")
            self.response = SimpleNamespace(status_code=500, headers={"Retry-After": "1"})
            self.request = SimpleNamespace(url="https://api", method="POST")

    attempts: list[int] = []

    class FlakyEmbeddings:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, model: str, input: list[str]) -> SimpleNamespace:
            self.calls += 1
            attempts.append(self.calls)
            if self.calls == 1:
                raise DummyError()
            return _make_embedding_response([[1.0, 0.0] for _ in input])

    def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(categorize, "MAX_RETRIES", 3, raising=False)
    monkeypatch.setattr(categorize, "RETRY_DELAY", 0.01, raising=False)
    monkeypatch.setattr(cast(Any, categorize).time, "sleep", no_sleep, raising=False)

    client = SimpleNamespace(embeddings=FlakyEmbeddings())
    arr = categorize.embed_chats_with_retry(cast(Any, client), ["only"], batch_size=1)
    assert arr.shape == (1, 2)
    assert attempts == [1, 2]


def test_embed_chats_with_retry_exhausts_and_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """If all attempts fail the original exception is raised."""

    class FailingEmbeddings:
        def create(self, model: str, input: list[str]) -> SimpleNamespace:
            raise RuntimeError("nope")

    def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(categorize, "MAX_RETRIES", 2, raising=False)
    monkeypatch.setattr(categorize, "RETRY_DELAY", 0.01, raising=False)
    monkeypatch.setattr(cast(Any, categorize).time, "sleep", no_sleep, raising=False)

    client = SimpleNamespace(embeddings=FailingEmbeddings())
    with pytest.raises(RuntimeError):
        categorize.embed_chats_with_retry(cast(Any, client), ["fail"], batch_size=1)


def test_embed_chats_with_retry_detects_empty_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    """A zero-dimensional embedding array triggers a runtime error."""

    class EmptyEmbeddings:
        def create(self, model: str, input: list[str]) -> SimpleNamespace:
            return _make_embedding_response([[] for _ in input])

    client = SimpleNamespace(embeddings=EmptyEmbeddings())
    with pytest.raises(RuntimeError):
        categorize.embed_chats_with_retry(cast(Any, client), ["anything"], batch_size=16)


def test_estimate_word_count_and_batch_builder() -> None:
    """Word counting and batch construction respect the configured limits."""

    texts = ["one two three", "four", "   ", "five six seven eight nine", "ten"]
    batches = categorize.build_embedding_batches(texts, words_per_batch=4)

    assert batches == [[0, 1], [2], [3], [4]]
    assert categorize.estimate_word_count("just two words") == 3
    assert categorize.estimate_word_count("   ") == 0


def test_build_embedding_batches_rejects_invalid_budget() -> None:
    """A non-positive word budget is rejected early."""

    with pytest.raises(ValueError):
        categorize.build_embedding_batches(["text"], words_per_batch=0)


def test_fetch_existing_embeddings_from_qdrant_returns_vectors() -> None:
    """Cached embeddings should be returned as float32 arrays."""

    calls: list[list[str]] = []

    class Client:
        def retrieve(
            self, collection_name: str, ids: Sequence[str], with_payload: bool, with_vectors: bool
        ) -> list[SimpleNamespace]:
            calls.append(list(ids))
            returned: list[SimpleNamespace] = []
            for cid in ids:
                if cid.endswith("dict"):
                    returned.append(SimpleNamespace(id=cid, vectors={"default": [0.5, 0.5]}))
                else:
                    returned.append(SimpleNamespace(id=cid, vector=[1.0, 0.0]))
            return returned

    result = categorize.fetch_existing_embeddings_from_qdrant(
        cast(Any, Client()),
        "col",
        ["a", "b-dict"],
        batch_size=1,
    )

    assert calls == [["a"], ["b-dict"]]
    assert set(result.keys()) == {"a", "b-dict"}
    assert all(isinstance(vec, np.ndarray) and vec.dtype == np.float32 for vec in result.values())


def test_fetch_existing_embeddings_from_qdrant_skips_missing_vectors() -> None:
    """Entries without vectors should be ignored."""

    class Client:
        def retrieve(
            self, collection_name: str, ids: Sequence[str], with_payload: bool, with_vectors: bool
        ) -> list[SimpleNamespace]:
            return [
                SimpleNamespace(id="a", vector=None),
                SimpleNamespace(id="b", vectors={}),
            ]

    result = categorize.fetch_existing_embeddings_from_qdrant(cast(Any, Client()), "col", ["a", "b"])
    assert result == {}


def test_fetch_existing_embeddings_from_qdrant_propagates_errors() -> None:
    """Retrieval failures should bubble up to the caller."""

    class Client:
        def retrieve(
            self, collection_name: str, ids: Sequence[str], with_payload: bool, with_vectors: bool
        ) -> list[SimpleNamespace]:
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        categorize.fetch_existing_embeddings_from_qdrant(cast(Any, Client()), "col", ["only"], batch_size=2)


def test_fetch_existing_embeddings_from_qdrant_handles_empty_ids() -> None:
    """No retrieval calls should happen when there are no chat IDs."""

    class Client:
        def retrieve(self, *args: object, **kwargs: object) -> list[SimpleNamespace]:
            raise AssertionError("retrieve should not be called")

    result = categorize.fetch_existing_embeddings_from_qdrant(cast(Any, Client()), "col", [])
    assert result == {}


def test_get_qdrant_client_with_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Qdrant client factory should forward configuration parameters."""

    constructed: dict[str, object] = {}

    class DummyQdrant:
        def __init__(self, **kwargs: object) -> None:
            constructed.update(kwargs)

    monkeypatch.setattr(categorize, "QdrantClient", DummyQdrant)
    monkeypatch.setattr(categorize, "QDRANT_URL", "http://qdrant", raising=False)
    monkeypatch.setattr(categorize, "QDRANT_API_KEY", "key", raising=False)
    monkeypatch.setattr(categorize, "QDRANT_TIMEOUT", 42, raising=False)

    client = categorize.get_qdrant_client_with_timeout()
    assert isinstance(client, DummyQdrant)
    assert constructed == {"url": "http://qdrant", "api_key": "key", "timeout": 42}


def test_ensure_qdrant_collection_already_exists() -> None:
    """No action is taken when the collection is already present."""

    class Client:
        def __init__(self) -> None:
            self.calls = 0

        def get_collection(self, name: str) -> dict[str, str]:
            self.calls += 1
            return {"name": name}

    client = Client()
    categorize.ensure_qdrant_collection(cast(Any, client), "col", 3)
    assert client.calls == 1


def test_ensure_qdrant_collection_creates_when_missing() -> None:
    """A missing collection should be created immediately."""

    events: list[str] = []

    class Client:
        def __init__(self) -> None:
            self.calls = 0

        def get_collection(self, name: str) -> dict[str, object]:
            self.calls += 1
            raise Exception("404")

        def recreate_collection(self, **kwargs: object) -> None:
            events.append("created")

    categorize.ensure_qdrant_collection(cast(Any, Client()), "col", 3)
    assert events == ["created"]


def test_ensure_qdrant_collection_retry_create_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Creation errors should retry until the maximum is hit."""

    class Client:
        def __init__(self) -> None:
            self.calls = 0

        def get_collection(self, name: str) -> dict[str, object]:
            raise Exception("404")

        def recreate_collection(self, **kwargs: object) -> None:
            self.calls += 1
            raise RuntimeError("still missing")

    def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(categorize, "MAX_RETRIES", 2, raising=False)
    monkeypatch.setattr(categorize, "RETRY_DELAY", 0.01, raising=False)
    monkeypatch.setattr(cast(Any, categorize).time, "sleep", no_sleep, raising=False)

    with pytest.raises(RuntimeError):
        categorize.ensure_qdrant_collection(cast(Any, Client()), "col", 3)


def test_ensure_qdrant_collection_other_error_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-404 errors retry and eventually succeed."""

    class Client:
        def __init__(self) -> None:
            self.calls = 0

        def get_collection(self, name: str) -> dict[str, bool]:
            self.calls += 1
            if self.calls < 2:
                raise RuntimeError("temporary")
            return {"ok": True}

    def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(categorize, "MAX_RETRIES", 3, raising=False)
    monkeypatch.setattr(categorize, "RETRY_DELAY", 0.01, raising=False)
    monkeypatch.setattr(cast(Any, categorize).time, "sleep", no_sleep, raising=False)

    categorize.ensure_qdrant_collection(cast(Any, Client()), "col", 3)


def test_ensure_qdrant_collection_other_error_exhaust(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the non-404 error persists it propagates on the last attempt."""

    class Client:
        def get_collection(self, name: str) -> dict[str, object]:
            raise RuntimeError("fatal")

    def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(categorize, "MAX_RETRIES", 2, raising=False)
    monkeypatch.setattr(categorize, "RETRY_DELAY", 0.01, raising=False)
    monkeypatch.setattr(cast(Any, categorize).time, "sleep", no_sleep, raising=False)

    with pytest.raises(RuntimeError):
        categorize.ensure_qdrant_collection(cast(Any, Client()), "col", 3)


def test_upsert_to_qdrant_batched_with_no_points(capsys: pytest.CaptureFixture[str]) -> None:
    """No upsert should occur when there are no chats."""

    def no_upsert(**_: object) -> None:
        return None

    categorize.upsert_to_qdrant_batched(cast(Any, SimpleNamespace(upsert=no_upsert)), "col", [], np.zeros((0, 3)))
    out = capsys.readouterr().out
    assert "No points" in out


def test_upsert_to_qdrant_batched_batches_and_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Batches are respected and sent to the client."""

    class Client:
        def __init__(self) -> None:
            self.calls: list[int] = []

        def upsert(self, collection_name: str, points: Iterable[SimpleNamespace]) -> None:
            self.calls.append(len(list(points)))

    chats = [categorize.Chat(id=str(i), title=f"Chat {i}") for i in range(3)]
    vectors = np.array([[1.0, 0.0], [0.5, 0.5], [0.1, 0.2]], dtype=float)

    def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(categorize, "QDRANT_BATCH_SIZE", 2, raising=False)
    monkeypatch.setattr(categorize, "MAX_RETRIES", 2, raising=False)
    monkeypatch.setattr(categorize, "RETRY_DELAY", 0.01, raising=False)
    monkeypatch.setattr(cast(Any, categorize).time, "sleep", no_sleep, raising=False)

    client = Client()
    categorize.upsert_to_qdrant_batched(cast(Any, client), "col", chats, vectors)
    assert client.calls == [2, 1]


def test_upsert_to_qdrant_batched_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Temporary upsert errors should retry before succeeding."""

    class Client:
        def __init__(self) -> None:
            self.calls = 0
            self.record: list[str] = []

        def upsert(self, collection_name: str, points: Iterable[SimpleNamespace]) -> None:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("retry")
            self.record.append(collection_name)

    chats = [categorize.Chat(id="1", title="A"), categorize.Chat(id="2", title="B")]
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)

    def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(categorize, "MAX_RETRIES", 2, raising=False)
    monkeypatch.setattr(categorize, "RETRY_DELAY", 0.01, raising=False)
    monkeypatch.setattr(cast(Any, categorize).time, "sleep", no_sleep, raising=False)

    client = Client()
    categorize.upsert_to_qdrant_batched(cast(Any, client), "col", chats, vectors)
    assert client.record == ["col"]


def test_upsert_to_qdrant_batched_exhausts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Persistent upsert failures should propagate."""

    class Client:
        def upsert(self, collection_name: str, points: Iterable[SimpleNamespace]) -> None:
            raise RuntimeError("fail")

    chats = [categorize.Chat(id="1", title="A"), categorize.Chat(id="2", title="B")]
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)

    def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(categorize, "MAX_RETRIES", 2, raising=False)
    monkeypatch.setattr(categorize, "RETRY_DELAY", 0.01, raising=False)
    monkeypatch.setattr(cast(Any, categorize).time, "sleep", no_sleep, raising=False)

    with pytest.raises(RuntimeError):
        categorize.upsert_to_qdrant_batched(cast(Any, Client()), "col", chats, vectors)


def test_upsert_to_qdrant_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    """Legacy wrapper should delegate to the batched implementation."""

    called: list[tuple[object, str, list[categorize.Chat], FloatArray]] = []

    def fake_batched(client: object, name: str, chats: list[categorize.Chat], vectors: FloatArray) -> None:
        called.append((client, name, chats, vectors))

    monkeypatch.setattr(categorize, "upsert_to_qdrant_batched", fake_batched)
    marker = object()
    categorize.upsert_to_qdrant(cast(Any, marker), "col", [], np.zeros((0, 3)))
    assert called and called[0][0] is marker
